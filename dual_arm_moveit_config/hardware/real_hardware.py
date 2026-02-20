#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty, SetBool
from xarm.wrapper import XArmAPI
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import math
import threading
import sys

# ================= CONFIGURATION =================
XARM_IP = '192.168.1.239'     
UF850_IP = '192.168.1.195'     
GRIPPER_IP = '192.168.1.1'    

RAD_CLOSE = -0.6109; RAD_OPEN = 0.6109; RAD_RANGE = RAD_OPEN - RAD_CLOSE
MM_CLOSE = 0.0; MM_OPEN = 160.0      

class RG:
    def __init__(self, gripper, ip, logger, port=502):
        self.logger = logger; self.ip = ip
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.lock = threading.Lock(); self.gripper = gripper
        if self.gripper == 'rg2': self.max_width=1100; self.max_force=400
        elif self.gripper == 'rg6': self.max_width=1600; self.max_force=1200
        if not self.client.connect(): pass 
    def move_gripper(self, width_mm, force_val=1200):
        val = int(width_mm * 10); val = max(0, min(val, self.max_width))
        with self.lock: 
            try: self.client.write_registers(address=0, values=[force_val, val, 16], unit=65)
            except: pass
    def get_width(self):
        with self.lock: 
            try:
                res = self.client.read_holding_registers(address=267, count=1, unit=65)
                if res and not res.isError(): return (res.registers[0] / 10.0) 
            except: pass
        return 0.0 
    def close_connection(self):
        with self.lock: self.client.close()

class RealRobotInterface:
    def __init__(self, ip, name, dof, logger, has_linear_track=False):
        self.ip=ip; self.name=name; self.dof=dof; self.logger=logger; self.has_linear_track=has_linear_track
        self.arm = XArmAPI(self.ip); self.connected=False
        
        # --- STABILITY PARAMETERS ---
        self.prev_pos = [0.0] * self.dof
        self.prev_time = time.time()
        
        # Velocity Smoothing: Moving Average Window (Last 5 samples)
        self.vel_window_size = 5 
        self.vel_history = [[0.0] * self.dof for _ in range(self.vel_window_size)]
        
        # Torque Smoothing & Deadband
        self.prev_eff = [0.0] * self.dof
        self.torque_deadband = 0.08  # Nm: Ignore fluctuations below this
        self.eff_alpha = 0.15         # Filter: Lower is smoother/slower
        
        self.connect()
    
    def connect(self):
        try:
            self.arm.connect(); self.arm.motion_enable(enable=True); self.arm.clean_error()
            self.arm.set_mode(1); self.arm.set_state(0)
            self.arm.set_report_tau_or_i(0) # 0 = Nm (Torque)
            if self.has_linear_track: self._init_linear_track()
            self.connected=True
        except Exception as e: raise e

    def _init_linear_track(self):
        self.arm.set_linear_track_back_origin(wait=True)
        self.arm.set_linear_track_enable(True); self.arm.set_linear_track_speed(200)

    def force_enable(self):
        if self.connected:
            self.arm.clean_error(); self.arm.motion_enable(enable=True); self.arm.set_mode(1); self.arm.set_state(0)

    def stop(self):
        if self.connected: self.arm.set_state(4)

    def get_full_state(self):
        if not self.connected: return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)
        try:
            now = time.time()
            dt = now - self.prev_time
            code_p, pos = self.arm.get_servo_angle(is_radian=True)
            code_t, effort = self.arm.get_joints_torque()
            
            if code_p == 0 and code_t == 0 and pos:
                curr_pos = pos[:self.dof]
                raw_eff = effort[:self.dof]
                
                # --- 1. STABLE VELOCITY (Moving Average) ---
                curr_vel = [0.0] * self.dof
                if dt > 0.001:
                    inst_vel = [(curr_pos[i] - self.prev_pos[i]) / dt for i in range(self.dof)]
                    self.vel_history.pop(0)
                    self.vel_history.append(inst_vel)
                    for i in range(self.dof):
                        curr_vel[i] = sum(h[i] for h in self.vel_history) / self.vel_window_size
                
                # --- 2. STABLE EFFORT (Deadband + Exponential Filter) ---
                stable_eff = []
                for i in range(self.dof):
                    # Only update if the change is significant (Deadband)
                    if abs(raw_eff[i] - self.prev_eff[i]) < self.torque_deadband:
                        val = self.prev_eff[i]
                    else:
                        # Smooth the transition (LPF)
                        val = (self.eff_alpha * raw_eff[i]) + ((1 - self.eff_alpha) * self.prev_eff[i])
                    stable_eff.append(val)

                self.prev_pos = curr_pos
                self.prev_eff = stable_eff
                self.prev_time = now
                return (curr_pos, curr_vel, stable_eff)
        except: pass
        return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)

    def get_linear_track_pos(self):
        if not self.connected or not self.has_linear_track: return 0.0
        c, p = self.arm.get_linear_track_pos()
        return -1.0 * (p / 1000.0) if c==0 and p else 0.0

    def set_servo_angle(self, angles):
        if self.connected: self.arm.set_servo_angle_j(angles=angles, is_radian=True)

    def set_linear_track(self, pos_meters):
        if self.connected and self.has_linear_track: self.arm.set_linear_track_pos(abs(pos_meters)*1000.0, wait=False)

    def disconnect(self):
        self.connected=False; self.arm.disconnect()

class RealHardware(Node):
    def __init__(self):
        super().__init__('real_hardware_driver')
        self.cb_group = ReentrantCallbackGroup()
        self.xarm = None; self.uf850 = None; self.gripper = None
        self.stop_current_motion = False 
        self.velocity_mode_active = False

        try:
            self.xarm = RealRobotInterface(XARM_IP, "xArm5", 5, self.get_logger(), has_linear_track=True)
            self.uf850 = RealRobotInterface(UF850_IP, "UF850", 6, self.get_logger())
            self.gripper = RG('rg6', GRIPPER_IP, self.get_logger())
        except Exception as e: sys.exit(1)

        # Services & Topics
        self._stop_service = self.create_service(Empty, '/xarm/stop_robot', self.handle_stop_request, callback_group=self.cb_group)
        self._reset_service = self.create_service(Empty, '/xarm/reset_robot', self.handle_reset_request, callback_group=self.cb_group)
        self._vel_mode_srv = self.create_service(SetBool, '/xarm/set_velocity_mode', self.handle_velocity_mode_request, callback_group=self.cb_group)
        self._vel_sub = self.create_subscription(Twist, '/xarm/velo_cmd', self.handle_velocity_command, 10)

        # Action Servers
        self._xarm_server = ActionServer(self, FollowJointTrajectory, '/xarm_controller/follow_joint_trajectory', execute_callback=self.execute_xarm_callback, callback_group=self.cb_group)
        self._uf_server = ActionServer(self, FollowJointTrajectory, '/uf_controller/follow_joint_trajectory', execute_callback=self.execute_uf_callback, callback_group=self.cb_group)
        self._rg6_server = ActionServer(self, FollowJointTrajectory, '/rg6_controller/follow_joint_trajectory', execute_callback=self.execute_gripper_callback, callback_group=self.cb_group)
        self._slider_server = ActionServer(self, FollowJointTrajectory, '/slider_controller/follow_joint_trajectory', execute_callback=self.execute_slider_callback, callback_group=self.cb_group)

        self.publisher_ = self.create_publisher(JointState, '/joint_states', 50)
        self.timer = self.create_timer(0.02, self.publish_real_states, callback_group=self.cb_group)
        self.loop_count = 0; self.cached_gripper_width = 0.0

    def handle_velocity_mode_request(self, request, response):
        if request.data: 
            self.xarm.arm.set_state(4); time.sleep(0.1)
            self.xarm.arm.set_mode(5); self.xarm.arm.set_state(0)
            self.velocity_mode_active = True
        else:
            self.xarm.arm.set_mode(1); self.xarm.arm.set_state(0)
            self.velocity_mode_active = False
        response.success = True
        return response

    def handle_velocity_command(self, msg):
        if self.velocity_mode_active and self.xarm and self.xarm.connected:
            vx, vy, vz = msg.linear.x * 1000.0, msg.linear.y * 1000.0, msg.linear.z * 1000.0
            v_yaw = math.degrees(msg.angular.z) 
            self.xarm.arm.vc_set_cartesian_velocity([vx, vy, vz, 0.0, 0.0, v_yaw])

    def handle_stop_request(self, request, response):
        self.stop_current_motion = True
        if self.xarm: self.xarm.stop()
        if self.uf850: self.uf850.stop()
        return response

    def handle_reset_request(self, request, response):
        self.stop_current_motion = False
        if self.xarm: self.xarm.force_enable()
        if self.uf850: self.uf850.force_enable()
        return response

    def execute_trajectory(self, goal_handle, robot_obj, joint_map_indices):
        if not robot_obj or not robot_obj.connected:
            goal_handle.abort(); return FollowJointTrajectory.Result()
        self.stop_current_motion = False
        if self.velocity_mode_active:
             robot_obj.arm.set_mode(1); robot_obj.arm.set_state(0); self.velocity_mode_active = False
        else: robot_obj.force_enable()

        traj = goal_handle.request.trajectory
        points = traj.points
        for i in range(len(points) - 1):
            if self.stop_current_motion or goal_handle.is_cancel_requested:
                goal_handle.canceled(); robot_obj.stop(); return FollowJointTrajectory.Result()
            start_pt, end_pt = points[i], points[i+1]
            duration = (end_pt.time_from_start.sec + end_pt.time_from_start.nanosec*1e-9) - (start_pt.time_from_start.sec + start_pt.time_from_start.nanosec*1e-9)
            if duration <= 0: continue
            step = 0.02; t = 0.0
            while t < duration:
                if self.stop_current_motion or goal_handle.is_cancel_requested:
                    goal_handle.canceled(); robot_obj.stop(); return FollowJointTrajectory.Result()
                fract = t / duration
                cmd_angles = [self.linear_interpolate(start_pt.positions[idx], end_pt.positions[idx], fract) for idx in joint_map_indices]
                robot_obj.set_servo_angle(cmd_angles)
                time.sleep(step); t += step

        if not self.stop_current_motion:
            robot_obj.set_servo_angle([points[-1].positions[idx] for idx in joint_map_indices])
            goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def rad_to_mm(self, radians): return max(0.0, min(MM_OPEN, ((radians - RAD_CLOSE)/RAD_RANGE)*MM_OPEN))
    def mm_to_rad(self, mm): return ((mm/MM_OPEN)*RAD_RANGE)+RAD_CLOSE
    def linear_interpolate(self, start, end, fract): return start + (end - start) * fract

    def execute_xarm_callback(self, goal_handle): return self.execute_trajectory(goal_handle, self.xarm, [0,1,2,3,4])
    def execute_uf_callback(self, goal_handle): return self.execute_trajectory(goal_handle, self.uf850, [0,1,2,3,4,5])
    def execute_gripper_callback(self, goal_handle): 
        if self.gripper: self.gripper.move_gripper(self.rad_to_mm(goal_handle.request.trajectory.points[-1].positions[0]))
        goal_handle.succeed(); return FollowJointTrajectory.Result()
    def execute_slider_callback(self, goal_handle):
        if self.xarm: self.xarm.set_linear_track(goal_handle.request.trajectory.points[-1].positions[0])
        goal_handle.succeed(); return FollowJointTrajectory.Result()

    def publish_real_states(self):
        msg = JointState(); msg.header.stamp = self.get_clock().now().to_msg()
        
        # Fetch Position, Velocity, and Effort for both robots
        x_p, x_v, x_e = self.xarm.get_full_state() if self.xarm else ([0.0]*5, [0.0]*5, [0.0]*5)
        u_p, u_v, u_e = self.uf850.get_full_state() if self.uf850 else ([0.0]*6, [0.0]*6, [0.0]*6)
        
        self.loop_count += 1
        if self.gripper and self.loop_count >= 10:
            try: self.cached_gripper_width = self.gripper.get_width()
            except: pass
            self.loop_count = 0
            
        g = self.mm_to_rad(self.cached_gripper_width)
        s = self.xarm.get_linear_track_pos() if self.xarm else 0.0
        
        msg.name = [
            'xarm5_joint1', 'xarm5_joint2', 'xarm5_joint3', 'xarm5_joint4', 'xarm5_joint5', 
            'u1_joint1', 'u1_joint2', 'u1_joint3', 'u1_joint4', 'u1_joint5', 'u1_joint6', 
            'slider_slider_joint', 'rg6_l_out', 'rg6_r_out', 'rg6_l_tip', 'rg6_r_tip', 
            'rg6_l_passive', 'rg6_r_passive'
        ]
        
        # Map all data to message arrays
        msg.position = x_p + u_p + [s] + [g, -g, g, -g, -g, -g]
        msg.velocity = x_v + u_v + [0.0] + [0.0]*6
        msg.effort = x_e + u_e + [0.0] + [0.0]*6
        
        self.publisher_.publish(msg)

    def destroy_node(self):
        if self.xarm: self.xarm.disconnect()
        if self.uf850: self.uf850.disconnect()
        if self.gripper: self.gripper.close_connection()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args); executor = MultiThreadedExecutor(); node = RealHardware()
    try: executor.add_node(node); executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()