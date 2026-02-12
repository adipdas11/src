#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist  # <--- NEW: For Velocity Control
from std_srvs.srv import Empty, Trigger, SetBool # <--- NEW: For Mode Switching
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
    def move_gripper(self, width_mm, force_val=400):
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
        self.arm = XArmAPI(self.ip); self.connected=False; self.connect()
    
    def connect(self):
        try:
            self.arm.connect(); self.arm.motion_enable(enable=True); self.arm.clean_error()
            self.arm.set_mode(1); self.arm.set_state(0); self.arm.set_report_tau_or_i(0) 
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
            c, a = self.arm.get_servo_angle(is_radian=True)
            if c!=0 or not a: return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)
            return (a[:self.dof], [0.0]*self.dof, [0.0]*self.dof)
        except: return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)
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
        self.velocity_mode_active = False # Flag for velocity mode

        try:
            self.xarm = RealRobotInterface(XARM_IP, "xArm5", 5, self.get_logger(), has_linear_track=True)
            self.uf850 = RealRobotInterface(UF850_IP, "UF850", 6, self.get_logger())
            self.gripper = RG('rg6', GRIPPER_IP, self.get_logger())
        except Exception as e: sys.exit(1)

        # --- SERVICES ---
        self._stop_service = self.create_service(Empty, '/xarm/stop_robot', self.handle_stop_request, callback_group=self.cb_group)
        self._reset_service = self.create_service(Empty, '/xarm/reset_robot', self.handle_reset_request, callback_group=self.cb_group)
        
        # NEW: Velocity Mode Switch
        self._vel_mode_srv = self.create_service(SetBool, '/xarm/set_velocity_mode', self.handle_velocity_mode_request, callback_group=self.cb_group)

        # --- SUBSCRIBERS ---
        # NEW: Velocity Command Topic
        self._vel_sub = self.create_subscription(Twist, '/xarm/velo_cmd', self.handle_velocity_command, 10)

        # --- ACTIONS ---
        self._xarm_server = ActionServer(self, FollowJointTrajectory, '/xarm_controller/follow_joint_trajectory', execute_callback=self.execute_xarm_callback, callback_group=self.cb_group)
        self._uf_server = ActionServer(self, FollowJointTrajectory, '/uf_controller/follow_joint_trajectory', execute_callback=self.execute_uf_callback, callback_group=self.cb_group)
        self._rg6_server = ActionServer(self, FollowJointTrajectory, '/rg6_controller/follow_joint_trajectory', execute_callback=self.execute_gripper_callback, callback_group=self.cb_group)
        self._slider_server = ActionServer(self, FollowJointTrajectory, '/slider_controller/follow_joint_trajectory', execute_callback=self.execute_slider_callback, callback_group=self.cb_group)

        self.publisher_ = self.create_publisher(JointState, '/joint_states', 50)
        self.timer = self.create_timer(0.02, self.publish_real_states, callback_group=self.cb_group)
        self.loop_count = 0; self.cached_gripper_width = 0.0

    def handle_velocity_mode_request(self, request, response):
        if request.data: 
            self.get_logger().info("🔄 Switching xArm to CARTESIAN VELOCITY MODE (5)...")
            self.xarm.arm.set_state(4)  # Stop 
            time.sleep(0.1)
            self.xarm.arm.set_mode(5)   # Mode 5 = Cartesian Velocity
            self.xarm.arm.set_state(0)  # Ready
            self.velocity_mode_active = True
            response.success = True
        else:
            self.get_logger().info("🛑 Disabling Velocity Mode (Back to Mode 1)...")
            self.xarm.arm.set_mode(1)
            self.xarm.arm.set_state(0)
            self.velocity_mode_active = False
            response.success = True
        return response

    def handle_velocity_command(self, msg):
        if self.velocity_mode_active and self.xarm and self.xarm.connected:
            # Linear: m/s -> mm/s
            vx = msg.linear.x * 1000.0  
            vy = msg.linear.y * 1000.0  
            vz = msg.linear.z * 1000.0  
            
            # Angular: rad/s -> deg/s (Maps to Joint 5 on xArm5)
            v_yaw = math.degrees(msg.angular.z) 

            # vc_set_cartesian_velocity([x, y, z, roll, pitch, yaw])
            self.xarm.arm.vc_set_cartesian_velocity([vx, vy, vz, 0.0, 0.0, v_yaw])

    # --- EXISTING HANDLERS ---
    def handle_stop_request(self, request, response):
        self.get_logger().error("🚨 STOP REQUEST")
        self.stop_current_motion = True
        if self.xarm: self.xarm.stop()
        if self.uf850: self.uf850.stop()
        return response

    def handle_reset_request(self, request, response):
        self.get_logger().info("♻️ RESET REQUEST")
        self.stop_current_motion = False
        if self.xarm: self.xarm.force_enable()
        if self.uf850: self.uf850.force_enable()
        return response

    def execute_trajectory(self, goal_handle, robot_obj, joint_map_indices):
        if not robot_obj or not robot_obj.connected:
            goal_handle.abort(); return FollowJointTrajectory.Result()

        self.stop_current_motion = False
        # Ensure we are in Position Mode (1) before trajectory execution
        if self.velocity_mode_active:
             self.get_logger().warn("⚠️ Auto-switching back to Position Mode for Trajectory")
             robot_obj.arm.set_mode(1); robot_obj.arm.set_state(0)
             self.velocity_mode_active = False
        else:
             robot_obj.force_enable()

        traj = goal_handle.request.trajectory
        points = traj.points
        
        for i in range(len(points) - 1):
            if self.stop_current_motion or goal_handle.is_cancel_requested:
                goal_handle.canceled(); robot_obj.stop(); return FollowJointTrajectory.Result()

            start_pt = points[i]; end_pt = points[i+1]
            duration = (end_pt.time_from_start.sec + end_pt.time_from_start.nanosec*1e-9) - (start_pt.time_from_start.sec + start_pt.time_from_start.nanosec*1e-9)
            if duration <= 0: continue
            step = 0.02; t = 0.0
            
            while t < duration:
                if self.stop_current_motion or goal_handle.is_cancel_requested:
                    goal_handle.canceled(); robot_obj.stop(); return FollowJointTrajectory.Result()
                fract = t / duration
                cmd_angles = []
                for idx in joint_map_indices:
                    s = start_pt.positions[idx]; e = end_pt.positions[idx]
                    cmd_angles.append(self.linear_interpolate(s, e, fract))
                robot_obj.set_servo_angle(cmd_angles)
                time.sleep(step); t += step

        if not self.stop_current_motion:
            final_angles = [points[-1].positions[idx] for idx in joint_map_indices]
            robot_obj.set_servo_angle(final_angles)
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
        x_p, _, _ = self.xarm.get_full_state() if self.xarm else ([0.0]*5, [], [])
        u_p, _, _ = self.uf850.get_full_state() if self.uf850 else ([0.0]*6, [], [])
        self.loop_count += 1
        if self.gripper and self.loop_count >= 10:
            try: self.cached_gripper_width = self.gripper.get_width()
            except: pass
            self.loop_count = 0
        g = self.mm_to_rad(self.cached_gripper_width)
        s = self.xarm.get_linear_track_pos() if self.xarm else 0.0
        msg.name = ['xarm5_joint1', 'xarm5_joint2', 'xarm5_joint3', 'xarm5_joint4', 'xarm5_joint5', 'u1_joint1', 'u1_joint2', 'u1_joint3', 'u1_joint4', 'u1_joint5', 'u1_joint6', 'slider_slider_joint', 'rg6_l_out', 'rg6_r_out', 'rg6_l_tip', 'rg6_r_tip', 'rg6_l_passive', 'rg6_r_passive']
        msg.position = x_p + u_p + [s] + [g, -g, g, -g, -g, -g]
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