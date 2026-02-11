#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from xarm.wrapper import XArmAPI
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import time
import math
import threading
import sys

# ==========================================
# CONFIGURATION
# ==========================================
XARM_IP = '192.168.1.239'     
UF850_IP = '192.168.1.195'     
GRIPPER_IP = '192.168.1.1'    

# --- GRIPPER MAPPING ---
RAD_CLOSE = -0.6109   # -35 deg
RAD_OPEN  =  0.6109   # +35 deg
RAD_RANGE = RAD_OPEN - RAD_CLOSE
MM_CLOSE = 0.0
MM_OPEN  = 160.0      

# ==========================================
# ROBUST DRIVER CLASSES
# ==========================================
class RG:
    def __init__(self, gripper, ip, logger, port=502):
        self.logger = logger
        self.ip = ip
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        self.lock = threading.Lock() 
        self.gripper = gripper
        
        if self.gripper == 'rg2':
            self.max_width = 1100 
            self.max_force = 400
        elif self.gripper == 'rg6':
            self.max_width = 1600 
            self.max_force = 1200
            
        if not self.client.connect():
            self.logger.error(f"[GRIPPER] CRITICAL: Could not connect to {ip}")
            raise ConnectionError(f"Gripper connection failed at {ip}")
        
        self.logger.info(f"[GRIPPER] Connected to {gripper.upper()} at {ip}")

    def move_gripper(self, width_mm, force_val=400):
        val = int(width_mm * 10) 
        val = max(0, min(val, self.max_width))
        params = [force_val, val, 16] 
        
        with self.lock: 
            try:
                self.client.write_registers(address=0, values=params, unit=65)
            except Exception as e:
                self.logger.error(f"[GRIPPER] Write Error: {e}")

    def get_width(self):
        with self.lock: 
            try:
                result = self.client.read_holding_registers(address=267, count=1, unit=65)
                if result and not result.isError():
                    return (result.registers[0] / 10.0) 
            except Exception as e:
                self.logger.warn(f"[GRIPPER] Read Error: {e}")
        return 0.0 
    
    def close_connection(self):
        with self.lock:
            self.client.close()

class RealRobotInterface:
    def __init__(self, ip, name, dof, logger, has_linear_track=False):
        self.ip = ip
        self.name = name
        self.dof = dof
        self.logger = logger
        self.has_linear_track = has_linear_track
        self.arm = XArmAPI(self.ip)
        self.connected = False
        
        # ATTEMPT CONNECTION
        self.connect()

    def connect(self):
        self.logger.info(f'[{self.name}] Connecting to {self.ip}...')
        try:
            self.arm.connect()
            if not self.arm.connected:
                raise ConnectionError("SDK returned connected=False")
            
            self.arm.motion_enable(enable=True)
            self.arm.clean_error()
            self.arm.set_mode(1) 
            self.arm.set_state(0)
            self.arm.set_report_tau_or_i(0) 
            
            # --- LINEAR TRACK INITIALIZATION ---
            if self.has_linear_track:
                self._init_linear_track()

            self.connected = True
            self.logger.info(f'[{self.name}] ✅ Connected & Ready.')
            
        except Exception as e:
            self.logger.error(f'[{self.name}] ❌ CRITICAL CONNECTION FAILURE: {e}')
            raise e

    def _init_linear_track(self):
        try:
            self.logger.info(f'[{self.name}] Starting Linear Track Homing...')
            code = self.arm.set_linear_track_back_origin(wait=True)
            if code != 0:
                self.logger.warn(f'[{self.name}] Linear Track Homing Warning Code: {code}')
            
            self.arm.set_linear_track_enable(True)
            self.arm.set_linear_track_speed(200) 
            self.logger.info(f'[{self.name}] Linear Track Ready.')
        except Exception as e:
            self.logger.error(f'[{self.name}] Linear Track Init Failed: {e}')
            raise e

    def get_full_state(self):
        if not self.connected: 
            return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)
        
        try:
            code, angles = self.arm.get_servo_angle(is_radian=True)
            if code != 0 or not angles: 
                # Check connection if reads fail repeatedly
                if self.arm.connected: 
                    return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)
                else:
                    self.logger.error(f"[{self.name}] Lost connection during read!")
                    self.connected = False
                    return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)

            # 2. Effort (Torque)
            code_t, torques = self.arm.get_joints_torque()
            if code_t != 0 or not torques: torques = [0.0]*self.dof

            # 3. Velocity (Approximation or 0)
            vels = [0.0]*self.dof

            return (angles[:self.dof], vels[:self.dof], torques[:self.dof])
        except:
            return ([0.0]*self.dof, [0.0]*self.dof, [0.0]*self.dof)

    def get_linear_track_pos(self):
        """ Returns position in METERS (Negative Range) """
        if not self.connected or not self.has_linear_track: return 0.0
        try:
            code, pos_mm = self.arm.get_linear_track_pos()
            if code == 0 and pos_mm is not None:
                return -1.0 * (pos_mm / 1000.0) 
        except: pass
        return 0.0

    def set_servo_angle(self, angles):
        if not self.connected: return
        ret = self.arm.set_servo_angle_j(angles=angles, is_radian=True)
        if ret != 0:
            self.logger.warn(f"[{self.name}] Servo Cmd Failed Code: {ret}")

    def set_linear_track(self, pos_meters):
        if not self.connected or not self.has_linear_track: return
        pos_mm = abs(pos_meters) * 1000.0
        pos_mm = max(0, min(700, pos_mm))
        self.arm.set_linear_track_pos(pos_mm, wait=False)

    def disconnect(self):
        self.connected = False
        self.arm.disconnect()

# ==========================================
# MAIN NODE
# ==========================================
class RealHardware(Node):
    def __init__(self):
        super().__init__('real_hardware_driver')
        
        # Use Reentrant Group to allow parallel callbacks (Trajectory + State Pub)
        self.cb_group = ReentrantCallbackGroup()

        self.xarm = None
        self.uf850 = None
        self.gripper = None
        
        # 1. CRITICAL CONNECTION BLOCK
        try:
            self.get_logger().info("--- INITIALIZING HARDWARE ---")
            
            # xArm5 (w/ Linear Track)
            self.xarm = RealRobotInterface(XARM_IP, "xArm5", 5, self.get_logger(), has_linear_track=True)
            
            # UF850
            self.uf850 = RealRobotInterface(UF850_IP, "UF850", 6, self.get_logger())
            
            # Gripper
            self.gripper = RG('rg6', GRIPPER_IP, self.get_logger())
            
            self.get_logger().info("--- HARDWARE INIT SUCCESSFUL ---")

        except Exception as e:
            self.get_logger().fatal(f"STARTUP FAILED: {e}")
            self.get_logger().fatal("Shutting down node due to hardware failure.")
            sys.exit(1)

        # Action Servers
        self._xarm_server = ActionServer(self, FollowJointTrajectory, '/xarm_controller/follow_joint_trajectory', 
                                         execute_callback=self.execute_xarm_callback,
                                         callback_group=self.cb_group)
        
        self._uf_server = ActionServer(self, FollowJointTrajectory, '/uf_controller/follow_joint_trajectory', 
                                       execute_callback=self.execute_uf_callback,
                                       callback_group=self.cb_group)
        
        self._rg6_server = ActionServer(self, FollowJointTrajectory, '/rg6_controller/follow_joint_trajectory', 
                                        execute_callback=self.execute_gripper_callback,
                                        callback_group=self.cb_group)
        
        self._slider_server = ActionServer(self, FollowJointTrajectory, '/slider_controller/follow_joint_trajectory', 
                                           execute_callback=self.execute_slider_callback,
                                           callback_group=self.cb_group)

        # Publisher
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 50)
        self.timer = self.create_timer(0.02, self.publish_real_states, callback_group=self.cb_group)
        
        self.loop_count = 0 
        self.cached_gripper_width = 0.0
        self.get_logger().info("REAL HARDWARE DRIVER STARTED & SPINNING.")

    # --- HELPERS ---
    def rad_to_mm(self, radians):
        ratio = (radians - RAD_CLOSE) / RAD_RANGE
        mm = ratio * MM_OPEN
        return max(0.0, min(MM_OPEN, mm))

    def mm_to_rad(self, mm):
        ratio = mm / MM_OPEN
        rad = (ratio * RAD_RANGE) + RAD_CLOSE
        return rad

    def linear_interpolate(self, start, end, fract):
        return start + (end - start) * fract

    # --- TRAJECTORY EXECUTION ---
    def execute_trajectory(self, goal_handle, robot_obj, joint_map_indices):
        if not robot_obj or not robot_obj.connected:
            self.get_logger().error(f"Aborting Trajectory: {robot_obj.name} is disconnected.")
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        traj = goal_handle.request.trajectory
        points = traj.points
        if len(points) < 2:
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        self.get_logger().info(f'Executing trajectory for {robot_obj.name} ({len(points)} pts)...')
        
        start_time = time.time()
        
        for i in range(len(points) - 1):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info(f"Trajectory Canceled for {robot_obj.name}")
                return FollowJointTrajectory.Result()

            start_pt = points[i]
            end_pt = points[i + 1]
            st = start_pt.time_from_start.sec + start_pt.time_from_start.nanosec * 1e-9
            et = end_pt.time_from_start.sec + end_pt.time_from_start.nanosec * 1e-9
            duration = et - st
            
            if duration <= 0: continue

            step = 0.02 # 50Hz control loop
            t = 0.0
            
            while t < duration:
                fract = t / duration
                cmd_angles = []
                for idx in joint_map_indices:
                    # Robust index check
                    if idx < len(start_pt.positions):
                        s_val = start_pt.positions[idx]
                        e_val = end_pt.positions[idx]
                        cmd_angles.append(self.linear_interpolate(s_val, e_val, fract))
                    else:
                        cmd_angles.append(0.0) # Fallback

                robot_obj.set_servo_angle(cmd_angles)
                time.sleep(step)
                t += step

        # Final Point Enforcement
        final_angles = []
        for idx in joint_map_indices:
             if idx < len(points[-1].positions):
                 final_angles.append(points[-1].positions[idx])
        
        robot_obj.set_servo_angle(final_angles)
        
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def execute_xarm_callback(self, goal_handle):
        return self.execute_trajectory(goal_handle, self.xarm, [0,1,2,3,4])

    def execute_uf_callback(self, goal_handle):
        return self.execute_trajectory(goal_handle, self.uf850, [0,1,2,3,4,5])

    def execute_gripper_callback(self, goal_handle):
        if not self.gripper:
            goal_handle.abort()
            return FollowJointTrajectory.Result()
        
        traj = goal_handle.request.trajectory
        if traj.points:
            target_rad = traj.points[-1].positions[0]
            target_mm = self.rad_to_mm(target_rad)
            self.get_logger().info(f"Gripper Cmd: {target_rad:.2f} rad -> {target_mm:.1f} mm")
            self.gripper.move_gripper(target_mm)
        
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def execute_slider_callback(self, goal_handle):
        if not self.xarm:
            goal_handle.abort()
            return FollowJointTrajectory.Result()
        
        traj = goal_handle.request.trajectory
        if traj.points:
            target_m = traj.points[-1].positions[0]
            self.get_logger().info(f"Slider Cmd: {target_m:.3f} m")
            self.xarm.set_linear_track(target_m)

        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    # --- PUBLISH STATES ---
    def publish_real_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Robust Get State
        x_p, x_v, x_e = self.xarm.get_full_state() if self.xarm else ([0.0]*5, [0.0]*5, [0.0]*5)
        u_p, u_v, u_e = self.uf850.get_full_state() if self.uf850 else ([0.0]*6, [0.0]*6, [0.0]*6)
        
        # Gripper polling (Limit rate to 5Hz to save bandwidth)
        self.loop_count += 1
        if self.gripper and self.loop_count >= 10:
            try:
                self.cached_gripper_width = self.gripper.get_width()
            except: pass
            self.loop_count = 0
            
        g_rad = self.mm_to_rad(self.cached_gripper_width)

        # Linear Track
        slider_pos = 0.0
        if self.xarm:
            slider_pos = self.xarm.get_linear_track_pos()

        # Mimic Joint Setup
        val_pos = g_rad
        val_neg = -g_rad

        msg.name = [
            'xarm5_joint1', 'xarm5_joint2', 'xarm5_joint3', 'xarm5_joint4', 'xarm5_joint5',
            'u1_joint1', 'u1_joint2', 'u1_joint3', 'u1_joint4', 'u1_joint5', 'u1_joint6',
            'slider_slider_joint',
            'rg6_l_out', 'rg6_r_out', 'rg6_l_tip', 'rg6_r_tip', 'rg6_l_passive', 'rg6_r_passive'   
        ]
        
        msg.position = x_p + u_p + [slider_pos] + [val_pos, val_neg, val_pos, val_neg, val_neg, val_neg]
        
        # Fill velocity/effort with zeros to match length
        msg.velocity = x_v + u_v + [0.0] + [0.0]*6
        msg.effort = x_e + u_e + [0.0] + [0.0]*6
        
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.get_logger().info("Stopping Hardware Drivers...")
        if self.xarm: self.xarm.disconnect()
        if self.uf850: self.uf850.disconnect()
        if self.gripper: self.gripper.close_connection()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor to prevent blocking callbacks
    executor = MultiThreadedExecutor()
    node = None
    
    try:
        node = RealHardware()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass # Expected from sys.exit(1)
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()