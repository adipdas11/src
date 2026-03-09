#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty, SetBool
import time
import math
import threading

class FakeHardware(Node):
    def __init__(self):
        super().__init__('fake_hardware_driver')
        
        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        
        # State Flags
        self.stop_current_motion = False
        self.velocity_mode_active = False

        # --- SERVICES (Mirrored from Real Hardware) ---
        self._stop_service = self.create_service(Empty, '/xarm/stop_robot', self.handle_stop_request, callback_group=self.cb_group)
        self._reset_service = self.create_service(Empty, '/xarm/reset_robot', self.handle_reset_request, callback_group=self.cb_group)
        self._vel_mode_srv = self.create_service(SetBool, '/xarm/set_velocity_mode', self.handle_velocity_mode_request, callback_group=self.cb_group)

        # --- SUBSCRIBERS (Mirrored from Real Hardware) ---
        self._vel_sub = self.create_subscription(Twist, '/xarm/velo_cmd', self.handle_velocity_command, 10, callback_group=self.cb_group)

        # --- ACTION SERVERS ---
        self._xarm_server = ActionServer(self, FollowJointTrajectory, '/xarm_controller/follow_joint_trajectory', execute_callback=self.execute_callback, callback_group=self.cb_group)
        self._uf_server = ActionServer(self, FollowJointTrajectory, '/uf_controller/follow_joint_trajectory', execute_callback=self.execute_callback, callback_group=self.cb_group)
        self._rg6_server = ActionServer(self, FollowJointTrajectory, '/rg6_controller/follow_joint_trajectory', execute_callback=self.execute_callback, callback_group=self.cb_group)
        self._slider_server = ActionServer(self, FollowJointTrajectory, '/slider_controller/follow_joint_trajectory', execute_callback=self.execute_callback, callback_group=self.cb_group)

        # --- PUBLISHERS ---
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 50) 
        self.timer = self.create_timer(0.02, self.publish_joints, callback_group=self.cb_group) # 50Hz
        
        # --- Internal State (Initial Positions) ---
        self.joint_positions = {
            'slider_slider_joint': 0.0,
            
            # xArm5
            'xarm5_joint1': 0.0, 
            'xarm5_joint2': 0.0, 
            'xarm5_joint3': -1.5708, 
            'xarm5_joint4': 1.5708, 
            'xarm5_joint5': 0.0,
            
            # UF850
            'u1_joint1': 0.0, 
            'u1_joint2': 0.0, 
            'u1_joint3': -1.5708, 
            'u1_joint4': 0.0, 
            'u1_joint5': -1.5708, 
            'u1_joint6': 0.0,
            
            # RG6 Main Joint
            'rg6_l_out': 0.0, 
        }
        
        # --- Mimic Definitions (For RG6 TF Fix) ---
        self.mimics = {
            'rg6_r_out':     ('rg6_l_out', -1.0),
            'rg6_l_tip':     ('rg6_l_out',  1.0),
            'rg6_r_tip':     ('rg6_l_out', -1.0),
            'rg6_l_passive': ('rg6_l_out', -1.0),
            'rg6_r_passive': ('rg6_l_out', -1.0),
        }
        
        self.get_logger().info("✅ Fake Hardware Ready. Interfaces mirrored from real hardware.")

    # ================= HANDLERS =================

    def handle_velocity_mode_request(self, request, response):
        if request.data: 
            self.get_logger().info("🔄 FAKE: Switching to CARTESIAN VELOCITY MODE...")
            self.velocity_mode_active = True
        else:
            self.get_logger().info("🛑 FAKE: Disabling Velocity Mode (Back to Position Mode)...")
            self.velocity_mode_active = False
        
        response.success = True
        return response

    def handle_velocity_command(self, msg):
        if self.velocity_mode_active:
            # In a real robot, IK handles this. For fake hardware, we just acknowledge receipt
            # so the ROS network doesn't break, but we won't physically move the fake joints here.
            pass

    def handle_stop_request(self, request, response):
        self.get_logger().error("🚨 FAKE: STOP REQUEST")
        self.stop_current_motion = True
        return response

    def handle_reset_request(self, request, response):
        self.get_logger().info("♻️ FAKE: RESET REQUEST")
        self.stop_current_motion = False
        return response

    # ================= TRAJECTORY EXECUTION =================

    def linear_interpolate(self, start_pos, end_pos, fraction):
        return start_pos + (end_pos - start_pos) * fraction

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'>>> GOAL RECEIVED for {goal_handle.request.trajectory.joint_names}...')
        
        # Auto-switch back to position mode if a trajectory arrives
        if self.velocity_mode_active:
             self.get_logger().warn("⚠️ FAKE: Auto-switching back to Position Mode for Trajectory")
             self.velocity_mode_active = False

        self.stop_current_motion = False
        traj = goal_handle.request.trajectory
        names = traj.joint_names
        points = traj.points
        
        if len(points) < 2:
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        for i in range(len(points) - 1):
            if self.stop_current_motion or goal_handle.is_cancel_requested:
                self.get_logger().warn('Motion canceled or stopped!')
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            start_point = points[i]
            end_point = points[i + 1]
            
            start_time = start_point.time_from_start.sec + start_point.time_from_start.nanosec * 1e-9
            end_time = end_point.time_from_start.sec + end_point.time_from_start.nanosec * 1e-9
            duration = end_time - start_time
            
            if duration <= 0:
                continue

            step_size = 0.02
            t = 0.0
            
            while t < duration:
                if self.stop_current_motion or goal_handle.is_cancel_requested:
                    self.get_logger().warn('Motion canceled or stopped mid-interpolation!')
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()

                fraction = t / duration
                with self.lock:
                    for j, name in enumerate(names):
                        if name in self.joint_positions:
                            start_val = start_point.positions[j]
                            end_val = end_point.positions[j]
                            self.joint_positions[name] = self.linear_interpolate(start_val, end_val, fraction)
                
                time.sleep(step_size)
                t += step_size
                
        # Snap to final position if not stopped
        if not self.stop_current_motion:
            with self.lock:
                for j, name in enumerate(names):
                    if name in self.joint_positions:
                        self.joint_positions[name] = points[-1].positions[j]
            self.get_logger().info('Motion Complete')
            goal_handle.succeed()

        return FollowJointTrajectory.Result()

    # ================= PUBLISHING =================

    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.lock:
            names = list(self.joint_positions.keys())
            positions = list(self.joint_positions.values())
            
            for mimic_name, (source_name, multiplier) in self.mimics.items():
                if source_name in self.joint_positions:
                    source_val = self.joint_positions[source_name]
                    mimic_val = source_val * multiplier
                    names.append(mimic_name)
                    positions.append(mimic_val)

            msg.name = names
            msg.position = positions
            
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeHardware()
    executor = MultiThreadedExecutor()
    
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()