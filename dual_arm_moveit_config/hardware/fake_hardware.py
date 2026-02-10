#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import time
import threading

class FakeHardware(Node):
    def __init__(self):
        super().__init__('fake_hardware_driver')
        
        # 1. Start Action Servers
        # xArm5
        self._xarm_server = ActionServer(
            self, FollowJointTrajectory, '/xarm_controller/follow_joint_trajectory', self.execute_callback
        )
        # UF850
        self._uf_server = ActionServer(
            self, FollowJointTrajectory, '/uf_controller/follow_joint_trajectory', self.execute_callback
        )
        # RG6 Gripper
        self._rg6_server = ActionServer(
            self, FollowJointTrajectory, '/rg6_controller/follow_joint_trajectory', self.execute_callback
        )
        # --- NEW: Slider Server ---
        self._slider_server = ActionServer(
            self, FollowJointTrajectory, '/slider_controller/follow_joint_trajectory', self.execute_callback
        )

        # 2. Publisher
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 50) 
        self.timer = self.create_timer(0.02, self.publish_joints) # 50Hz
        
        # 3. Internal State (Initial Positions)
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
        
        # 4. Mimic Definitions (For RG6 TF Fix)
        # Maps mimic_joint_name -> (source_joint_name, multiplier)
        self.mimics = {
            'rg6_r_out':     ('rg6_l_out', -1.0),
            'rg6_l_tip':     ('rg6_l_out',  1.0),
            'rg6_r_tip':     ('rg6_l_out', -1.0),
            'rg6_l_passive': ('rg6_l_out', -1.0),
            'rg6_r_passive': ('rg6_l_out', -1.0),
        }
        
        self.lock = threading.Lock()
        self.get_logger().info("Fake Hardware Ready. SLIDER & MIMICS ENABLED.")

    def linear_interpolate(self, start_pos, end_pos, fraction):
        return start_pos + (end_pos - start_pos) * fraction

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'>>> GOAL RECEIVED for {goal_handle.request.trajectory.joint_names}...')
        
        traj = goal_handle.request.trajectory
        names = traj.joint_names
        points = traj.points
        
        if len(points) < 2:
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        # Iterate through trajectory segments
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            
            start_time = start_point.time_from_start.sec + start_point.time_from_start.nanosec * 1e-9
            end_time = end_point.time_from_start.sec + end_point.time_from_start.nanosec * 1e-9
            duration = end_time - start_time
            
            if duration <= 0:
                continue

            # Interpolation Loop
            step_size = 0.02
            t = 0.0
            
            while t < duration:
                fraction = t / duration
                
                with self.lock:
                    for j, name in enumerate(names):
                        if name in self.joint_positions:
                            start_val = start_point.positions[j]
                            end_val = end_point.positions[j]
                            self.joint_positions[name] = self.linear_interpolate(start_val, end_val, fraction)
                
                time.sleep(step_size)
                t += step_size
                
        # Snap to final position
        with self.lock:
            for j, name in enumerate(names):
                if name in self.joint_positions:
                    self.joint_positions[name] = points[-1].positions[j]

        self.get_logger().info('Motion Complete')
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.lock:
            # 1. Get Main Joints
            names = list(self.joint_positions.keys())
            positions = list(self.joint_positions.values())
            
            # 2. Calculate Mimic Joints (Fixes TF Errors)
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
    
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()