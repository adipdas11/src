#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time, math
from disassembly_skills.motion_backend import MotionBackend

class XArmDiagnostic(Node):
    def __init__(self):
        super().__init__('xarm_diagnostic_node')
        # 🦾 Use the existing robust backend
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        
        # Configuration
        self.TARGET = {'x': 0.85, 'y': 0.04, 'z': 1.07}
        self.HOME_JOINTS = {
            'xarm5_joint1': 0.0,
            'xarm5_joint2': -0.5,
            'xarm5_joint3': -1.0,
            'xarm5_joint4': 0.0,
            'xarm5_joint5': 0.0
        }

    def run_test(self):
        # 1. Warm up TF and Controllers
        self.get_logger().info("⏳ Resetting hardware and warming up...")
        self.moveit_backend.reset_robot()
        time.sleep(2.0)

        # 2. Go to a safe Joint Home first (Avoids IK confusion)
        self.get_logger().info("🏠 Moving to JOINT HOME...")
        self.moveit_backend.move_to_joint_positions(self.HOME_JOINTS, velocity=0.2)
        time.sleep(1.0)

        # 3. Move to your specific Cartesian target
        # Using link5 (the flange) to ensure the 5-DOF IK is simple
        self.get_logger().info(f"🎯 Moving to TARGET: {self.TARGET}")
        
        # We pass q_dict=None so the backend triggers the 5-DOF Yaw Sweep
        # if the default Top-Down orientation is blocked.
        success = self.moveit_backend.move_to_pose_robust(
            x=self.TARGET['x'], 
            y=self.TARGET['y'], 
            z=self.TARGET['z'], 
            q_dict=None, 
            velocity=0.1
        )

        if success:
            self.get_logger().info("✅ Target Reached! Holding for 3 seconds...")
            time.sleep(3.0)
            self.get_logger().info("🏠 Returning HOME...")
            self.moveit_backend.move_to_joint_positions(self.HOME_JOINTS, velocity=0.2)
        else:
            self.get_logger().error("❌ IK Failed even for a simple move. Check if the point is within 700mm of the base.")

def main(args=None):
    rclpy.init(args=args)
    node = XArmDiagnostic()
    
    # We use a simple spin_once approach for this one-shot test
    import threading
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    node.run_test()
    
    node.get_logger().info("🏁 Test Complete.")
    rclpy.shutdown()

if __name__ == '__main__':
    main()