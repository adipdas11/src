#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import math
import time

# Ensure this matches your package structure
from disassembly_skills.motion_backend import MotionBackend

# --- TEST COORDINATES ---
WORLD_X = 0.84
WORLD_Y = 0.07
WORLD_Z = 0.95
Z_OFFSET = 0.30  # Using your successful 30cm offset

# TCP Links
LINK_XARM5 = "xarm5_tool0" 
LINK_UF850 = "u1_tool0"    

# Home Poses for Reset
HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0,
    'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57,
    'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0
}

class SequentialWorldTester(Node):
    def __init__(self):
        super().__init__('sequential_world_tester')
        self.xarm5 = MotionBackend(self, "xarm_arm")
        self.uf850 = MotionBackend(self, "uf_arm")
        self.get_logger().info(f"Targeting: Z={WORLD_Z + Z_OFFSET}m (30cm Offset)")

    def run_test(self):
        try:
            # --- PHASE 1: xArm5 Position ---
            print("\n" + "="*50)
            print("STEP 1: xArm5 to Position Only")
            input("👉 Press ENTER to move xArm5...")
            success = self.xarm5.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict={}, # Relaxation logic
                link_name=LINK_XARM5
            )
            if success:
                time.sleep(1.0)
                print("Returning xArm5 to Home...")
                self.xarm5.move_to_joint_positions(HOME_JOINTS, filter_prefix="xarm5")
                time.sleep(2.0)

            # --- PHASE 2: UF850 Position ---
            print("\n" + "="*50)
            print("STEP 2: UF850 to Position Only (Relaxed)")
            input("👉 Press ENTER to move UF850...")
            success = self.uf850.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict={}, # Relaxation logic
                link_name=LINK_UF850
            )
            if success:
                time.sleep(1.0)
                print("Returning UF850 to Home...")
                self.uf850.move_to_joint_positions(HOME_JOINTS, filter_prefix="u1")
                time.sleep(2.0)

            # --- PHASE 3: UF850 Full Pose (Position + Orientation) ---
            print("\n" + "="*50)
            print("STEP 3: UF850 to Full Pose (Image-Captured Orientation)")
            print("Targeting exact orientation from u1_tool0 screenshot")
            input("👉 Press ENTER to move UF850...")
            
            # Quaternions from image_1ca47b.png
            u1_q_image = {
                'qx': 0.55531, 'qy': 0.52615, 'qz': 0.46753, 'qw': -0.44295
            }
            
            success = self.uf850.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict=u1_q_image, # Strict pose logic
                link_name=LINK_UF850
            )
            if success:
                time.sleep(1.0)
                print("Returning UF850 to Home...")
                self.uf850.move_to_joint_positions(HOME_JOINTS, filter_prefix="u1")

        except Exception as e:
            self.get_logger().error(f"Sequence interrupted: {e}")

    def report(self, name, success):
        if success: self.get_logger().info(f"✅ {name}: SUCCESS")
        else: self.get_logger().error(f"❌ {name}: FAILED")

def main(args=None):
    rclpy.init(args=args)
    node = SequentialWorldTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()