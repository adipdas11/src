#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import math
import time

# Ensure this matches your package structure
from disassembly_skills.motion_backend import MotionBackend

# ==========================================
# GLOBAL CONFIGURATION & COORDINATES
# ==========================================
MOVE_SPEED = 0.1      
HOME_SPEED = 0.2      
GRIPPER_SPEED = 0.5   

WORLD_X = 0.84
WORLD_Y = 0.07
WORLD_Z = 0.95
Z_OFFSET = 0.30       

OPEN_DEG = 35.0
CLOSE_DEG = -35.0
HOME_DEG = 0.0

LINK_XARM5 = "xarm5_tool0" 
LINK_UF850 = "u1_tool0"    
JOINT_GRIPPER = "rg6_l_out"

# --- FIXED: Global Home Definition ---
HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0,
    'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57,
    'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0
}
# ==========================================

class SequentialWorldTester(Node):
    def __init__(self):
        super().__init__('sequential_world_tester')
        self.xarm5 = MotionBackend(self, "xarm_arm")
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.get_logger().info(f"🚀 Tester Active: Move={MOVE_SPEED}, Home={HOME_SPEED}")

    def run_test(self):
        try:
            # --- PHASE 1: xArm5 Move and Return ---
            print("\n" + "="*50)
            print(f"STEP 1: xArm5 to Position Only (Speed: {MOVE_SPEED})")
            input("👉 Press ENTER to move xArm5...")
            success = self.xarm5.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict={}, 
                link_name=LINK_XARM5,
                velocity=MOVE_SPEED
            )
            if success:
                time.sleep(1.0)
                print(f"Returning xArm5 to Home (Speed: {HOME_SPEED})...")
                # Using the global HOME_JOINTS
                self.xarm5.move_to_joint_positions(HOME_JOINTS, filter_prefix="xarm5", velocity=HOME_SPEED)
                time.sleep(2.0)

            # --- PHASE 2: UF850 + Gripper Test ---
            print("\n" + "="*50)
            print("STEP 2: UF850 Position + Gripper Action")
            input("👉 Press ENTER to approach and test GRIPPER...")
            
            success_move = self.uf850.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict={}, 
                link_name=LINK_UF850,
                velocity=MOVE_SPEED
            )
            
            if success_move:
                # 1. OPEN
                print("Opening Gripper...")
                self.gripper.move_to_joint_positions({JOINT_GRIPPER: math.radians(OPEN_DEG)}, 
                                                     filter_prefix="rg6", velocity=GRIPPER_SPEED)
                time.sleep(2.0) # Increased delay for mechanical stabilization
                
                # 2. CLOSE
                print("Closing Gripper...")
                self.gripper.move_to_joint_positions({JOINT_GRIPPER: math.radians(CLOSE_DEG)}, 
                                                     filter_prefix="rg6", velocity=GRIPPER_SPEED)
                time.sleep(2.0)

                # 3. EXPLICIT HOME RESET
                print("Force Resetting Gripper to 0.0...")
                # We move back to 0.0 specifically to clear the controller "Busy" state
                self.gripper.move_to_joint_positions({JOINT_GRIPPER: 0.0}, 
                                                     filter_prefix="rg6", velocity=GRIPPER_SPEED)
                time.sleep(1.0)

            # --- PHASE 3: UF850 Full Pose ---
            print("\n" + "="*50)
            print("STEP 3: UF850 Full Pose (Orientation + Position)")
            input("👉 Press ENTER to move UF850...")
            
            u1_q_image = {'qx': 0.55531, 'qy': 0.52615, 'qz': 0.46753, 'qw': -0.44295} #
            success = self.uf850.move_to_pose_robust(
                x=WORLD_X, y=WORLD_Y, z=WORLD_Z + Z_OFFSET,
                q_dict=u1_q_image, 
                link_name=LINK_UF850,
                velocity=MOVE_SPEED
            )
            if success:
                time.sleep(1.0)
                self.uf850.move_to_joint_positions(HOME_JOINTS, filter_prefix="u1", velocity=HOME_SPEED)

        except Exception as e:
            self.get_logger().error(f"Sequence interrupted: {e}")

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