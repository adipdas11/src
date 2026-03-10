#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import math

# Ensure this matches your package structure
from disassembly_skills.motion_backend import MotionBackend

# ==========================================
# CONFIGURATION
# ==========================================
HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0,
    'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57,
    'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0
}

class JointMotionTester(Node):
    def __init__(self):
        super().__init__('joint_motion_tester')
        
        # Initialize backends with group names matching your SRDF
        self.xarm5 = MotionBackend(self, "xarm_arm")
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.get_logger().info("✅ Motion Tester Initialized (xArm5 + UF850 + Gripper).")

    def run_sequence(self):
        """Main test sequence covering joint-level, pose-level, gripper, and SDK control."""
        try:
            # --- TEST 1: xArm5 Robust Pose (Vertical) ---
            print("\n" + "="*50)
            print("TEST 1: xArm5 Robust Pose (Vertical Point Down)")
            print("Target: xarm5_link5 at (0.83, 0.11, 1.1)")
            input("👉 Press ENTER to execute...")
            
            success = self.xarm5.move_to_pose_robust(
                x=0.83, y=0.11, z=1.1, 
                q_dict={},
                link_name="xarm5_link5"
            )
            self.report_status("xArm5 Pose Move", success)

            # --- TEST 2: UF850 Pose ---
            print("\n" + "="*50)
            print("TEST 2: UF850 Pose Move")
            input("👉 Press ENTER to execute...")
            
            u1_q = {'qx': 0.55531, 'qy': 0.52615, 'qz': 0.46753, 'qw': -0.44295}
            success = self.uf850.move_to_pose_robust(
                x=0.82887, y=-0.20629, z=1.0172, 
                q_dict=u1_q, 
                link_name="u1_tool0"
            )
            self.report_status("UF850 Pose Move", success)

            # --- TEST 3: Joint Filtering Check ---
            print("\n" + "="*50)
            print("TEST 3: Independent Joint Control (xArm5 J1 ONLY)")
            input("👉 Press ENTER to execute...")
            
            success = self.xarm5.move_to_joint_positions(
                target_joints={'xarm5_joint1': 0.7},
                filter_prefix="xarm5"
            )
            self.report_status("Filtered Joint Move", success)

            # --- TEST 4: Gripper Open/Close ---
            print("\n" + "="*50)
            print("TEST 4: Gripper Open/Close Cycle")
            input("👉 Press ENTER to execute...")
            
            print("Opening gripper...")
            self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(35.0)}, "rg6", velocity=0.3)
            time.sleep(1.0)
            print("Closing gripper...")
            self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(-35.0)}, "rg6", velocity=0.3)
            time.sleep(1.0)
            print("Neutral...")
            self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(0.0)}, "rg6", velocity=0.3)
            self.report_status("Gripper Cycle", True)

            # --- TEST 5: SDK Cartesian Jog (xArm5) ---
            print("\n" + "="*50)
            print("TEST 5: xArm5 SDK Cartesian Jog (+10mm Z, then -10mm Z)")
            input("👉 Press ENTER to execute...")
            
            success = self.xarm5.jog_cartesian_sdk(0, 0, 0.01, speed_mm_s=20.0)
            self.report_status("SDK Jog UP 10mm", success)
            time.sleep(0.5)
            success = self.xarm5.jog_cartesian_sdk(0, 0, -0.01, speed_mm_s=20.0)
            self.report_status("SDK Jog DOWN 10mm", success)

            # --- TEST 6: Home All ---
            print("\n" + "="*50)
            print("TEST 6: Sequential Home — xArm5 then UF850")
            input("👉 Press ENTER to execute...")
            
            success1 = self.xarm5.move_to_joint_positions(HOME_JOINTS, "xarm5", velocity=0.2)
            self.report_status("xArm5 Home", success1)
            
            success2 = self.uf850.move_to_joint_positions(HOME_JOINTS, "u1", velocity=0.2)
            self.report_status("UF850 Home", success2)

            print("\n✅ Full Test Sequence Finished.")

        except Exception as e:
            self.get_logger().error(f"Test sequence interrupted: {e}")

    def report_status(self, task, success):
        if success:
            print(f"✅ {task}: SUCCESS")
        else:
            print(f"❌ {task}: FAILED")

def main(args=None):
    rclpy.init(args=args)
    tester = JointMotionTester()
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    tester.run_sequence()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()