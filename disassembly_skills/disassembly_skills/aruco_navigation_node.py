#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion
import tf2_ros
import tf_transformations
import math

# --- IMPORT YOUR UPDATED BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
TARGET_FRAME = 'aruco_target_world' 
PLANNING_FRAME = 'world' # Absolute global origin
FINAL_Z_OFFSET = 0.01 

GROUP_UF850 = "uf_arm"
LINK_UF850 = "rg6_hand_tcp" 

GROUP_XARM5 = "xarm_arm"
LINK_XARM5 = "screwdriver_tcp"

# --- DEFINING START JOINTS FROM SRDF ---
START_JOINTS = {
    "xarm5": ["xarm5_joint1", "xarm5_joint2", "xarm5_joint3", "xarm5_joint4", "xarm5_joint5"],
    "xarm5_vals": [0.0, 0.0, -1.5708, 1.5708, 0.0],
    
    "u1": ["u1_joint1", "u1_joint2", "u1_joint3", "u1_joint4", "u1_joint5", "u1_joint6"],
    "u1_vals": [0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0]
}
# =================================================

class StableDisassemblyNavigator(Node):
    def __init__(self):
        super().__init__('stable_disassembly_navigator')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Motion Backends for dual-arm setup
        self.uf850 = MotionBackend(self, GROUP_UF850)
        self.xarm5 = MotionBackend(self, GROUP_XARM5)
        
        self.get_logger().info("🤖 Stable Navigator Active: Using RPY Logic & World Coordinates.")

    def get_world_target(self):
        """Acquires target pose from TF and returns raw coordinates for RPY processing."""
        try:
            now = rclpy.time.Time()
            t = self.tf_buffer.lookup_transform(PLANNING_FRAME, TARGET_FRAME, now, timeout=rclpy.duration.Duration(seconds=1.0))
            
            # Return as a simple dictionary for the move_to_pose_rpy method
            target_data = {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'z': t.transform.translation.z + FINAL_Z_OFFSET
            }
            
            print(f"\n🌍 GLOBAL TARGET LOCKED (Frame: {PLANNING_FRAME})")
            print(f"   X: {target_data['x']:.4f} | Y: {target_data['y']:.4f} | Z: {target_data['z']:.4f}")
            return target_data
        except Exception as e:
            self.get_logger().warn(f"Waiting for ArUco TF: {e}", throttle_duration_sec=2.0)
            return None

    def run_sequence(self):
        """Main execution logic for the stable disassembly sequence."""
        
        # 1. UF850 Phase (6-DOF)
        target = None
        while rclpy.ok() and target is None:
            target = self.get_world_target()
            rclpy.spin_once(self, timeout_sec=0.1)

        input(f"👉 Press ENTER to move UF850 to final target...")
        # Pointing down (Roll=pi, Pitch=0, Yaw=0)
        if self.uf850.move_to_pose_rpy(target['x'], target['y'], target['z'], math.pi, 0.0, 0.0, LINK_UF850, frame_id=PLANNING_FRAME):
            print("✅ UF850 SUCCESS.")
            self.uf850.move_to_named_target(START_JOINTS["u1"], START_JOINTS["u1_vals"])
        
        # 2. xArm5 Phase (5-DOF Stability)
        print("\n⏳ Waiting for xArm5 target lock...")
        target = None 
        while rclpy.ok() and target is None:
            target = self.get_world_target()
            rclpy.spin_once(self, timeout_sec=0.1)

        input(f"👉 Press ENTER to move xArm5 (Screwdriver) to final target...")
        # Pointing down (Roll=pi, Pitch=0, Yaw=0) using stable RPY logic
        if self.xarm5.move_to_pose_rpy(target['x'], target['y'], target['z'], math.pi, 0.0, 0.0, LINK_XARM5, frame_id=PLANNING_FRAME):
            print("✅ xArm5 SUCCESS.")
            self.xarm5.move_to_named_target(START_JOINTS["xarm5"], START_JOINTS["xarm5_vals"])

        print("\n🏁 Stable Dual-arm sequence finished.")

def main():
    rclpy.init()
    node = StableDisassemblyNavigator()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()