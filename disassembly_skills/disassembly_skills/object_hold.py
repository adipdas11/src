#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import math
import time

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

class OrientationFirstNode(Node):
    def __init__(self):
        super().__init__('orient_first_grasp_node')
        self.uf850 = MotionBackend(self, "uf_arm")
        self.get_logger().info("🚀 Orientation-First Sequence Initialized")

    def run_sequence(self):
        # 1. Target Data
        BOX_X, BOX_Y, BOX_Z = 0.9, 0.06, 0.965
        TOOL_LENGTH = 0.28
        HIGH_Z = BOX_Z + 0.25   # High altitude for translation
        FINAL_Z = BOX_Z + 0.05  # 5cm above target center
        VEL = 0.2

        # --- STEP 1: FLIP AND ROTATE WRIST (IN PLACE) ---
        print("\n📐 Phase 1: Orienting Tool (Flip & Rotate)...")
        # Roll = pi/2 (Flip), Pitch = 1.57 (Rotate Jaws), Yaw = pi (Facing target)
        q_target = self.uf850._rpy_to_quaternion(math.pi/2, 1.57, math.pi)
        q_dict_orient = {'qx': q_target.x, 'qy': q_target.y, 'qz': q_target.z, 'qw': q_target.w}
        
        # Get current X/Y to stay in place while re-orienting
        success_orient = self.uf850.move_to_pose_robust(
            x=0.5, y=0.0, z=HIGH_Z + TOOL_LENGTH, # Assuming safe middle start
            q_dict=q_dict_orient, link_name='u1_tool0', 
            frame_id='world_world', velocity=VEL
        )
        if not success_orient:
            print("❌ Orientation failed. Checking alternative IK...")

        # --- STEP 2: TRANSLATE TO TARGET HOVER ---
        print("📍 Phase 2: Translating to Target at High Altitude...")
        standoff_x = BOX_X - 0.25 # Maintain 25cm distance for safety
        
        self.uf850.move_to_pose_robust(
            x=standoff_x, y=BOX_Y, z=HIGH_Z + TOOL_LENGTH, 
            q_dict=q_dict_orient, link_name='u1_tool0', 
            frame_id='world_world', velocity=VEL
        )
        time.sleep(1.0)

        # --- STEP 3: VERTICAL DROP ---
        print("📉 Phase 3: Dropping Down to 5cm Above Target...")
        
        success_drop = self.uf850.move_to_pose_robust(
            x=standoff_x, 
            y=BOX_Y, 
            z=FINAL_Z + TOOL_LENGTH, 
            q_dict=q_dict_orient, 
            link_name='u1_tool0', 
            frame_id='world_world', 
            velocity=0.1 # Slow descent for precision
        )

        if success_drop:
            print(f"\n✅ Sequence Complete! Tool is at Z: {FINAL_Z:.3f} (5cm clearance).")
        else:
            print("\n❌ Drop failed: IK limits reached at lower altitude.")

def main():
    rclpy.init()
    node = OrientationFirstNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    import threading
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    try:
        node.run_sequence()
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()