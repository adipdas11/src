#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import threading
import time

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
SPEED = 0.1           
WAIT_TIME = 3.0         
HOME_SPEED = 0.1

# Tool Lengths
OFFSET_XARM_TOOL = 0.24      
OFFSET_UF850_TOOL = 0.28     

# Safety Offset (Stop above marker)
ACTUAL_TARGET_Z_OFFSET = 0.01

# --- FRAMES (MATCHING YOUR ORIGINAL EXAMPLE) ---
TARGET_FRAME = 'aruco_target_world'   # The marker
PLANNING_FRAME = 'world_world'        # The frame MoveIt expects

# Home Joints
HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0,
    'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57,
    'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0
}
# =================================================

class SequentialTester(Node):
    def __init__(self):
        super().__init__('sequential_tester')
        
        # 1. Motion Backends
        self.xarm5 = MotionBackend(self, "xarm_arm")
        self.uf850 = MotionBackend(self, "uf_arm")
        
        # 2. TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("🚀 Sequential Tester Ready.")

    def check_connections(self):
        """Verifies that MoveIt Action Servers are actually reachable."""
        print("⏳ Checking connection to MoveGroup servers...")
        if not self.xarm5._action_client.wait_for_server(timeout_sec=3.0):
            print("❌ ERROR: xArm5 MoveGroup Action Server NOT found!")
            return False
        if not self.uf850._action_client.wait_for_server(timeout_sec=3.0):
            print("❌ ERROR: UF850 MoveGroup Action Server NOT found!")
            return False
        print("✅ MoveGroup Servers Connected.")
        return True

    def get_target_coordinates(self):
        """Looks up the Aruco position relative to 'world_world'."""
        try:
            # We transform FROM aruco_target_world TO world_world
            if self.tf_buffer.can_transform(PLANNING_FRAME, TARGET_FRAME, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.tf_buffer.lookup_transform(PLANNING_FRAME, TARGET_FRAME, rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            else:
                self.get_logger().warn(f"Wait: Transform {PLANNING_FRAME} -> {TARGET_FRAME} missing.")
                return None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    def run_sequence(self):
        time.sleep(1.0) 
        
        # 1. Verify Connections
        if not self.check_connections():
            return

        print("\n" + "="*50)
        print(f"🔎 Looking for Target TF in frame: {PLANNING_FRAME}...")
        
        coords = self.get_target_coordinates()
        if coords is None:
            print("❌ Target not found! Is 'detect_aruco' running?")
            return

        tx, ty, tz = coords
        print(f"✅ FOUND ARUCO MARKER AT (in {PLANNING_FRAME}):\n   X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}")

        # ==========================================
        # STEP 1: UF850 Motion (FIRST)
        # ==========================================
        print("\n" + "-"*30)
        print("🤖 STEP 1: UF850 Motion Prep")

        uf_z = tz + OFFSET_UF850_TOOL + ACTUAL_TARGET_Z_OFFSET
        
        print(f"   Target: {uf_z:.3f} (Marker Z={tz:.3f} + Tool={OFFSET_UF850_TOOL} + Offset={ACTUAL_TARGET_Z_OFFSET})")
        
        # --- USER INPUT ---
        input("👉 Press ENTER to move UF850...")

        print(f"   🚀 Sending Goal to MoveIt in frame '{PLANNING_FRAME}'...")
        success = self.uf850.move_to_pose_robust(
            tx, ty, uf_z, {}, 
            link_name="u1_tool0", 
            frame_id=PLANNING_FRAME,  # <--- CRITICAL: Using world_world
            velocity=SPEED
        )

        if success:
            print(f"   ✅ Reached! Waiting {WAIT_TIME}s...")
            time.sleep(WAIT_TIME)
            print("   🏠 UF850 Returning Home...")
            self.uf850.move_to_joint_positions(HOME_JOINTS, "u1", velocity=HOME_SPEED)
        else:
            print("   ❌ UF850 Move Failed! (Check MoveIt Console for errors)")
            return

        time.sleep(1.0) 

        # ==========================================
        # STEP 2: xArm5 Motion (SECOND)
        # ==========================================
        print("\n" + "-"*30)
        print("🤖 STEP 2: xArm5 Motion Prep")
        
        xarm_z = tz + OFFSET_XARM_TOOL + ACTUAL_TARGET_Z_OFFSET
        
        print(f"   Target: {xarm_z:.3f} (Marker Z={tz:.3f} + Tool={OFFSET_XARM_TOOL} + Offset={ACTUAL_TARGET_Z_OFFSET})")
        
        # --- USER INPUT ---
        input("👉 Press ENTER to move xArm5...")

        print(f"   🚀 Sending Goal to MoveIt in frame '{PLANNING_FRAME}'...")
        success = self.xarm5.move_to_pose_robust(
            tx, ty, xarm_z, {}, 
            link_name="xarm5_link5", 
            frame_id=PLANNING_FRAME,  
            velocity=SPEED
        )

        if success:
            print(f"   ✅ Reached! Waiting {WAIT_TIME}s...")
            time.sleep(WAIT_TIME)
            print("   🏠 xArm Returning Home...")
            self.xarm5.move_to_joint_positions(HOME_JOINTS, "xarm5", velocity=HOME_SPEED)
        else:
            print("   ❌ xArm Move Failed! (Check MoveIt Console for errors)")

        print("\n✅ Sequence Complete.")

def main(args=None):
    rclpy.init(args=args)
    node = SequentialTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    node.run_sequence()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()