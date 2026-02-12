#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Pose, TransformStamped
import tf2_ros

import json
import time
import threading
import sys
import copy

from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
TOOL_LENGTH_XARM = 0.24       
SAFETY_BOUNDARY_Z = 0.05      

# --- CRITICAL FIX: MATCHING THE WORKING EXAMPLE ---
# The JSON xyz data is in the optical frame (Z-forward), NOT camera_link
CAMERA_FRAME = 'camera_color_optical_frame'   

# The robot expects commands in this frame
PLANNING_FRAME = 'world_world' 

VISUALIZE_TF = True  

HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0
}
# =================================================

class ScrewZoneTargeter(Node):
    def __init__(self):
        super().__init__('screw_zone_targeter')
        
        self.xarm = MotionBackend(self, "xarm_arm")
        
        self.subscription = self.create_subscription(
            String,
            '/vision/agent_state', 
            self.vision_callback,
            10
        )
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.latest_screw_targets = []
        self.data_lock = threading.Lock()
        
        self.get_logger().info(f"✅ ScrewTargeter Ready. Listening to {CAMERA_FRAME}...")

    def publish_debug_tf(self, x, y, z, screw_id):
        if not VISUALIZE_TF: return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = PLANNING_FRAME
        t.child_frame_id = f"screw_target_{screw_id}"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def check_connections(self):
        print("⏳ Checking connection to MoveGroup servers...")
        if not self.xarm._action_client.wait_for_server(timeout_sec=3.0):
            print("❌ ERROR: xArm5 MoveGroup Action Server NOT found!")
            return False
        print("✅ MoveGroup Server Connected.")
        return True

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            objects = data.get("global_view", {}).get("objects", [])
            
            new_targets = []
            for obj in objects:
                label = obj.get("label", "")
                if "Screw_Zone" in label and "xyz" in obj:
                    target = {
                        "id": obj.get("id"),
                        "label": label,
                        "xyz": obj.get("xyz"), 
                        "confidence": obj.get("confidence", 0.0)
                    }
                    new_targets.append(target)
            
            with self.data_lock:
                self.latest_screw_targets = new_targets
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def run_logic(self):
        if not self.check_connections(): return

        print(f"\n⏳ Waiting for Vision Data from {CAMERA_FRAME}...")
        
        while rclpy.ok():
            current_targets = []
            with self.data_lock:
                current_targets = copy.deepcopy(self.latest_screw_targets)
            
            if not current_targets:
                time.sleep(1.0)
                continue

            print(f"\n🔎 FOUND {len(current_targets)} SCREW ZONES. PROCESSING...")
            
            for screw in current_targets:
                if not rclpy.ok(): return
                self.process_single_target(screw)
                
            print("\n✅ Batch Complete. Waiting 5s...")
            time.sleep(5.0) 

    def process_single_target(self, part_data):
        p_id = part_data['id']
        label = part_data['label']
        raw_xyz = part_data['xyz']
        
        print("\n" + "="*60)
        print(f"🔩 TARGET DETECTED: {label} (ID: {p_id})")
        
        source_pose = Pose()
        source_pose.position.x = raw_xyz[0]
        source_pose.position.y = raw_xyz[1]
        source_pose.position.z = raw_xyz[2]
        source_pose.orientation.w = 1.0

        # --- TRANSFORM LOGIC ---
        # 1. Source: camera_color_optical_frame (Z-forward)
        # 2. Target: world_world (Robot Frame)
        # TF Tree Path: optical_frame -> u1_base_link -> world_world
        target_stamped = None
        for _ in range(5):
            target_stamped = self.xarm.get_transformed_pose(
                source_pose, 
                CAMERA_FRAME,     # <--- Corrected Source Frame
                PLANNING_FRAME    # <--- Corrected Target Frame
            )
            if target_stamped: break
            time.sleep(0.2)

        if not target_stamped:
            print(f"❌ TF Error: Could not transform {CAMERA_FRAME} -> {PLANNING_FRAME}")
            print("   (Check if your calibration publisher is running)")
            return

        world_x = target_stamped.pose.position.x
        world_y = target_stamped.pose.position.y
        base_z  = target_stamped.pose.position.z
        final_z = base_z + TOOL_LENGTH_XARM + SAFETY_BOUNDARY_Z

        # Publish Debug TF
        self.publish_debug_tf(world_x, world_y, final_z, p_id)

        print(f"   📸 Raw ({CAMERA_FRAME}): {raw_xyz}")
        print(f"   🤖 Transformed ({PLANNING_FRAME}): X={world_x:.4f} Y={world_y:.4f} Z={base_z:.4f}")
        print(f"   🛑 Final Target (+Offset): Z={final_z:.4f}")
        
        user_input = input(f"👉 Press ENTER to move to ID {p_id} ('s'=skip, 'q'=quit): ")
        if user_input.lower() == 's': return
        if user_input.lower() == 'q': sys.exit(0)

        print(f"   🚀 Moving...")
        success = self.xarm.move_to_pose_robust(
            world_x, world_y, final_z, {}, 
            link_name="xarm5_link5", 
            frame_id=PLANNING_FRAME, 
            velocity=0.1
        )

        if success:
            print(f"   ✅ Success! Waiting 3s...")
            time.sleep(3.0)
            print("   🏠 Homing...")
            self.xarm.move_to_joint_positions(HOME_JOINTS, filter_prefix="xarm5", velocity=0.2)
            time.sleep(1.0)
        else:
            print(f"   ❌ Move Failed!")

def main(args=None):
    rclpy.init(args=args)
    node = ScrewZoneTargeter()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    try:
        node.run_logic()
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()