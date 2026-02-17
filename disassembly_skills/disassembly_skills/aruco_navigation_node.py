#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
import tf2_ros
import threading
import json
import math
import time

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
SPEED = 0.2           
TOOL_LENGTH = 0.028     
HOVER_OFFSET = 0.03   # 3cm above the part

# --- FRAMES ---
ROBOT_BASE_FRAME = 'u1_base_link'     # Where the camera data originates
TARGET_FRAME = 'target_part_world'    # The broadcasted object frame
PLANNING_FRAME = 'world_world'        # MoveIt's universal floor frame
# =================================================

class VisionToMoveItTester(Node):
    def __init__(self):
        super().__init__('vision_tf_sequence_tester')
        
        # 1. Motion Backends
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # 2. TF Broadcaster 
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.broadcast_target = None  
        self.broadcast_count = 0
        self.tf_timer = self.create_timer(0.1, self.broadcast_timer_cb) # 10Hz
        
        # 3. TF Listener 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 4. Vision Subscriber
        self.vision_data_received = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        self.get_logger().info(f"🚀 Architecture Active. Planning in {PLANNING_FRAME}.")

    def vision_callback(self, msg):
        """Reads JSON and updates the target for the TF Broadcaster."""
        try:
            data = json.loads(msg.data)
            for obj in data.get("global_view", {}).get("objects", []):
                if obj.get("id") == 0: 
                    self.broadcast_target = obj
                    self.vision_data_received.set()
                    break
        except json.JSONDecodeError:
            self.get_logger().error("❌ JSON parse error.")

    def broadcast_timer_cb(self):
        """Continuously broadcasts the object's position as a TF frame."""
        if not self.broadcast_target: 
            return
        
        try:
            x, y, z = self.broadcast_target["xyz"]
            angle_deg = self.broadcast_target["angle"]
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = ROBOT_BASE_FRAME  
            t.child_frame_id = TARGET_FRAME       
            
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z)
            
            yaw_rad = math.radians(angle_deg)
            q = self.uf850._rpy_to_quaternion(0.0, 0.0, yaw_rad)
            t.transform.rotation = q
            
            self.tf_broadcaster.sendTransform(t)
            
            # Print a debug message once every 2 seconds to prove it's broadcasting
            self.broadcast_count += 1
            if self.broadcast_count % 20 == 0:
                self.get_logger().info(f"📡 Broadcasting '{TARGET_FRAME}' at X:{x:.3f} Y:{y:.3f} Z:{z:.3f}")
                
        except Exception as e:
            self.get_logger().error(f"Broadcaster Error: {e}")

    def get_target_coordinates(self):
        """Looks up the target frame relative to world_world."""
        try:
            if self.tf_buffer.can_transform(PLANNING_FRAME, TARGET_FRAME, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.tf_buffer.lookup_transform(PLANNING_FRAME, TARGET_FRAME, rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            else:
                return None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    def run_sequence(self):
        """The step-by-step interactive execution thread."""
        print("\n" + "="*50)
        print("🤖 STEP 1: Waiting for Vision Data...")
        
        if not self.vision_data_received.wait(timeout=10.0):
            print("❌ No vision data received for Part ID 0.")
            return

        label = self.broadcast_target.get("label", "Unknown")
        print(f"✅ Found: {label}. Frame '{TARGET_FRAME}' should now appear in RViz.")
        
        # --- WAIT FOR USER TO TRIGGER TF LOOKUP ---
        input(f"\n👉 Press ENTER to lookup '{TARGET_FRAME}' coordinates in '{PLANNING_FRAME}'...")
        
        coords = self.get_target_coordinates()
        if coords is None:
            print(f"❌ TF Lookup Failed! Cannot map to {PLANNING_FRAME}.")
            return
            
        tx, ty, tz = coords
        print("\n" + "="*50)
        print("🎯 [DEBUG] TF2 MAPPED COORDINATES")
        print("="*50)
        print(f"  Mapped X : {tx:.4f} m")
        print(f"  Mapped Y : {ty:.4f} m")
        print(f"  Mapped Z : {tz:.4f} m")
        
        # --- CALCULATE FINAL POSITIONS ---
        target_z = tz + TOOL_LENGTH + HOVER_OFFSET
        
        obj_yaw_rad = math.radians(self.broadcast_target["angle"])
        q_target = self.uf850._rpy_to_quaternion(math.pi, 0.0, obj_yaw_rad)
        q_dict = {'qx': q_target.x, 'qy': q_target.y, 'qz': q_target.z, 'qw': q_target.w}

        print("\n" + "="*50)
        print("🚀 [DEBUG] UF850 FLIGHT PLAN")
        print("="*50)
        print(f"  Target Z : {target_z:.4f} m (Includes hover + tool length)")

        # --- WAIT FOR USER TO TRIGGER MOTION ---
        input(f"\n👉 Press ENTER to move UF850 to TOP-DOWN HOVER (Z={target_z:.3f})...")
        print(f"   🚀 Sending Goal to MoveIt in frame '{PLANNING_FRAME}'...")
        
        success = self.uf850.move_to_pose_robust(
            tx, ty, target_z, q_dict, 
            link_name="u1_tool0", 
            frame_id=PLANNING_FRAME,  
            velocity=SPEED
        )

        if success:
            print("\n✅ Sequence Complete! Verify the 3cm gap and X/Y alignment.")
        else:
            print("\n❌ MoveIt failed to find a valid trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = VisionToMoveItTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # --- THREADING FIX ---
    # Put the blocking run_sequence in the background thread
    sequence_thread = threading.Thread(target=node.run_sequence, daemon=True)
    sequence_thread.start()
    
    # Let rclpy and TF spin unhindered on the main thread
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()