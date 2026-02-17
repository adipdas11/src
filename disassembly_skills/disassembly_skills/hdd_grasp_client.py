#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import Point, TransformStamped
from gpd_ros2_msgs.srv import DetectConstrainedGrasps
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import threading
import json
import time
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
SPEED = 0.2           
TOOL_LENGTH = 0.28    # 28cm RG6 Gripper length offset
HOVER_DIST = 0.05     # 5cm hover gap

# --- FRAMES ---
# Fixed to match the valid frame from your RViz TF Tree
CAMERA_FRAME = 'camera_color_optical_frame' 
PLANNING_FRAME = 'world_world'        
TOOL_LINK = 'u1_tool0'
# =================================================

class UnifiedGraspAgent(Node):
    def __init__(self):
        super().__init__('unified_grasp_agent')
        
        # 1. Backends
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # 2. State & Events
        self.lid_segments = None
        self.masked_cloud = None
        self.grasp_target = None 
        
        self.vision_event = threading.Event()
        self.cloud_event = threading.Event()
        self.grasp_event = threading.Event()
        
        # 3. Communications
        self.gpd_cli = self.create_client(DetectConstrainedGrasps, 'detect_constrained_grasps')
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_cb, 10)
        self.cloud_sub = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.cloud_cb, 1)
        
        # 4. TF Architecture
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_timer = self.create_timer(0.05, self.broadcast_grasp_frames) 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("🚀 Unified Perception-to-Action Agent Online.")

    # ================== PERCEPTION CALLBACKS ==================
    def vision_cb(self, msg):
        if self.vision_event.is_set(): return
        try:
            data = json.loads(msg.data.strip("'"))
            for obj in data.get('global_view', {}).get('objects', []):
                if obj['label'] == 'Top_Lid':
                    self.lid_segments = obj['segments']
                    self.vision_event.set()
        except Exception: pass

    def cloud_cb(self, msg):
        if not self.vision_event.is_set() or self.cloud_event.is_set(): return
        
        from shapely.geometry import Point as ShapePoint
        from shapely.geometry.polygon import Polygon
        
        poly = Polygon(self.lid_segments)
        # D455 Intrinsics
        fx, fy, cx, cy = 605.0, 605.0, 320.0, 240.0 
        
        points_gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        filtered = []
        raw_count = 0

        for p in points_gen:
            raw_count += 1
            x, y, z = p[0], p[1], p[2]
            if z <= 0: continue
            
            u, v = int((x * fx / z) + cx), int((y * fy / z) + cy)
            if poly.contains(ShapePoint(u, v)):
                filtered.append([x, y, z])

        if filtered:
            self.masked_cloud = pc2.create_cloud_xyz32(msg.header, filtered)
            self.get_logger().info(f"✨ Masking Complete! Dropped {raw_count - len(filtered)} background points.")
            self.cloud_event.set()

    # ================== GPD MATH ==================
    def process_gpd_response(self, future):
        try:
            grasps = future.result().grasp_configs.grasps
            if not grasps:
                self.get_logger().error("❌ GPD found no grasps on the Lid.")
                return

            print("\n" + "="*50)
            print("📋 ALL GRASP CANDIDATES (CAMERA FRAME)")
            print("="*50)
            for i, g in enumerate(grasps):
                print(f" [{i:02d}] Score: {g.score.data:.2f} | Pos: X={g.position.x:.3f}, Y={g.position.y:.3f}, Z={g.position.z:.3f}")

            best = grasps[0]
            
            print("\n" + "="*50)
            print("🏆 BEST CANDIDATE RAW DATA")
            print("="*50)
            print(f" Approach (Z) : [{best.approach.x:.3f}, {best.approach.y:.3f}, {best.approach.z:.3f}]")
            print(f" Binormal (Y) : [{best.binormal.x:.3f}, {best.binormal.y:.3f}, {best.binormal.z:.3f}]")
            print(f" Axis     (X) : [{best.axis.x:.3f}, {best.axis.y:.3f}, {best.axis.z:.3f}]")
            
            # Bulletproof RHS Math
            z = np.array([best.approach.x, best.approach.y, best.approach.z])
            x_raw = np.array([best.axis.x, best.axis.y, best.axis.z])
            
            if np.linalg.norm(z) < 1e-6: z = np.array([0.0, 0.0, -1.0])
            else: z = z / np.linalg.norm(z)
            
            y_calc = np.cross(z, x_raw)
            if np.linalg.norm(y_calc) > 1e-4:
                y = y_calc / np.linalg.norm(y_calc)
                x = np.cross(y, z)
            else:
                dummy = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                y = np.cross(z, dummy)
                y = y / np.linalg.norm(y)
                x = np.cross(y, z)

            rot = np.column_stack((x, y, z))
            if np.linalg.det(rot) < 0: rot[:, 0] = -rot[:, 0]
            
            q = R.from_matrix(rot).as_quat()
            
            print(f" RHS Quat     : [x={q[0]:.3f}, y={q[1]:.3f}, z={q[2]:.3f}, w={q[3]:.3f}]")
            
            self.grasp_target = {
                'x': best.position.x, 'y': best.position.y, 'z': best.position.z,
                'qx': q[0], 'qy': q[1], 'qz': q[2], 'qw': q[3], 'score': best.score.data
            }
            self.grasp_event.set()
            
        except Exception as e:
            self.get_logger().error(f"GPD Processing Failed: {e}")

    # ================== TF BROADCASTER ==================
    def broadcast_grasp_frames(self):
        if not self.grasp_target: return
        try:
            now = self.get_clock().now().to_msg()
            
            # 1. Fingertip Target
            t1 = TransformStamped()
            t1.header.stamp, t1.header.frame_id, t1.child_frame_id = now, CAMERA_FRAME, 'gpd_tcp_target'
            t1.transform.translation.x, t1.transform.translation.y, t1.transform.translation.z = self.grasp_target['x'], self.grasp_target['y'], self.grasp_target['z']
            t1.transform.rotation.x, t1.transform.rotation.y, t1.transform.rotation.z, t1.transform.rotation.w = self.grasp_target['qx'], self.grasp_target['qy'], self.grasp_target['qz'], self.grasp_target['qw']
            self.tf_broadcaster.sendTransform(t1)

            # 2. Wrist Final Target (Compensated for Gripper Length)
            t2 = TransformStamped()
            t2.header.stamp, t2.header.frame_id, t2.child_frame_id = now, 'gpd_tcp_target', 'wrist_target_final'
            t2.transform.translation.z = -TOOL_LENGTH
            t2.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t2)

            # 3. Wrist Hover Target (Compensated + 5cm Hover Gap)
            t3 = TransformStamped()
            t3.header.stamp, t3.header.frame_id, t3.child_frame_id = now, 'gpd_tcp_target', 'wrist_target_hover'
            t3.transform.translation.z = -(TOOL_LENGTH + HOVER_DIST)
            t3.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t3)
        except Exception: pass

    # ================== INTERACTIVE SEQUENCE ==================
    def get_world_pose(self, frame_name):
        """Robust lookup that retries if the TF tree is lagging."""
        for attempt in range(4):
            if self.tf_buffer.can_transform(PLANNING_FRAME, frame_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.tf_buffer.lookup_transform(PLANNING_FRAME, frame_name, rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z, {
                    'qx': t.transform.rotation.x, 'qy': t.transform.rotation.y, 'qz': t.transform.rotation.z, 'qw': t.transform.rotation.w
                }
            print(f"  [TF Retry {attempt+1}/4] Waiting for '{frame_name}' to reach '{PLANNING_FRAME}'...")
            time.sleep(1.0)
        return None, None, None, None

    def run_sequence(self):
        print("\n" + "="*50)
        print("🤖 STEP 1: PERCEPTION ACQUISITION")
        print("="*50)
        print("⏳ Waiting for /vision/agent_state 'Top_Lid'...")
        self.vision_event.wait()
        print("✅ Vision locked. Waiting for PointCloud...")
        self.cloud_event.wait()
        
        input("\n👉 [INPUT] Cloud masked successfully. Press ENTER to query GPD Server...")
        
        while not self.gpd_cli.wait_for_service(timeout_sec=1.0):
            print("⏳ Waiting for GPD Service...")
        
        req = DetectConstrainedGrasps.Request()
        req.params_policy = 0 
        req.cloud_indexed.cloud_sources.cloud = self.masked_cloud
        req.cloud_indexed.cloud_sources.view_points = [Point(x=0.0, y=0.0, z=0.0)]
        
        print("🧠 Processing constraints in C++ Neural Net...")
        future = self.gpd_cli.call_async(req)
        future.add_done_callback(self.process_gpd_response)
        
        self.grasp_event.wait() 

        print("\n" + "="*50)
        print("🤖 STEP 2: RVIZ VERIFICATION & TF TRANSFORM")
        print("="*50)
        print("Broadcasted virtual frames compensating for 0.28m RG6 length.")
        input("👉 [INPUT] Ensure Rosbag is PLAYING. Open RViz to verify axes. Press ENTER to map coordinates...")

        # --- HOVER MOTION ---
        hx, hy, hz, hq = self.get_world_pose('wrist_target_hover')
        if hx is None:
            print(f"❌ TF Error: Could not resolve Hover coordinates to {PLANNING_FRAME}.")
            print("   Hint: Is your rosbag paused? Press space in the rosbag terminal to resume play.")
            return
            
        print("\n" + "="*50)
        print("🌍 WORLD_WORLD TRANSFORMED COORDINATES")
        print("="*50)
        print(f" Pre-Grasp Hover : X={hx:.3f}, Y={hy:.3f}, Z={hz:.3f}")
        
        print("\n" + "="*50)
        print("🤖 STEP 3: PRE-GRASP HOVER")
        print("="*50)
        input("👉 [INPUT] Press ENTER to command UF850 to Hover Pose...")
        
        if not self.uf850.move_to_pose_robust(hx, hy, hz, hq, link_name=TOOL_LINK, frame_id=PLANNING_FRAME, velocity=SPEED):
            print("❌ MoveIt Planning Failed.")
            return

        # --- FINAL GRASP MOTION ---
        gx, gy, gz, gq = self.get_world_pose('wrist_target_final')
        print(f"\n🌍 Final Insertion : X={gx:.3f}, Y={gy:.3f}, Z={gz:.3f}")
        
        print("\n" + "="*50)
        print("🤖 STEP 4: FINAL GRASP INSERTION")
        print("="*50)
        input("👉 [INPUT] Press ENTER to command UF850 to slide onto HDD Lid...")
        
        if not self.uf850.move_to_pose_robust(gx, gy, gz, gq, link_name=TOOL_LINK, frame_id=PLANNING_FRAME, velocity=SPEED * 0.3):
            print("❌ MoveIt Planning Failed.")
            return

        # --- GRIPPER ACTUATION ---
        print("\n" + "="*50)
        print("🤖 STEP 5: GRIPPER ACTUATION")
        print("="*50)
        input("👉 [INPUT] Position reached. Press ENTER to close OnRobot RG6...")
        # self.gripper.close() # Uncomment when physically running the hardware
        print("🗜️ GRIPPER COMMAND SENT.")
        
        
        
        print("\n✅ Sequence Fully Complete. Agent ready for extraction trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedGraspAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    thread = threading.Thread(target=node.run_sequence, daemon=True)
    thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()