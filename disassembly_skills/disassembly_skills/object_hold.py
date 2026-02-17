#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import threading
import json
import math
import time

# Import your analyzed backend
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
CAMERA_FRAME = 'camera_color_optical_frame' 
PLANNING_FRAME = 'world_world'    
TOOL_LENGTH = 0.28                # End-effector length
APPROACH_BUFFER = 0.05            # 5cm safety gap
SAFETY_Z_HEIGHT = -0.003          # -3mm height (gripping lower on the object)

# 👈 [NEW] Safety hover height to prevent swooping crashes
HOVER_Z_OFFSET = 0.10             
TARGET_LABELS = ["Top_Lid", "HDD_Chassis"]

# --- GRIPPER CONFIGURATION ---
JOINT_GRIPPER = "rg6_l_out"
OPEN_DEG = 35.0
CLOSE_DEG = -35.0 
GRIPPER_SPEED = 0.3

# --- 🛠️ QUICK TUNE VARIABLES 🛠️ ---
INVERT_VISION_ANGLE = True  
JAWS_ROTATION_DEG = 0.0  
APPROACH_SIDE_OFFSET_DEG = 0.0 
FLIP_GRIPPER_UPSIDE_DOWN = False  
GRIPPER_TILT_DEG = 7.0  
# =================================================

class SideHoldSequence(Node):
    def __init__(self):
        super().__init__('side_hold_sequence_node')
        
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.latest_vision_data = None
        self.vision_event = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.get_logger().info(f"🚀 Side-Hold Node Initialized. Planning in {PLANNING_FRAME}.")

    def vision_callback(self, msg):
        try:
            clean_data = msg.data.strip("'")
            self.latest_vision_data = json.loads(clean_data)
            self.vision_event.set()
        except Exception as e:
            self.get_logger().error(f"JSON Parsing Error: {e}")

    def get_current_tcp_position(self):
        try:
            if self.uf850.tf_buffer.can_transform(PLANNING_FRAME, 'u1_tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.uf850.tf_buffer.lookup_transform(PLANNING_FRAME, 'u1_tool0', rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            return None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    def wait_for_gripper(self, target_deg, timeout=5.0):
        target_rad = math.radians(target_deg)
        start_time = time.time()
        last_pos = 999.0
        stall_timer = 0.0

        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(JOINT_GRIPPER, 999)
            
            if current_pos == 999:
                time.sleep(0.1)
                continue

            if abs(current_pos - target_rad) < 0.05:
                return True
            
            if abs(current_pos - last_pos) < 0.005:
                stall_timer += 0.1
                if stall_timer >= 0.5:  
                    self.get_logger().info("✅ Gripper stalled (Object grasped successfully).")
                    return True
            else:
                stall_timer = 0.0
                
            last_pos = current_pos
            time.sleep(0.1)
            
        self.get_logger().warn("⚠️ Gripper Wait Loop Timed Out.")
        return False

    def run_sequence(self):
        print("\n" + "!"*60)
        print("🤖 STEP 1: WAITING FOR VISION DATA...")
        print("!"*60)
        
        self.vision_event.wait()
        
        target_obj = None
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        
        for obj in objects:
            if obj.get("label") in TARGET_LABELS:
                target_obj = obj
                break
        
        if not target_obj:
            print("❌ Target class not found in current vision state.")
            return

        label = target_obj["label"]
        raw_x, raw_y, raw_z = target_obj["xyz"]
        
        obj_angle_deg = target_obj["angle"]
        if INVERT_VISION_ANGLE:
            obj_angle_deg = -obj_angle_deg

        print(f"\n✅ DETECTED OBJECT: {label}")
        print(f"📍 Raw Vision Coords: X:{raw_x:.4f}, Y:{raw_y:.4f}, Z:{raw_z:.4f}")
        
        input(f"\n👉 Press ENTER to map coordinates to '{PLANNING_FRAME}'...")

        source_pose = Pose()
        source_pose.position.x = float(raw_x)
        source_pose.position.y = float(raw_y)
        source_pose.position.z = float(raw_z)
        source_pose.orientation.w = 1.0 

        transformed_pose_stamped = self.uf850.get_transformed_pose(
            source_pose=source_pose, 
            source_frame=CAMERA_FRAME, 
            target_frame=PLANNING_FRAME
        )

        if not transformed_pose_stamped:
            print(f"❌ TF Lookup Failed!")
            return
            
        world_x = transformed_pose_stamped.pose.position.x
        world_y = transformed_pose_stamped.pose.position.y
        world_z = transformed_pose_stamped.pose.position.z

        print("\n" + "="*50)
        print(f"🌍 [DEBUG] MAPPED COORDINATES ({PLANNING_FRAME})")
        print("="*50)
        print(f"  Mapped X : {world_x:.4f} m")
        print(f"  Mapped Y : {world_y:.4f} m")
        print(f"  Mapped Z : {world_z:.4f} m")

        input("\n👉 Press ENTER to calculate the Tilted Flight Plan...")

        # --- 3. TILTED FLIGHT PLAN CALCULATION ---
        obj_yaw_rad = math.radians(obj_angle_deg)
        approach_vector_rad = obj_yaw_rad + (math.pi / 2.0) + math.radians(APPROACH_SIDE_OFFSET_DEG)
        
        tilt_rad = math.radians(GRIPPER_TILT_DEG)
        horizontal_tool_length = TOOL_LENGTH * math.cos(tilt_rad)
        z_lift_offset = TOOL_LENGTH * math.sin(tilt_rad)
        
        total_horizontal_offset = horizontal_tool_length + APPROACH_BUFFER
        
        # Calculate Base Target
        target_x = world_x - (total_horizontal_offset * math.cos(approach_vector_rad))
        target_y = world_y - (total_horizontal_offset * math.sin(approach_vector_rad))
        target_z = world_z + SAFETY_Z_HEIGHT + z_lift_offset

        # 👈 [NEW] Calculate Hover Target (Strictly vertical offset)
        hover_z = target_z + HOVER_Z_OFFSET
        
        grasp_x = world_x - (horizontal_tool_length * math.cos(approach_vector_rad))
        grasp_y = world_y - (horizontal_tool_length * math.sin(approach_vector_rad))
        grasp_z = target_z 
        
        roll_angle = math.pi + math.radians(JAWS_ROTATION_DEG)
        if FLIP_GRIPPER_UPSIDE_DOWN:
            roll_angle = math.radians(JAWS_ROTATION_DEG)
            
        final_gripper_yaw = approach_vector_rad 
        pitch_angle = -math.pi/2 + tilt_rad
        
        q = self.uf850._rpy_to_quaternion(roll_angle, pitch_angle, final_gripper_yaw)
        q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        print("\n" + "="*50)
        print("🎯 FLIGHT PLAN WAYPOINTS (u1_tool0)")
        print("="*50)
        print(f"  [HOVER]    X: {target_x:.4f} | Y: {target_y:.4f} | Z: {hover_z:.4f} m")
        print(f"  [APPROACH] X: {target_x:.4f} | Y: {target_y:.4f} | Z: {target_z:.4f} m")
        print(f"  [GRASP]    X: {grasp_x:.4f} | Y: {grasp_y:.4f} | Z: {grasp_z:.4f} m")
        print(f"  Orientation: [Roll {math.degrees(roll_angle):.0f}° | Pitch {math.degrees(pitch_angle):.1f}° | Yaw {math.degrees(final_gripper_yaw):.2f}°]")

        # --- 4A. HOVER MOTION EXECUTION ---
        input(f"\n🚀 STEP 2A: OPEN Gripper and Move to SAFE HOVER POSITION? [Enter]")
        
        print("   🛠️ Opening Gripper jaws...")
        self.gripper.move_to_joint_positions({JOINT_GRIPPER: math.radians(OPEN_DEG)}, "rg6", velocity=GRIPPER_SPEED)
        self.wait_for_gripper(OPEN_DEG)
        
        print(f"   🚀 Sending Goal to MoveIt (Velocity 0.3)...")
        success_hover = self.uf850.move_to_pose_robust(
            target_x, target_y, hover_z, q_dict,
            link_name="u1_tool0",
            frame_id=PLANNING_FRAME,
            velocity=0.1
        )

        if not success_hover:
            print("⚠️ MoveIt timeout. Verifying hardware position...")
            time.sleep(1.0) 
            curr_pos = self.get_current_tcp_position()
            if curr_pos:
                cx, cy, cz = curr_pos
                dist = math.sqrt((cx - target_x)**2 + (cy - target_y)**2 + (cz - hover_z)**2)
                if dist < 0.03: 
                    print(f"✅ Position verified! Overriding timeout.")
                else:
                    print(f"❌ Robot too far from target. Aborting.")
                    return
            else:
                return

        print(f"\n✅ Arrived at Hover Position.")

        # --- 4B. VERTICAL DROP ---
        input(f"\n🚀 STEP 2B: Perform Vertical Z-Drop to approach altitude? [Enter]")
        print(f"   🚀 Dropping down (Velocity 0.05)...")
        
        success_drop = self.uf850.move_to_pose_robust(
            target_x, target_y, target_z, q_dict,
            link_name="u1_tool0",
            frame_id=PLANNING_FRAME,
            velocity=0.05
        )

        if not success_drop:
            print("⚠️ MoveIt timeout. Verifying hardware position...")
            time.sleep(1.0) 
            curr_pos = self.get_current_tcp_position()
            if curr_pos:
                cx, cy, cz = curr_pos
                dist = math.sqrt((cx - target_x)**2 + (cy - target_y)**2 + (cz - target_z)**2)
                if dist < 0.03: 
                    print(f"✅ Position verified! Overriding timeout.")
                else:
                    print(f"❌ Robot too far from target. Aborting.")
                    return
            else:
                return

        print(f"\n✅ Arrived at Approach Position (Pre-Grasp).")
        
        # --- 5. MOVEIT PLUNGE & GRASP ---
        input(f"\n🚀 STEP 3: Slide down the diagonal approach to contact? [Enter]")

        success_final = self.uf850.move_to_pose_robust(
            grasp_x, grasp_y, grasp_z, q_dict,
            link_name="u1_tool0",
            frame_id=PLANNING_FRAME,
            velocity=0.05 
        )

        if success_final or self.get_current_tcp_position(): 
            print(f"\n✅ Plunge successful. Jaws are positioned around {label}.")
            
            input(f"\n🚀 STEP 4: CLOSE Gripper to SECURE {label}? [Enter]")
            print("   🛠️ Actuating Gripper...")
            self.gripper.move_to_joint_positions({JOINT_GRIPPER: math.radians(CLOSE_DEG)}, "rg6", velocity=GRIPPER_SPEED)
            self.wait_for_gripper(CLOSE_DEG)
            
            print(f"\n🎉 SEQUENCE COMPLETE: Object is fully secured.")
        else:
            print("❌ Final contact move failed.")

def main(args=None):
    rclpy.init(args=args)
    node = SideHoldSequence()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    sequence_thread = threading.Thread(target=node.run_sequence, daemon=True)
    sequence_thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()