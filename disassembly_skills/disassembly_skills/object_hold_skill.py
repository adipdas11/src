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

class ObjectHoldSkill(Node):
    def __init__(self):
        """
        Initializes the hold skill as a self-contained ROS 2 Node.
        """
        super().__init__('object_hold_skill_node')
        
        # 1. Backends initialized under this specific node
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # 2. Dedicated Vision Subscriber
        self.latest_vision_data = None
        self.vision_event = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        self.get_logger().info("🚀 Object Hold Skill Node Active.")

        # ================= CONFIGURATION =================
        self.CAMERA_FRAME = 'camera_color_optical_frame' 
        self.PLANNING_FRAME = 'world_world'    
        self.TOOL_LENGTH = 0.28                
        self.APPROACH_BUFFER = 0.05            
        self.SAFETY_Z_HEIGHT = -0.003          
        self.HOVER_Z_OFFSET = 0.10             
        
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0 
        self.GRIPPER_SPEED = 0.3

        self.INVERT_VISION_ANGLE = True  
        self.JAWS_ROTATION_DEG = 0.0  
        self.APPROACH_SIDE_OFFSET_DEG = 0.0 
        self.FLIP_GRIPPER_UPSIDE_DOWN = False  
        self.GRIPPER_TILT_DEG = 7.0  
        # =================================================

    def vision_callback(self, msg):
        try:
            clean_data = msg.data.strip("'")
            self.latest_vision_data = json.loads(clean_data)
            self.vision_event.set()
        except Exception as e:
            self.get_logger().error(f"JSON Parsing Error: {e}")

    def get_current_tcp_position(self):
        try:
            if self.uf850.tf_buffer.can_transform(self.PLANNING_FRAME, 'u1_tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, 'u1_tool0', rclpy.time.Time())
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
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
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

    def execute_hold(self, part_id: int, target_label: str, interactive=True):
        """
        The main public function to be called by the Master Node or Standalone runner.
        """
        print("\n" + "!"*60)
        print(f"🤖 INITIATING HOLD SKILL FOR: {target_label} (ID: {part_id})")
        print("!"*60)

        # 1. Clear the event so we wait for a FRESH vision frame
        self.vision_event.clear()
        print("   Waiting for fresh vision data...")
        if not self.vision_event.wait(timeout=10.0):
            print("❌ Vision data timeout. Is the camera node running?")
            return False
        
        # 2. Find the object in the global view
        target_obj = None
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        
        for obj in objects:
            if obj.get("id") == part_id and obj.get("label") == target_label:
                target_obj = obj
                break
        
        if not target_obj:
            print(f"❌ {target_label} (ID: {part_id}) not found in current vision state.")
            return False

        raw_x, raw_y, raw_z = target_obj["xyz"]
        obj_angle_deg = target_obj["angle"]
        if self.INVERT_VISION_ANGLE:
            obj_angle_deg = -obj_angle_deg

        print(f"\n✅ Target Locked: {target_label}")
        print(f"📍 Raw Vision Coords: X:{raw_x:.4f}, Y:{raw_y:.4f}, Z:{raw_z:.4f}")
        
        if interactive: input(f"\n👉 Press ENTER to map coordinates to '{self.PLANNING_FRAME}'...")

        # --- 3. COORDINATE TRANSFORMATION ---
        source_pose = Pose()
        source_pose.position.x = float(raw_x)
        source_pose.position.y = float(raw_y)
        source_pose.position.z = float(raw_z)
        source_pose.orientation.w = 1.0 

        transformed_pose_stamped = self.uf850.get_transformed_pose(
            source_pose=source_pose, 
            source_frame=self.CAMERA_FRAME, 
            target_frame=self.PLANNING_FRAME
        )

        if not transformed_pose_stamped:
            print(f"❌ TF Lookup Failed!")
            return False
            
        world_x = transformed_pose_stamped.pose.position.x
        world_y = transformed_pose_stamped.pose.position.y
        world_z = transformed_pose_stamped.pose.position.z

        if interactive: input("\n👉 Press ENTER to calculate the Tilted Flight Plan...")

        # --- 4. TILTED FLIGHT PLAN CALCULATION ---
        obj_yaw_rad = math.radians(obj_angle_deg)
        approach_vector_rad = obj_yaw_rad + (math.pi / 2.0) + math.radians(self.APPROACH_SIDE_OFFSET_DEG)
        
        tilt_rad = math.radians(self.GRIPPER_TILT_DEG)
        horizontal_tool_length = self.TOOL_LENGTH * math.cos(tilt_rad)
        z_lift_offset = self.TOOL_LENGTH * math.sin(tilt_rad)
        
        total_horizontal_offset = horizontal_tool_length + self.APPROACH_BUFFER
        
        target_x = world_x - (total_horizontal_offset * math.cos(approach_vector_rad))
        target_y = world_y - (total_horizontal_offset * math.sin(approach_vector_rad))
        target_z = world_z + self.SAFETY_Z_HEIGHT + z_lift_offset

        hover_z = target_z + self.HOVER_Z_OFFSET
        
        grasp_x = world_x - (horizontal_tool_length * math.cos(approach_vector_rad))
        grasp_y = world_y - (horizontal_tool_length * math.sin(approach_vector_rad))
        grasp_z = target_z 
        
        roll_angle = math.pi + math.radians(self.JAWS_ROTATION_DEG)
        if self.FLIP_GRIPPER_UPSIDE_DOWN:
            roll_angle = math.radians(self.JAWS_ROTATION_DEG)
            
        final_gripper_yaw = approach_vector_rad 
        pitch_angle = -math.pi/2 + tilt_rad
        
        q = self.uf850._rpy_to_quaternion(roll_angle, pitch_angle, final_gripper_yaw)
        q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        # --- 5A. HOVER MOTION EXECUTION ---
        if interactive: input(f"\n🚀 STEP 2A: OPEN Gripper and Move to SAFE HOVER POSITION? [Enter]")
        
        print("   🛠️ Opening Gripper jaws...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        
        print(f"   🚀 Sending Goal to MoveIt (Velocity 0.3)...")
        success_hover = self.uf850.move_to_pose_robust(
            target_x, target_y, hover_z, q_dict,
            link_name="u1_tool0", frame_id=self.PLANNING_FRAME, velocity=0.1
        )

        if not success_hover:
            time.sleep(1.0) 
            curr_pos = self.get_current_tcp_position()
            if curr_pos and math.sqrt((curr_pos[0]-target_x)**2 + (curr_pos[1]-target_y)**2 + (curr_pos[2]-hover_z)**2) < 0.03: 
                pass
            else:
                print(f"❌ Robot too far from hover target. Aborting.")
                return False

        # --- 5B. VERTICAL DROP ---
        if interactive: input(f"\n🚀 STEP 2B: Perform Vertical Z-Drop to approach altitude? [Enter]")
        
        success_drop = self.uf850.move_to_pose_robust(
            target_x, target_y, target_z, q_dict,
            link_name="u1_tool0", frame_id=self.PLANNING_FRAME, velocity=0.05
        )

        if not success_drop:
            time.sleep(1.0) 
            curr_pos = self.get_current_tcp_position()
            if curr_pos and math.sqrt((curr_pos[0]-target_x)**2 + (curr_pos[1]-target_y)**2 + (curr_pos[2]-target_z)**2) < 0.03: 
                pass
            else:
                print(f"❌ Robot too far from approach target. Aborting.")
                return False
        
        # --- 6. MOVEIT PLUNGE & GRASP ---
        if interactive: input(f"\n🚀 STEP 3: Slide down the diagonal approach to contact? [Enter]")

        success_final = self.uf850.move_to_pose_robust(
            grasp_x, grasp_y, grasp_z, q_dict,
            link_name="u1_tool0", frame_id=self.PLANNING_FRAME, velocity=0.05 
        )

        if success_final or self.get_current_tcp_position(): 
            if interactive: input(f"\n🚀 STEP 4: CLOSE Gripper to SECURE {target_label}? [Enter]")
            print(f"   🛠️ Actuating Gripper to secure {target_label}...")
            self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
            self.wait_for_gripper(self.CLOSE_DEG)
            
            print(f"\n🎉 HOLD SKILL COMPLETE: {target_label} is fully secured.")
            return True
        else:
            print("❌ Final contact move failed.")
            return False

# =====================================================================
# STANDALONE EXECUTION BLOCK (When run directly via 'ros2 run')
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    
    # Initialize the Skill Node
    hold_node = ObjectHoldSkill()
    
    # Spin it in a background thread so the interactive loop isn't blocked
    executor = MultiThreadedExecutor()
    executor.add_node(hold_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # Run the sequence interactively
    try:
        # Default test parameters
        hold_node.execute_hold(part_id=0, target_label="Top_Lid", interactive=True)
    except KeyboardInterrupt:
        pass
        
    hold_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()