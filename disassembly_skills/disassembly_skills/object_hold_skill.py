#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
import threading
import json
import math
import time

# Import the backend containing SDK controllers & dynamic IP switching
from disassembly_skills.motion_backend import MotionBackend

class ObjectHoldSkill(Node):
    def __init__(self):
        super().__init__('object_hold_skill_node')
        
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.latest_vision_data = None
        self.vision_event = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)

        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        
        self.get_logger().info("🚀 Object Hold Skill Node: Native SDK Tactile Mode Active.")

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
        self.GRIPPER_TILT_DEG = 8.0  

        self.CONTACT_JOINT = "u1_joint5"
        self.TORQUE_THRESHOLD =1.0    
        self.RETRACT_DISTANCE = 0.02  
        self.ROBOT_EE_LINK = "u1_tool0"

        self.FINAL_ALIGN_YAW_DEG = 90.0       
        self.REMOVE_TILT_ON_ALIGN = False    
        # =================================================

    def publish_hold_status(self, is_held: bool):
        msg = Bool()
        msg.data = is_held
        self.hold_status_pub.publish(msg)
        self.get_logger().info(f"📢 Published Hold Status: {is_held}")

    def vision_callback(self, msg):
        try:
            clean_data = msg.data.strip("'")
            self.latest_vision_data = json.loads(clean_data)
            self.vision_event.set()
        except Exception as e:
            self.get_logger().error(f"JSON Parsing Error: {e}")

    def get_current_tcp_position(self):
        try:
            if self.uf850.tf_buffer.can_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
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
                time.sleep(0.1); continue
            if abs(current_pos - target_rad) < 0.05: return True
            if abs(current_pos - last_pos) < 0.005:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    self.get_logger().info("✅ Gripper stalled (Object grasped).")
                    return True
            else: stall_timer = 0.0
            last_pos = current_pos
            time.sleep(0.1)
        return False

    def descend_until_contact(self, q_dict, custom_retract=None):
        retract_dist = custom_retract if custom_retract is not None else self.RETRACT_DISTANCE

        self.get_logger().info(f"⬇️ Starting Native SDK Tactile Descent (Target Joint: {self.CONTACT_JOINT})")
        self.contact_detected = False
        last_print_time = time.time()

        def check_force():
            nonlocal last_print_time
            effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, -99.0)
            
            if time.time() - last_print_time > 0.1:
                print(f"   [Live Debug] Current {self.CONTACT_JOINT} Effort: {effort:.4f} Nm")
                last_print_time = time.time()

            if effort > self.TORQUE_THRESHOLD:
                self.get_logger().warn(f"💥 THRESHOLD CROSSED! Actual Spike: {effort:.4f} Nm")
                self.contact_detected = True
                return True 
            return False

        # 1. Perform the SDK Descent
        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=0.15, 
            speed_mm_s=10.0, 
            check_force_callback=check_force
        )

        if not self.contact_detected:
            self.get_logger().error("❌ Max descent reached without sensing contact.")
            self.uf850.reset_robot() 
            return False

        # 2. Clear the hardware collision state
        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        # 3. Use SDK to Retract (Bypassing MoveIt completely)
        self.get_logger().info(f"⬆️ Attempting SDK Retraction: {retract_dist*1000}mm UP...")
        
        # jog_cartesian_sdk takes relative dx, dy, dz in meters. Moving UP is positive Z.
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=retract_dist, speed_mm_s=20.0)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed to execute.")
            return False

        # Let the robot physically settle before confirming
        time.sleep(0.5)
        
        check_pos = self.get_current_tcp_position()
        if check_pos:
            self.get_logger().info(f"✅ SDK Retraction complete! Current Z: {check_pos[2]:.4f}")
            
        return True

    def execute_hold(self, part_id: int, target_label: str, interactive=True):
        """
        Wrapper function to GUARANTEE the final status is published, 
        even if the sequence fails early.
        """
        # Small delay to ensure ROS 2 publisher is fully registered before sending first message
        time.sleep(0.5) 
        self.publish_hold_status(False)
        
        success = False
        try:
            success = self._run_hold_sequence(part_id, target_label, interactive)
        except Exception as e:
            self.get_logger().error(f"❌ Hold Sequence crashed: {e}")
        finally:
            # This ensures that no matter what happens, the network gets the final True/False result
            self.publish_hold_status(success)
            return success

    def _run_hold_sequence(self, part_id, target_label, interactive):
        """
        The actual motion logic.
        """
        print("\n" + "!"*60)
        print(f"🤖 INITIATING TACTILE HOLD SKILL: {target_label} (ID: {part_id})")
        print("!"*60)

        print("   👀 Waiting for fresh vision data on '/vision/agent_state'...")
        self.vision_event.clear()
        if not self.vision_event.wait(timeout=10.0): 
            self.get_logger().error("❌ Vision data timeout! Is the camera node running?")
            return False
        
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        target_obj = next((obj for obj in objects if obj.get("id") == part_id), None)
        if not target_obj: 
            self.get_logger().error(f"❌ Target object (ID: {part_id}) not found in current vision state!")
            return False

        raw_x, raw_y, raw_z = target_obj["xyz"]
        obj_angle_deg = -target_obj["angle"] if self.INVERT_VISION_ANGLE else target_obj["angle"]

        print(f"\n✅ Target Locked: {target_label}")
        print(f"📍 Raw Vision Coords: X:{raw_x:.4f}, Y:{raw_y:.4f}, Z:{raw_z:.4f}")

        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = float(raw_x), float(raw_y), float(raw_z)
        source_pose.orientation.w = 1.0 
        transformed_pose = self.uf850.get_transformed_pose(source_pose, self.CAMERA_FRAME, self.PLANNING_FRAME)
        
        if not transformed_pose: 
            self.get_logger().error("❌ TF Transformation Failed! Cannot map camera to world.")
            return False
            
        world_x, world_y, world_z = transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z

        # --- TILTED FLIGHT PLAN ---
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
        
        roll_angle = math.pi + math.radians(self.JAWS_ROTATION_DEG) if not self.FLIP_GRIPPER_UPSIDE_DOWN else math.radians(self.JAWS_ROTATION_DEG)
        pitch_angle = -math.pi/2 + tilt_rad
        
        q = self.uf850._rpy_to_quaternion(roll_angle, pitch_angle, approach_vector_rad)
        q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        aligned_yaw_rad = math.radians(self.FINAL_ALIGN_YAW_DEG)
        aligned_pitch_rad = -math.pi/2 if self.REMOVE_TILT_ON_ALIGN else pitch_angle
        q_aligned = self.uf850._rpy_to_quaternion(roll_angle, aligned_pitch_rad, aligned_yaw_rad)
        q_dict_aligned = {'qx': q_aligned.x, 'qy': q_aligned.y, 'qz': q_aligned.z, 'qw': q_aligned.w}

        # --- EXECUTION FLOW ---
        if interactive: input(f"\n🚀 STEP 1: Move to SAFE HOVER Position? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        self.uf850.move_to_pose_robust(target_x, target_y, hover_z, q_dict, link_name=self.ROBOT_EE_LINK, velocity=0.1)

        if interactive: input(f"\n🚀 STEP 2: Start 1st SDK TACTILE DESCENT (Retracts {self.RETRACT_DISTANCE*1000}mm)? [Enter]")
        if not self.descend_until_contact(q_dict): return False
        
        if interactive: input(f"\n🚀 STEP 3: Slide Forward into Grasp Position? [Enter]")
        current_safe_z = self.get_current_tcp_position()[2]
        self.uf850.move_to_pose_robust(grasp_x, grasp_y, current_safe_z, q_dict, link_name=self.ROBOT_EE_LINK, velocity=0.05)

        if interactive: input(f"\n🚀 STEP 4: Start 2nd (FINAL) SDK TACTILE DESCENT (Retracts 2mm)? [Enter]")
        if not self.descend_until_contact(q_dict, custom_retract=0.005): return False

        if interactive: input(f"\n🚀 STEP 5: CLOSE Gripper to Secure {target_label}? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        
        if not is_held:
            self.get_logger().error("❌ Failed to grasp the object.")
            return False
            
        print(f"\n🎉 SECURED: {target_label} (2mm above sensed surface).")

        if interactive: input(f"\n🚀 STEP 6: Square Object to World Axes (Yaw: {self.FINAL_ALIGN_YAW_DEG}°)? [Enter]")
        current_pose = self.get_current_tcp_position()
        if current_pose:
            self.get_logger().info("🔄 Rotating to align object perfectly straight...")
            self.uf850.move_to_pose_robust(
                current_pose[0], current_pose[1], current_pose[2], q_dict_aligned, 
                link_name=self.ROBOT_EE_LINK, velocity=0.03, frame_id=self.PLANNING_FRAME
            )

        return True

# =====================================================================
# STANDALONE EXECUTION BLOCK (When run directly via 'ros2 run')
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    hold_node = ObjectHoldSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(hold_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # Capture the final boolean result of the hold sequence
        final_status = hold_node.execute_hold(part_id=0, target_label="Top_Lid", interactive=True)
        
        print("\n⏳ Hold Sequence Complete.")
        print(f"📡 Continuously broadcasting final status ({final_status}) at 1Hz...")
        
        # Continuously publish the status so late-joining nodes (like the Flip Skill) catch it
        while rclpy.ok():
            hold_node.publish_hold_status(final_status)
            time.sleep(1.0)
            
    except KeyboardInterrupt: 
        pass
    
    hold_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()