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
    """
    ROS 2 Node responsible for securely holding an object using tactile feedback.
    It uses a camera to locate the object, approaches it, and gently descends 
    until physical contact is felt (via torque spikes) before clamping the gripper.
    """
    def __init__(self):
        super().__init__('object_hold_skill_node')
        
        # --- Hardware Backends ---
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # --- Vision & Sync Variables ---
        self.latest_vision_data = None
        self.vision_event = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)

        # --- Publishers ---
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        self.get_logger().info("🚀 Object Hold Skill Node: Native SDK Tactile Mode Active.")

        # =====================================================================
        # CONFIGURATION & PARAMETERS
        # =====================================================================
        
        # --- Frames & Geometry ---
        self.CAMERA_FRAME = 'camera_color_optical_frame' # Frame of the vision system
        self.PLANNING_FRAME = 'world_world'              # Base frame for coordinate transformations
        self.ROBOT_EE_LINK = "u1_tool0"                  # End effector link name
        
        # --- Approach Distances & Offsets ---
        self.TOOL_LENGTH = 0.28                # Effective length of the gripper tool (meters)
        self.APPROACH_BUFFER = 0.05            # Horizontal distance to stay back before sliding in (meters)
        self.SAFETY_Z_HEIGHT = -0.003          # Base Z height target for the bottom of the object (meters)
        self.HOVER_Z_OFFSET = 0.10             # Vertical distance to hover above target before descent (meters)
        
        # --- Gripper Configuration ---
        self.JOINT_GRIPPER = "rg6_l_out"       # Joint name for the gripper actuation
        self.OPEN_DEG = 35.0                   # Jaw open angle (degrees)
        self.CLOSE_DEG = -35.0                 # Jaw closed angle (degrees)
        self.GRIPPER_SPEED = 0.3               # Gripper actuation speed
        
        # --- Orientation & Alignment Angles ---
        self.INVERT_VISION_ANGLE = True        # Flips the yaw angle received from vision
        self.JAWS_ROTATION_DEG = 0.0           # Base rotation of the jaws
        self.APPROACH_SIDE_OFFSET_DEG = 0.0    # Angle offset for the side approach vector
        self.FLIP_GRIPPER_UPSIDE_DOWN = False  # Flips the gripper 180 degrees if True
        self.GRIPPER_TILT_DEG = 8.0            # Downward tilt angle of the gripper during approach
        
        # --- Tactile Feedback (Force/Torque) Settings ---
        self.CONTACT_JOINT = "u1_joint5"       # Specific robot joint to monitor for contact spikes
        self.TORQUE_THRESHOLD = 1.0            # Torque spike threshold indicating physical contact (Nm)
        self.RETRACT_DISTANCE = 0.02           # Distance to retract up after hitting the table (meters)
        
        # --- Post-Grasp Alignment ---
        self.FINAL_ALIGN_YAW_DEG = 90.0        # Final squaring yaw angle for the object
        self.REMOVE_TILT_ON_ALIGN = False      # Whether to level out the 8-degree tilt after grasping
        # =====================================================================
        
        # Initialize default state
        self.publish_state("IDLE")

    # -------------------------------------------------------------------------
    # Helper & Communication Functions
    # -------------------------------------------------------------------------
    def publish_state(self, state_str: str):
        """
        Publishes the current operational state of the manipulator arm to the central State Manager.
        """
        msg = String()
        msg.data = state_str
        self.state_update_pub.publish(msg)
        self.get_logger().info(f"🔄 State Manager Update -> {state_str}")

    def publish_hold_status(self, is_held: bool):
        """
        Publishes a boolean indicating whether the object is currently secured in the gripper.
        """
        msg = Bool()
        msg.data = is_held
        self.hold_status_pub.publish(msg)
        # self.get_logger().info(f"📢 Published Hold Status: {is_held}")

    def vision_callback(self, msg):
        """
        Receives and parses the JSON string from the camera system containing object coordinates.
        Triggers the vision_event to unblock waiting processes.
        """
        try:
            clean_data = msg.data.strip("'")
            self.latest_vision_data = json.loads(clean_data)
            self.vision_event.set()
        except Exception as e:
            self.get_logger().error(f"JSON Parsing Error: {e}")

    def get_current_tcp_position(self):
        """
        Looks up the current physical position of the robot's end-effector using TF2.
        Returns a tuple of (x, y, z) in the planning frame, or None if it fails.
        """
        try:
            if self.uf850.tf_buffer.can_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            return None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Core Hardware Interaction Functions
    # -------------------------------------------------------------------------
    def wait_for_gripper(self, target_deg, timeout=5.0):
        """
        Monitors the gripper's joint position and blocks until it either reaches 
        the target angle or physically stalls (indicating it grasped an object).
        """
        target_rad = math.radians(target_deg)
        start_time = time.time()
        last_pos = 999.0
        stall_timer = 0.0
        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if current_pos == 999:
                time.sleep(0.1); continue
            
            # Reached Target exactly
            if abs(current_pos - target_rad) < 0.05: return True
            
            # Stall Detection (Object grasped before reaching target angle)
            if abs(current_pos - last_pos) < 0.005:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    self.get_logger().info("✅ Gripper stalled (Object grasped).")
                    return True
            else: 
                stall_timer = 0.0
            
            last_pos = current_pos
            time.sleep(0.1)
        return False

    def descend_until_contact(self, q_dict, custom_retract=None):
        self.publish_state("MOVING")
        
        # 1. Capture the BASELINE before moving
        # We take a small average or a fresh reading to see what the joint feels like 'idle'
        time.sleep(0.2) 
        baseline_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
        self.get_logger().info(f"📊 Baseline established for {self.CONTACT_JOINT}: {baseline_effort:.4f} Nm")

        retract_dist = custom_retract if custom_retract is not None else self.RETRACT_DISTANCE
        self.contact_detected = False
        last_print_time = time.time()

        def check_force():
            nonlocal last_print_time
            current_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
            
            # CALCULATE THE SPIKE (Current - Baseline)
            actual_spike = abs(current_effort - baseline_effort)

            if time.time() - last_print_time > 0.1:
                print(f"   [Live Debug] Spike: {actual_spike:.4f} Nm (Current: {current_effort:.4f})")
                last_print_time = time.time()

            # Compare the SPIKE to the threshold, not the raw value
            if actual_spike > self.TORQUE_THRESHOLD:
                self.get_logger().warn(f"💥 CONTACT DETECTED! Spike: {actual_spike:.4f} Nm")
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
            self.publish_state("ERROR") # STATE UPDATE
            self.uf850.reset_robot() 
            return False

        # 2. Clear the hardware collision state generated by the intentional contact
        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        # 3. Use SDK to Retract safely (Bypassing MoveIt completely)
        self.get_logger().info(f"⬆️ Attempting SDK Retraction: {retract_dist*1000}mm UP...")
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=retract_dist, speed_mm_s=20.0)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed to execute.")
            self.publish_state("ERROR") # STATE UPDATE
            return False

        time.sleep(0.5)
        
        check_pos = self.get_current_tcp_position()
        if check_pos:
            self.get_logger().info(f"✅ SDK Retraction complete! Current Z: {check_pos[2]:.4f}")
            
        return True

    # -------------------------------------------------------------------------
    # Core Sequence Execution
    # -------------------------------------------------------------------------
    def execute_hold(self, part_id: int, target_label: str, interactive=True):
        """
        Public wrapper function for the hold sequence. 
        Guarantees the final success/failure status is always published correctly to the network.
        """
        time.sleep(0.5) 
        self.publish_hold_status(False)
        self.publish_state("MOVING") # STATE UPDATE
        
        success = False
        try:
            success = self._run_hold_sequence(part_id, target_label, interactive)
        except Exception as e:
            self.get_logger().error(f"❌ Hold Sequence crashed: {e}")
            self.publish_state("ERROR") # STATE UPDATE
        finally:
            self.publish_hold_status(success)
            self.publish_state("HOLDING" if success else "IDLE") # STATE UPDATE
            return success

    def _run_hold_sequence(self, part_id, target_label, interactive):
        """
        The internal state machine managing the actual sequence of operations:
        Vision Acquisition -> Trajectory Calculation -> Hover -> Tactile Descent 1 -> 
        Slide Forward -> Tactile Descent 2 -> Grasp -> Final Alignment.
        """
        print("\n" + "!"*60)
        print(f"🤖 INITIATING TACTILE HOLD SKILL: {target_label} (ID: {part_id})")
        print("!"*60)

        # --- Vision Check ---
        print("   👀 Waiting for fresh vision data on '/vision/agent_state'...")
        self.vision_event.clear()
        if not self.vision_event.wait(timeout=10.0): 
            self.get_logger().error("❌ Vision data timeout! Is the camera node running?")
            self.publish_state("ERROR") # STATE UPDATE
            return False
        
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        target_obj = next((obj for obj in objects if obj.get("id") == part_id), None)
        if not target_obj: 
            self.get_logger().error(f"❌ Target object (ID: {part_id}) not found in current vision state!")
            self.publish_state("ERROR") # STATE UPDATE
            return False

        raw_x, raw_y, raw_z = target_obj["xyz"]
        obj_angle_deg = -target_obj["angle"] if self.INVERT_VISION_ANGLE else target_obj["angle"]

        print(f"\n✅ Target Locked: {target_label}")
        print(f"📍 Raw Vision Coords: X:{raw_x:.4f}, Y:{raw_y:.4f}, Z:{raw_z:.4f}")

        # --- Trajectory Transformation ---
        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = float(raw_x), float(raw_y), float(raw_z)
        source_pose.orientation.w = 1.0 
        transformed_pose = self.uf850.get_transformed_pose(source_pose, self.CAMERA_FRAME, self.PLANNING_FRAME)
        
        if not transformed_pose: 
            self.get_logger().error("❌ TF Transformation Failed! Cannot map camera to world.")
            self.publish_state("ERROR") # STATE UPDATE
            return False
            
        world_x, world_y, world_z = transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z

        # --- TILTED FLIGHT PLAN CALCULATIONS ---
        obj_yaw_rad = math.radians(obj_angle_deg)
        approach_vector_rad = obj_yaw_rad + (math.pi / 2.0) + math.radians(self.APPROACH_SIDE_OFFSET_DEG)
        tilt_rad = math.radians(self.GRIPPER_TILT_DEG)
        
        horizontal_tool_length = self.TOOL_LENGTH * math.cos(tilt_rad)
        z_lift_offset = self.TOOL_LENGTH * math.sin(tilt_rad)
        total_horizontal_offset = horizontal_tool_length + self.APPROACH_BUFFER
        
        # Hover Position Setup
        target_x = world_x - (total_horizontal_offset * math.cos(approach_vector_rad))
        target_y = world_y - (total_horizontal_offset * math.sin(approach_vector_rad))
        target_z = world_z + self.SAFETY_Z_HEIGHT + z_lift_offset
        hover_z = target_z + self.HOVER_Z_OFFSET
        
        # Grasp Position Setup
        grasp_x = world_x - (horizontal_tool_length * math.cos(approach_vector_rad))
        grasp_y = world_y - (horizontal_tool_length * math.sin(approach_vector_rad))
        
        # Orientation setup (Quaternions)
        roll_angle = math.pi + math.radians(self.JAWS_ROTATION_DEG) if not self.FLIP_GRIPPER_UPSIDE_DOWN else math.radians(self.JAWS_ROTATION_DEG)
        pitch_angle = -math.pi/2 + tilt_rad
        q = self.uf850._rpy_to_quaternion(roll_angle, pitch_angle, approach_vector_rad)
        q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        # Final squaring orientation
        aligned_yaw_rad = math.radians(self.FINAL_ALIGN_YAW_DEG)
        aligned_pitch_rad = -math.pi/2 if self.REMOVE_TILT_ON_ALIGN else pitch_angle
        q_aligned = self.uf850._rpy_to_quaternion(roll_angle, aligned_pitch_rad, aligned_yaw_rad)
        q_dict_aligned = {'qx': q_aligned.x, 'qy': q_aligned.y, 'qz': q_aligned.z, 'qw': q_aligned.w}

        # --- EXECUTION FLOW ---
        
        # Step 1: Move above target and open jaws
        if interactive: input(f"\n🚀 STEP 1: Move to SAFE HOVER Position? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        self.uf850.move_to_pose_robust(target_x, target_y, hover_z, q_dict, link_name=self.ROBOT_EE_LINK, velocity=0.1)

        # Step 2: First descent to find the table/base level
        if interactive: input(f"\n🚀 STEP 2: Start 1st SDK TACTILE DESCENT (Retracts {self.RETRACT_DISTANCE*1000}mm)? [Enter]")
        if not self.descend_until_contact(q_dict): return False
        
        # Step 3: Slide horizontally to encapsulate object
        if interactive: input(f"\n🚀 STEP 3: Slide Forward into Grasp Position? [Enter]")
        current_safe_z = self.get_current_tcp_position()[2]
        self.uf850.move_to_pose_robust(grasp_x, grasp_y, current_safe_z, q_dict, link_name=self.ROBOT_EE_LINK, velocity=0.05)

        # Step 4: Second, very small descent to secure perfectly against the surface
        if interactive: input(f"\n🚀 STEP 4: Start 2nd (FINAL) SDK TACTILE DESCENT (Retracts 2mm)? [Enter]")
        if not self.descend_until_contact(q_dict, custom_retract=0.005): return False

        # Step 5: Clamping
        if interactive: input(f"\n🚀 STEP 5: CLOSE Gripper to Secure {target_label}? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        
        if not is_held:
            self.get_logger().error("❌ Failed to grasp the object.")
            self.publish_state("ERROR") # STATE UPDATE
            return False
            
        print(f"\n🎉 SECURED: {target_label} (2mm above sensed surface).")
        self.publish_state("HOLDING") # STATE UPDATE

        # Step 6: Post-grasp squared alignment
        if interactive: input(f"\n🚀 STEP 6: Square Object to World Axes (Yaw: {self.FINAL_ALIGN_YAW_DEG}°)? [Enter]")
        current_pose = self.get_current_tcp_position()
        if current_pose:
            self.publish_state("MOVING") # STATE UPDATE for the alignment move
            self.get_logger().info("🔄 Rotating to align object perfectly straight...")
            self.uf850.move_to_pose_robust(
                current_pose[0], current_pose[1], current_pose[2], q_dict_aligned, 
                link_name=self.ROBOT_EE_LINK, velocity=0.03, frame_id=self.PLANNING_FRAME
            )
            self.publish_state("HOLDING") # STATE UPDATE back to holding after align

        return True

# =====================================================================
# STANDALONE EXECUTION BLOCK (When run directly via 'ros2 run')
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    hold_node = ObjectHoldSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(hold_node)
    
    # Spin node continuously in background thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # Trigger the hold operation in standalone mode
        final_status = hold_node.execute_hold(part_id=0, target_label="Top_Lid", interactive=True)
        
        print("\n⏳ Hold Sequence Complete.")
        print(f"📡 Continuously broadcasting final status ({final_status}) at 1Hz...")
        
        # Keep node alive to broadcast its hold status over the topic
        while rclpy.ok():
            hold_node.publish_hold_status(final_status)
            time.sleep(1.0)
            
    except KeyboardInterrupt: 
        pass
    
    hold_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()