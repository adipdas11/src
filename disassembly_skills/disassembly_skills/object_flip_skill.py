#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
import threading
import math
import time

# Import the backend containing SDK controllers & dynamic IP switching
from disassembly_skills.motion_backend import MotionBackend

class ObjectFlipSkill(Node):
    """
    ROS 2 Node responsible for flipping an object that is currently held by the robot.
    It lifts the object, rotates the wrist joint alternatingly (+180/-180 degrees) to 
    prevent cable wind-up, lowers it using tactile feedback until it touches the table, 
    releases it to settle, and re-grasps it.
    """
    def __init__(self):
        super().__init__('object_flip_skill_node')
        
        # --- Hardware Backends ---
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # --- Subscriptions & Synchronization ---
        # Used only for standalone trigger to ensure an object is held before flipping
        self.is_holding_object = False
        self.last_logged_status = None  
        self.hold_event = threading.Event()
        self.create_subscription(Bool, '/object_hold_status', self.hold_status_callback, 10)
        
        # --- Publishers ---
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        self.get_logger().info("🚀 Object Flip Skill Node Active.")

        # =====================================================================
        # CONFIGURATION & PARAMETERS
        # =====================================================================
        
        # --- Frames & Geometry ---
        self.PLANNING_FRAME = 'world_world'    # Base coordinate frame for TF operations
        self.ROBOT_EE_LINK = "u1_tool0"        # End-effector link name for the manipulator arm
        
        # --- Gripper Configuration ---
        self.JOINT_GRIPPER = "rg6_l_out"       # Joint name controlling the gripper jaws
        self.OPEN_DEG = 35.0                   # Angle (degrees) to fully open the jaws
        self.CLOSE_DEG = -35.0                 # Angle (degrees) to fully close/clamp the jaws
        self.GRIPPER_SPEED = 0.3               # Velocity multiplier for gripper actuation
        
        # --- Flip Motion Parameters ---
        self.RETRACT_Z_HEIGHT = 0.15           # Distance (meters) to lift the object before rotating (15cm)
        
        # --- Tactile Feedback Configuration ---
        self.CONTACT_JOINT = "u1_joint5"       # Joint monitored for torque spikes to detect table contact
        self.TORQUE_THRESHOLD = 1.0            # Torque threshold (Nm) that registers as physical contact
        self.RETRACT_DISTANCE = 0.005          # Distance (meters) to lift up after hitting the table (5mm)
        # =====================================================================

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

    def hold_status_callback(self, msg):
        """
        Listens to the /object_hold_status topic. Triggers the threading event 
        so the standalone block knows when it is safe to execute the flip.
        """
        self.is_holding_object = msg.data
        
        if self.last_logged_status != self.is_holding_object:
            if self.is_holding_object:
                self.get_logger().info("✅ Hold Status changed to TRUE. Object detected in jaws.")
            else:
                self.get_logger().warn("⚠️ Hold Status changed to FALSE. Waiting...")
            self.last_logged_status = self.is_holding_object
        
        self.hold_event.set() 

    def get_current_tcp_pose(self):
        """
        Looks up the current physical pose (Position and Quaternion) of the robot's end-effector.
        Returns ((x, y, z), (qx, qy, qz, qw)) or (None, None) if the lookup fails.
        """
        try:
            if self.uf850.tf_buffer.can_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
                pos = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                quat = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                return pos, quat
            return None, None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None, None

    # -------------------------------------------------------------------------
    # Core Hardware Interaction Functions
    # -------------------------------------------------------------------------
    def wait_for_gripper(self, target_deg, timeout=5.0):
        """
        Monitors the gripper's joint position and blocks until it either reaches 
        the target angle or physically stalls (indicating it grasped the object).
        """
        target_rad = math.radians(target_deg)
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if current_pos == 999:
                time.sleep(0.1); continue
            
            # Exact angle reached
            if abs(current_pos - target_rad) < 0.05: return True
            
            # Stall detection for re-grasping
            stall_timer = getattr(self, '_stall_timer', 0.0)
            last_pos = getattr(self, '_last_pos', 999.0)
            
            if abs(current_pos - last_pos) < 0.005:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    self.get_logger().info("✅ Gripper stalled (Object grasped).")
                    self._stall_timer = 0.0
                    return True
            else: 
                stall_timer = 0.0
                
            self._last_pos = current_pos
            self._stall_timer = stall_timer
            time.sleep(0.1)
        return False

    def descend_until_contact(self):
        """
        Performs a downward SDK linear movement while monitoring joint torque.
        Stops the robot immediately when the torque spike exceeds TORQUE_THRESHOLD,
        clears the generated collision error, and retracts slightly to un-pin the object.
        """
        self.publish_state("MOVING") # STATE UPDATE
        self.get_logger().info(f"⬇️ Starting Native SDK Tactile Descent (Target Joint: {self.CONTACT_JOINT})")
        self.contact_detected = False
        last_print_time = time.time()

        def check_force():
            # Evaluated continuously during the SDK move block
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

        # Execute descent
        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=0.25, 
            speed_mm_s=10.0, 
            check_force_callback=check_force
        )

        if not self.contact_detected:
            self.get_logger().error("❌ Max descent reached without sensing contact.")
            self.publish_state("ERROR") # STATE UPDATE
            self.uf850.reset_robot() 
            return False

        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        # Retract to avoid pressing the object into the table when opening the jaws
        self.get_logger().info(f"⬆️ Attempting SDK Retraction: {self.RETRACT_DISTANCE*1000}mm UP...")
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=self.RETRACT_DISTANCE, speed_mm_s=20.0)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed to execute.")
            self.publish_state("ERROR") # STATE UPDATE
            return False

        time.sleep(0.5)
        return True

    # -------------------------------------------------------------------------
    # Core Sequence Execution
    # -------------------------------------------------------------------------
    def execute_flip(self, interactive=False):
        """
        Main state machine for the flip operation. 
        Sequence: Lift -> Unwind/Wind Joint 6 (180deg) -> Tactile Descent -> Open Jaws -> Re-grasp.
        """
        print("\n" + "!"*60)
        print("🤖 INITIATING FLIP SKILL")
        print("!"*60)

        pos, quat = self.get_current_tcp_pose()
        if not pos or not quat:
            self.get_logger().error("❌ Failed to get current TCP pose.")
            self.publish_state("ERROR") # STATE UPDATE
            return False

        q_dict_current = {'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3]}
        retract_z = pos[2] + self.RETRACT_Z_HEIGHT

        # --- STEP 1: Retract ---
        self.publish_state("MOVING") # STATE UPDATE
        if interactive: input(f"\n🚀 STEP 1: Retract {self.RETRACT_Z_HEIGHT*100}cm in Z-Axis? [Enter]")
        self.uf850.move_to_pose_robust(
            pos[0], pos[1], retract_z, q_dict_current, 
            link_name=self.ROBOT_EE_LINK, velocity=0.07, frame_id=self.PLANNING_FRAME
        )

        # --- STEP 2: Flip (Alternating Logic) ---
        self.publish_state("FLIPPING") # STATE UPDATE
        if interactive: input(f"\n🚀 STEP 2: Rotate u1_joint6 180° to flip object? [Enter]")
        
        current_joints = self.uf850.current_joint_positions.copy()
        if "u1_joint6" not in current_joints:
            self.get_logger().error("❌ Could not read current state of u1_joint6!")
            self.publish_state("ERROR") # STATE UPDATE
            return False
            
        current_j6 = current_joints["u1_joint6"]
        
        # If already rotated positively past 90 degrees, subtract 180 degrees to unwind
        if current_j6 > (math.pi / 2.0):
            self.get_logger().info(f"🔄 Flipping backward (-180°) to unwind joint 6 (Current: {math.degrees(current_j6):.1f}°)...")
            current_joints["u1_joint6"] -= math.pi
        # Otherwise, add 180 degrees
        else:
            self.get_logger().info(f"🔄 Flipping forward (+180°) on joint 6 (Current: {math.degrees(current_j6):.1f}°)...")
            current_joints["u1_joint6"] += math.pi
            
        self.uf850.move_to_joint_positions(current_joints, filter_prefix="u1", velocity=0.1)

        # --- STEP 3: Descent ---
        if interactive: input(f"\n🚀 STEP 3: Start SDK Tactile Descent? [Enter]")
        # Note: descend_until_contact() internally sets "MOVING" and "ERROR" if it fails.
        if not self.descend_until_contact(): 
            return False

        # --- STEP 4: Open Gripper ---
        self.publish_state("MOVING") # STATE UPDATE (Gripper actuation)
        if interactive: input(f"\n🚀 STEP 4: Open Gripper to Release Object? [Enter]")
        self.get_logger().info("👐 Opening jaws to rest object on surface...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        
        # Give the object half a second to settle perfectly flat
        self.publish_state("IDLE") # STATE UPDATE (Object is free on the table)
        time.sleep(0.5)

        # --- STEP 5: Close Gripper (Re-grasp) ---
        self.publish_state("MOVING") # STATE UPDATE (Gripper actuation)
        if interactive: input(f"\n🚀 STEP 5: Close Gripper to Re-grasp Object? [Enter]")
        self.get_logger().info("✊ Closing jaws to re-grasp flipped object...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        
        if not is_held:
            self.get_logger().error("❌ Failed to re-grasp the object after flipping.")
            self.publish_state("ERROR") # STATE UPDATE
            return False

        print("\n🎉 FLIP SKILL COMPLETE: Object successfully flipped and re-grasped.")
        self.publish_state("HOLDING") # STATE UPDATE
        return True


# =====================================================================
# STANDALONE EXECUTION BLOCK (One-Shot)
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    flip_node = ObjectFlipSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(flip_node)
    
    # Spin the node in a background thread to allow callbacks to process
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        print("\n" + "="*60)
        print("🕒 Waiting for Hold Skill to finish (/object_hold_status update)...")
        print("="*60)
        
        # Block until the hold_event is triggered by the subscriber
        flip_node.hold_event.clear()
        flip_node.hold_event.wait() 
        
        if flip_node.is_holding_object:
            # Execute exactly once
            success = flip_node.execute_flip(interactive=True)
            if success:
                print("✅ Object successfully flipped and secured. Exiting Node.")
            else:
                print("❌ Flip sequence failed. Exiting Node.")
        else:
            flip_node.get_logger().error("❌ Cannot execute flip. The object was not securely held.")
            
    except KeyboardInterrupt: 
        pass
        
    # Shutdown gracefully after the single execution
    flip_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()