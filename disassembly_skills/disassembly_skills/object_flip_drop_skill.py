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

class FlipDropSkill(Node):
    """
    ROS 2 Node responsible for moving an object to a safe zone, inverting it,
    wiggling it to shake out contents, reverting its orientation, returning to 
    the original zone, and releasing it via tactile sensing.
    """
    def __init__(self):
        super().__init__('flip_drop_skill_node')
        
        # --- Hardware Backends ---
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # --- Subscriptions & Synchronization ---
        self.is_holding_object = False
        self.last_logged_status = None  
        self.hold_event = threading.Event()
        
        # Updated to listen to the new Central State Manager!
        self.create_subscription(Bool, '/object_hold_state/is_held', self.hold_status_callback, 10)
        
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        self.get_logger().info("🚀 Flip Drop (Shake) Skill Node Active.")

        # =====================================================================
        # CONFIGURATION & PARAMETERS
        # =====================================================================
        self.PLANNING_FRAME = 'world_world'    
        self.ROBOT_EE_LINK = "u1_tool0"        
        
        self.JOINT_GRIPPER = "rg6_l_out"       
        self.OPEN_DEG = 35.0                   
        self.CLOSE_DEG = -35.0                 
        self.GRIPPER_SPEED = 0.3               
        
        self.RETRACT_Z_HEIGHT = 0.15           # 15cm Retract at the starting location
        
        # --- INTERMEDIATE FLIP ZONE POSE (From Image) ---
        self.INTERMEDIATE_POSE = {
            'x': 0.929872, 'y': -0.633943, 'z': 1.0977,
            'qx': -0.495652, 'qy': -0.505050, 'qz': -0.491028, 'qw': 0.508080
        }

        # --- Shake / Wiggle Configuration ---
        self.WIGGLE_Z_DIST = 0.02              # 2cm up and down movement
        self.WIGGLE_SPEED = 100.0              # Fast cartesian speed for snapping motion
        self.WIGGLE_CYCLES = 2                 # Number of up-down bounces
        
        # --- Tactile Feedback Configuration ---
        self.CONTACT_JOINT = "u1_joint5"       
        self.TORQUE_THRESHOLD = 2.0            
        self.RETRACT_DISTANCE = 0.005          # 5mm Retract after hitting the table
        # =====================================================================

    def publish_state(self, state_str: str):
        msg = String()
        msg.data = state_str
        self.state_update_pub.publish(msg)
        self.get_logger().info(f"🔄 State Manager Update -> {state_str}")

    def hold_status_callback(self, msg):
        self.is_holding_object = msg.data
        if self.last_logged_status != self.is_holding_object:
            if self.is_holding_object:
                self.get_logger().info("✅ Hold Status changed to TRUE. Object detected in jaws.")
            else:
                self.get_logger().warn("⚠️ Hold Status is FALSE. Waiting for object...")
            self.last_logged_status = self.is_holding_object
        self.hold_event.set() 

    def get_current_tcp_pose(self):
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

    def wait_for_gripper(self, target_deg, timeout=5.0):
        target_rad = math.radians(target_deg)
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if current_pos == 999:
                time.sleep(0.1); continue
            if abs(current_pos - target_rad) < 0.05: return True
            time.sleep(0.1)
        return False

    def descend_until_contact(self):
        self.publish_state("MOVING") 
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

        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=0.25, 
            speed_mm_s=10.0, 
            check_force_callback=check_force
        )

        if not self.contact_detected:
            self.get_logger().error("❌ Max descent reached without sensing contact.")
            self.publish_state("ERROR") 
            self.uf850.reset_robot() 
            return False

        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        self.get_logger().info(f"⬆️ Attempting SDK Retraction: {self.RETRACT_DISTANCE*1000}mm UP...")
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=self.RETRACT_DISTANCE, speed_mm_s=20.0)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed.")
            self.publish_state("ERROR") 
            return False

        time.sleep(0.5)
        return True

    def execute_flip_drop(self, interactive=False):
        """
        Main state machine for the Flip Drop operation. 
        """
        print("\n" + "!"*60)
        print("🤖 INITIATING FLIP DROP (SHAKE) SKILL")
        print("!"*60)

        # --- Hard Check against library-mode execution when empty ---
        if not self.is_holding_object:
            self.get_logger().error("❌ Aborting: The State Manager reports the gripper is EMPTY.")
            return False

        # 1. Capture Original Data
        orig_pos, orig_quat = self.get_current_tcp_pose()
        if not orig_pos or not orig_quat:
            self.get_logger().error("❌ Failed to get current TCP pose.")
            self.publish_state("ERROR") 
            return False

        q_dict_orig = {'qx': orig_quat[0], 'qy': orig_quat[1], 'qz': orig_quat[2], 'qw': orig_quat[3]}
        orig_retract_z = orig_pos[2] + self.RETRACT_Z_HEIGHT

        # --- STEP 1: Retract ---
        self.publish_state("MOVING") 
        if interactive: input(f"\n🚀 STEP 1: Retract {self.RETRACT_Z_HEIGHT*100}cm in Z-Axis? [Enter]")
        self.uf850.move_to_pose_robust(
            orig_pos[0], orig_pos[1], orig_retract_z, q_dict_orig, 
            link_name=self.ROBOT_EE_LINK, velocity=0.1, frame_id=self.PLANNING_FRAME
        )

        # --- STEP 2: Move to Intermediate Pose (Maintaining Original Orientation) ---
        if interactive: input(f"\n🚀 STEP 2: Move to Intermediate Flip Zone? [Enter]")
        self.get_logger().info("🔄 Moving to Flip Zone while maintaining picked orientation...")
        self.uf850.move_to_pose_robust(
            self.INTERMEDIATE_POSE['x'], self.INTERMEDIATE_POSE['y'], self.INTERMEDIATE_POSE['z'], q_dict_orig, 
            link_name=self.ROBOT_EE_LINK, velocity=0.1, frame_id=self.PLANNING_FRAME
        )

        # --- STEP 3: Flip, Shake, and Un-Flip ---
        self.publish_state("FLIPPING") 
        if interactive: input(f"\n🚀 STEP 3: Rotate 180°, Shake, and Un-Flip? [Enter]")
        
        current_joints = self.uf850.current_joint_positions.copy()
        if "u1_joint6" not in current_joints:
            self.get_logger().error("❌ Could not read current state of u1_joint6!")
            self.publish_state("ERROR") 
            return False
            
        original_j6 = current_joints["u1_joint6"]
        
        # 3A. Flip Upside Down (Anti-windup logic)
        if original_j6 > (math.pi / 2.0):
            self.get_logger().info("🔄 Flipping backward (-180°)...")
            current_joints["u1_joint6"] -= math.pi
        else:
            self.get_logger().info("🔄 Flipping forward (+180°)...")
            current_joints["u1_joint6"] += math.pi
        self.uf850.move_to_joint_positions(current_joints, filter_prefix="u1", velocity=0.1)

        # 3B. Wiggle (Up / Down) via SDK
        self.get_logger().info("🪀 Performing up-down wiggle to shake out contents...")
        for _ in range(self.WIGGLE_CYCLES):
            # Jog UP
            self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=self.WIGGLE_Z_DIST, speed_mm_s=self.WIGGLE_SPEED)
            # Jog DOWN
            self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=-self.WIGGLE_Z_DIST, speed_mm_s=self.WIGGLE_SPEED)

        # 3C. Un-Flip (Return to Original Orientation)
        self.get_logger().info("🔄 Flipping back to original upright orientation...")
        current_joints["u1_joint6"] = original_j6
        self.uf850.move_to_joint_positions(current_joints, filter_prefix="u1", velocity=0.1)

        # --- STEP 4: Return to Original Drop Zone ---
        self.publish_state("MOVING") 
        if interactive: input(f"\n🚀 STEP 4: Return to Original Zone? [Enter]")
        self.get_logger().info("🔄 Returning to original XY with original orientation...")
        self.uf850.move_to_pose_robust(
            orig_pos[0], orig_pos[1], orig_retract_z, q_dict_orig, 
            link_name=self.ROBOT_EE_LINK, velocity=0.1, frame_id=self.PLANNING_FRAME
        )

        # --- STEP 5: Descent ---
        if interactive: input(f"\n🚀 STEP 5: Start SDK Tactile Descent? [Enter]")
        if not self.descend_until_contact(): 
            return False

        # --- STEP 6: Drop Object ---
        self.publish_state("MOVING") 
        if interactive: input(f"\n🚀 STEP 6: Open Gripper to Drop Object? [Enter]")
        self.get_logger().info("👐 Opening jaws to drop object on surface...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        
        # Give the object half a second to fall completely free
        time.sleep(0.5)

        # --- STEP 7: Close Gripper ---
        if interactive: input(f"\n🚀 STEP 7: Close Gripper jaws? [Enter]")
        self.get_logger().info("✊ Closing empty jaws...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.CLOSE_DEG)
        
        print("\n🎉 FLIP DROP SKILL COMPLETE: Object successfully shaken, dropped, and jaws closed.")
        self.publish_state("IDLE") 
        return True


# =====================================================================
# STANDALONE EXECUTION BLOCK (One-Shot)
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    flip_node = FlipDropSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(flip_node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        print("\n" + "="*60)
        print("🕒 Waiting for State Manager to report HOLDING (/object_hold_state/is_held)...")
        print("="*60)
        
        # Because the State Manager broadcasts at 2Hz, we must continuously ignore
        # the "False" messages and wait until the gripper actually secures something.
        while rclpy.ok() and not flip_node.is_holding_object:
            flip_node.hold_event.clear()
            flip_node.hold_event.wait() 
        
        # Once it breaks out of the loop, execute the skill!
        if rclpy.ok() and flip_node.is_holding_object:
            success = flip_node.execute_flip_drop(interactive=True)
            if success:
                print("✅ Flip Drop sequence complete. Exiting Node.")
            else:
                print("❌ Flip Drop sequence failed. Exiting Node.")
                
    except KeyboardInterrupt: 
        pass
        
    flip_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()