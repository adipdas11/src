#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String
import threading
import threading
import math
import time

# Import the backend containing SDK controllers & dynamic IP switching
from disassembly_skills.motion_backend import MotionBackend

class ObjectFlipSkill(Node):
    def __init__(self):
        super().__init__('object_flip_skill_node')
        
        # 1. Backends
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # 2. State Publisher
        self.state_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        self.get_logger().info("🚀 Object Flip Skill Node Active.")

        # ================= CONFIGURATION =================
        self.PLANNING_FRAME = 'world_world'    
        self.ROBOT_EE_LINK = "u1_tool0"
        
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0 
        self.GRIPPER_SPEED = 0.3

        self.RETRACT_Z_HEIGHT = 0.15   # 15cm Retract
        
        # --- TACTILE CONFIGURATION ---
        self.CONTACT_JOINT = "u1_joint5"
        self.TORQUE_THRESHOLD = 1.0    
        self.RETRACT_DISTANCE = 0.005  # 5mm Retract for the drop
        # =================================================

    def publish_arm_state(self, state_str: str):
        msg = String()
        msg.data = state_str
        self.state_pub.publish(msg)

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
        baseline_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
        self.get_logger().info(f"⬇️ Starting Native SDK Tactile Descent (Target Joint: {self.CONTACT_JOINT}, Baseline: {baseline_effort:.4f}Nm)")
        self.contact_detected = False
        last_print_time = time.time()
        start_t = time.time()

        def check_force():
            nonlocal last_print_time, start_t
            curr_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, baseline_effort)
            spike = abs(curr_effort - baseline_effort)
            
            if time.time() - last_print_time > 0.1:
                print(f"   [Live Debug] Current {self.CONTACT_JOINT} Effort: {curr_effort:.4f} Nm (Spike: {spike:.4f} Nm)")
                last_print_time = time.time()

            if (time.time() - start_t) > 0.2:
                if spike > self.TORQUE_THRESHOLD:
                    self.get_logger().warn(f"💥 THRESHOLD CROSSED! Actual Spike: {spike:.4f} Nm")
                    self.contact_detected = True
                    return True 
            return False

        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=0.25, 
            speed_mm_s=26.0, 
            check_force_callback=check_force
        )

        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        self.get_logger().info(f"⬆️ Attempting SDK Retraction: {self.RETRACT_DISTANCE*1000}mm UP...")
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=self.RETRACT_DISTANCE, speed_mm_s=26.0)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed to execute.")
            return False

        time.sleep(0.5)
        return True

    def execute_flip(self, interactive=False):
        """
        Pure motion logic. Can be called directly as a library function.
        Defaults to interactive=False for automated script execution.
        """
        self.publish_arm_state("FLIPPING")
        print("\n" + "!"*60)
        print("🤖 INITIATING FLIP SKILL")
        print("!"*60)

        pos, quat = self.get_current_tcp_pose()
        if not pos or not quat:
            self.get_logger().error("❌ Failed to get current TCP pose.")
            return False

        q_dict_current = {'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3]}
        retract_z = pos[2] + self.RETRACT_Z_HEIGHT

        # --- STEP 1: Retract ---
        if interactive: input(f"\n🚀 STEP 1: Retract {self.RETRACT_Z_HEIGHT*100}cm in Z-Axis? [Enter]")
        self.uf850.move_to_pose_robust(
            pos[0], pos[1], retract_z, q_dict_current, 
            link_name=self.ROBOT_EE_LINK, velocity=0.1, frame_id=self.PLANNING_FRAME
        )

        # --- STEP 2: Flip ---
        if interactive: input(f"\n🚀 STEP 2: Rotate u1_joint6 180° to flip object? [Enter]")
        self.get_logger().info("🔄 Flipping object upside down by directly rotating u1_joint6...")
        
        current_joints = self.uf850.current_joint_positions.copy()
        if "u1_joint6" not in current_joints:
            self.get_logger().error("❌ Could not read current state of u1_joint6!")
            return False
            
        current_joints["u1_joint6"] += math.pi
        self.uf850.move_to_joint_positions(current_joints, filter_prefix="u1", velocity=0.1)

        # --- STEP 3: Descent ---
        if interactive: input(f"\n🚀 STEP 3: Start SDK Tactile Descent? [Enter]")
        if not self.descend_until_contact(): 
            return False

        # --- STEP 4: Open Gripper ---
        if interactive: input(f"\n🚀 STEP 4: Open Gripper to Release Object? [Enter]")
        self.get_logger().info("👐 Opening jaws to rest object on surface...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        
        # Give the object half a second to settle perfectly flat
        time.sleep(0.5)

        # --- STEP 5: Close Gripper (Re-grasp) ---
        if interactive: input(f"\n🚀 STEP 5: Close Gripper to Re-grasp Object? [Enter]")
        self.get_logger().info("✊ Closing jaws to re-grasp flipped object...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        
        if not is_held:
            self.get_logger().error("❌ Failed to re-grasp the object after flipping.")
            self.publish_arm_state("IDLE")
            return False

        print("\n🎉 FLIP SKILL COMPLETE: Object successfully flipped and re-grasped.")
        self.publish_arm_state("IDLE")
        return True
# =====================================================================
# STANDALONE EXECUTION BLOCK (Event-Driven)
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    flip_node = ObjectFlipSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(flip_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        final_status = flip_node.execute_flip(interactive=False)
        print(f"\n⏳ Flip Sequence Complete. Status: {final_status}")
        time.sleep(1.0)

    except KeyboardInterrupt:
        pass

    flip_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()