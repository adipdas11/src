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
        self.state_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        self.get_logger().info("🚀 Object Hold Skill Node: SDK Tactile Mode Active.")

        # ================= CONFIGURATION =================
        self.CAMERA_FRAME = 'camera_color_optical_frame' 
        self.PLANNING_FRAME = 'world_world'    
        self.TOOL_LENGTH = 0.28                
        self.HOVER_Z_OFFSET = 0.05             
        
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0 
        self.GRIPPER_SPEED = 0.3

        self.INVERT_VISION_ANGLE = True  
        self.GRIPPER_TILT_DEG = 7.0  

        self.CONTACT_JOINT = "u1_joint5"
        self.TORQUE_THRESHOLD = 3.0     # Nm — matches v4
        self.RETRACT_DISTANCE = 0.005   # 5mm retract after contact
        self.ROBOT_EE_LINK = "u1_tool0"
        
        # SDK MINIMUM SPEED: 26mm/s to guarantee robot execution
        self.SDK_MIN_SPEED = 26.0
        self.SDK_DESCENT_SPEED = 30.0   # Tactile descent speed
        self.SDK_RETRACT_SPEED = 30.0   # Retract speed
        # =================================================

    def publish_arm_state(self, state_str: str):
        msg = String()
        msg.data = state_str
        self.state_pub.publish(msg)

    def publish_hold_status(self, is_held: bool, quiet=False):
        msg = Bool()
        msg.data = is_held
        self.hold_status_pub.publish(msg)
        if not quiet:
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
        is_closing = target_deg < 0
        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if current_pos == 999:
                time.sleep(0.1); continue
            if abs(current_pos - target_rad) < 0.05: return True
            if abs(current_pos - last_pos) < 0.002:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    if is_closing:
                        self.get_logger().info(f"📦 Grasp Secured: Stalled at {current_pos:.3f} rad.")
                        return True
                    else:
                        return False
            else: stall_timer = 0.0
            last_pos = current_pos
            time.sleep(0.1)
        return False

    def wait_for_arm_settled(self, timeout=20.0):
        time.sleep(0.2)
        start_t = time.time()
        settle_timer = 0.0
        last_positions = {}
        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr_positions = self.uf850.current_joint_positions.copy()
            if not curr_positions:
                time.sleep(0.1); continue
            if last_positions:
                max_delta = max(abs(curr_positions[n] - last_positions[n]) 
                                for n in curr_positions if n in last_positions)
                if max_delta <= 0.006:
                    settle_timer += 0.1
                    if settle_timer >= 0.4: return True
                else: settle_timer = 0.0
            last_positions = curr_positions
            time.sleep(0.1)
        return True

    def descend_until_contact(self):
        """SDK-based tactile descent. Uses move_linear_z_sdk_with_force_stop with correct speed."""
        baseline_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
        self.get_logger().info(f"⬇️ SDK Tactile Descent (Joint: {self.CONTACT_JOINT}, Threshold: {self.TORQUE_THRESHOLD}Nm, Speed: {self.SDK_DESCENT_SPEED}mm/s, Baseline: {baseline_effort:.3f}Nm)")
        self.contact_detected = False
        last_print_time = time.time()
        start_t = time.time()

        def check_force():
            nonlocal last_print_time, start_t
            curr_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, baseline_effort)
            spike = abs(curr_effort - baseline_effort)
            
            if time.time() - last_print_time > 0.1:
                print(f"   [Live] {self.CONTACT_JOINT} Effort: {curr_effort:.4f} Nm (Spike: {spike:.4f} Nm)")
                last_print_time = time.time()

            # 0.2s blanking time to ignore initial movement jerk
            if (time.time() - start_t) > 0.2:
                if spike > self.TORQUE_THRESHOLD:
                    self.get_logger().warn(f"💥 CONTACT! Spike: {spike:.4f} Nm")
                    self.contact_detected = True
                    return True 
            return False

        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=0.15, 
            speed_mm_s=self.SDK_DESCENT_SPEED,
            check_force_callback=check_force
        )

        self.get_logger().info("🔄 Clearing hardware error state after contact stop...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        # Retract after contact
        self.get_logger().info(f"⬆️ SDK Retract: {self.RETRACT_DISTANCE*1000}mm UP at {self.SDK_RETRACT_SPEED}mm/s")
        success = self.uf850.jog_cartesian_sdk(dx_m=0.0, dy_m=0.0, dz_m=self.RETRACT_DISTANCE, speed_mm_s=self.SDK_RETRACT_SPEED)
        
        if not success:
            self.get_logger().error("❌ SDK Retraction command failed.")
            return False

        time.sleep(0.5)
        return True

    def execute_hold(self, part_id: int, target_label: str, interactive=False):
        """Wrapper ensuring hold status is always published."""
        time.sleep(0.5) 
        self.publish_hold_status(False)
        
        success = False
        try:
            success = self._run_hold_sequence(part_id, target_label, interactive)
        except Exception as e:
            self.get_logger().error(f"❌ Hold Sequence crashed: {e}")
        finally:
            self.publish_hold_status(success)
            return success

    def _run_hold_sequence(self, part_id, target_label, interactive=False):
        """
        Simplified hold sequence (matching v4 approach):
        1. Hover above target with tilt
        2. Tactile descent → contact → retract 5mm
        3. Close gripper
        No slide-in motion.
        """
        print("\n" + "!"*60)
        print(f"🤖 INITIATING HOLD SKILL: {target_label} (ID: {part_id})")
        print("!"*60)

        self.publish_arm_state("MOVING")

        print("   👀 Waiting for fresh vision data...")
        self.vision_event.clear()
        if not self.vision_event.wait(timeout=10.0): 
            self.get_logger().error("❌ Vision data timeout!")
            return False
        
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        target_obj = next((obj for obj in objects if obj.get("id") == part_id), None)
        if not target_obj: 
            self.get_logger().error(f"❌ Target (ID: {part_id}) not found in vision state!")
            return False

        raw_x, raw_y, raw_z = target_obj["xyz"]

        print(f"\n✅ Target Locked: {target_label}")
        print(f"📍 Raw Vision Coords: X:{raw_x:.4f}, Y:{raw_y:.4f}, Z:{raw_z:.4f}")

        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = float(raw_x), float(raw_y), float(raw_z)
        source_pose.orientation.w = 1.0 
        transformed_pose = self.uf850.get_transformed_pose(source_pose, self.CAMERA_FRAME, self.PLANNING_FRAME)
        
        if not transformed_pose: 
            self.get_logger().error("❌ TF Transformation Failed!")
            return False
            
        world_x = transformed_pose.pose.position.x
        world_y = transformed_pose.pose.position.y
        world_z = transformed_pose.pose.position.z

        # --- TILTED HOVER POSE CALCULATION (matches v4) ---
        tilt_rad = math.radians(self.GRIPPER_TILT_DEG)
        v_rad = math.radians(90.0)  # Fixed approach angle

        q = self.uf850._rpy_to_quaternion(math.pi, -math.pi/2.0 + tilt_rad, v_rad)
        q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        # Wrist offset from tool length + tilt
        off = self.TOOL_LENGTH * math.cos(tilt_rad)
        tx = world_x - (off * math.cos(v_rad))
        ty = world_y - (off * math.sin(v_rad))
        hz = world_z + self.HOVER_Z_OFFSET + (self.TOOL_LENGTH * math.sin(tilt_rad))

        # --- STEP 1: HOVER ---
        if interactive: input(f"\n🚀 STEP 1: Move to Hover Position? [Enter]")
        print("🔓 Opening gripper...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)

        print(f"🚁 Moving to tilted hover pose (Z: {hz:.3f})...")
        success = self.uf850.move_to_pose_robust(tx - 0.01, ty - 0.005, hz, q_dict, link_name=self.ROBOT_EE_LINK, velocity=0.1)
        if not success:
            self.get_logger().warn("MoveIt reported failure, but proceeding to verify physically...")
        self.wait_for_arm_settled()

        # --- STEP 2: TACTILE DESCENT ---
        if interactive: input(f"\n🚀 STEP 2: SDK Tactile Descent? [Enter]")
        if not self.descend_until_contact(): 
            return False
        self.wait_for_arm_settled()

        # --- STEP 3: CLOSE GRIPPER ---
        if interactive: input(f"\n🚀 STEP 3: CLOSE Gripper to Secure {target_label}? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        
        if not is_held:
            self.get_logger().error("❌ Failed to grasp the object.")
            self.publish_arm_state("IDLE")
            return False
            
        print(f"\n🎉 SECURED: {target_label}")
        self.publish_arm_state("HOLDING")
        return True


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    hold_node = ObjectHoldSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(hold_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        final_status = hold_node.execute_hold(part_id=0, target_label="Top_Lid", interactive=False)
        
        print(f"\n⏳ Hold Sequence Complete. Status: {final_status}")
        hold_node.publish_hold_status(final_status)
        
        # Give ROS time to broadcast before exiting cleanly
        time.sleep(1.0)
            
    except KeyboardInterrupt: 
        pass
    
    hold_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()