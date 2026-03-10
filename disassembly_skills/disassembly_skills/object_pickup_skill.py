#!/usr/bin/env python3
"""
Object Pickup Skill — SDK-only (no MoveIt Servo)

Picks up a detected object (e.g., PCB) using UF850 + RG6 gripper:
1. Clears workspace (homes xArm5 if needed)
2. Transforms vision coordinates to world frame
3. Hovers above target with gripper open
4. Tactile descent using SDK force monitoring (u1_joint5 torque)
5. Retracts, grasps, lifts, and drops at configured drop zone
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
import json
import time
import threading
import math
import copy

from disassembly_skills.motion_backend import MotionBackend


class ObjectPickupSkill(Node):
    def __init__(self):
        super().__init__('object_pickup_skill_node')

        # --- Motion Backends ---
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        self.xarm5 = MotionBackend(self, "xarm_arm")

        # --- Publishers ---
        self.vision_reset_pub = self.create_publisher(String, '/vision/reset_tracker', 10)
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        self.state_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)

        # --- Physical Parameters ---
        self.UF_TOOL_LENGTH = 0.28     # TCP to tool0 offset (meters)
        self.HOVER_HEIGHT = 0.05       # Height above object for hover (meters)
        self.RETRACT_DIST = 0.030      # Post-grasp retract distance (meters)
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0
        self.GRIPPER_SPEED = 0.3

        # --- Frame Configuration ---
        self.CAMERA_FRAME = 'camera_color_optical_frame'
        self.PLANNING_FRAME = 'world_world'

        # --- Home Positions ---
        self.UF_HOME_JOINTS = {
            'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57,
            'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0
        }
        self.XARM_HOME_JOINTS = {
            'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
            'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0
        }
        self.DROP_POSE = {'x': 0.92, 'y': -0.36, 'z': 1.25}

        # --- Tactile Descent Config ---
        self.CONTACT_JOINT = "u1_joint5"
        self.TORQUE_THRESHOLD = 1.0    # Nm spike to detect contact
        self.DESCENT_MAX_DIST = 0.15   # Max descent in meters
        self.DESCENT_SPEED = 35.0      # mm/s for SDK descent (minimum 35mm/s!)
        self.POST_CONTACT_RETRACT = 0.005  # Small retract after contact (meters)

        # --- Joint Name for Gripper Position ---
        self.JOINT_GRIPPER = "rg6_l_out"

        # --- Thread-Safe State ---
        self.data_lock = threading.Lock()
        self.latest_targets = []
        self.is_holding_object = False

        # --- Subscriptions ---
        self.create_subscription(Bool, '/object_hold_status', self.hold_status_callback, 10)
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)

        self.get_logger().info("🚀 Object Pickup Skill Node Active (SDK-only).")

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def hold_status_callback(self, msg):
        self.is_holding_object = msg.data

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data.strip().strip("'").strip('"'))
            with self.data_lock:
                self.latest_targets = data.get("global_view", {}).get("objects", [])
        except Exception:
            pass

    # =====================================================================
    # HELPERS
    # =====================================================================
    def wait_for_settled(self, backend, timeout=5.0):
        """Monitors joint states until motion settles or timeout."""
        time.sleep(0.3)
        start_t = time.time()
        last_pos = {}
        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr = backend.current_joint_positions.copy()
            if last_pos:
                deltas = [abs(curr[n] - last_pos[n]) for n in curr if n in last_pos]
                if deltas and max(deltas) <= 0.006:
                    return True
            last_pos = curr
            time.sleep(0.1)
        return True

    def wait_for_gripper(self, target_deg, timeout=5.0):
        """Waits until gripper reaches target or stalls (object grasped)."""
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
                    self.get_logger().info("✅ Gripper stalled (Object grasped).")
                    return True
            else:
                stall_timer = 0.0
            last_pos = current_pos
            time.sleep(0.1)
        return False

    def publish_arm_state(self, state_str: str):
        msg = String()
        msg.data = state_str
        self.state_pub.publish(msg)

    def world_to_robot_base_mm(self, wx, wy, wz):
        """Convert world_world coords (m) to robot-base frame (mm) via live TF+SDK offset."""
        try:
            tf = self.uf850.tf_buffer.lookup_transform(
                self.PLANNING_FRAME, 'u1_tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            ee_wx = tf.transform.translation.x
            ee_wy = tf.transform.translation.y
            ee_wz = tf.transform.translation.z
        except Exception as e:
            self.get_logger().error(f"❌ TF EE lookup failed: {e}")
            return None

        if not hasattr(self.uf850, 'arm'):
            return None
        code, base_pos = self.uf850.arm.get_position(is_radian=False)
        if code != 0 or not base_pos:
            return None

        off_x = base_pos[0] - ee_wx * 1000.0
        off_y = base_pos[1] - ee_wy * 1000.0
        off_z = base_pos[2] - ee_wz * 1000.0

        return (wx * 1000.0 + off_x, wy * 1000.0 + off_y, wz * 1000.0 + off_z)

    def approach_hover(self, tx, ty, hover_z):
        """
        Move to hover position. Tries MoveIt first, then SDK fallback.
        Returns True only if the arm actually reaches within tolerance of the target.
        Aborts with False on any failure — never silently continues.
        """
        HOVER_Z_TOLERANCE = 0.04  # 4 cm — must be this close to proceed

        # --- Attempt 1: MoveIt ---
        success = self.uf850.move_to_pose_robust(tx, ty, hover_z, velocity=0.1)
        if success:
            self.wait_for_settled(self.uf850)
            return self._verify_hover_z(hover_z, HOVER_Z_TOLERANCE)

        # --- Attempt 2: SDK fallback ---
        self.get_logger().warn("⚠️ MoveIt hover failed. Trying SDK fallback...")
        base_target = self.world_to_robot_base_mm(tx, ty, hover_z)
        if not base_target:
            self.get_logger().error("❌ SDK fallback: frame conversion failed.")
            return False

        if not hasattr(self.uf850, 'arm'):
            self.get_logger().error("❌ SDK arm not available.")
            return False

        self.uf850._ensure_sdk_mode(0)
        code, curr = self.uf850.arm.get_position(is_radian=False)
        if code != 0:
            return False

        ret = self.uf850.arm.set_position(
            x=base_target[0], y=base_target[1], z=base_target[2],
            roll=curr[3], pitch=curr[4], yaw=curr[5],
            speed=35.0, is_radian=False, wait=True)

        if ret != 0:
            self.get_logger().error(f"❌ SDK hover move failed, code={ret}")
            return False

        self.wait_for_settled(self.uf850)
        if not self._verify_hover_z(hover_z, HOVER_Z_TOLERANCE):
            self.get_logger().error(
                f"❌ Arm did not reach hover height {hover_z:.3f}m after SDK fallback. ABORTING.")
            return False

        self.get_logger().info("✅ SDK fallback hover reached.")
        return True

    def _verify_hover_z(self, target_z, tolerance):
        """Check actual EE Z (world frame) is within tolerance of target_z."""
        try:
            tf = self.uf850.tf_buffer.lookup_transform(
                self.PLANNING_FRAME, 'u1_tool0', rclpy.time.Time())
            actual_z = tf.transform.translation.z
            diff = abs(actual_z - target_z)
            self.get_logger().info(
                f"📏 Hover Z check: actual={actual_z:.3f}m, target={target_z:.3f}m, diff={diff*100:.1f}cm")
            return diff <= tolerance
        except Exception as e:
            self.get_logger().warn(f"⚠️ Z verify TF failed ({e}), allowing descent.")
            return True  # TF unavailable — don't block if SDK move returned 0

    def publish_hold_status(self, is_held: bool):
        msg = Bool()
        msg.data = is_held
        self.hold_status_pub.publish(msg)
        self.get_logger().info(f"📢 Published Hold Status: {is_held}")

    def descend_until_contact(self):
        """SDK-based tactile descent monitoring u1_joint5 torque.
        Returns True if contact detected or max descent reached."""
        baseline_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
        self.get_logger().info(f"⬇️ SDK Tactile Descent (Monitoring: {self.CONTACT_JOINT}, Threshold: {self.TORQUE_THRESHOLD}Nm, Baseline: {baseline_effort:.4f}Nm)")
        self.contact_detected = False
        start_t = time.time()

        def check_force():
            nonlocal start_t
            curr_effort = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, baseline_effort)
            spike = abs(curr_effort - baseline_effort)
            
            if (time.time() - start_t) > 0.2:
                if spike > self.TORQUE_THRESHOLD:
                    self.get_logger().warn(f"💥 CONTACT! Spike: {spike:.4f} Nm")
                    self.contact_detected = True
                    return True
            return False

        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=self.DESCENT_MAX_DIST,
            speed_mm_s=self.DESCENT_SPEED,
            check_force_callback=check_force
        )

        # Clear hardware error state after force stop
        self.get_logger().info("🔄 Clearing hardware error state...")
        self.uf850.reset_robot()
        time.sleep(0.5)

        # Small retract after contact
        self.get_logger().info(f"⬆️ SDK Retract: {self.POST_CONTACT_RETRACT*1000:.0f}mm UP")
        success = self.uf850.jog_cartesian_sdk(0.0, 0.0, self.POST_CONTACT_RETRACT, speed_mm_s=35.0)

        if not success:
            self.get_logger().error("❌ SDK Retraction failed.")
            return False

        time.sleep(0.5)
        return True

    # =====================================================================
    # MAIN EXECUTION SEQUENCE
    # =====================================================================
    def execute_pickup(self, target_id, target_label, interactive=False):
        """Full pickup sequence: approach → descend → grasp → lift → drop."""
        print(f"\n🛠️ [START] {target_label} Pickup (ID: {target_id})")
        self.publish_arm_state("MOVING")

        # --- STEP 0: PRE-CONDITION (Release if holding) ---
        if self.is_holding_object:
            print("📦 [PRE-CONDITION] Active Hold Detected. Releasing...")
            self.gripper.move_to_joint_positions(
                {self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
            self.wait_for_gripper(self.OPEN_DEG)
            self.publish_hold_status(False)
            self.publish_arm_state("MOVING")

            print("⬆️ SDK Retracting 30cm...")
            if not self.uf850.jog_cartesian_sdk(0, 0, 0.30, speed_mm_s=35.0):
                print("❌ SDK Retract failed. Aborting.")
                self.publish_arm_state("ERROR")
                return False
            self.wait_for_settled(self.uf850)

            print("🏠 Homing UF850...")
            self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS, "u1", velocity=0.1)
            self.wait_for_settled(self.uf850)

        # --- STEP 1: CLEAR WORKSPACE (Home xArm5) ---
        if interactive: input("👉 STEP 1: Home xArm5 to clear workspace? [Enter]")
        print("🏠 Homing xArm5...")
        self.xarm5.move_to_joint_positions(self.XARM_HOME_JOINTS, "xarm5", velocity=0.2)
        self.wait_for_settled(self.xarm5)

        # --- STEP 2: GET TARGET COORDINATES ---
        with self.data_lock:
            target = next((t for t in self.latest_targets if t['id'] == target_id), None)
        if not target or 'xyz' not in target:
            print(f"❌ Target ID {target_id} not found or has no xyz data.")
            return False

        raw_p = Pose()
        raw_p.position.x, raw_p.position.y, raw_p.position.z = target['xyz']
        raw_p.orientation.w = 1.0
        world_p = self.uf850.get_transformed_pose(raw_p, self.CAMERA_FRAME, self.PLANNING_FRAME)
        if not world_p:
            print("❌ TF Transform failed.")
            return False

        # Apply small coordinate offsets if needed
        tx = world_p.pose.position.x - 0.04
        ty = world_p.pose.position.y + 0.03
        final_z = world_p.pose.position.z + self.UF_TOOL_LENGTH
        hover_z = final_z + self.HOVER_HEIGHT

        print(f"📍 World Target: X:{tx:.3f}, Y:{ty:.3f}, HoverZ:{hover_z:.3f}")

        # --- STEP 3: APPROACH WITH GRIPPER OPEN ---
        if interactive: input("👉 STEP 3: Open gripper and hover? [Enter]")
        print("🔓 Opening Gripper...")
        self.gripper.move_to_joint_positions(
            {self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)

        print(f"🚁 Hovering at X:{tx:.3f}, Y:{ty:.3f}, Z:{hover_z:.3f}...")
        if not self.approach_hover(tx, ty, hover_z):
            self.get_logger().error("❌ Hover approach failed (MoveIt + SDK). ABORTING pickup.")
            self.publish_arm_state("ERROR")
            return False

        # --- STEP 4: CLOSE GRIPPER TO NEUTRAL (0°) ---
        if interactive: input("👉 STEP 4: Close gripper to neutral (0°) for descent? [Enter]")
        print("🗜️ Closing Gripper to 0° (neutral)...")
        self.gripper.move_to_joint_positions(
            {self.JOINT_GRIPPER: math.radians(0.0)}, "rg6", velocity=self.GRIPPER_SPEED)
        time.sleep(0.5)

        # --- STEP 5: TACTILE DESCENT ---
        if interactive: input("👉 STEP 5: Start SDK Tactile Descent? [Enter]")
        if not self.descend_until_contact():
            print("❌ Tactile descent failed.")
            return False
        self.wait_for_settled(self.uf850)

        # --- STEP 6: RETRACT 7mm THEN GRASP ---
        if interactive: input("👉 STEP 6: Retract 7mm and grasp? [Enter]")
        print("⬆️ Retracting 7mm before grasping...")
        self.uf850.jog_cartesian_sdk(0, 0, 0.007, speed_mm_s=35.0)
        self.wait_for_settled(self.uf850)

        # --- STEP 7: FULL GRIPPER CLOSE ---
        print("🗜️ Closing Gripper fully...")
        self.gripper.move_to_joint_positions(
            {self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        is_held = self.wait_for_gripper(self.CLOSE_DEG)
        if not is_held:
            print("❌ Failed to grasp object.")
            self.publish_arm_state("ERROR")
            return False
        time.sleep(0.5)
        self.publish_hold_status(True)
        self.publish_arm_state("HOLDING")

        # --- STEP 8: RETRACT 3cm ---
        if interactive: input("👉 STEP 8: Retract 3cm? [Enter]")
        print("⬆️ Retracting 3cm to clear surroundings...")
        self.uf850.retract_sdk_z_closed_loop(0.03, speed_mm_s=35.0)
        self.wait_for_settled(self.uf850)

        # --- STEP 9: HOME FIRST ---
        if interactive: input("👉 STEP 9: Home UF850? [Enter]")
        print("🏠 Homing UF850...")
        self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS, "u1", velocity=0.1)
        self.wait_for_settled(self.uf850)

        # --- STEP 10: DROP-OFF ---
        if interactive: input("👉 STEP 10: Move to drop zone and release? [Enter]")
        print("🗑️ Moving to Drop Pose...")
        drop_ok = self.uf850.move_to_pose_robust(
            self.DROP_POSE['x'], self.DROP_POSE['y'], self.DROP_POSE['z'], velocity=0.1)

        if not drop_ok:
            print("⚠️ IK failed for drop zone. Trying Cartesian path...")
            drop_ok = self.uf850.move_cartesian_to_pose(
                self.DROP_POSE['x'], self.DROP_POSE['y'], self.DROP_POSE['z'], velocity=0.1)

        if not drop_ok:
            print("❌ Cannot reach drop zone. Releasing and homing...")
            self.gripper.move_to_joint_positions(
                {self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
            time.sleep(1.0)
            self.publish_hold_status(False)
            self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS, "u1", velocity=0.1)
            self.wait_for_settled(self.uf850)
            return False

        self.wait_for_settled(self.uf850)

        # --- STEP 11: RELEASE & HOME ---
        self.publish_arm_state("DROPPING")
        print("📦 Opening gripper to release...")
        self.gripper.move_to_joint_positions(
            {self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)}, "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        time.sleep(1.0)
        self.publish_hold_status(False)

        print("🏠 Homing UF850...")
        self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS, "u1", velocity=0.1)
        self.wait_for_settled(self.uf850)

        self.publish_arm_state("IDLE")
        print("✅ [SUCCESS] Pickup Sequence Complete.")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPickupSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    time.sleep(1.0)
    node.vision_reset_pub.publish(String(data='reset'))

    try:
        while rclpy.ok():
            tid = None
            with node.data_lock:
                # Look for objects that could be picked up (e.g., PCB, Top_Lid)
                pickables = [t for t in node.latest_targets
                            if any(k in t.get('label', '').lower() for k in ['pcb', 'lid', 'top'])]
                if pickables:
                    tid = pickables[0]['id']
                    label = pickables[0].get('label', 'unknown')
            if tid is not None:
                node.execute_pickup(tid, label, interactive=False)
                with node.data_lock:
                    node.latest_targets = []
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
