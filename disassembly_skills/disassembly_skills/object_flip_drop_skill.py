#!/usr/bin/env python3
"""
Object Flip-Drop Skill — Pure SDK (no MoveIt, no Servo)

Called after ObjectFlipSkill has re-grasped the flipped object.

Sequence (all via arm.set_position / set_servo_angle):
  1. Save start TCP pose (robot base frame, via arm.get_position)
  2. Retract 10 cm up   (jog_cartesian_sdk)
  3. Save retract pose
  4. Move to bin hover  (set_position in robot-base frame — computed via TF+SDK offset)
  5. Open gripper       (drop)
  6. Flip wrist +180°   (set_servo_angle on joint-6)
  7. Flip wrist back    (set_servo_angle −180°)
  8. Return to retract pose (set_position)
  9. Tactile descent    (move_linear_z_sdk_with_force_stop)
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool

import threading
import time
import math

from disassembly_skills.motion_backend import MotionBackend


class ObjectFlipDropSkill(Node):
    def __init__(self):
        super().__init__('object_flip_drop_skill_node')

        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")

        self.state_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)

        self.get_logger().info("🚀 Object Flip-Drop Skill Node Active (SDK-only).")

        # ================= CONFIGURATION =================
        self.PLANNING_FRAME = 'world_world'
        self.CAMERA_FRAME = 'camera_color_optical_frame'
        self.ROBOT_EE_LINK = "u1_tool0"

        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.GRIPPER_SPEED = 0.3

        self.TOOL_LENGTH = 0.28           # TCP-to-flange offset (m)
        self.RETRACT_DIST = 0.10         # 10 cm retract before transit (m)

        # Fixed drop location in world_world frame (metres)
        self.DROP_POSE = {'x': 0.929872, 'y': -0.633943, 'z': 1.0977}

        self.CONTACT_JOINT = "u1_joint5"
        self.TORQUE_THRESHOLD = 1.0      # Nm spike threshold for tactile descent
        self.DESCENT_MAX_DIST = 0.15     # Max descent distance (m)

        # SDK speeds (all ≥ 26 mm/s minimum)
        self.RETRACT_SPEED = 50.0
        self.TRANSIT_SPEED = 50.0
        self.JOINT_FLIP_SPEED_DEG = 30.0  # deg/s for joint rotation
        self.DESCENT_SPEED = 26.0
        # =================================================

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def publish_arm_state(self, state_str: str):
        msg = String()
        msg.data = state_str
        self.state_pub.publish(msg)

    def publish_hold_status(self, is_held: bool):
        msg = Bool()
        msg.data = is_held
        self.hold_status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def wait_for_gripper(self, target_deg, timeout=5.0):
        target_rad = math.radians(target_deg)
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout:
            current_pos = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if current_pos == 999:
                time.sleep(0.1)
                continue
            if abs(current_pos - target_rad) < 0.05:
                return True
            time.sleep(0.1)
        return False

    def world_to_robot_base_mm(self, wx, wy, wz):
        """
        Convert a point in world_world frame (metres) to robot-base frame (mm).

        Uses live SDK + TF readings to compute world→base offset:
          offset = arm.get_position() [mm] − TF EE in world [m × 1000]
        Valid when world_world and robot-base share the same rotation (standard mount).
        """
        # Current EE in world_world via TF
        try:
            tf = self.uf850.tf_buffer.lookup_transform(
                self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            ee_wx = tf.transform.translation.x
            ee_wy = tf.transform.translation.y
            ee_wz = tf.transform.translation.z
        except Exception as e:
            self.get_logger().error(f"❌ TF EE lookup failed: {e}")
            return None

        # Current EE in robot-base via SDK (mm)
        code, base_pos = self.uf850.arm.get_position(is_radian=False)
        if code != 0 or not base_pos:
            self.get_logger().error(f"❌ arm.get_position failed, code={code}")
            return None

        # world→base offset (mm)
        off_x = base_pos[0] - ee_wx * 1000.0
        off_y = base_pos[1] - ee_wy * 1000.0
        off_z = base_pos[2] - ee_wz * 1000.0

        tgt_x = wx * 1000.0 + off_x
        tgt_y = wy * 1000.0 + off_y
        tgt_z = wz * 1000.0 + off_z

        self.get_logger().info(
            f"📐 Drop target (robot-base, mm): X:{tgt_x:.1f} Y:{tgt_y:.1f} Z:{tgt_z:.1f}")
        return (tgt_x, tgt_y, tgt_z)

    def flip_wrist(self, delta_rad, speed_deg=None):
        """
        Rotates joint-6 of UF850 by delta_rad via SDK set_servo_angle.
        Uses mode 0 (position control).
        """
        if speed_deg is None:
            speed_deg = self.JOINT_FLIP_SPEED_DEG

        if not hasattr(self.uf850, 'arm'):
            self.get_logger().error("❌ SDK arm not available")
            return False

        self.uf850._ensure_sdk_mode(0)
        code, angles = self.uf850.arm.get_servo_angle(is_radian=True)
        if code != 0 or not angles:
            self.get_logger().error(f"❌ get_servo_angle failed, code={code}")
            return False

        target = list(angles)
        target[5] += delta_rad  # joint-6 is index 5

        ret = self.uf850.arm.set_servo_angle(
            angle=target, speed=speed_deg, is_radian=True, wait=True)
        return ret == 0

    def descend_until_contact(self):
        """SDK tactile descent on u1_joint5 torque."""
        baseline = self.uf850.current_joint_efforts.get(self.CONTACT_JOINT, 0.0)
        self.get_logger().info(
            f"⬇️ Tactile Descent (joint:{self.CONTACT_JOINT}, "
            f"baseline:{baseline:.3f}Nm, thresh:{self.TORQUE_THRESHOLD}Nm)")
        self.contact_detected = False
        start_t = time.time()

        def check_force():
            nonlocal start_t
            effort = self.uf850.current_joint_efforts.get(
                self.CONTACT_JOINT, baseline)
            spike = abs(effort - baseline)
            if (time.time() - start_t) > 0.2 and spike > self.TORQUE_THRESHOLD:
                self.get_logger().warn(f"💥 CONTACT! Spike: {spike:.3f}Nm")
                self.contact_detected = True
                return True
            return False

        self.uf850.move_linear_z_sdk_with_force_stop(
            distance_down_m=self.DESCENT_MAX_DIST,
            speed_mm_s=self.DESCENT_SPEED,
            check_force_callback=check_force)

        self.uf850.reset_robot()
        time.sleep(0.3)
        return self.contact_detected

    # ------------------------------------------------------------------
    # Main Sequence
    # ------------------------------------------------------------------
    def execute_flip_drop(self, bin_name="bin_1", interactive=False):
        print("\n" + "!" * 60)
        print(f"🤖 INITIATING FLIP-DROP SKILL → {bin_name}")
        print("!" * 60)

        if not hasattr(self.uf850, 'arm'):
            self.get_logger().error("❌ SDK arm not available (is hardware_type=real?)")
            return False

        self.publish_arm_state("MOVING")

        # ── 1. Ensure SDK mode, save start position ──────────────────────
        self.uf850._ensure_sdk_mode(0)
        code, start_pos = self.uf850.arm.get_position(is_radian=False)
        if code != 0:
            self.get_logger().error("❌ Cannot read start position.")
            self.publish_arm_state("ERROR")
            return False
        print(f"📌 Start (robot-base mm): X:{start_pos[0]:.1f} Y:{start_pos[1]:.1f} Z:{start_pos[2]:.1f}")

        # ── 2. Retract 10 cm up ──────────────────────────────────────────
        if interactive:
            input("\n🚀 STEP 1: Retract 10cm? [Enter]")
        print(f"⬆️  SDK Retract {self.RETRACT_DIST * 100:.0f}cm...")
        if not self.uf850.jog_cartesian_sdk(0, 0, self.RETRACT_DIST,
                                             speed_mm_s=self.RETRACT_SPEED):
            self.get_logger().error("❌ Retract failed.")
            self.publish_arm_state("ERROR")
            return False

        code, retract_pos = self.uf850.arm.get_position(is_radian=False)
        if code != 0:
            self.get_logger().error("❌ Cannot read retract position.")
            self.publish_arm_state("ERROR")
            return False
        print(f"📌 Retract pose saved: Z:{retract_pos[2]:.1f}mm")

        # ── 3. Convert fixed DROP_POSE to robot-base frame ───────────────
        if interactive:
            input("\n🚀 STEP 2: Move to drop pose? [Enter]")
        print(f"   📍 DROP_POSE (world_world): {self.DROP_POSE}")
        drop_base = self.world_to_robot_base_mm(
            self.DROP_POSE['x'], self.DROP_POSE['y'], self.DROP_POSE['z'])

        if not drop_base:
            self.get_logger().error("❌ Could not convert DROP_POSE to robot-base frame.")
            self.publish_arm_state("ERROR")
            return False

        tgt_x, tgt_y, tgt_z = drop_base

        # ── 4. Move to drop pose (pure SDK set_position, keep retract orientation) ──
        print(f"🚁 SDK Move to drop pose: X:{tgt_x:.1f} Y:{tgt_y:.1f} Z:{tgt_z:.1f}mm")
        self.uf850._ensure_sdk_mode(0)
        ret = self.uf850.arm.set_position(
            x=tgt_x, y=tgt_y, z=tgt_z,
            roll=retract_pos[3], pitch=retract_pos[4], yaw=retract_pos[5],
            speed=self.TRANSIT_SPEED, is_radian=False, wait=True)
        if ret != 0:
            self.get_logger().error(f"❌ SDK move to bin failed, code={ret}")
            self.publish_arm_state("ERROR")
            return False

        # ── 5. Open gripper — drop object ────────────────────────────────
        if interactive:
            input("\n🚀 STEP 3: Open gripper to drop? [Enter]")
        self.publish_arm_state("DROPPING")
        print("🪣 Opening gripper — releasing object...")
        self.gripper.move_to_joint_positions(
            {self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)},
            "rg6", velocity=self.GRIPPER_SPEED)
        self.wait_for_gripper(self.OPEN_DEG)
        self.publish_hold_status(False)
        time.sleep(0.2)

        # ── 6. Wrist flip +180° (shake to release) ───────────────────────
        if interactive:
            input("\n🚀 STEP 4: Flip wrist +180°? [Enter]")
        print("🔄 Wrist flip +180°...")
        if not self.flip_wrist(+math.pi):
            self.get_logger().warn("⚠️ Wrist flip failed — continuing.")

        time.sleep(0.2)

        # ── 7. Wrist flip back -180° ─────────────────────────────────────
        if interactive:
            input("\n🚀 STEP 5: Flip wrist back -180°? [Enter]")
        print("🔄 Wrist flip back −180°...")
        if not self.flip_wrist(-math.pi):
            self.get_logger().warn("⚠️ Wrist flip-back failed — continuing.")

        time.sleep(0.2)

        # ── 8. Return to retract pose ─────────────────────────────────────
        if interactive:
            input("\n🚀 STEP 6: Return to retract position? [Enter]")
        print(f"🔙 SDK Return to retract pose: Z:{retract_pos[2]:.1f}mm")
        self.publish_arm_state("MOVING")
        self.uf850._ensure_sdk_mode(0)
        ret = self.uf850.arm.set_position(
            x=retract_pos[0], y=retract_pos[1], z=retract_pos[2],
            roll=retract_pos[3], pitch=retract_pos[4], yaw=retract_pos[5],
            speed=self.TRANSIT_SPEED, is_radian=False, wait=True)
        if ret != 0:
            self.get_logger().warn(f"⚠️ Return to retract pose failed, code={ret}. Continuing.")

        # ── 9. Tactile descent ────────────────────────────────────────────
        if interactive:
            input("\n🚀 STEP 7: Tactile descent? [Enter]")
        print("⬇️  SDK Tactile descent to surface...")
        self.descend_until_contact()

        print(f"\n🎉 FLIP-DROP COMPLETE: Object placed in {bin_name}.")
        self.publish_arm_state("IDLE")
        return True


def main(args=None):
    rclpy.init(args=args)
    drop_node = ObjectFlipDropSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(drop_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        final_status = drop_node.execute_flip_drop(bin_name="bin_1", interactive=False)
        print(f"\n⏳ Flip-Drop Sequence Complete. Status: {final_status}")
        time.sleep(1.0)

    except KeyboardInterrupt:
        pass

    drop_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
