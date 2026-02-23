#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
import threading, math, time
from disassembly_skills.motion_backend import MotionBackend

class FlipDropSkill(Node):
    def __init__(self):
        super().__init__('flip_drop_skill_node')
        
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.is_holding_object = False
        self.hold_event = threading.Event()
        self.create_subscription(Bool, '/object_hold_state/is_held', self.hold_status_callback, 10)
        
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        
        # Configuration
        self.PLANNING_FRAME = 'world_world'
        self.ROBOT_EE_LINK = "u1_tool0"
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG, self.CLOSE_DEG = 35.0, -35.0
        
        self.RETRACT_Z_HEIGHT = 0.15           
        self.INTERMEDIATE_POSE = {'x': 0.929872, 'y': -0.633943, 'z': 1.0977}
        self.TORQUE_THRESHOLD = 3.0            
        self.DESCENT_SPEED = 0.1
        
        self.get_logger().info("🚀 Flip-Drop Skill: Fixed Sequential Mode Active.")
        self.publish_state("IDLE")

    def publish_state(self, s): self.state_update_pub.publish(String(data=s))
    def publish_hold_status(self, h): self.hold_status_pub.publish(Bool(data=h))
    def hold_status_callback(self, msg): self.is_holding_object = msg.data; self.hold_event.set()

    def wait_for_gripper(self, target_deg, timeout=5.0):
        target_rad = math.radians(target_deg)
        start_t = time.time(); last_pos = 999.0; stall_t = 0.0
        is_closing = target_deg < 0 
        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if curr == 999: time.sleep(0.1); continue
            if abs(curr - target_rad) < 0.05: return True
            if abs(curr - last_pos) < 0.002:
                stall_t += 0.1
                if stall_t >= 0.5 and is_closing: return True
            else: stall_t = 0.0
            last_pos = curr; time.sleep(0.1)
        return False

    def execute_flip_drop(self, interactive=True):
        if not self.is_holding_object:
            self.get_logger().error("❌ Gripper Empty. Cannot Flip."); return False

        # Capture START pose (The pose immediately after picking)
        try:
            start_tf = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
            sx, sy, sz = start_tf.transform.translation.x, start_tf.transform.translation.y, start_tf.transform.translation.z
            sq = start_tf.transform.rotation
            q_start = {'qx': sq.x, 'qy': sq.y, 'qz': sq.z, 'qw': sq.w}
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}"); return False

        # --- STEP 1: RETRACT (Stay at pick XY) ---
        self.publish_state("MOVING")
        if interactive: input(f"🚀 STEP 1: Retract {self.RETRACT_Z_HEIGHT*100}cm? [Enter]")
        rz = sz + self.RETRACT_Z_HEIGHT
        if not self.uf850.move_to_pose_robust(sx, sy, rz, q_start, velocity=0.1): return False

        # --- STEP 2: TRAVEL (Maintain pick orientation) ---
        if interactive: input(f"🚀 STEP 2: Travel to Flip Zone (Maintain Orientation)? [Enter]")
        # We use the Target X/Y/Z but keep q_start so it doesn't flip while moving
        if not self.uf850.move_to_pose_robust(self.INTERMEDIATE_POSE['x'], self.INTERMEDIATE_POSE['y'], self.INTERMEDIATE_POSE['z'], q_start, velocity=0.1): return False

        # --- STEP 3: FLIP (In-place rotation) ---
        self.publish_state("FLIPPING")
        if interactive: input(f"🚀 STEP 3: Flip Object 180°? [Enter]")
        curr_joints = self.uf850.current_joint_positions.copy()
        orig_j6 = curr_joints.get("u1_joint6", 0.0)
        # Toggle 180 deg
        curr_joints["u1_joint6"] += math.pi if orig_j6 <= (math.pi / 2.0) else -math.pi
        if not self.uf850.move_to_joint_positions(curr_joints): return False
        
        # --- STEP 4: FLIP BACK (Reset orientation) ---
        if interactive: input(f"🚀 STEP 4: Flip Back to Original? [Enter]")
        curr_joints["u1_joint6"] = orig_j6
        if not self.uf850.move_to_joint_positions(curr_joints): return False

        # --- STEP 5: RETURN (To the pose after Step 1) ---
        self.publish_state("MOVING")
        if interactive: input(f"🚀 STEP 5: Return to Pick Location? [Enter]")
        if not self.uf850.move_to_pose_robust(sx, sy, rz, q_start, velocity=0.1): return False

        # --- STEP 6: TACTILE DESCENT ---
        if interactive: input(f"🚀 STEP 6: Tactile Descent to Surface? [Enter]")
        if not self.uf850.move_linear_z_with_torque_stop(self.DESCENT_SPEED, self.TORQUE_THRESHOLD): return False
        
        # Lift 5mm for clearance before release
        time.sleep(0.5)
        self.uf850.jog_cartesian_servo(0.0, 0.0, 0.005, duration=0.5)

        # --- STEP 7: RELEASE & RESET ---
        if interactive: input(f"🚀 STEP 7: Release Object? [Enter]")
        self.publish_hold_status(False)
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)})
        self.wait_for_gripper(self.OPEN_DEG)
        
        time.sleep(0.5)
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)})
        self.wait_for_gripper(self.CLOSE_DEG)

        self.get_logger().info("🎉 Skill Finished: Flipped and Dropped.")
        self.publish_state("IDLE")
        return True

def main(args=None):
    rclpy.init(args=args); node = FlipDropSkill()
    executor = MultiThreadedExecutor(); executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        while rclpy.ok() and not node.is_holding_object:
            node.get_logger().info("🕒 Waiting for HOLDING state..."); node.hold_event.clear(); node.hold_event.wait(timeout=2.0)
        if rclpy.ok() and node.is_holding_object: node.execute_flip_drop(interactive=True)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()