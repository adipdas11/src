#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger  
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
        
        # --- ⏰ NEW: Servo Start Service Client ---
        self.uf_servo_start_client = self.create_client(Trigger, '/uf_servo_node/start_servo')
        
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

    # --- 🛑 NEW DYNAMIC SETTLE FUNCTION ---
    def wait_for_arm_settled(self, timeout=20.0):
        """
        Dynamically monitors joint states. Proceeds only when all joints stop moving.
        """
        print("⏳ Waiting for arm to physically settle (monitoring joint states)...")
        time.sleep(0.2) # Allow ROS 2 message buffer to catch up post-trajectory
        
        start_t = time.time()
        settle_timer = 0.0
        last_positions = {}
        NOISE_TOLERANCE = 0.006 

        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr_positions = self.uf850.current_joint_positions.copy()
            if not curr_positions:
                time.sleep(0.1)
                continue
                
            if last_positions:
                max_delta = 0.0
                for j_name, j_pos in curr_positions.items():
                    if j_name in last_positions:
                        delta = abs(j_pos - last_positions[j_name])
                        if delta > max_delta:
                            max_delta = delta
                            
                if max_delta <= NOISE_TOLERANCE:
                    settle_timer += 0.1
                    if settle_timer >= 0.4: 
                        print("✅ Arm has completely settled.")
                        return True
                else:
                    settle_timer = 0.0 
                    
            last_positions = curr_positions
            time.sleep(0.1)
            
        print("⚠️ Warning: Arm settle timeout reached. Proceeding anyway.")
        return True

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
        self.wait_for_arm_settled() # 🛑 ADDED POST-RETRACT SETTLE

        # --- STEP 2: TRAVEL (Maintain pick orientation) ---
        if interactive: input(f"🚀 STEP 2: Travel to Flip Zone (Maintain Orientation)? [Enter]")
        # We use the Target X/Y/Z but keep q_start so it doesn't flip while moving
        if not self.uf850.move_to_pose_robust(self.INTERMEDIATE_POSE['x'], self.INTERMEDIATE_POSE['y'], self.INTERMEDIATE_POSE['z'], q_start, velocity=0.1): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-TRAVEL SETTLE

        # --- STEP 3: FLIP (In-place rotation) ---
        self.publish_state("FLIPPING")
        if interactive: input(f"🚀 STEP 3: Flip Object 180°? [Enter]")
        curr_joints = self.uf850.current_joint_positions.copy()
        orig_j6 = curr_joints.get("u1_joint6", 0.0)
        # Toggle 180 deg
        curr_joints["u1_joint6"] += math.pi if orig_j6 <= (math.pi / 2.0) else -math.pi
        if not self.uf850.move_to_joint_positions(curr_joints): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-FLIP SETTLE
        
        # --- STEP 4: FLIP BACK (Reset orientation) ---
        if interactive: input(f"🚀 STEP 4: Flip Back to Original? [Enter]")
        curr_joints["u1_joint6"] = orig_j6
        if not self.uf850.move_to_joint_positions(curr_joints): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-FLIP-BACK SETTLE

        # --- STEP 5: RETURN (To the pose after Step 1) ---
        self.publish_state("MOVING")
        if interactive: input(f"🚀 STEP 5: Return to Pick Location? [Enter]")
        if not self.uf850.move_to_pose_robust(sx, sy, rz, q_start, velocity=0.1): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-RETURN SETTLE

        # --- ⏰ WAKE UP UF850 SERVO NODE ---
        print("⏰ Requesting UF850 Servo Node to Activate...")
        if self.uf_servo_start_client.wait_for_service(timeout_sec=2.0):
            self.uf_servo_start_client.call_async(Trigger.Request())
        else:
            self.get_logger().warn("⚠️ /uf_servo_node/start_servo service not available!")
        time.sleep(1.0) # Controller swap delay

        # --- STEP 6: TACTILE DESCENT ---
        if interactive: input(f"🚀 STEP 6: Tactile Descent to Surface? [Enter]")
        if not self.uf850.move_linear_z_with_torque_stop(self.DESCENT_SPEED, self.TORQUE_THRESHOLD): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-DESCENT SETTLE
        
        # Lift 5mm for clearance before release
        time.sleep(0.5)
        self.uf850.jog_cartesian_servo(0.0, 0.0, 0.005, duration=0.5)
        self.wait_for_arm_settled() # 🛑 ADDED POST-LIFT SETTLE

        # --- STEP 7: RELEASE & RESET ---
        if interactive: input(f"🚀 STEP 7: Open slightly to drop parts, then re-grasp Chassis? [Enter]")
        
        # 🛑 FIX: We DO NOT publish hold_status(False) here because we are keeping the chassis!
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)})
        self.wait_for_gripper(self.OPEN_DEG)
        
        time.sleep(0.5)
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)})
        self.wait_for_gripper(self.CLOSE_DEG)

        self.get_logger().info("🎉 Skill Finished: Loose parts dropped, Chassis re-grasped.")
        
        # 🛑 FIX: Broadcast that the arm is STILL holding the object
        self.publish_state("HOLDING")
        self.publish_hold_status(True)
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