#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger  # <--- NEW: Imported Trigger for the Servo Service
import threading
import math
import time
from disassembly_skills.motion_backend import MotionBackend

class ObjectFlipSkill(Node):
    def __init__(self):
        super().__init__('object_flip_skill_node')
        
        # --- Hardware Backends (Updated to use robust Planning Groups) ---
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        # --- Subscriptions & Synchronization ---
        self.is_holding_object = False
        self.hold_event = threading.Event()
        self.create_subscription(Bool, '/object_hold_state/is_held', self.hold_status_callback, 10)
        
        # --- Publishers ---
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10) # Added to update manager
        
        # --- ⏰ NEW: Servo Start Service Client ---
        self.uf_servo_start_client = self.create_client(Trigger, '/uf_servo_node/start_servo')

        # --- Configuration ---
        self.PLANNING_FRAME = 'world_world'
        self.ROBOT_EE_LINK = "u1_tool0"
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0
        
        self.RETRACT_Z_HEIGHT = 0.15           
        self.TORQUE_THRESHOLD = 3.0            
        self.DESCENT_SPEED = 0.1
        
        self.get_logger().info("🚀 Object Flip Skill: Robust IROS Setup Active.")
        self.publish_state("IDLE")

    def publish_state(self, s): 
        self.state_update_pub.publish(String(data=s))

    def publish_hold_status(self, h): 
        self.hold_status_pub.publish(Bool(data=h))

    def hold_status_callback(self, msg):
        self.is_holding_object = msg.data
        self.hold_event.set() 

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
        
        # 0.006 rad is ~0.34 degrees. Ignores motor hum and minor vibrations.
        NOISE_TOLERANCE = 0.006 

        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr_positions = self.uf850.current_joint_positions.copy()
            if not curr_positions:
                time.sleep(0.1)
                continue
                
            if last_positions:
                # Find max delta across all tracked arm joints
                max_delta = 0.0
                for j_name, j_pos in curr_positions.items():
                    if j_name in last_positions:
                        delta = abs(j_pos - last_positions[j_name])
                        if delta > max_delta:
                            max_delta = delta
                            
                # If movement is within the noise tolerance
                if max_delta <= NOISE_TOLERANCE:
                    settle_timer += 0.1
                    if settle_timer >= 0.4: # Needs to be still for 0.4 seconds
                        print("✅ Arm has completely settled.")
                        return True
                else:
                    settle_timer = 0.0 # Reset if a joint moves beyond tolerance
                    
            last_positions = curr_positions
            time.sleep(0.1)
            
        print("⚠️ Warning: Arm settle timeout reached. Proceeding anyway.")
        return True

    def wait_for_gripper(self, target_deg, timeout=5.0):
        """Stall-aware gripper monitoring (Correct for grabbing lid)."""
        target_rad = math.radians(target_deg)
        start_t = time.time(); last_pos = 999.0; stall_timer = 0.0
        is_closing = target_deg < 0 

        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if curr == 999:
                time.sleep(0.1); continue
            
            if abs(curr - target_rad) < 0.05: return True
            
            if abs(curr - last_pos) < 0.002:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    if is_closing:
                        self.get_logger().info(f"✅ Grasp Verified at {curr:.3f} rad.")
                        return True
                    else:
                        return False
            else: stall_timer = 0.0
            last_pos = curr; time.sleep(0.1)
        return False

    def execute_flip(self, interactive=True):
        """Sequential Flip: Lift -> Rotate J6 -> Tactile Descent -> Re-grasp."""
        
        # 0. Safety Check
        if not self.is_holding_object:
            self.get_logger().error("❌ Empty Gripper: Cannot perform Flip.")
            return False

        # Get current state via TF2
        try:
            current_tf = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
            pos = current_tf.transform.translation
            rot = current_tf.transform.rotation
            q_dict = {'qx': rot.x, 'qy': rot.y, 'qz': rot.z, 'qw': rot.w}
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}"); return False

        # --- STEP 1: LIFT ---
        self.publish_state("MOVING")
        if interactive: input(f"🚀 STEP 1: Lift {self.RETRACT_Z_HEIGHT*100}cm? [Enter]")
        target_z = pos.z + self.RETRACT_Z_HEIGHT
        if not self.uf850.move_to_pose_robust(pos.x, pos.y, target_z, q_dict, velocity=0.1): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-LIFT SETTLE

        # --- STEP 2: ROTATE JOINT 6 (Flip Logic) ---
        self.publish_state("FLIPPING")
        if interactive: input(f"🚀 STEP 2: Rotate u1_joint6 180°? [Enter]")
        
        current_joints = self.uf850.current_joint_positions.copy()
        current_j6 = current_joints.get("u1_joint6", 0.0)
        
        # Logic to decide rotation direction based on cable limit
        if current_j6 > (math.pi / 2.0):
            self.get_logger().info("🔄 Unwinding Joint 6 (-180°)...")
            current_joints["u1_joint6"] -= math.pi
        else:
            self.get_logger().info("🔄 Winding Joint 6 (+180°)...")
            current_joints["u1_joint6"] += math.pi
            
        if not self.uf850.move_to_joint_positions(current_joints): return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-ROTATE SETTLE

        # --- ⏰ WAKE UP UF850 SERVO NODE ---
        print("⏰ Requesting UF850 Servo Node to Activate...")
        if self.uf_servo_start_client.wait_for_service(timeout_sec=2.0):
            self.uf_servo_start_client.call_async(Trigger.Request())
        else:
            self.get_logger().warn("⚠️ /uf_servo_node/start_servo service not available!")
        time.sleep(1.0) # Controller swap delay

        # --- STEP 3: TACTILE DESCENT ---
        if interactive: input(f"🚀 STEP 3: Tactile Descent to Table? [Enter]")
        self.publish_state("MOVING")
        # move_linear_z_with_torque_stop handles the baseline-subtraction internally
        if not self.uf850.move_linear_z_with_torque_stop(self.DESCENT_SPEED, self.TORQUE_THRESHOLD): 
            return False
        self.wait_for_arm_settled() # 🛑 ADDED POST-DESCENT SETTLE
        
        # Retract 5mm to clear pressure
        time.sleep(0.5)
        self.uf850.jog_cartesian_servo(0.0, 0.0, 0.005, duration=0.5)
        self.wait_for_arm_settled() # 🛑 ADDED POST-RETRACT SETTLE

        # --- STEP 4: OPEN ---
        if interactive: input(f"🚀 STEP 4: Open Gripper? [Enter]")
        self.publish_hold_status(False) # Manager will show EMPTY
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)})
        self.wait_for_gripper(self.OPEN_DEG)
        
        # Object is now free on table
        self.publish_state("IDLE")
        time.sleep(1.0)

        # --- STEP 5: RE-GRASP ---
        if interactive: input(f"🚀 STEP 5: Re-grasp Flipped Object? [Enter]")
        self.publish_state("HOLDING")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)})
        
        success = self.wait_for_gripper(self.CLOSE_DEG)
        self.publish_hold_status(success)
        
        if success:
            self.get_logger().info("🎉 FLIP SUCCESSFUL")
            self.publish_state("HOLDING")
        else:
            self.publish_state("ERROR")
            
        return success

def main(args=None):
    rclpy.init(args=args)
    flip_node = ObjectFlipSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(flip_node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # Wait for centralized manager to say we have an object
        while rclpy.ok() and not flip_node.is_holding_object:
            flip_node.get_logger().info("🕒 Waiting for HOLDING state...")
            flip_node.hold_event.clear()
            flip_event_tripped = flip_node.hold_event.wait(timeout=2.0)
        
        if rclpy.ok() and flip_node.is_holding_object:
            flip_node.execute_flip(interactive=True)
            
    except KeyboardInterrupt: pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__': main()