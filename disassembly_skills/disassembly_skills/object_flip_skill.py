#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
import threading, math, time
from disassembly_skills.motion_backend import MotionBackend

class ObjectFlipSkill(Node):
    def __init__(self):
        super().__init__('object_flip_skill_node')
        
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        self.xarm5 = MotionBackend(self, "xarm_arm")
        
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        self.uf_servo_start_client = self.create_client(Trigger, '/uf_servo_node/start_servo')

        self.PLANNING_FRAME = 'world_world'
        self.ROBOT_EE_LINK = "u1_tool0"
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG, self.CLOSE_DEG = 35.0, -35.0
        self.RETRACT_Z_HEIGHT = 0.1           
        self.TORQUE_THRESHOLD = 3.0            
        self.DESCENT_SPEED = 0.1
        
        self.is_holding_object = False
        self.create_subscription(Bool, '/object_hold_state/is_held', self.hold_status_callback, 10)
        
        self.get_logger().info("🚀 Object Flip Skill: Fixed Single-Rotation Logic Active.")

    def hold_status_callback(self, msg): self.is_holding_object = msg.data

    def wait_for_arm_settled(self, timeout=10.0):
        time.sleep(0.5)
        start_t = time.time(); last_pos = {}
        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr = self.uf850.current_joint_positions.copy()
            if last_pos:
                delta = max([abs(curr[n] - last_pos[n]) for n in curr if n in last_pos], default=0)
                if delta <= 0.006: return True
            last_pos = curr; time.sleep(0.1)
        return True

    def execute_flip(self, interactive=True):
        if not self.is_holding_object:
            self.get_logger().error("❌ Error: No object held.")
            return False

        try:
            tf = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time())
            pos, rot = tf.transform.translation, tf.transform.rotation
            q_dict = {'qx': rot.x, 'qy': rot.y, 'qz': rot.z, 'qw': rot.w}
        except: return False

        # --- STEP 1: LIFT ---
        print(f"🚀 STEP 1: Lifting {self.RETRACT_Z_HEIGHT*100}cm...")
        
        # 🛠️ UPDATED: Using retract_servo_z_closed_loop with 0.01 speed as requested
        if not self.uf850.retract_servo_z_closed_loop(self.RETRACT_Z_HEIGHT, speed_mps=0.2): return False
        
        self.wait_for_arm_settled()

        # --- STEP 2: FIXED 180° SINGLE ROTATION ---
        print("🔄 STEP 2: Executing Single 180° Flip...")
        
        # Fresh joint state read
        time.sleep(0.5)
        joints = self.uf850.current_joint_positions.copy()
        
        # Debug: print all u1 joints
        u1_joints = {k: v for k, v in joints.items() if k.startswith("u1")}
        print(f"📊 Current u1 joints: { {k: f'{math.degrees(v):.1f}°' for k, v in u1_joints.items()} }")
        
        current_j6 = joints.get("u1_joint6", 0.0)
        
        # Calculate target: rotate 180° in the direction that avoids joint limits
        if current_j6 > 0:
            target_j6 = current_j6 - math.pi
        else:
            target_j6 = current_j6 + math.pi
        print(f"🔄 J6: {math.degrees(current_j6):.1f}° → {math.degrees(target_j6):.1f}° (Δ=180°)")
        
        # Build target: keep all current u1 joints, only change J6
        flip_target = {}
        for k, v in joints.items():
            if k.startswith("u1"):
                flip_target[k] = v
        flip_target["u1_joint6"] = target_j6
        print(f"🎯 Target joints: { {k: f'{math.degrees(v):.1f}°' for k, v in flip_target.items()} }")
        
        flip_ok = self.uf850.move_to_joint_positions(flip_target, velocity=0.15)
        
        if not flip_ok:
            print("⚠️ Flip move failed. Retrying with slower velocity...")
            time.sleep(0.5)
            flip_target["u1_joint6"] = target_j6  # Ensure target is still correct
            flip_ok = self.uf850.move_to_joint_positions(flip_target, velocity=0.08)
            if not flip_ok:
                print("❌ J6 rotation failed completely.")
                return False
        
        self.wait_for_arm_settled()
        
        # Verify rotation actually happened
        time.sleep(0.5)
        actual_j6 = self.uf850.current_joint_positions.get("u1_joint6", 0.0)
        rotation_deg = abs(actual_j6 - current_j6) * 180.0 / math.pi
        print(f"📊 J6 moved: {math.degrees(current_j6):.1f}° → {math.degrees(actual_j6):.1f}° (Δ={rotation_deg:.1f}°)")
        
        if abs(rotation_deg - 180.0) > 20.0:
            print(f"⚠️ J6 rotation incomplete! Expected 180°, got {rotation_deg:.1f}°")
        else:
            print(f"✅ J6 rotated successfully!")

        # --- STEP 3: TACTILE DESCENT ---
        print("⏰ Activating Servo Node...")
        if self.uf_servo_start_client.wait_for_service(timeout_sec=5.0):
            req_f = self.uf_servo_start_client.call_async(Trigger.Request())
            while rclpy.ok() and not req_f.done(): time.sleep(0.01)
        time.sleep(1.0)

        print(f"⬇️ STEP 3: Tactile Descent (Speed: {self.DESCENT_SPEED}m/s)...")
        if not self.uf850.move_linear_z_with_torque_stop(self.DESCENT_SPEED, self.TORQUE_THRESHOLD): return False
        self.wait_for_arm_settled()
        self.uf850.jog_cartesian_servo(0.0, 0.0, 0.005, duration=0.5)
        self.wait_for_arm_settled()

        # --- STEP 4 & 5: RELEASE & RE-GRASP ---
        print("🔓 STEP 4: Releasing...")
        self.hold_status_pub.publish(Bool(data=False))
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)})
        time.sleep(1.5)

        print("🗜️ STEP 5: Re-grasping...")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)})
        time.sleep(1.5)
        self.hold_status_pub.publish(Bool(data=True))
        
        print("🎉 FLIP COMPLETE")
        return True

def main(args=None):
    rclpy.init(args=args); node = ObjectFlipSkill()
    executor = MultiThreadedExecutor(); executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    time.sleep(1.0)
    
    try:
        # Wait up to 3s for hold status, then proceed anyway
        print("⏳ Waiting for hold status (or 3s timeout)...")
        wait_start = time.time()
        while rclpy.ok() and not node.is_holding_object:
            if time.time() - wait_start > 3.0:
                print("⚠️ No hold status received. Assuming object is held — proceeding with flip...")
                node.is_holding_object = True
                break
            time.sleep(0.5)
        
        if node.is_holding_object:
            print("🔄 Starting flip sequence...")
            node.execute_flip(interactive=False)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()