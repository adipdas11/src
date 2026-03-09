#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
import json, time, threading, math
from disassembly_skills.motion_backend import MotionBackend

class PickupSkill(Node):
    def __init__(self):
        super().__init__('pickup_skill_node')
        # Motion Backends
        self.uf850 = MotionBackend(self, "uf_arm")
        self.gripper = MotionBackend(self, "rg6_gripper")
        self.xarm5 = MotionBackend(self, "xarm_arm")
        
        # Service & Topic Interfaces
        self.uf_servo_start_client = self.create_client(Trigger, '/uf_servo_node/start_servo')
        self.vision_reset_pub = self.create_publisher(String, '/vision/reset_tracker', 10)
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        
        # Physical Parameters
        self.UF_TOOL_LENGTH = 0.26
        self.HOVER_HEIGHT = 0.05    
        self.DESCENT_SPEED = 0.1    
        self.RETRACT_DIST = 0.030   
        self.OPEN_DEG, self.CLOSE_DEG = 33.0, -35.0
        
        self.UF_HOME_JOINTS = {'u1_joint1': 0.0, 'u1_joint2': 0.0, 'u1_joint3': -1.57, 'u1_joint4': 0.0, 'u1_joint5': -1.57, 'u1_joint6': 0.0}
        self.DROP_POSE = {'x': 0.92, 'y': -0.36, 'z': 1.25}

        # Thread Safety & State
        self.data_lock = threading.Lock()
        self.latest_targets = []
        self.is_holding_object = False
        
        # Subscriptions
        self.create_subscription(Bool, '/object_hold_state/is_held', self.hold_status_callback, 10)
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        self.get_logger().info("🚀 PCB Pickup Skill: Safety-Conditioned Release Logic Active.")

    def hold_status_callback(self, msg): 
        self.is_holding_object = msg.data

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data.strip().strip("'").strip('"'))
            with self.data_lock: self.latest_targets = data.get("global_view", {}).get("objects", [])
        except: pass

    def wait_for_settled(self, backend):
        time.sleep(0.5)
        start_t = time.time(); last_pos = {}
        while rclpy.ok() and (time.time() - start_t) < 5.0:
            curr = backend.current_joint_positions.copy()
            if last_pos:
                if max([abs(curr[n] - last_pos[n]) for n in curr if n in last_pos], default=0) <= 0.006: return True
            last_pos = curr; time.sleep(0.1)
        return True

    def execute_pickup(self, target_id, target_label):
        print(f"\n🛠️ [START] {target_label} Sequence (ID: {target_id})")

        # --- STEP 0: PRE-CONDITION (Release -> 30cm Retract -> Home) ---
        if self.is_holding_object:
            print("📦 [PRE-CONDITION] Active Hold Detected. Releasing...")
            if not self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(self.OPEN_DEG)}): 
                return False
            time.sleep(1.0)
            self.hold_status_pub.publish(Bool(data=False))
            
            print("⬆️ Retracting 30cm to clear workspace...")
            # 🎯 MOVE TO HOME ONLY IF RETRACT SUCCEEDS
            if self.uf850.retract_relative_z(0.30):
                self.wait_for_settled(self.uf850)
                print("🏠 Retract Successful. Homing UF850...")
                if not self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS): 
                    return False
                self.wait_for_settled(self.uf850)
            else:
                print("❌ Retract failed. Aborting to prevent collision.")
                return False

        # --- STEP 1: CLEAR WORKSPACE ---
        print("🏠 Clearing xArm5 workspace...")
        if not self.xarm5.move_to_joint_positions({'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57, 'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0}):
            return False
        self.wait_for_settled(self.xarm5)

        # --- STEP 2: COORDINATE TRANSFORM ---
        with self.data_lock: target = next((t for t in self.latest_targets if t['id'] == target_id), None)
        if not target: return False
        
        raw_p = Pose()
        raw_p.position.x, raw_p.position.y, raw_p.position.z = target['xyz']
        world_p = self.uf850.get_transformed_pose(raw_p, 'camera_color_optical_frame', 'world_world')
        if not world_p: return False

        tx, ty = world_p.pose.position.x - 0.04, world_p.pose.position.y + 0.03
        final_z = world_p.pose.position.z + self.UF_TOOL_LENGTH
        hover_z = final_z + self.HOVER_HEIGHT

        # --- STEP 3: APPROACH & CLOSED-LOOP DESCENT ---
        print("🔓 Opening Gripper for Approach...")
        if not self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(self.OPEN_DEG)}): return False
        
        print(f"🚁 Hovering at {tx:.3f}, {ty:.3f}...")
        if not self.uf850.move_to_pose_robust(tx, ty, hover_z, velocity=0.1): return False
        self.wait_for_settled(self.uf850)

        print("⏰ Activating Servo & Closed-Loop Descent (0.1m/s)...")
        if self.uf_servo_start_client.wait_for_service(timeout_sec=5.0):
            req_f = self.uf_servo_start_client.call_async(Trigger.Request())
            while rclpy.ok() and not req_f.done(): time.sleep(0.01)
        else: return False
        time.sleep(1.0)  # Allow controller transition to complete

        if not self.uf850.retract_servo_z_closed_loop(-self.HOVER_HEIGHT, speed_mps=self.DESCENT_SPEED): return False
        self.wait_for_settled(self.uf850)

        # --- STEP 4: GRASP & RETRACT ---
        print("🗜️ Closing Gripper on PCB...")
        if not self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(self.CLOSE_DEG)}): return False
        time.sleep(1.5); self.hold_status_pub.publish(Bool(data=True))

        print(f"⬆️ Retracting {self.RETRACT_DIST*1000}mm...")
        if not self.uf850.retract_servo_z_closed_loop(self.RETRACT_DIST): return False
        self.wait_for_settled(self.uf850)

        # --- STEP 5: DROP-OFF ---
        print("🗑️ Moving to Drop Pose...")
        if not self.uf850.move_to_pose_robust(self.DROP_POSE['x'], self.DROP_POSE['y'], self.DROP_POSE['z'], velocity=0.1): return False
        self.wait_for_settled(self.uf850)

        print("🎉 Finalizing: Release & Home...")
        if not self.gripper.move_to_joint_positions({"rg6_l_out": math.radians(self.OPEN_DEG)}): return False
        time.sleep(1.0); self.hold_status_pub.publish(Bool(data=False))
        
        if not self.uf850.move_to_joint_positions(self.UF_HOME_JOINTS): return False
        
        print("✅ [SUCCESS] Sequence Complete.")
        return True

def main(args=None):
    rclpy.init(args=args); node = PickupSkill()
    executor = MultiThreadedExecutor(); executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    time.sleep(1.0); node.vision_reset_pub.publish(String(data='reset'))
    try:
        while rclpy.ok():
            tid = None
            with node.data_lock:
                pcbs = [t for t in node.latest_targets if 'pcb_main' in t.get('label', '').lower()]
                if pcbs: tid = pcbs[0]['id']
            if tid and node.execute_pickup(tid, "pcb_main"): break
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()