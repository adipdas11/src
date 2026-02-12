#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import json, time, threading
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
START_X = 0.9345
START_Y = 0.45712
START_Z = 1.5184

FORCE_DROP_TRIGGER = 1.0  # N
MAX_DESCENT_DISTANCE = 0.15 
DESCENT_VELOCITY = 0.01

class SmoothForceGuard(Node):
    def __init__(self):
        super().__init__('smooth_force_guard')
        self.xarm = MotionBackend(self, "xarm_arm")
        self.subscription = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        self.current_force_z = 0.0
        self.calculated_threshold = None
        self.data_lock = threading.Lock()
        
        self.get_logger().info("✅ Node Active.")

    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            z_val = data.get("force_torque", {}).get("force", {}).get("z")
            if z_val is not None:
                with self.data_lock: self.current_force_z = float(z_val)
        except: pass

    def calibrate_threshold(self):
        print("⏳ Calibrating...", end="", flush=True)
        time.sleep(1.0) 
        with self.data_lock: baseline = self.current_force_z
        self.calculated_threshold = baseline - FORCE_DROP_TRIGGER
        print(f" Done. Baseline: {baseline:.2f} | Limit: {self.calculated_threshold:.2f}")

    def check_force_threshold(self):
        if self.calculated_threshold is None: return False
        f_z = 0.0
        with self.data_lock: f_z = self.current_force_z
        
        # --- COMMENTED OUT TO PREVENT FLICKERING WITH BACKEND PRINT ---
        # print(f"   ⚖️ Force: {f_z:.2f} N  |  Limit: {self.calculated_threshold:.2f} N", end='\r')
        
        if f_z < self.calculated_threshold: 
            # We can print ONLY when triggered
            print(f"\n🛑 TRIGGER! Force {f_z:.2f} dropped below {self.calculated_threshold:.2f}")
            return True 
        return False

    def run_sequence(self):
        print("🔓 Resetting Robot...")
        self.xarm.reset_robot()
        time.sleep(1.0)

        if not self.xarm._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ Action server missing!"); return

        print(f"\n🚀 PHASE 1: Move to Start ({START_X}, {START_Y}, {START_Z})")
        success = self.xarm.move_to_pose_robust(START_X, START_Y, START_Z, {}, "xarm5_link5", "world_world", 0.1)
        if not success: print("❌ Failed to reach Start."); return

        print(f"\n📏 PHASE 1.5: Calibration")
        self.calibrate_threshold()

        print(f"\n🚀 PHASE 2: Linear Descent (Debug Mode)")
        input("👉 Press ENTER to descend...")
        
        try:
            success = self.xarm.move_linear_z_with_force_stop(MAX_DESCENT_DISTANCE, DESCENT_VELOCITY, self.check_force_threshold)
            if success: print("\n✅ Stopped Safely.")
            else: print("\n❌ Motion Failed.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.xarm.stop_immediately()

def main(args=None):
    rclpy.init(args=args); node = SmoothForceGuard()
    executor = MultiThreadedExecutor(); executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True); thread.start()
    try: node.run_sequence()
    except KeyboardInterrupt: print("\n⚠️ Interrupt.")
    finally:
        if hasattr(node, 'xarm'): node.xarm.stop_immediately()
        node.destroy_node(); time.sleep(0.5); rclpy.shutdown()

if __name__ == '__main__': main()