#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Pose
import json, time, threading, copy, math
from disassembly_skills.motion_backend import MotionBackend

class UnscrewSkill(Node):
    def __init__(self):
        super().__init__('unscrew_skill_node')
        
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        
        # --- Physical Parameters ---
        self.TOOL_LENGTH = 0.240       
        self.HOVER_GAP = 0.015         
        self.TRANSIT_LIFT = 0.050      
        self.REACH_LIMIT = 0.680       
        self.MM_PER_PIX = 0.000130     
        
        # --- TF Frame Configuration ---
        self.WORLD_FRAME = 'world_world'      
        self.XARM_BASE_FRAME = 'xarm5_base_link'  
        self.CAMERA_FRAME = 'camera_color_optical_frame'
        
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.state_pub = self.create_publisher(String, '/robot_state/tool_arm/update', 10)
        
        self.data_lock = threading.Lock()
        self.latest_targets = []
        self.local_view = {}
        
        self.get_logger().info("🚀 Full Unscrew Skill: Velocity-Controlled Staircase Mode Active.")

    def vision_callback(self, msg):
        try:
            raw_data = msg.data.strip().strip("'").strip('"')
            data = json.loads(raw_data)
            with self.data_lock:
                self.latest_targets = [obj for obj in data.get("global_view", {}).get("objects", []) 
                                     if "screw" in obj.get("label", "").lower()]
                self.local_view = data  
        except: pass

    # =========================================================================
    # STAIRCASE: CONTINUOUS XY ALIGN + FORCE DESCENT
    # =========================================================================
    def perform_staircase_descent(self):
        """Velocity-controlled descent with Singularity Runaway protection."""
        print("\n🔍 [STAIRCASE] Starting Force-Controlled Descent...")
        
        # --- ⚙️ SPEED & VELOCITY CONTROLS ---
        XY_SPEED_GAIN = 0.8        # Multiplier (0.1 to 1.0). Lowers aggressive jerks.
        Z_DESCENT_SPEED = 0.0015   # 1.5mm absolute descent per loop
        MAX_XY_STEP = 0.004        # LIMITER: Never move more than 4mm in XY per loop
        
        # --- 🦾 CAMERA-TO-ROBOT MAPPING ---
        SWAP_AXES = False  
        X_SENSE = 1.0      
        Y_SENSE = -1.0      
        
        FORCE_SPIKE_THRESHOLD = 1.0  
        
        with self.data_lock:
            baseline_force_z = self.local_view.get('force_torque', {}).get('force', {}).get('z', 0.0)
        print(f"⚖️ Baseline Force Z: {baseline_force_z:.2f} N")
        
        spiral_idx = 0
        while rclpy.ok():
            with self.data_lock:
                local = copy.deepcopy(self.local_view)
            
            # Force Check
            current_force_z = local.get('force_torque', {}).get('force', {}).get('z', 0.0)
            force_diff = abs(current_force_z - baseline_force_z)
            
            if force_diff > FORCE_SPIKE_THRESHOLD:
                print(f"🎯 [STOP] Force Spike Detected: {force_diff:.2f}N. Bit is seated.")
                return True

            local_cam_view = local.get('local_view', {})
            tool = local_cam_view.get('tool_tips', [])
            screw = local_cam_view.get('screw_heads', [])

            if not tool or not screw:
                spiral_idx += 1
                # Slightly slower search to prevent servo jerks
                dx = (spiral_idx * 0.001) * math.cos(spiral_idx * 0.5)
                dy = (spiral_idx * 0.001) * math.sin(spiral_idx * 0.5)
                
                # Apply Max Step Clamp to Search
                dx = max(min(dx, MAX_XY_STEP), -MAX_XY_STEP)
                dy = max(min(dy, MAX_XY_STEP), -MAX_XY_STEP)
                
                print(f"⚠️ Vision lost. Searching... Step {spiral_idx}")
                self.moveit_backend.jog_cartesian_servo(dx, dy, 0.0, duration=0.1)
                time.sleep(0.3)
                continue

            spiral_idx = 0 
            target_px = tool[0].get('contact_point', [320, 240])
            current_px = screw[0].get('center', [320, 240])
            
            err_x = target_px[0] - current_px[0]
            err_y = target_px[1] - current_px[1]
            dist_px = math.hypot(err_x, err_y)
            
            # --- CALCULATE RAW VELOCITY ---
            if SWAP_AXES:
                raw_joy_x = (err_y * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_x * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            else:
                raw_joy_x = (err_x * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_y * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            
            # --- 🛑 THE FIX: CLAMP TO PREVENT SINGULARITY AVOIDANCE RUNAWAY ---
            joy_x = max(min(raw_joy_x, MAX_XY_STEP), -MAX_XY_STEP)
            joy_y = max(min(raw_joy_y, MAX_XY_STEP), -MAX_XY_STEP)
            
            print(f"📉 Descend [{Z_DESCENT_SPEED:.4f}m] | Force: {force_diff:.2f}N | Err: {dist_px:.1f}px")
            self.moveit_backend.jog_cartesian_servo(joy_x, joy_y, -Z_DESCENT_SPEED, duration=0.2)
            time.sleep(0.3) 

        return False

    # =========================================================================
    # EXECUTION SEQUENCE
    # =========================================================================
    def execute_unscrew_sequence(self, target_id):
        print(f"\n🛠️ [START] Sequence on ID: {target_id}")
        time.sleep(0.1)
        
        with self.data_lock:
            target_data = next((t for t in self.latest_targets if t['id'] == target_id), None)
        if not target_data: return False

        raw_pose = Pose()
        raw_pose.position.x, raw_pose.position.y, raw_pose.position.z = target_data['xyz']

        world_pose = self.moveit_backend.get_transformed_pose(raw_pose, self.CAMERA_FRAME, self.WORLD_FRAME)
        base_pose = self.moveit_backend.get_transformed_pose(raw_pose, self.CAMERA_FRAME, self.XARM_BASE_FRAME)
        
        if world_pose is None or base_pose is None: return False

        tx_world, ty_world = world_pose.pose.position.x, world_pose.pose.position.y
        dist_base = math.hypot(base_pose.pose.position.x, base_pose.pose.position.y)
        tz_flange_hover = world_pose.pose.position.z + self.TOOL_LENGTH + self.HOVER_GAP

        print(f"📍 World Target: X:{tx_world:.3f}, Y:{ty_world:.3f}")
        print(f"🦾 Reach Dist: {dist_base:.3f}m | Flange Z: {tz_flange_hover:.3f}m")

        if dist_base > self.REACH_LIMIT:
            print(f"❌ ABORT: Reach {dist_base:.3f}m exceeds limit.")
            return False

        input("👉 GATE 1: Press [ENTER] to Approach Hover...")
        self.moveit_backend.retract_relative_z(self.TRANSIT_LIFT)
        
        if self.moveit_backend.move_to_pose_robust(tx_world, ty_world, tz_flange_hover):
            print("✅ [STATUS] Hover Complete.")
            
            input("👉 GATE 2: Press [ENTER] to start Force-Controlled Staircase Descent...")
            if self.perform_staircase_descent():
                print("🏁 [FINISH] Bit is seated. Ready for unscrewing.")
                input("👉 Sequence Finished. Press [ENTER] to return to monitoring...")
                return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = UnscrewSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        node.moveit_backend.reset_robot()
        time.sleep(2.0)
        while rclpy.ok():
            target_id = None
            with node.data_lock:
                if node.latest_targets: target_id = node.latest_targets[0]['id']
            if target_id is not None:
                node.execute_unscrew_sequence(target_id)
                with node.data_lock: node.latest_targets = []
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()