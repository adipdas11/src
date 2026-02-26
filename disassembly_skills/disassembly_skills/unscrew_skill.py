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
        
        # --- ROS 2 Interfaces ---
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.bin_sub = self.create_subscription(String, '/vision/bin_coordinates', self.bin_callback, 10)
        self.state_pub = self.create_publisher(String, '/robot_state/tool_arm/update', 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        
        self.data_lock = threading.Lock()
        self.latest_targets = []
        self.local_view = {}
        self.cached_bin1_xyz = None  
        
        self.get_logger().info("🚀 Full Unscrew Skill: Align, Wiggle, Extraction & Drop-Off Active.")

    def vision_callback(self, msg):
        try:
            raw_data = msg.data.strip().strip("'").strip('"')
            data = json.loads(raw_data)
            with self.data_lock:
                self.latest_targets = [obj for obj in data.get("global_view", {}).get("objects", []) 
                                     if "screw" in obj.get("label", "").lower()]
                self.local_view = data  
        except: pass

    def bin_callback(self, msg):
        try:
            raw_data = msg.data.strip().strip("'").strip('"')
            data = json.loads(raw_data)
            if "bin_1" in data and "xyz" in data["bin_1"]:
                with self.data_lock:
                    self.cached_bin1_xyz = data["bin_1"]["xyz"]
        except: pass

    # =========================================================================
    # 1. STAIRCASE: CROSSHAIR SEQUENTIAL ALIGN + FORCE DESCENT + WIGGLE CHECK
    # =========================================================================
    def perform_staircase_descent(self):
        print("\n🔍 [STAIRCASE] Starting Sequential Crosshair Alignment & Descent...")
        
        # --- ⚙️ SPEED & VELOCITY CONTROLS ---
        XY_SPEED_GAIN = 7.0        
        Z_DESCENT_SPEED = 0.01   
        MAX_XY_STEP = 0.015        
        
        # --- 🌀 AGGRESSIVE FAST SPIRAL SEARCH CONTROLS ---
        SPIRAL_GAP_MM = 15.0         
        SPIRAL_ANGULAR_STEP = 0.5    
        SPIRAL_VELOCITY_GAIN = 6.0   
        MAX_SPIRAL_STEP = 0.060      
        
        # --- 🎯 SEQUENTIAL ALIGNMENT CONTROL ---
        Y_TOLERANCE_PX = 10.0        
        
        SWAP_AXES = True             
        X_SENSE = -1.0      
        Y_SENSE = -1.0      
        
        # --- 🛑 STRICT FAIL CONDITIONS ---
        FORCE_SPIKE_THRESHOLD = 3.0  
        ALIGN_TOLERANCE_PX = 15.0    
        RETRACT_DIST = 0.005         
        
        # --- 🔄 WIGGLE TEST CONTROLS ---
        WIGGLE_MOVEMENT_MM = 0.015   
        WIGGLE_SPIKE_VALUE = 1.0     
        
        with self.data_lock:
            force_data = self.local_view.get('force_torque', {}).get('force', {})
            base_fx = force_data.get('x', 0.0)
            base_fy = force_data.get('y', 0.0)
            base_fz = force_data.get('z', 0.0)
            
        print(f"⚖️ Baselines -> Fx: {base_fx:.2f}N | Fy: {base_fy:.2f}N | Fz: {base_fz:.2f}N")
        
        spiral_idx = 0
        while rclpy.ok():
            with self.data_lock:
                local = copy.deepcopy(self.local_view)
            
            current_force = local.get('force_torque', {}).get('force', {})
            diff_fz = abs(current_force.get('z', 0.0) - base_fz)
            
            local_cam_view = local.get('local_view', {})
            screw = local_cam_view.get('screw_heads', [])
            crosshair = local_cam_view.get('crosshair', [])

            if not crosshair: crosshair = [320, 240]

            dist_px = 999.0
            err_x = 999.0
            err_y = 999.0
            
            if screw:
                err_x = crosshair[0] - screw[0].get('center', [320, 240])[0]
                err_y = crosshair[1] - screw[0].get('center', [320, 240])[1]
                dist_px = math.hypot(err_x, err_y)

            # --- Z-CONTACT & WIGGLE CHECK ---
            if diff_fz > FORCE_SPIKE_THRESHOLD:
                print(f"🎯 [CONTACT] Z-Force Spike Detected: {diff_fz:.2f}N.")
                
                if dist_px > ALIGN_TOLERANCE_PX:
                    print(f"❌ [MISALIGNED] Hit surface but error is {dist_px:.1f}px. Retracting 5mm.")
                    self.moveit_backend.jog_cartesian_servo(0.0, 0.0, RETRACT_DIST, duration=0.5)
                    return False
                
                print("🔩 [SEATING] Rotating screwdriver briefly to seat the bit...")
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.5) 
                
                cmd_msg = Int8()
                cmd_msg.data = -1
                self.tool_pub.publish(cmd_msg)
                time.sleep(0.3)  
                
                cmd_msg.data = 0
                self.tool_pub.publish(cmd_msg)
                time.sleep(0.5)  
                
                print("🔄 [WIGGLE] XY Aligned and Bit Seated. Performing 4-Direction Wiggle Test...")
                with self.data_lock:
                    wiggle_base_fx = self.local_view.get('force_torque', {}).get('force', {}).get('x', 0.0)
                    wiggle_base_fy = self.local_view.get('force_torque', {}).get('force', {}).get('y', 0.0)
                    
                print(f"⚖️ Wiggle Baselines -> Fx: {wiggle_base_fx:.2f}N | Fy: {wiggle_base_fy:.2f}N")
                
                wiggle_directions = [
                    (WIGGLE_MOVEMENT_MM, 0.0, 'x', "+X"),
                    (-WIGGLE_MOVEMENT_MM, 0.0, 'x', "-X"),
                    (0.0, WIGGLE_MOVEMENT_MM, 'y', "+Y"),
                    (0.0, -WIGGLE_MOVEMENT_MM, 'y', "-Y")
                ]
                
                successful_wiggles = 0
                
                for w_dx, w_dy, axis, name in wiggle_directions:
                    self.moveit_backend.jog_cartesian_servo(w_dx, w_dy, 0.0, duration=0.5)
                    time.sleep(0.5) 
                    
                    with self.data_lock:
                        curr_fx = self.local_view.get('force_torque', {}).get('force', {}).get('x', 0.0)
                        curr_fy = self.local_view.get('force_torque', {}).get('force', {}).get('y', 0.0)
                        
                    if axis == 'x':
                        spike = abs(curr_fx - wiggle_base_fx)
                    else:
                        spike = abs(curr_fy - wiggle_base_fy)
                        
                    if spike > WIGGLE_SPIKE_VALUE:
                        print(f"  ✅ Wiggle {name} SUCCESS | Spike: {spike:.2f}N (Required: {WIGGLE_SPIKE_VALUE}N)")
                        successful_wiggles += 1
                    else:
                        print(f"  ❌ Wiggle {name} FAIL | Spike: {spike:.2f}N (Required: {WIGGLE_SPIKE_VALUE}N)")
                        
                    self.moveit_backend.jog_cartesian_servo(-w_dx, -w_dy, 0.0, duration=0.5)
                    time.sleep(0.5)

                if successful_wiggles < 3:
                    print(f"❌ [WIGGLE FAIL] Only {successful_wiggles}/4 wiggles succeeded. Retracting 5mm.")
                    self.moveit_backend.jog_cartesian_servo(0.0, 0.0, RETRACT_DIST, duration=0.5)
                    return False
                
                print("✅ [WIGGLE PASS] Bit is fully seated and locked into screw head.")
                return True

            # --- VISION SERVOING & SPIRAL SEARCH ---
            if not screw:
                spiral_idx += 1
                
                theta = spiral_idx * SPIRAL_ANGULAR_STEP
                r = theta * ((SPIRAL_GAP_MM / 1000.0) / (2 * math.pi))
                
                prev_theta = (spiral_idx - 1) * SPIRAL_ANGULAR_STEP
                prev_r = prev_theta * ((SPIRAL_GAP_MM / 1000.0) / (2 * math.pi))
                
                delta_x = (r * math.cos(theta)) - (prev_r * math.cos(prev_theta))
                delta_y = (r * math.sin(theta)) - (prev_r * math.sin(prev_theta))
                
                dx = delta_x * SPIRAL_VELOCITY_GAIN
                dy = delta_y * SPIRAL_VELOCITY_GAIN
                
                dx = max(min(dx, MAX_SPIRAL_STEP), -MAX_SPIRAL_STEP)
                dy = max(min(dy, MAX_SPIRAL_STEP), -MAX_SPIRAL_STEP)
                
                print(f"⚠️ Vision lost. Spiral search Step {spiral_idx} (Gap: {SPIRAL_GAP_MM}mm)")
                self.moveit_backend.jog_cartesian_servo(dx, dy, 0.0, duration=0.1)
                time.sleep(0.3)
                continue

            spiral_idx = 0 
            
            if SWAP_AXES:
                raw_joy_x = (err_y * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_x * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            else:
                raw_joy_x = (err_x * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_y * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            
            align_state = ""
            if abs(err_y) > Y_TOLERANCE_PX:
                align_state = "[Align Y]"
                if SWAP_AXES:
                    joy_x = max(min(raw_joy_x, MAX_XY_STEP), -MAX_XY_STEP)
                    joy_y = 0.0 
                else:
                    joy_x = 0.0
                    joy_y = max(min(raw_joy_y, MAX_XY_STEP), -MAX_XY_STEP)
            else:
                align_state = "[Align X]"
                joy_x = max(min(raw_joy_x, MAX_XY_STEP), -MAX_XY_STEP)
                joy_y = max(min(raw_joy_y, MAX_XY_STEP), -MAX_XY_STEP)
            
            print(f"📉 {align_state} | Fz: {diff_fz:.2f}N | ErrX: {err_x:>5.1f} | ErrY: {err_y:>5.1f} | Total: {dist_px:.1f}px")
            
            self.moveit_backend.jog_cartesian_servo(joy_x, joy_y, -Z_DESCENT_SPEED, duration=0.2)
            time.sleep(0.3) 

        return False

    # =========================================================================
    # 2. EXTRACTION: FORCE-COMPLIANT UNSCREWING & SLOW RETRACT
    # =========================================================================
    def perform_compliant_extraction(self):
        print("\n🔄 [EXTRACTION] Starting Force-Compliant Unscrewing & Grab...")
        
        K_P_COMPLIANCE = 0.001       
        MAX_RETRACT_STEP = 0.005     
        EXTRACTION_TIME = 4.0        
        FORCE_DEADBAND = 0.5         
        
        # --- 🐢 CONTINUOUS SLOW RETRACT CONTROLS ---
        SLOW_RETRACT_TIME = 4.0      # Duration to continuously retract
        Z_RETRACT_SPEED = 0.04      # Speed to move up (Matches descent speed)
        
        with self.data_lock:
            baseline_fz = self.local_view.get('force_torque', {}).get('force', {}).get('z', 0.0)
            
        print(f"⚖️ Extraction Baseline Fz: {baseline_fz:.2f}N")
        
        cmd_msg = Int8()
        cmd_msg.data = -1 # Unscrew motor
        self.tool_pub.publish(cmd_msg)
        
        time.sleep(0.1) 
        cmd_msg.data = 2 # Grab with 2-finger onrobot rg6 gripper
        self.tool_pub.publish(cmd_msg)
        
        # --- The Compliance Loop (First 4 Seconds) ---
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < EXTRACTION_TIME:
            with self.data_lock:
                current_fz = self.local_view.get('force_torque', {}).get('force', {}).get('z', 0.0)
                
            force_error = current_fz - baseline_fz 
            z_step = force_error * K_P_COMPLIANCE
            z_step = max(min(z_step, MAX_RETRACT_STEP), -MAX_RETRACT_STEP)
            
            if abs(force_error) > FORCE_DEADBAND:
                print(f"🔩 Unscrewing... Force diff: {force_error:+.2f}N | Retracting Z: {z_step*1000:+.2f}mm")
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, z_step, duration=0.1)
            else:
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.1)
            
            time.sleep(0.1)
            
        # --- NEW: Smooth Continuous Slow Retract (Next 4 Seconds) ---
        print(f"🐢 [SLOW RETRACT] Gently pulling screw out smoothly over {SLOW_RETRACT_TIME}s...")
        retract_start = time.time()
        
        while rclpy.ok() and (time.time() - retract_start) < SLOW_RETRACT_TIME:
            # Continually command an upward Z movement while motor is still spinning
            self.moveit_backend.jog_cartesian_servo(0.0, 0.0, Z_RETRACT_SPEED, duration=0.2)
            time.sleep(0.25)
            
        print("✅ [EXTRACTION] Thread cleared. Stopping motor...")
        cmd_msg.data = 0
        self.tool_pub.publish(cmd_msg)
        time.sleep(0.5)
        
        print("⬆️ [RETRACT] Lifting 50mm to clear the workspace...")
        self.moveit_backend.retract_relative_z(0.050) 
        
        return True

    # =========================================================================
    # EXECUTION SEQUENCE (WITH DROP-OFF)
    # =========================================================================
    def execute_unscrew_sequence(self, target_id):
        print(f"\n🛠️ [START] Sequence on ID: {target_id}")
        time.sleep(0.1)
        
        with self.data_lock:
            bin1_raw = copy.deepcopy(self.cached_bin1_xyz)
            
        if not bin1_raw:
            print("⚠️ WARNING: Bin 1 location not yet received from /vision/bin_coordinates! Aborting sequence.")
            return False
            
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
                
                input("👉 GATE 3: Press [ENTER] to execute Force-Compliant Extraction...")
                if self.perform_compliant_extraction():
                    print("🎉 [SUCCESS] Screw successfully removed!")
                    
                    print("🗑️ [DROP-OFF] Navigating to Bin 1...")
                    bin_pose = Pose()
                    bin_pose.position.x, bin_pose.position.y, bin_pose.position.z = bin1_raw
                    world_bin_pose = self.moveit_backend.get_transformed_pose(bin_pose, self.CAMERA_FRAME, self.WORLD_FRAME)
                    
                    if world_bin_pose:
                        bx = world_bin_pose.pose.position.x
                        by = world_bin_pose.pose.position.y
                        # DROP-OFF HEIGHT: 30mm safe hover offset
                        bz = world_bin_pose.pose.position.z + self.TOOL_LENGTH + 0.030 
                        
                        if self.moveit_backend.move_to_pose_robust(bx, by, bz):
                            print("⏬ [RELEASE] Dropping screw...")
                            drop_cmd = Int8()
                            drop_cmd.data = 3
                            self.tool_pub.publish(drop_cmd)
                            time.sleep(1.0)
                            
                            drop_cmd.data = 0
                            self.tool_pub.publish(drop_cmd)
                            print("♻️ [RESET] Ready for next target.")
                    
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