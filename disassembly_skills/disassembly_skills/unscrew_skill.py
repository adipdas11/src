#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int8
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose
import json, time, threading, copy, math
from disassembly_skills.motion_backend import MotionBackend

class UnscrewSkill(Node):
    def __init__(self):
        super().__init__('unscrew_skill_node')
        
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        
        # --- Physical Parameters ---
        self.TOOL_LENGTH = 0.240       
        self.HOVER_GAP = 0.007         
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
        self.xarm_servo_start_client = self.create_client(Trigger, '/xarm_servo_node/start_servo')

        self.data_lock = threading.Lock()
        self.latest_targets = []
        self.local_view = {}
        self.cached_bin1_xyz = None  
        
        self.get_logger().info("🚀 Full Unscrew Skill: Align, Wiggle, Dynamic Velocity Extraction & Drop-Off Active.")

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
            curr_positions = self.moveit_backend.current_joint_positions.copy()
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

    # =========================================================================
    # 1. STAIRCASE: CROSSHAIR SEQUENTIAL ALIGN + FORCE DESCENT + WIGGLE CHECK
    # =========================================================================
    def perform_staircase_descent(self):
        print("\n🔍 [STAIRCASE] Starting Sequential Crosshair Alignment & Descent...")
        
        # --- ⚙️ SPEED & VELOCITY CONTROLS ---
        XY_SPEED_GAIN = 2.5          # Halved from 5.0 to reduce overshoot
        Z_MAX_SPEED = 0.005          # 5mm/s → 1.5mm/s actual
        MAX_XY_STEP = 0.006          # 6mm/s max → 1.8mm/s actual (was 10mm/s)
        
        # --- 🌀 SPIRAL SEARCH CONTROLS (SMOOTH) ---
        SPIRAL_GAP_MM = 8.0          # Tighter spiral rings (was 15)
        SPIRAL_ANGULAR_STEP = 0.5    
        SPIRAL_VELOCITY_GAIN = 2.0   # Gentler amplification (was 6)
        MAX_SPIRAL_STEP = 0.012      # 12mm/s max speed (was 60mm/s)
        
        # --- 🎯 SEQUENTIAL ALIGNMENT CONTROL ---
        Y_TOLERANCE_PX = 10.0        
        
        SWAP_AXES = True             
        X_SENSE = -1.0      
        Y_SENSE = -1.0      
        
        # --- 🛑 STRICT FAIL CONDITIONS ---
        FORCE_SPIKE_THRESHOLD = 3.0  
        ALIGN_TOLERANCE_PX = 10.0    
        RETRACT_DIST = 0.005         
        # --- 🔄 WIGGLE TEST CONTROLS ---
        WIGGLE_VELOCITY = 0.020     # m/s — gentle wiggle (~3mm/s with 0.15 servo scale)
        WIGGLE_SPIKE_VALUE = 0.5
        
        with self.data_lock:
            force_data = self.local_view.get('force_torque', {}).get('force', {})
            base_fx = force_data.get('x', 0.0)
            base_fy = force_data.get('y', 0.0)
            base_fz = force_data.get('z', 0.0)
            
        print(f"⚖️ Baselines -> Fx: {base_fx:.2f}N | Fy: {base_fy:.2f}N | Fz: {base_fz:.2f}N")
        
        spiral_idx = 0
        spiral_start_time = None
        retry_count = 0
        MAX_RETRIES = 3
        
        # Smoothing state for visual servo controller
        prev_joy_x = 0.0
        prev_joy_y = 0.0
        
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
                
                is_misaligned = (dist_px > ALIGN_TOLERANCE_PX)
                
                if is_misaligned:
                    print(f"⚠️ [MISALIGNED] Error is {dist_px:.1f}px. Attempting wiggle test to force seating...")

                print("🔩 [SEATING] Rotating screwdriver briefly to seat the bit...")
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.5) 
                
                cmd_msg = Int8()
                cmd_msg.data = -1
                self.tool_pub.publish(cmd_msg)
                time.sleep(0.3)  
                
                cmd_msg.data = 0
                self.tool_pub.publish(cmd_msg)
                time.sleep(0.5)  
                
                print("🔄 [WIGGLE] Performing 4-Direction Wiggle Test...")
                with self.data_lock:
                    wiggle_base_fx = self.local_view.get('force_torque', {}).get('force', {}).get('x', 0.0)
                    wiggle_base_fy = self.local_view.get('force_torque', {}).get('force', {}).get('y', 0.0)
                    
                print(f"⚖️ Wiggle Baselines -> Fx: {wiggle_base_fx:.2f}N | Fy: {wiggle_base_fy:.2f}N")
                
                wiggle_directions = [
                    (WIGGLE_VELOCITY, 0.0, 'x', "+X"),
                    (-WIGGLE_VELOCITY, 0.0, 'x', "-X"),
                    (0.0, WIGGLE_VELOCITY, 'y', "+Y"),
                    (0.0, -WIGGLE_VELOCITY, 'y', "-Y")
                ]
                
                successful_wiggles = 0
                
                for w_dx, w_dy, axis, name in wiggle_directions:
                    # Ramp up smoothly over 0.3s, hold 0.2s, then ramp down
                    RAMP_STEPS = 6
                    RAMP_DT = 0.05
                    for step in range(1, RAMP_STEPS + 1):
                        frac = step / RAMP_STEPS
                        self.moveit_backend.jog_cartesian_servo(
                            w_dx * frac, w_dy * frac, 0.0, duration=RAMP_DT, stop_after=False)
                    # Hold at full speed briefly
                    self.moveit_backend.jog_cartesian_servo(w_dx, w_dy, 0.0, duration=0.2, stop_after=False)
                    # Stop and read force
                    self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.1)
                    time.sleep(0.1)

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

                    # Return to center with ramp
                    for step in range(1, RAMP_STEPS + 1):
                        frac = step / RAMP_STEPS
                        self.moveit_backend.jog_cartesian_servo(
                            -w_dx * frac, -w_dy * frac, 0.0, duration=RAMP_DT, stop_after=False)
                    self.moveit_backend.jog_cartesian_servo(-w_dx, -w_dy, 0.0, duration=0.2, stop_after=False)
                    self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.1)
                    time.sleep(0.1)

                # --- RETRY LOGIC FOR WIGGLE FAIL ---
                if successful_wiggles < 3:
                    retry_count += 1
                    if is_misaligned:
                        print(f"❌ [MISALIGNED & WIGGLE FAIL] Error was {dist_px:.1f}px. Retry {retry_count}/{MAX_RETRIES}. Retracting 10mm...")
                    else:
                        print(f"❌ [WIGGLE FAIL] Only {successful_wiggles}/4 wiggles succeeded. Retry {retry_count}/{MAX_RETRIES}. Retracting 10mm...")
                    
                    if retry_count > MAX_RETRIES:
                        print("❌ Max retries reached. Aborting target.")
                        return False
                    
                    # Servo closed-loop retract (10mm) — more reliable than planned retract
                    print("⬆️ Servo retracting 10mm...")
                    self.moveit_backend.retract_servo_z_closed_loop(0.005, speed_mps=0.02)
                    time.sleep(1.0)
                    
                    # 🔄 Reset force baselines after retract (stale baselines cause false triggers)
                    with self.data_lock:
                        force_data = self.local_view.get('force_torque', {}).get('force', {})
                        base_fz = force_data.get('z', 0.0)
                    print(f"⚖️ Baselines RESET -> Fz: {base_fz:.2f}N")
                    
                    # Reset smoothing state so XY alignment starts fresh
                    prev_joy_x = 0.0
                    prev_joy_y = 0.0
                    continue 
                
                print("✅ [WIGGLE PASS] Bit is fully seated and locked into screw head.")
                return True

            # --- VISION SERVOING & SPIRAL SEARCH ---
            if not screw:
                # 🛑 NEW FIX: 3-Second Timeout for Spiral Search
                if spiral_start_time is None:
                    spiral_start_time = time.time()
                elif (time.time() - spiral_start_time) > 15.0:
                    print("⚠️ [TIMEOUT] Spiral search exceeded 15 seconds without finding screw. Proceeding directly to Bin 1...")
                    return "HOLE_TIMEOUT"
                
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
                
                print(f"⚠️ Vision lost. Spiral search Step {spiral_idx} | dx:{dx*1000:.1f} dy:{dy*1000:.1f} mm/s")
                # stop_after=False keeps commands flowing — avoids start-stop jerk
                self.moveit_backend.jog_cartesian_servo(dx, dy, 0.0, duration=0.15, stop_after=False)
                continue

            # Reset spiral variables if we found the screw
            spiral_idx = 0 
            spiral_start_time = None 
            
            # --- 🎯 ROBUST VISUAL SERVO CONTROLLER ---
            # Deadband: ignore sub-pixel noise near the target
            DEADBAND_PX = 5.0
            
            if SWAP_AXES:
                raw_joy_x = (err_y * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_x * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            else:
                raw_joy_x = (err_x * self.MM_PER_PIX) * X_SENSE * XY_SPEED_GAIN
                raw_joy_y = (err_y * self.MM_PER_PIX) * Y_SENSE * XY_SPEED_GAIN
            
            # Apply deadband — zero out command if error is small
            if abs(err_y if SWAP_AXES else err_x) < DEADBAND_PX:
                raw_joy_x = 0.0
            if abs(err_x if SWAP_AXES else err_y) < DEADBAND_PX:
                raw_joy_y = 0.0
            
            align_state = ""
            if abs(err_y) > Y_TOLERANCE_PX:
                align_state = "[Align Y]"
                if SWAP_AXES:
                    target_joy_x = max(min(raw_joy_x, MAX_XY_STEP), -MAX_XY_STEP)
                    target_joy_y = 0.0 
                else:
                    target_joy_x = 0.0
                    target_joy_y = max(min(raw_joy_y, MAX_XY_STEP), -MAX_XY_STEP)
            else:
                align_state = "[Align X]"
                target_joy_x = max(min(raw_joy_x, MAX_XY_STEP), -MAX_XY_STEP)
                target_joy_y = max(min(raw_joy_y, MAX_XY_STEP), -MAX_XY_STEP)
            
            # Exponential smoothing: prevents sudden velocity reversals
            # α=0.3 → new command is 30% target + 70% previous (heavy damping)
            SMOOTH_ALPHA = 0.3
            smooth_joy_x = SMOOTH_ALPHA * target_joy_x + (1.0 - SMOOTH_ALPHA) * prev_joy_x
            smooth_joy_y = SMOOTH_ALPHA * target_joy_y + (1.0 - SMOOTH_ALPHA) * prev_joy_y
            
            # Acceleration limiter: cap velocity change per cycle
            MAX_ACCEL = 0.002  # m/s per cycle
            smooth_joy_x = max(min(smooth_joy_x, prev_joy_x + MAX_ACCEL), prev_joy_x - MAX_ACCEL)
            smooth_joy_y = max(min(smooth_joy_y, prev_joy_y + MAX_ACCEL), prev_joy_y - MAX_ACCEL)
            
            prev_joy_x = smooth_joy_x
            prev_joy_y = smooth_joy_y
            
            # --- 🌪️ FUNNEL LOGIC FOR Z DESCENT ---
            # Pause Z completely when very misaligned to avoid dragging bit sideways
            if dist_px > 30.0:
                dynamic_z_speed = 0.0  # Pure XY correction first
            elif dist_px <= ALIGN_TOLERANCE_PX:
                dynamic_z_speed = Z_MAX_SPEED
            else:
                dynamic_z_speed = max(0.001, Z_MAX_SPEED * (ALIGN_TOLERANCE_PX / dist_px))
            
            print(f"📉 {align_state} | Fz: {diff_fz:.2f}N | ErrX: {err_x:>5.1f} | ErrY: {err_y:>5.1f} | Total: {dist_px:.1f}px | Vx:{smooth_joy_x*1000:.1f} Vy:{smooth_joy_y*1000:.1f} Vz:{dynamic_z_speed*1000:.1f} mm/s")
            
            self.moveit_backend.jog_cartesian_servo(smooth_joy_x, smooth_joy_y, -dynamic_z_speed, duration=0.15, stop_after=False)

        return False

    # =========================================================================
    # 2. EXTRACTION: DYNAMIC VELOCITY-SERVOING FORCE COMPLIANCE & SLOW RETRACT
    # =========================================================================
    def perform_compliant_extraction(self):
        print("\n🔄 [EXTRACTION] Starting Dynamic Velocity-Servoing Force Compliance & Grab...")
        
        # --- ⚙️ COMPLIANCE TUNING CONTROLS (VELOCITY-BASED) ---
        K_V_COMPLIANCE = 0.002       
        MAX_Z_SPEED = 0.0005          
        FORCE_DEADBAND = 0.3         
        
        # --- ⏱️ DYNAMIC EXTRACTION CONTROLS ---
        STABLE_DURATION = 3.0          # Time to wait with no force increase to declare success
        FORCE_FLUCTUATION_MARGIN = 1.0 # Allowable force bounce (N) before timer resets
        MAX_EXTRACTION_TIME = 20.0     # Absolute max time (safety net)
        
        with self.data_lock:
            baseline_fz = self.local_view.get('force_torque', {}).get('force', {}).get('z', 0.0)
            
        print(f"⚖️ Extraction Baseline Fz: {baseline_fz:.2f}N")
        
        cmd_msg = Int8()
        cmd_msg.data = -1 # Unscrew motor
        self.tool_pub.publish(cmd_msg)
        
        time.sleep(0.1) 
        cmd_msg.data = 2 # Grab with 2-finger onrobot rg6 gripper
        self.tool_pub.publish(cmd_msg)
        
        # Variables to track when the force plateaus
        peak_upward_force = 0.0
        last_increase_time = time.time()
        start_time = time.time()
        
        # --- The Dynamic Velocity-Servoing Compliance Loop ---
        while rclpy.ok() and (time.time() - start_time) < MAX_EXTRACTION_TIME:
            with self.data_lock:
                current_fz = self.local_view.get('force_torque', {}).get('force', {}).get('z', 0.0)
                
            raw_force_error = current_fz - baseline_fz 
            
            # Since upward pressure results in a negative value in the sensor, flip it to positive
            upward_force = -raw_force_error
            
            # --- DYNAMIC EXTRACTION CHECK ---
            if upward_force > (peak_upward_force + FORCE_FLUCTUATION_MARGIN):
                peak_upward_force = upward_force
                last_increase_time = time.time() # Reset the 3-second timer
                
            if (time.time() - last_increase_time) >= STABLE_DURATION:
                print(f"🎉 [FREE] Upward force stabilized for {STABLE_DURATION}s (Peak: {peak_upward_force:.2f}N).")
                break # Break out of the compliance loop early!
            
            # Calculate the compliant speed (0.0 prevents downward movement)
            z_speed = upward_force * K_V_COMPLIANCE
            z_speed = max(min(z_speed, MAX_Z_SPEED), 0.0)
            
            if upward_force > FORCE_DEADBAND:
                print(f"🔩 Yielding... Upward Force: {upward_force:+.2f}N | Z-Speed: {z_speed*1000:+.2f}mm/tick")
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, z_speed, duration=0.1)
            else:
                self.moveit_backend.jog_cartesian_servo(0.0, 0.0, 0.0, duration=0.1)
            
            time.sleep(0.1)
            
        # --- STOP UNSCREWING MOTOR ---
        print("✅ [EXTRACTION] Thread cleared. Stopping unscrew motor before retracting...")
        cmd_msg.data = 0
        self.tool_pub.publish(cmd_msg)
        time.sleep(0.5) 
            
        # --- NEW: CONTINUOUS SLOW SERVO RETRACT WITH WIGGLE (15 SECONDS) ---
        Z_SLOW_RETRACT_SPEED = 0.01  # Matches Z_DESCENT_SPEED for consistent slow motion
        SLOW_RETRACT_DURATION = 15.0 # Run for exactly 15 seconds
        
        # Wiggle parameters: Swirls 3mm/s in a circle at 1.5 Hz
        WIGGLE_XY_SPEED = 0.003      
        WIGGLE_FREQ = 1.5            
        
        print(f"🐢 [SLOW RETRACT + WIGGLE] Gently pulling and swirling screw out at {Z_SLOW_RETRACT_SPEED*1000}mm/s for {SLOW_RETRACT_DURATION}s...")
        
        retract_start = time.time()
        while rclpy.ok() and (time.time() - retract_start) < SLOW_RETRACT_DURATION:
            elapsed = time.time() - retract_start
            
            # Calculate smooth circular wiggle velocities using sine and cosine waves
            wx = WIGGLE_XY_SPEED * math.cos(2 * math.pi * WIGGLE_FREQ * elapsed)
            wy = WIGGLE_XY_SPEED * math.sin(2 * math.pi * WIGGLE_FREQ * elapsed)
            
            # Inject wx and wy alongside the continuous Z retract
            self.moveit_backend.jog_cartesian_servo(wx, wy, Z_SLOW_RETRACT_SPEED, duration=0.1)
            time.sleep(0.1)
        
        print("⬆️ [FAST RETRACT] Lifting 50mm to clear the workspace...")
        self.moveit_backend.retract_relative_z(0.050) 
        self.wait_for_arm_settled() # 🛑 ADDED POST-RETRACT SETTLE
        
        return True

    # =========================================================================
    # EXECUTION SEQUENCE (WITH DROP-OFF)
    # =========================================================================
    def execute_unscrew_command(self, target_id, target_label=None, interactive=True):
        print(f"\n🛠️ [START] Sequence on ID: {target_id} (Label: {target_label})")
        time.sleep(0.1)
        
        with self.data_lock:
            bin1_raw = copy.deepcopy(self.cached_bin1_xyz)
            
        if not bin1_raw:
            print("⚠️ WARNING: Bin 1 location not yet received from /vision/bin_coordinates! Aborting sequence.")
            return False
            
        with self.data_lock:
            target_data = next((t for t in self.latest_targets if t['id'] == target_id), None)
        if not target_data:
            print(f"❌ Target ID {target_id} not found in vision data.")
            return False
        if 'xyz' not in target_data:
            print(f"❌ Target ID {target_id} has no 'xyz' position data. Skipping.")
            return False

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

        if interactive: input("👉 GATE 1: Press [ENTER] to Approach Hover...")
        self.moveit_backend.retract_relative_z(self.TRANSIT_LIFT)
        self.wait_for_arm_settled()

        # Try Cartesian path first (incremental IK — works where single-shot fails on 5-DOF)
        hover_ok = self.moveit_backend.move_cartesian_to_pose(tx_world, ty_world, tz_flange_hover)
        if not hover_ok:
            print("⚠️ Cartesian path failed. Falling back to IK + yaw sweep...")
            hover_ok = self.moveit_backend.move_to_pose_robust(tx_world, ty_world, tz_flange_hover)

        if hover_ok:
            print("✅ [STATUS] Hover Complete.")

            self.wait_for_arm_settled()

            # Start xArm servo controller before staircase descent
            print("⏰ Activating xArm Servo Controller...")
            if self.xarm_servo_start_client.wait_for_service(timeout_sec=5.0):
                req_f = self.xarm_servo_start_client.call_async(Trigger.Request())
                while rclpy.ok() and not req_f.done(): time.sleep(0.01)
            else:
                print("⚠️ xArm servo start service not available. Proceeding anyway.")
            time.sleep(1.0)  # Allow controller transition to complete

            if interactive: input("👉 GATE 2: Press [ENTER] to start Force-Controlled Staircase Descent...")
            
            # 🛑 NEW: Capture the staircase result to check for the HOLE_TIMEOUT
            staircase_result = self.perform_staircase_descent()
            
            if staircase_result == False:
                # Wiggle test failed after max retries — retract and go to bin for next target
                print("⚠️ [STAIRCASE FAILED] Could not seat bit. Retracting and moving to bin...")
                self.moveit_backend.retract_relative_z(0.050)
                self.wait_for_arm_settled()
                self._navigate_to_bin(bin1_raw)
                return True  # Return True so caller continues to next screw
            
            # If vision timed out on a hole, skip the extraction and safely retract
            if staircase_result == "HOLE_TIMEOUT":
                print("⏭️ [SKIP EXTRACTION] Hole timeout reached. Retracting safely before moving to Bin 1...")
                self.moveit_backend.retract_relative_z(0.050)
                self.wait_for_arm_settled()
                self._navigate_to_bin(bin1_raw)
                return True
            
            # staircase_result == True → bit is seated
            print("🏁 [FINISH] Bit is seated. Ready for unscrewing.")
            if interactive: input("👉 GATE 3: Press [ENTER] to execute Dynamic Force-Compliant Extraction...")
            extraction_success = self.perform_compliant_extraction()
            
            if extraction_success:
                print("🎉 [SUCCESS] Screw successfully removed!")
                self._navigate_to_bin(bin1_raw)
                
            if interactive: input("👉 Sequence Finished. Press [ENTER] to return to monitoring...")
            return True
        return False
    
    def _navigate_to_bin(self, bin1_raw):
        """Navigate to bin 1 and release the screw."""
        print("🗑️ [DROP-OFF] Navigating to Bin 1...")
        bin_pose = Pose()
        bin_pose.position.x, bin_pose.position.y, bin_pose.position.z = bin1_raw
        world_bin_pose = self.moveit_backend.get_transformed_pose(bin_pose, self.CAMERA_FRAME, self.WORLD_FRAME)
        
        if world_bin_pose:
            bx = world_bin_pose.pose.position.x
            by = world_bin_pose.pose.position.y
            bz = world_bin_pose.pose.position.z + self.TOOL_LENGTH + 0.030 
            
            if self.moveit_backend.move_to_pose_robust(bx, by, bz):
                print("⏬ [WAITING] Allowing arm to settle over bin...")
                self.wait_for_arm_settled()
                
                print("⏬ [RELEASE] Dropping screw...")
                drop_cmd = Int8()
                drop_cmd.data = 3
                self.tool_pub.publish(drop_cmd)
                time.sleep(1.0)
                
                drop_cmd.data = 0
                self.tool_pub.publish(drop_cmd)
                print("♻️ [RESET] Ready for next target.")

def main(args=None):
    rclpy.init(args=args)
    node = UnscrewSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        time.sleep(2.0)
        while rclpy.ok():
            target_id = None
            with node.data_lock:
                if node.latest_targets: target_id = node.latest_targets[0]['id']
            if target_id is not None:
                # In standalone mode, we can test it interactively
                node.execute_unscrew_command(target_id, target_label="screw", interactive=False)
                with node.data_lock: node.latest_targets = []
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()