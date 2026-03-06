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
        self.HOVER_GAP = 0.008         
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
        XY_SPEED_GAIN = 10.0      
        Z_MAX_SPEED = 0.010       # 🛑 CHANGED: Max 10mm/s cap for funnel logic 
        MAX_XY_STEP = 0.025       
        
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
        ALIGN_TOLERANCE_PX = 10.0    
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
        spiral_start_time = None  # 🛑 NEW: Timer for spiral search
        retry_count = 0
        MAX_RETRIES = 3
        
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

                # --- RETRY LOGIC FOR WIGGLE FAIL (Handles both Misaligned and Aligned cases) ---
                if successful_wiggles < 3:
                    retry_count += 1
                    if is_misaligned:
                        print(f"❌ [MISALIGNED & WIGGLE FAIL] Error was {dist_px:.1f}px. Retry {retry_count}/{MAX_RETRIES}. Retracting 5mm...")
                    else:
                        print(f"❌ [WIGGLE FAIL] Only {successful_wiggles}/4 wiggles succeeded. Retry {retry_count}/{MAX_RETRIES}. Retracting 5mm...")
                        
                    self.moveit_backend.retract_relative_z(0.005) # Guaranteed 5mm lift
                    time.sleep(1.0) # Let forces completely settle before re-aligning
                    if retry_count > MAX_RETRIES:
                        print("❌ Max retries reached. Aborting target.")
                        return False
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
                
                print(f"⚠️ Vision lost. Spiral search Step {spiral_idx} (Gap: {SPIRAL_GAP_MM}mm)")
                self.moveit_backend.jog_cartesian_servo(dx, dy, 0.0, duration=0.1)
                time.sleep(0.3)
                continue

            # Reset spiral variables if we found the screw
            spiral_idx = 0 
            spiral_start_time = None 
            
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
            
            # --- 🌪️ FUNNEL LOGIC FOR Z DESCENT ---
            if dist_px <= ALIGN_TOLERANCE_PX:
                dynamic_z_speed = Z_MAX_SPEED
            else:
                # Scale speed proportionally down. Floor it at 2mm/s so it doesn't freeze.
                dynamic_z_speed = max(0.002, Z_MAX_SPEED * (ALIGN_TOLERANCE_PX / dist_px))
            
            print(f"📉 {align_state} | Fz: {diff_fz:.2f}N | ErrX: {err_x:>5.1f} | ErrY: {err_y:>5.1f} | Total: {dist_px:.1f}px | Z-Speed: {dynamic_z_speed*1000:.1f}mm/s")
            
            self.moveit_backend.jog_cartesian_servo(joy_x, joy_y, -dynamic_z_speed, duration=0.2)
            time.sleep(0.3) 

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

        if interactive: input("👉 GATE 1: Press [ENTER] to Approach Hover...")
        self.moveit_backend.retract_relative_z(self.TRANSIT_LIFT)
        
        if self.moveit_backend.move_to_pose_robust(tx_world, ty_world, tz_flange_hover):
            print("✅ [STATUS] Hover Complete.")
            
            # 🛑 NEW FIX: Wait for xArm to settle ONLY after arriving at the hover pose
            self.wait_for_arm_settled()
            
            if interactive: input("👉 GATE 2: Press [ENTER] to start Force-Controlled Staircase Descent...")
            
            # 🛑 NEW: Capture the staircase result to check for the HOLE_TIMEOUT
            staircase_result = self.perform_staircase_descent()
            
            if staircase_result == True or staircase_result == "HOLE_TIMEOUT":
                
                # If vision timed out on a hole, skip the extraction and safely retract
                if staircase_result == "HOLE_TIMEOUT":
                    print("⏭️ [SKIP EXTRACTION] Hole timeout reached. Retracting safely before moving to Bin 1...")
                    self.moveit_backend.retract_relative_z(0.050)
                    self.wait_for_arm_settled()
                    extraction_success = True  # Proceed to bin
                else:
                    print("🏁 [FINISH] Bit is seated. Ready for unscrewing.")
                    if interactive: input("👉 GATE 3: Press [ENTER] to execute Dynamic Force-Compliant Extraction...")
                    extraction_success = self.perform_compliant_extraction()
                
                # Proceed to Drop-Off
                if extraction_success:
                    if staircase_result == True:
                        print("🎉 [SUCCESS] Screw successfully removed!")
                    
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
                            self.wait_for_arm_settled() # 🛑 REPLACED TIME.SLEEP(2.0)
                            
                            print("⏬ [RELEASE] Dropping screw...")
                            drop_cmd = Int8()
                            drop_cmd.data = 3
                            self.tool_pub.publish(drop_cmd)
                            time.sleep(1.0)
                            
                            drop_cmd.data = 0
                            self.tool_pub.publish(drop_cmd)
                            print("♻️ [RESET] Ready for next target.")
                    
                if interactive: input("👉 Sequence Finished. Press [ENTER] to return to monitoring...")
                return True
        return False

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
                node.execute_unscrew_command(target_id, target_label="screw", interactive=True)
                with node.data_lock: node.latest_targets = []
            time.sleep(0.5)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()