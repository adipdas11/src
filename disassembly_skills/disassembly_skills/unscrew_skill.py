#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Pose
import tf2_ros
import json
import time
import threading
import sys
import copy
import math
import numpy as np
from xarm.wrapper import XArmAPI

# ==========================================
# DUAL-ARM SYSTEM CONFIG
# ==========================================
XARM_IP = '192.168.1.239'

class UnscrewSkill(Node):
    def __init__(self):
        super().__init__('unscrew_skill_node')
        
        # 1. Backends
        # Direct SDK connection for precision control
        self.get_logger().info(f"🔌 Connecting Direct SDK to xArm at {XARM_IP}...")
        self.arm = XArmAPI(XARM_IP, is_radian=False)
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.get_logger().info("✅ Direct SDK Connected & Armed.")

        # 2. Communication
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.bin_sub = self.create_subscription(String, '/vision/bin_coordinates', self.bin_callback, 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        self.state_pub = self.create_publisher(String, '/robot_state/tool_arm/update', 10)
        self.send_tool_cmd(0) # Start Neutral
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 3. Memory
        self.data_lock = threading.Lock()
        self.latest_screw_targets = []
        self.latest_local_data = None
        self.latest_full_data = {}
        self.latest_bin_coords = {} 
        self.ft_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.cached_bin_pose = None 
        self.vision_msg_count = 0
        self.last_sync_count = 0

        # =====================================================================
        # CONFIGURATION VARIABLES (Fine-Tune these)
        # =====================================================================
        self.SDK_MIN_SPEED = 30.0           # mm/s
        self.SDK_MOVE_SPEED = 30.0          # mm/s for hops
        self.SDK_SLOW_SPEED = 15.0          # mm/s for seating/contact? Use 26 for safety if needed.
        self.SDK_RETRACT_SPEED = 50.0       # Fast retract for departures (mm/s)
        
        # XY ALIGNMENT
        self.XY_MAX_STEP_MM = 0.5           # Allow larger steps if fast
        self.XY_MIN_STEP_MM = 0.1           # Smaller min jump to prevent oscillation
        self.XY_GAIN = 0.4                  # LOWER GAIN (0.4) to fix overshoot latency
        self.MM_PER_PIXEL = 0.00013         # Calibration (1mm per px @ 1m)
        self.ALIGN_TOLERANCE_PX = 5.0      # Lock-on tolerance
        self.ALIGN_LOCKED_THRESHOLD_PX = 10.0 # Threshold for high-speed Z descent
        self.DEADBAND_PX = 3.0              # Stop jitter
        self.SDK_XY_MAX_SPEED = 1.0         # Max mapping (Large error)
        self.SDK_XY_MIN_SPEED = 0.1         # Min mapping (Small error)
        self.SDK_XY_ALIGN_SPEED = 0.1       # Default/Legacy speed

        # FUNNEL DESCENT
        self.FUNNEL_START_THRESH_PX = 50.0  # Only start Z descent below this error
        self.Z_MAX_STEP_MM = 1.0            # 1mm descent per hop
        self.Z_MIN_STEP_MM = 0.5            # 0.5mm descent when misaligned
        
        # SPIRAL SEARCH
        self.SPIRAL_RING_GAP_MM = 1.0       # 2mm expansion per rotation
        self.SPIRAL_STEP_LEN_MM = 1.0       # 5mm hop distance
        self.SPIRAL_SPEED = 10.0            # Faster spiral speed mm/s
        self.SPIRAL_TIMEOUT = 15.0          # 15s vision recovery timeout
        
        # CONTACT & SEATING
        self.Z_FORCE_SPIKE_THRESH = 3.0     # N
        self.SEAT_ROTATE_TIME = 1.0         # 1s seating rotation
        self.SEAT_CHECK_TIME = 0.5          # 0.5s check rotation
        self.WIGGLE_SPEED = 1.0             # 1mm/s as requested
        self.WIGGLE_AMP_MM = 2.0            # 5mm wiggle as requested
        self.WIGGLE_LOCK_THRESH = 1.0       # N (force required to confirm seat)
        self.WIGGLE_PASS_MIN = 3            # Pass out of 4 (1-4)

        # EXTRACTION
        self.EXTRACTION_RELIEF_MM = 1.0     # 1mm hop up on force spike
        self.EXTRACTION_FORCE_SPIKE = 0.5   # N spike during unscrew
        self.EXTRACTION_DONE_TIMEOUT = 5.0  # 5s timeout as requested
        
        # FRAME MAPPING
        self.PLANNING_FRAME = 'world_world'
        self.CAMERA_FRAME = 'camera_color_optical_frame'
        self.ROBOT_BASE_FRAME = 'xarm5_base_link'
        self.SWAP_XY = True                 # Transformed vision err_x -> robot_y, etc.
        self.INVERT_X = True
        self.INVERT_Y = True
        
        # SAFETY HEIGHTS
        self.BEFORE_TARGET_RETRACT = 0.050  # 50mm lift before approach
        self.HOVER_Z_HEIGHT = 0.015         # 15mm hover above part
        self.TOOL_LENGTH = 0.24             # 240mm Tool Offset
        self.SUCCESS_RETRACT = 0.015        # 15mm slow retract after unscrew
        self.VISION_SETTLE_TIME = 0.3        # 300ms to allow vision pipeline to refresh
        # =====================================================================

        self.get_logger().info("🛠️ Unscrew Skill: Scratch-Rewrite Initialized.")

    # =========================================================================
    # 1. CORE EXECUTION ENGINE
    # =========================================================================
    def execute_unscrew_command(self, part_id: int, label="", interactive=False):
        self.publish_arm_state("MOVING")
        self.get_logger().info(f"🔗 TARGET: {label} [ID: {part_id}]")
        
        # 1. RETRACT 50mm BEFORE ANY MOVE
        self.get_logger().info(f"⬆️ Safe Retract: {self.BEFORE_TARGET_RETRACT*1000}mm")
        self._ensure_sdk_mode(0)
        self.jog_absolute_dk(0, 0, self.BEFORE_TARGET_RETRACT*1000, speed=self.SDK_RETRACT_SPEED)

        # 2. LOCATE TARGET IN WORLD
        target_xyz = self.locate_part_in_world(part_id)
        if not target_xyz:
            self.send_tool_cmd(0)
            self.publish_arm_state("ERROR")
            return False

        # 3. SDK HOVER APPROACH
        world_x, world_y, surface_z = target_xyz
        hover_flange_z = surface_z + self.TOOL_LENGTH + self.HOVER_Z_HEIGHT
        
        p = Pose()
        p.position.x, p.position.y, p.position.z = world_x, world_y, hover_flange_z
        p.orientation.w = 1.0
        
        trans_p = self.get_transformed_pose(p, self.PLANNING_FRAME, self.ROBOT_BASE_FRAME)
        if not trans_p:
            self.send_tool_cmd(0)
            self.get_logger().error("❌ Failed to transform hover pose to base frame.")
            self.publish_arm_state("ERROR")
            return False
            
        tx = trans_p.pose.position.x * 1000.0
        ty = trans_p.pose.position.y * 1000.0
        tz = trans_p.pose.position.z * 1000.0
        
        self.get_logger().info(f"🛰️ SDK Planning to Hover: X:{tx:.1f} Y:{ty:.1f} Z:{tz:.1f}...")
        self._ensure_sdk_mode(0)
        
        # Fast transit to hover using retract speed
        move_code = self.arm.set_position(x=tx, y=ty, z=tz, roll=180, pitch=0, yaw=0, speed=self.SDK_RETRACT_SPEED, wait=True)
        if move_code != 0:
            self.send_tool_cmd(0)
            self.get_logger().error("❌ SDK Hover move failed.")
            self.publish_arm_state("ERROR")
            return False

        # 4. START ALIGNMENT & DESCENT (Direct SDK)
        self.publish_arm_state("UNSCREWING")
        seated = self.sdk_staircase_descent(interactive)
        
        if seated == "HOLE_ONLY":
            self.get_logger().info("⏹️ Hole found. Navigating to Bin 1 directly.")
            self.dispose_to_bin("bin_1")
            return True
            
        if not seated:
            self.send_tool_cmd(0)
            self.get_logger().error("❌ Failed to seat bit.")
            self.publish_arm_state("ERROR")
            return False

        # 5. UNSCREWING PROCESS
        self.publish_arm_state("UNSCREWING")
        self.sdk_reactive_extraction(interactive)

        # 6. DISPOSE
        self.dispose_to_bin("bin_1")
        return True

    # =========================================================================
    # 2. SDK STAIRCASE & SPIRAL (The Deep Logic)
    # =========================================================================
    def sdk_staircase_descent(self, interactive=False):
        self.get_logger().info("🔍 Starting SDK Funnel Alignment...")
        self._ensure_sdk_mode(0)
        
        # Initialize baselines
        bx, by, bz = self.get_fresh_ft()
        spiral_start_t = None
        spiral_cnt = 0
        
        # Check if we should start with spiral (Hole Only logic)
        with self.data_lock:
            local = copy.deepcopy(self.latest_local_data)
        if local and local.get("holes") and not local.get("screw_heads"):
            self.get_logger().info("🕳️ Only Hole detected. Starting direct Hole Spiral Search...")
            res = self.sdk_spiral_loop()
            if res == "TIMEOUT": return "HOLE_ONLY"
            # If spiral found a screw, continue

        while rclpy.ok():
            # 1. Contact Monitoring (Global Surface Detection)
            _, _, curr_z = self.get_fresh_ft()
            spike = abs(curr_z - bz)
            if spike > self.Z_FORCE_SPIKE_THRESH:
                self.get_logger().info(f"🎯 [SURFACE CONTACT] Z-Spike: {spike:.2f}N. Verifying bit seat...")
                
                # User: "when z force / surface contact detected do a quick unscrew of 0.5 sec just to check... than do wiggle test"
                self.send_tool_cmd(-1)
                time.sleep(0.5)
                self.send_tool_cmd(0)
                time.sleep(0.5)
                
                if self.sdk_wiggle_check():
                    self.publish_arm_state("UNSCREWING_COMPLETE")
                    return True
                else:
                    self.publish_arm_state("UNSCREWING_RETRY")
                    self.get_logger().warn("⚠️ Wiggle failed. Retrying descent...")
                    self.jog_absolute_dk(0, 0, 5.0, speed=self.SDK_RETRACT_SPEED) # Faster retract
                    bz = self.get_fresh_ft()[2] # Re-zero
                    continue

            # 2. Vision Check
            err_x, err_y, dist_px = self.get_target_pixel_error()
            
            if err_x is None:
                # Vision lost -> Spiral Search
                self.get_logger().warn("⚠️ Vision lost! Entering Spiral Recovery...")
                res = self.sdk_spiral_loop()
                if res == "TIMEOUT":
                    return "HOLE_ONLY"
                elif res == "CONTACT":
                    # Force a contact cycle in the next loop iteration
                    continue
                continue

            # 3. Calculate XY Jump
            # Map pixels to robot meters
            raw_dx = (err_y * self.MM_PER_PIXEL) * self.XY_GAIN if self.SWAP_XY else (err_x * self.MM_PER_PIXEL) * self.XY_GAIN
            raw_dy = (err_x * self.MM_PER_PIXEL) * self.XY_GAIN if self.SWAP_XY else (err_y * self.MM_PER_PIXEL) * self.XY_GAIN
            
            if self.INVERT_X: raw_dx = -raw_dx
            if self.INVERT_Y: raw_dy = -raw_dy
            
            # Clamp Step size (mm)
            dx_mm = self.clamp_step(raw_dx * 1000.0, self.XY_MIN_STEP_MM, self.XY_MAX_STEP_MM)
            dy_mm = self.clamp_step(raw_dy * 1000.0, self.XY_MIN_STEP_MM, self.XY_MAX_STEP_MM)
            
            # Apply deadband
            if dist_px < self.DEADBAND_PX: dx_mm = 0.0; dy_mm = 0.0

            # 4. Calculate Z Descent (Funnel)
            dz_mm = 0.0
            if dist_px < self.FUNNEL_START_THRESH_PX:
                # User instructions: "close it it to the target higher the step size"
                # If error < 10px -> 1.0mm, if error > 20px -> 0.5mm
                if dist_px < self.ALIGN_LOCKED_THRESHOLD_PX:
                    dz_mm = -self.Z_MAX_STEP_MM
                else:
                    dz_mm = -self.Z_MIN_STEP_MM
            
            # 5. Execute Command with Dynamic Speed
            # Less error -> Less speed.
            dyn_speed = max(self.SDK_XY_MIN_SPEED, min(self.SDK_XY_MAX_SPEED, (dist_px / 30.0) * self.SDK_XY_MAX_SPEED))
            
            self.get_logger().info(f"📉 Err:{dist_px:.1f}px | dX:{dx_mm:.2f} dY:{dy_mm:.2f} dZ:{dz_mm:.2f} Spd:{dyn_speed:.2f}")
            self.jog_absolute_dk(dx_mm, dy_mm, dz_mm, speed=dyn_speed)
            
            # Anti-Overshoot Sync: Wait for fresh frames (usually 2 msgs to clear buffer)
            self.wait_for_vision_update(count=2)
            time.sleep(self.VISION_SETTLE_TIME)
            
        return False

    def sdk_spiral_loop(self):
        """Archimedean spiral hop loop with force monitoring."""
        start_t = time.time()
        idx = 0
        _, _, bz = self.get_fresh_ft()
        
        while (time.time() - start_t) < self.SPIRAL_TIMEOUT:
            # Check Force Spike (Surface Detection during spiral)
            _, _, cz = self.get_fresh_ft()
            if abs(cz - bz) > self.Z_FORCE_SPIKE_THRESH:
                self.get_logger().info(f"🎯 [SPIRAL CONTACT] Z-Spike: {abs(cz-bz):.2f}N")
                return "CONTACT"

            idx += 1
            # Archimedean math: r = a * theta
            # Ring gap 1mm -> 0.001m
            theta = idx * 0.5 # radians per step
            r = theta * ((self.SPIRAL_RING_GAP_MM / 1000.0) / (2 * math.pi))
            
            p_theta = (idx - 1) * 0.5
            p_r = p_theta * ((self.SPIRAL_RING_GAP_MM / 1000.0) / (2 * math.pi))
            
            # Step in robot space
            delta_x = (r * math.cos(theta)) - (p_r * math.cos(p_theta))
            delta_y = (r * math.sin(theta)) - (p_r * math.sin(p_theta))
            
            # User wants 1mm steps
            dx_mm = delta_x * 1000.0
            dy_mm = delta_y * 1000.0
            
            # Scale to user's requirement
            scale = self.SPIRAL_STEP_LEN_MM / math.hypot(dx_mm, dy_mm)
            dx_mm *= scale; dy_mm *= scale
            
            self.get_logger().info(f"🌀 Spiral hop {idx} | dX:{dx_mm:.2f} dY:{dy_mm:.2f} Spd:{self.SPIRAL_SPEED}")
            
            # Map axes for spiral
            raw_dx = dy_mm if self.SWAP_XY else dx_mm
            raw_dy = dx_mm if self.SWAP_XY else dy_mm
            if self.INVERT_X: raw_dx = -raw_dx
            if self.INVERT_Y: raw_dy = -raw_dy

            self.jog_absolute_dk(raw_dx, raw_dy, 0, speed=self.SPIRAL_SPEED)
            time.sleep(0.1)
            
            if self.get_target_pixel_error()[0] is not None:
                self.get_logger().info("✅ Vision Recovered!")
                return "FOUND"
                
        return "TIMEOUT"

    def sdk_wiggle_check(self):
        """Perform 4-way force barrier check with detailed UI states."""
        self.get_logger().info(f"🔄 Performing 3/4 Wiggle Pass ({self.WIGGLE_AMP_MM}mm Hops)...")
        passes = 0
        amp = self.WIGGLE_AMP_MM
        
        # Directions: List of (dx, dy, axis_index, label)
        directions = [
            (amp, 0, 0, "WIGGLE_POS_X"),
            (-amp, 0, 0, "WIGGLE_NEG_X"),
            (0, amp, 1, "WIGGLE_POS_Y"),
            (0, -amp, 1, "WIGGLE_NEG_Y")
        ]
        
        self._ensure_sdk_mode(0)
        
        for dx, dy, axis_idx, label in directions:
            self.publish_arm_state(label) # Show in UI one by one
            baselines = self.get_fresh_ft()
            bx, by = baselines[0], baselines[1]
            
            # Wiggle at 1mm/s for 5mm
            self.jog_absolute_dk(dx, dy, 0, speed=self.WIGGLE_SPEED)
            time.sleep(0.15)
            
            currents = self.get_fresh_ft()
            cx, cy = currents[0], currents[1]
            
            spike = abs(cx - bx) if axis_idx == 0 else abs(cy - by)
            self.get_logger().info(f"   [{label}] Spike:{spike:.2f}N")
                
            if spike > 1.0: # 1.0N spike threshold
                passes += 1
            
            # Return to center
            self.jog_absolute_dk(-dx, -dy, 0, speed=self.WIGGLE_SPEED)
            time.sleep(0.1)
            
        success = passes >= 3
        self.get_logger().info(f"📊 Wiggle result: {passes}/4. {'PASS' if success else 'FAIL'}")
        return success

    def sdk_reactive_extraction(self, interactive=False):
        """Unscrew with relief hops on force spikes."""
        self.get_logger().info("🔥 Starting Unscrewing Force Compliance...")
        bx, by, bz = self.get_fresh_ft()
        
        # User: "send -1... always before starting to unscrew"
        self.send_tool_cmd(-1) 
        time.sleep(0.2)
        
        # User: "while unscrew grab the screw as well"
        self.send_tool_cmd(2) 
        time.sleep(0.5)
        
        last_spike_t = time.time()
        while rclpy.ok():
            _, _, cz = self.get_fresh_ft()
            spike = abs(cz - bz) # Use absolute spike to handle rising force correctly
            
            if spike > self.EXTRACTION_FORCE_SPIKE:
                self.get_logger().info(f"📈 Screw rising! Relief hop: {self.EXTRACTION_RELIEF_MM}mm (Spike: {spike:.2f}N)")
                self.jog_absolute_dk(0, 0, self.EXTRACTION_RELIEF_MM)
                # Re-baseline slightly higher
                bx, by, bz = self.get_fresh_ft()
                last_spike_t = time.time()
            
            if (time.time() - last_spike_t) > self.EXTRACTION_DONE_TIMEOUT:
                self.get_logger().info("✅ Force stabilized. Unscrew complete.")
                break
            time.sleep(0.05)
            
        self.send_tool_cmd(0) # Stop motor
        time.sleep(0.5)
        
        # SLOW 15mm RETRACT
        self.get_logger().info(f"⬆️ Slow Retract: {self.SUCCESS_RETRACT*1000}mm")
        self.jog_absolute_dk(0, 0, self.SUCCESS_RETRACT*1000)

    def dispose_to_bin(self, bin_name="bin_1"):
        """SDK-based bin disposal to follow 'MoveIt only for hover' rule."""
        self.publish_arm_state("MOVING")
        self.get_logger().info(f"🗑️ Navigating to {bin_name}...")
        
        # 0. Lift 50mm before transit
        self.jog_absolute_dk(0, 0, self.BEFORE_TARGET_RETRACT*1000, speed=self.SDK_RETRACT_SPEED)
        
        bin_xyz = self.calculate_bin_pose(bin_name)
        if not bin_xyz:
            self.get_logger().error(f"❌ {bin_name} coords unknown.")
            return False
            
        wx, wy, wz = bin_xyz
        target_z = wz + self.TOOL_LENGTH + 0.050 # Drop 50mm above bin surface
        
        # Convert World (wx, wy, target_z) to Robot Base Frame
        p = Pose()
        p.position.x, p.position.y, p.position.z = wx, wy, target_z
        p.orientation.w = 1.0
        
        trans_p = self.get_transformed_pose(p, self.PLANNING_FRAME, self.ROBOT_BASE_FRAME)
        if not trans_p:
            self.get_logger().error("❌ Failed to transform bin pose to base frame.")
            return False
            
        tx, ty, tz = trans_p.pose.position.x * 1000.0, trans_p.pose.position.y * 1000.0, trans_p.pose.position.z * 1000.0
        
        # Absolute SDK move to bin
        self._ensure_sdk_mode(0)
        self.get_logger().info(f"🚚 SDK Move to Bin: X:{tx:.1f} Y:{ty:.1f} Z:{tz:.1f}")
        self.arm.set_position(tx, ty, tz, 180, 0, 0, speed=50.0, wait=True)
        
        self.get_logger().info(f"👐 Dropping screw in {bin_name}...")
        self.publish_arm_state("DROPPING")
        
        # User: "send 0 to stop the unscrew and after that 3 to open"
        self.send_tool_cmd(0)
        time.sleep(0.2)
        self.send_tool_cmd(3) 
        time.sleep(1.5)
        self.send_tool_cmd(0)
        
        # Retract 50mm after drop
        self.jog_absolute_dk(0, 0, self.BEFORE_TARGET_RETRACT*1000, speed=self.SDK_RETRACT_SPEED)
        self.publish_arm_state("IDLE")
        return True

    # =========================================================================
    # 3. TRANSFORMATION & VISION
    # =========================================================================
    def locate_part_in_world(self, part_id):
        target_data = None
        wait_start = time.time()
        while rclpy.ok() and (time.time() - wait_start) < 5.0:
            with self.data_lock:
                for part in self.latest_screw_targets:
                    if part['id'] == part_id: target_data = copy.deepcopy(part); break
            if target_data: break
            time.sleep(0.5)
        if not target_data: return None

        # Pixel-to-World Transform
        p = Pose()
        p.position.x, p.position.y, p.position.z = target_data['xyz']
        p.orientation.w = 1.0
        
        # Average 5 samples
        valid_x, valid_y, valid_z = [], [], []
        for _ in range(5):
            t = self.get_transformed_pose(p, self.CAMERA_FRAME, self.PLANNING_FRAME)
            if t:
                valid_x.append(t.pose.position.x)
                valid_y.append(t.pose.position.y)
                valid_z.append(t.pose.position.z)
            time.sleep(0.05)
        
        return (np.median(valid_x), np.median(valid_y), np.median(valid_z)) if valid_x else None

    def get_target_pixel_error(self):
        with self.data_lock:
            local = copy.deepcopy(self.latest_local_data)
        if not local or not local.get("screw_heads"):
            return None, None, None
        
        # Use center of best screw bounding box
        best = max(local["screw_heads"], key=lambda x: x.get("conf", 0))
        box = best.get("box")
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        
        # Crosshair center (usually 320, 240)
        ch = local.get("crosshair", [320, 240])
        err_x = ch[0] - cx
        err_y = ch[1] - cy
        return err_x, err_y, math.hypot(err_x, err_y)

    # =========================================================================
    # 4. LOW-LEVEL SDK JOGGERS (The "Scratch" part)
    # =========================================================================
    def _ensure_sdk_mode(self, mode=0):
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(mode)
        self.arm.set_state(0)
        time.sleep(0.1)

    def jog_absolute_dk(self, dx_mm, dy_mm, dz_mm, speed=None):
        """Calculate absolute target from current joint positions and command absolute move."""
        code, curr_pos = self.arm.get_position(is_radian=False)
        if code != 0: return False
        
        tx = curr_pos[0] + dx_mm
        ty = curr_pos[1] + dy_mm
        tz = curr_pos[2] + dz_mm
        
        cmd_speed = speed if speed is not None else self.SDK_XY_ALIGN_SPEED

        # Use wait=True for precise blocking steps
        self.arm.set_position(x=tx, y=ty, z=tz, roll=curr_pos[3], pitch=curr_pos[4], yaw=curr_pos[5],
                              speed=cmd_speed, wait=True)
        return True

    def get_fresh_ft(self):
        with self.data_lock:
            return self.ft_force['x'], self.ft_force['y'], self.ft_force['z']

    # =========================================================================
    # HELPERS
    # =========================================================================
    def clamp_step(self, val, min_s, max_s):
        if abs(val) < 0.001: return 0.0
        sign = 1.0 if val >= 0 else -1.0
        return sign * max(min_s, min(abs(val), max_s))

    def publish_arm_state(self, state):
        msg = String(); msg.data = state.upper(); self.state_pub.publish(msg)

    def bin_callback(self, msg):
        try:
            data = json.loads(msg.data.strip("'"))
            with self.data_lock: self.latest_bin_coords = data
        except Exception: pass

    def vision_callback(self, msg):
        self.vision_msg_count += 1
        try:
            data = json.loads(msg.data.strip("'"))
            objects = data.get("global_view", {}).get("objects", [])
            new_targets = []
            for obj in objects:
                if "Screw_Zone" in obj.get("label", "") and "xyz" in obj:
                    new_targets.append({"id": obj.get("id"), "label": obj.get("label"), "xyz": obj.get("xyz")})
            with self.data_lock:
                self.latest_screw_targets = new_targets
                self.latest_local_data = data.get("local_view", {})
                self.latest_full_data = data
                ft = data.get("force_torque", {}).get("force", {})
                self.ft_force = {'x': ft.get("x", 0.0), 'y': ft.get("y", 0.0), 'z': ft.get("z", 0.0)}
        except Exception: pass 

    def send_tool_cmd(self, val):
        msg = Int8(); msg.data = int(val); self.tool_pub.publish(msg)

    def wait_for_vision_update(self, count=2, timeout=2.0):
        """Wait for n new messages to arrive to ensure we aren't using stale frames."""
        start_count = self.vision_msg_count
        start_t = time.time()
        while rclpy.ok() and (self.vision_msg_count - start_count) < count:
            if (time.time() - start_t) > timeout: break
            time.sleep(0.01)

    def calculate_bin_pose(self, name="bin_1"):
        if self.cached_bin_pose: return self.cached_bin_pose
        with self.data_lock: data = self.latest_bin_coords.get(name)
        if not data: return None
        p = Pose()
        p.position.x, p.position.y, p.position.z = data['xyz']
        p.orientation.w = 1.0
        res = self.locate_part_in_world_from_pose(p)
        if res: self.cached_bin_pose = res
        return res

    def locate_part_in_world_from_pose(self, p):
        valid_x, valid_y, valid_z = [], [], []
        for _ in range(5):
            t = self.get_transformed_pose(p, self.CAMERA_FRAME, self.PLANNING_FRAME)
            if t:
                valid_x.append(t.pose.position.x)
                valid_y.append(t.pose.position.y)
                valid_z.append(t.pose.position.z)
            time.sleep(0.05)
        if not valid_x: return None
        return (np.median(valid_x), np.median(valid_y), np.median(valid_z))

    def get_transformed_pose(self, source_pose, source_frame: str, target_frame: str):
        from geometry_msgs.msg import PoseStamped
        import tf2_geometry_msgs
        try:
            real_pose = source_pose.pose if hasattr(source_pose, 'pose') else source_pose
            p = PoseStamped()
            p.header.frame_id = source_frame
            p.header.stamp = rclpy.time.Time().to_msg()
            p.pose = real_pose

            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=1.0)):
                return None
            t = self.tf_buffer.transform(p, target_frame)
            return t
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = UnscrewSkill()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # Run loop in thread
    loop_thread = threading.Thread(target=executor.spin, daemon=True)
    loop_thread.start()
    
    try:
        while rclpy.ok():
            with node.data_lock:
                targets = copy.deepcopy(node.latest_screw_targets)
            if not targets:
                time.sleep(1.0)
                continue
            
            for part in targets:
                node.execute_unscrew_command(part['id'], part['label'], interactive=False)
            time.sleep(2.0)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()