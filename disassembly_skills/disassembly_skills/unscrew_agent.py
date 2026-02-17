#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Pose, TransformStamped
import tf2_ros

import json
import time
import threading
import sys
import copy
import math

from disassembly_skills.motion_backend import MotionBackend

# =========================================================
#                    CONFIGURATION
# =========================================================

# --- FRAMES & VISUALIZATION ---
CAMERA_FRAME = 'camera_color_optical_frame'   # Source frame from the RealSense
PLANNING_FRAME = 'world_world'                # Target global base frame
VISUALIZE_TF = True                           # Broadcast targets to RViz

# --- ARUCO / VISION CALIBRATION ---
MM_PER_PIXEL = 0.000130   # Ratio to map camera pixel error to physical distance

# --- CAMERA TO ROBOT AXIS TUNING ---
INVERT_X = False          # True if U error should move -X
INVERT_Y = False          # True if V error should move -Y
SWAP_XY = True            # True if U maps to Y, and V maps to X (90-deg rotation)

# --- APPROACH & RETRACT SETTINGS ---
TOOL_LENGTH_XARM = 0.24   # Physical length of the screwdriver bit (meters)
APPROACH_HOVER_Z = 0.010  # Safe initial height above screw to begin staircase (meters)
APPROACH_SPEED = 30.0     # Fast macro descent speed to hover height (mm/s)

# 👈 [UPDATED] RETRACT DISTANCES
SUCCESS_RETRACT = 0.050   # 50mm retract after a successful unscrew
RETRY_RETRACT = 0.005     # 5mm retract if the tool slips and needs to try aligning again
EMERGENCY_RETRACT = 0.010 # 10mm retract if an unexpected mid-air collision occurs

# --- STEP-WISE MANIPULATION SETTINGS ---
XY_ALIGN_SPEED = 25.0     # Speed of micro-adjustments during visual alignment (mm/s)
Z_STEP_DOWN_DIST = -0.002 # 2mm drop per staircase step (meters)
Z_STEP_SPEED = 5.0        # Speed of downward tactile probes (mm/s)
WIGGLE_DIST = 0.002       # Distance to move during tactile seating check (meters)
UNSCREW_RELIEF_STEP = 0.0005 # Upward step to relieve pressure during unscrewing (meters)

# --- FORCE THRESHOLDS (Delta from Baseline) ---
APPROACH_COLLISION_THRESH = 3.0 # N (Total XYZ magnitude spike indicating mid-air collision)
SURFACE_CONTACT_THRESH = 2.0    # N (Z spike indicating tool has touched the object)
WIGGLE_LOCK_THRESH = 3.5        # N (XY spike indicating tool is inside screw slot walls)
UNSCREW_SPIKE_THRESH = 1.5      # N (Z spike indicating screw threads are pushing up)
UNSCREW_DONE_TIMEOUT = 4.0      # Seconds without a Z-spike to consider extraction complete

# --- SPIRAL SEARCH ---
SPIRAL_STEPS = 20         # Maximum number of legs in the outward expansion
SPIRAL_GAP = 0.002        # Distance between concentric spiral loops (meters)

# =========================================================

class MasterDisassemblyAgent(Node):
    """
    Master control node for adaptive dual-arm disassembly sequences.
    Manages the state machine for precision visual-tactile unscrewing.
    """
    def __init__(self):
        super().__init__('master_disassembly_agent')
        
        # Initialize hardware backend
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        
        # ROS 2 Pubs/Subs
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Thread-safe data storage for asynchronous callbacks
        self.data_lock = threading.Lock()
        self.latest_screw_targets = []
        self.latest_local_data = None
        self.ft_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        self.get_logger().info("✅ Infinite Micro-Staircase Disassembly Agent Ready.")

    def vision_callback(self, msg):
        """Asynchronously parses incoming JSON vision states and localizes targets and forces."""
        try:
            data = json.loads(msg.data)
            
            # Extract global screw coordinates
            objects = data.get("global_view", {}).get("objects", [])
            new_targets = []
            for obj in objects:
                label = obj.get("label", "")
                if "Screw_Zone" in label and "xyz" in obj:
                    new_targets.append({
                        "id": obj.get("id"),
                        "label": label,
                        "xyz": obj.get("xyz")
                    })
            
            # Extract Real-Time Force Torque Data
            ft = data.get("force_torque", {}).get("force", {})
            local_ft = {'x': ft.get("x", 0.0), 'y': ft.get("y", 0.0), 'z': ft.get("z", 0.0)}
            
            # Safely update state variables
            with self.data_lock:
                self.latest_screw_targets = new_targets
                self.latest_local_data = data.get("local_view", {})
                self.ft_force = local_ft
                    
        except Exception as e:
            pass 

    def send_tool_cmd(self, val):
        """Publishes integer command to tool motor (1: Twist, -1: Reverse/Unscrew, 0: Stop)."""
        msg = Int8()
        msg.data = int(val)
        self.tool_pub.publish(msg)

    def publish_debug_tf(self, x, y, z, screw_id):
        """Broadcasts calculated global screw coordinates to TF tree for RViz visualization."""
        if not VISUALIZE_TF: return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = PLANNING_FRAME
        t.child_frame_id = f"screw_target_{screw_id}"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def get_fresh_baselines(self, settle_time=0.2):
        """Pauses briefly to let hardware vibrations settle, then locks in current XYZ forces."""
        time.sleep(settle_time)
        with self.data_lock:
            return self.ft_force['x'], self.ft_force['y'], self.ft_force['z']

    def get_fresh_vision_error(self):
        """Wipes old vision data and blocks until a brand new camera frame is received to prevent stale tracking."""
        with self.data_lock:
            self.latest_local_data = None 
            
        wait_start = time.time()
        # Wait up to 2 seconds for fresh inference
        while rclpy.ok() and (time.time() - wait_start) < 2.0:
            with self.data_lock:
                if self.latest_local_data is not None:
                    screws = self.latest_local_data.get("screw_heads", [])
                    tools = self.latest_local_data.get("tool_tips", [])
                    if screws and tools:
                        t = max(tools, key=lambda x: x.get("conf", 0))
                        s = max(screws, key=lambda x: x.get("conf", 0))
                        tx, ty = t.get("contact_point", [None, None])
                        sx, sy = s.get("center", [None, None])
                        if None not in [tx, ty, sx, sy]:
                            return (sx - tx), (sy - ty)
            time.sleep(0.1) # Poll for new data
        return None, None

    def run_logic(self):
        """Main continuous loop. Waits for targets and sequentially executes the unscrew skill."""
        self.moveit_backend.reset_robot()

        while rclpy.ok():
            current_targets = []
            with self.data_lock:
                current_targets = copy.deepcopy(self.latest_screw_targets)
            
            # Wait if no targets found
            if not current_targets:
                time.sleep(1.0); continue

            print(f"\n🔎 FOUND {len(current_targets)} SCREW ZONES.")
            
            for part_data in current_targets:
                if not rclpy.ok(): return
                
                # Execute the main unscrewing state machine
                success = self.unscrew(part_data, command="unscrew")
                
                if success:
                    print(f"✅ Successfully processed {part_data['id']}.")
                else:
                    print(f"⏭️ Failed or aborted {part_data['id']}. Moving to next target.")
                
                ui_next = input(f"\n👉 GATE 4: Press ENTER to move to the next part in batch ('q'=quit): ")
                if ui_next.lower() == 'q': sys.exit(0)
                
            print("\n✅ Batch Complete. Waiting 5s for new vision data...")
            time.sleep(5.0) 

    def calculate_target_pose(self, part_data):
        """Transforms raw camera XYZ to world frame, accounting for tool tip length."""
        raw_xyz = part_data['xyz']
        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = raw_xyz[0], raw_xyz[1], raw_xyz[2]
        source_pose.orientation.w = 1.0

        target_stamped = None
        for _ in range(5):
            target_stamped = self.moveit_backend.get_transformed_pose(source_pose, CAMERA_FRAME, PLANNING_FRAME)
            if target_stamped: break
            time.sleep(0.2)

        if not target_stamped: return None
        
        wx = target_stamped.pose.position.x
        wy = target_stamped.pose.position.y
        # Compensate for physical tool length
        wz = target_stamped.pose.position.z + TOOL_LENGTH_XARM 
        
        self.publish_debug_tf(wx, wy, wz, part_data['id'])
        return (wx, wy, wz)

    def execute_unscrew_command(self, target_id, target_label=""):
        """
        LIBRARY METHOD: Called by the Master Orchestrator/VLA.
        Takes the ID and Label, finds the freshest XYZ coordinates in the 
        live vision stream, and executes the extraction.
        """
        target_data = None
        
        # Look up the ID in our latest vision data to get the freshest XYZ
        with self.data_lock:
            for part in self.latest_screw_targets:
                if part['id'] == target_id:
                    # Safety check: Warn if the VLA mixed up the labels
                    if target_label and part['label'] != target_label:
                        print(f"⚠️ [WARNING] ID matched ({target_id}), but label mismatched! Expected '{target_label}', saw '{part['label']}'.")
                    
                    target_data = copy.deepcopy(part)
                    break
                    
        if not target_data:
            print(f"❌ [ERROR] Orchestrator requested ID {target_id}, but it is not currently visible in the camera frame.")
            return False
            
        print(f"🎯 [LIBRARY] Request approved. Fetching fresh XYZ for ID {target_id} ({target_data['label']})...")
        
        # Call the main state machine with the fully populated dictionary
        return self.unscrew(target_data, command="unscrew")

    def unscrew(self, part_data, command="unscrew"):
        """
        Master State Machine for extracting a specific screw.
        Handles Approach -> Staircase Descent -> Wiggle Verification -> Reactive Extraction.
        """
        print("\n" + "="*70)
        print(f"🛠️  [SKILL START] Executing '{command}' on: {part_data['label']} (ID: {part_data['id']})")
        
        target_pose = self.calculate_target_pose(part_data)
        if not target_pose: 
            return False
            
        world_x, world_y, target_z = target_pose
        hover_z = target_z + APPROACH_HOVER_Z

        print(f"   🤖 [DEBUG] Global Coords: X={world_x:.4f} Y={world_y:.4f} Z={target_z:.4f}")
        print(f"   🤖 [DEBUG] Hover Z height calculated at: {hover_z:.4f}")
        
        ui1 = input(f"\n👉 GATE 1: Press ENTER to begin guarded approach to 3cm hover ('s'=skip, 'q'=quit): ")
        if ui1.lower() == 's': return False
        if ui1.lower() == 'q': sys.exit(0)

        # Execute coarse approach over the target
        if not self.guarded_approach(world_x, world_y, hover_z):
            return False 

        ui2 = input(f"\n👉 GATE 2: Arrived at 3cm. Press ENTER to begin autonomous Infinite Staircase: ")

        max_tactile_retries = 3
        for attempt in range(max_tactile_retries):
            print(f"\n   --- Extraction Attempt {attempt + 1}/{max_tactile_retries} ---")
            
            # PHASE 1 & 2: Dynamic Visual-Tactile Search
            if not self.staircase_align_and_descend():
                return False 

            # PHASE 3: Physical Lock Verification
            if self.verify_seating():
                ui3 = input(f"\n👉 GATE 3: Lock verified! Press ENTER to begin reactive unscrewing: ")
                # PHASE 4: Execute final rotation skill
                self.reactive_unscrew()
                return True 
            else:
                # Recovery: Retract and re-attempt the entire staircase
                print(f"   💨 [DEBUG] Wiggle Slipped! Retracting {RETRY_RETRACT*1000}mm and automatically retrying staircase...")
                self.moveit_backend.jog_cartesian_sdk(0, 0, RETRY_RETRACT, speed_mm_s=20.0)
                
        print("   ❌ [DEBUG] Max tactile retries reached. Aborting skill.")
        return False

    def guarded_approach(self, x, y, hover_z):
        """Moves quickly via MoveIt in XY plane, then plunges in Z while polling force sensor for safety."""
        print("   🚀 [PHASE 0] Guarded Approach to Hover Height...")
        
        curr_pose = self.moveit_backend.get_transformed_pose(Pose(), 'xarm5_link5', 'world_world')
        if not curr_pose: return False
        safe_z = curr_pose.pose.position.z
        
        # MoveIt XY transit
        print(f"      [DEBUG] Planning MoveIt trajectory to XY ({x:.4f}, {y:.4f})...")
        self.moveit_backend.move_to_pose_robust(x, y, safe_z, {}, link_name="xarm5_link5", velocity=0.1)
        bx, by, bz = self.get_fresh_baselines(0.5)
        
        # Safety callback for sudden mid-air collisions
        def approach_collision_check():
            cx, cy, cz = self.ft_force['x'], self.ft_force['y'], self.ft_force['z']
            return math.sqrt((cx-bx)**2 + (cy-by)**2 + (cz-bz)**2) > APPROACH_COLLISION_THRESH

        z_drop_dist = safe_z - hover_z
        success = self.moveit_backend.move_linear_z_sdk_with_force_stop(
            distance_down_m=z_drop_dist, 
            speed_mm_s=APPROACH_SPEED,
            check_force_callback=approach_collision_check
        )
        
        if approach_collision_check():
            print("\n   💥 [FATAL] Unexpected collision during approach! Emergency Retract.")
            self.moveit_backend.jog_cartesian_sdk(0, 0, EMERGENCY_RETRACT, speed_mm_s=20.0)
            return False
        return True

    def staircase_align_and_descend(self):
        """
        Infinite Staircase Logic: 
        1. Aligns XY using camera until center is achieved.
        2. Drops Z blindly by a set increment (2mm).
        3. Repeats infinitely until tactile surface contact is achieved.
        """
        print("   🔭⬇️ [PHASE 1 & 2] Starting Infinite Staircase...")
        step_count = 0
        
        while rclpy.ok():
            step_count += 1
            print(f"\n      --- Staircase Step {step_count} ---")
            
            # --- STEP A: ALIGN XY ---
            aligned = False
            is_first_align_jump = True 
            
            while not aligned and rclpy.ok():
                # Force a fresh frame to avoid latency loops
                err_u, err_v = self.get_fresh_vision_error()
                
                # Auto-recovery if vision is lost (e.g. occlusion or yeeted)
                if err_u is None:
                    print("         [DEBUG] Vision lost. Spiraling...")
                    self.run_visual_spiral()
                    is_first_align_jump = True 
                    continue
                    
                dist_px = math.sqrt(err_u**2 + err_v**2)
                print(f"         [VISION] U:{err_u:.1f}px V:{err_v:.1f}px | Dist: {dist_px:.1f}px")
                
                # Break condition for alignment loop
                if dist_px < 5.0:
                    print("         ✅ Perfectly Aligned.")
                    aligned = True
                    break
                
                # Apply configured Camera-to-Robot mapping
                raw_dx = err_v * MM_PER_PIXEL if SWAP_XY else err_u * MM_PER_PIXEL
                raw_dy = err_u * MM_PER_PIXEL if SWAP_XY else err_v * MM_PER_PIXEL
                dx_m = -raw_dx if INVERT_X else raw_dx
                dy_m = -raw_dy if INVERT_Y else raw_dy
                
                # MICRO-STEP LOGIC
                # First jump is allowed 5mm to close gap, all subsequent adjustments locked to 0.5mm
                max_step = 0.005 if is_first_align_jump else 0.0005
                move_dist = math.hypot(dx_m, dy_m)
                
                if move_dist > max_step:
                    scale = max_step / move_dist
                    dx_m *= scale
                    dy_m *= scale
                    print(f"         [DEBUG] Move capped to {max_step*1000}mm.")
                
                print(f"         [ACTION] Jogging -> dX: {dx_m*1000:.2f}mm, dY: {dy_m*1000:.2f}mm")
                self.moveit_backend.jog_cartesian_sdk(dx_m, dy_m, 0.0, speed_mm_s=XY_ALIGN_SPEED)
                
                is_first_align_jump = False 

            # --- STEP B: DROP Z (2mm) ---
            bx, by, bz = self.get_fresh_baselines(0.2)
            
            self.moveit_backend.jog_cartesian_sdk(0, 0, Z_STEP_DOWN_DIST, speed_mm_s=Z_STEP_SPEED)
            
            cx, cy, cz = self.get_fresh_baselines(0.2)
            spike = abs(cz - bz)
            
            print(f"      [DEBUG] Z-Drop 2mm | Base_Z: {bz:.2f}N | Curr_Z: {cz:.2f}N | Spike: {spike:.2f}N")
            
            # Break condition for full staircase (tactile hit)
            if spike > SURFACE_CONTACT_THRESH:
                print(f"   ✅ [DEBUG] Surface Contact Detected! (Spike > {SURFACE_CONTACT_THRESH}N)")
                return True
                
        return False

    def verify_seating(self):
        """Wiggles tool in +/- XY to check if it hits walls (indicating successful screw insertion)."""
        print("   🔄 [PHASE 3] Pre-seating twist & Tactile Wiggle Verification...")
        
        # Brief spin to encourage bit to slip into crosshead/torx slots
        self.send_tool_cmd(1); time.sleep(0.3)
        self.send_tool_cmd(0); time.sleep(0.5)
        
        moves = [
            ('X+', WIGGLE_DIST, 0.0), 
            ('X-', -WIGGLE_DIST, 0.0),
            ('Y+', 0.0, WIGGLE_DIST), 
            ('Y-', 0.0, -WIGGLE_DIST)
        ]
        
        locks = 0
        for name, dx, dy in moves:
            bx, by, bz = self.get_fresh_baselines(0.2)
            
            self.moveit_backend.jog_cartesian_sdk(dx, dy, 0.0, speed_mm_s=Z_STEP_SPEED)
            
            cx, cy, cz = self.get_fresh_baselines(0.2)
            force_diff = math.sqrt((cx-bx)**2 + (cy-by)**2)
            
            if force_diff > WIGGLE_LOCK_THRESH:
                locks += 1
                print(f"      ✔️ [LOCKED] {name} | Spike: {force_diff:.2f}N")
            else:
                print(f"      ❌ [SLIPPED] {name} | Spike: {force_diff:.2f}N")
                
            # Return to center
            self.moveit_backend.jog_cartesian_sdk(-dx, -dy, 0.0, speed_mm_s=Z_STEP_SPEED)
            
        total_lock = locks >= 3
        print(f"   📊 [DEBUG] Wiggle Summary: {locks}/4 locked. Result: {'PASS' if total_lock else 'FAIL'}")
        return total_lock

    def reactive_unscrew(self):
        """Engages threads and intelligently rises in Z to prevent stripping as screw lifts."""
        print("   🌊 [PHASE 4] Starting Reactive Unscrewing Control...")
        
        bx, by, bz = self.get_fresh_baselines(0.5)
        print(f"      [DEBUG] Initial Unscrew Baseline Z: {bz:.2f}N")
        
        self.send_tool_cmd(-1) 
        
        last_spike_time = time.time()
        total_lift_mm = 0.0
        
        while rclpy.ok():
            curr_z = self.ft_force['z']
            spike = curr_z - bz 
            
            print(f"      [MONITOR] Base: {bz:.2f}N | Live: {curr_z:.2f}N | Spike: {spike:.2f}N | Total Lift: {total_lift_mm:.1f}mm", end='\r')
            
            # If upward pushing force is detected, relieve pressure by stepping up
            if spike > UNSCREW_SPIKE_THRESH:
                print(f"\n      📈 [ACTION] Z-Spike ({spike:.2f}N) > {UNSCREW_SPIKE_THRESH}N limit. Stepping up {UNSCREW_RELIEF_STEP*1000}mm to relieve pressure...")
                self.moveit_backend.jog_cartesian_sdk(0, 0, UNSCREW_RELIEF_STEP, speed_mm_s=Z_STEP_SPEED)
                total_lift_mm += (UNSCREW_RELIEF_STEP * 1000)
                
                bx, by, bz = self.get_fresh_baselines(0.1)
                last_spike_time = time.time()
                
            # If no upward pressure is felt for timeout duration, assume threads are free
            elif (time.time() - last_spike_time) > UNSCREW_DONE_TIMEOUT:
                print(f"\n   🏆 [SUCCESS] No spikes for {UNSCREW_DONE_TIMEOUT}s. Threads disengaged. Total Lift: {total_lift_mm:.1f}mm.")
                break
                
            time.sleep(0.05)
            
        self.send_tool_cmd(0)
        print(f"   ⬆️ [ACTION] Retracting {SUCCESS_RETRACT*1000}mm to safe height...")
        self.moveit_backend.jog_cartesian_sdk(0, 0, SUCCESS_RETRACT, speed_mm_s=20.0)

    def run_visual_spiral(self):
        """Executes a 2D expanding square spiral to blindly reacquire a lost visual target."""
        print(f"\n🌀 [DEBUG] Visual Spiral Recovery Initiated ({SPIRAL_STEPS} steps, {SPIRAL_GAP*1000}mm gap)...")
        
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0]]
        leg = 0
        current_gap_multiplier = 1
        
        while leg < SPIRAL_STEPS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: 
                current_gap_multiplier += 1
                
            step_dist = SPIRAL_GAP * current_gap_multiplier
            raw_dx = dirs[leg%4][0] * step_dist
            raw_dy = dirs[leg%4][1] * step_dist
            
            # Apply configured Camera-to-Robot mapping
            if SWAP_XY:
                raw_dx, raw_dy = raw_dy, raw_dx
                
            dx_m = -raw_dx if INVERT_X else raw_dx
            dy_m = -raw_dy if INVERT_Y else raw_dy
            
            print(f"   🔄 [DEBUG] Spiral Leg {leg+1}/{SPIRAL_STEPS} | Stepping: dX={dx_m*1000:.1f}mm, dY={dy_m*1000:.1f}mm")
            self.moveit_backend.jog_cartesian_sdk(dx_m, dy_m, 0.0, speed_mm_s=XY_ALIGN_SPEED)
            
            # Wait for fresh vision after moving
            err_u, _ = self.get_fresh_vision_error()
            if err_u is not None: 
                print(f"   👁️ [DEBUG] Target Reacquired!")
                return True
                
            leg += 1
            
        return False

def main(args=None):
    rclpy.init(args=args)
    node = MasterDisassemblyAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    try:
        node.run_logic()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()