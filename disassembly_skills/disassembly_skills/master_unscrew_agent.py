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
from disassembly_skills.velocity_backend import VelocityBackend

# ================= COARSE TARGETING CONFIG =================
TOOL_LENGTH_XARM = 0.24       
SAFETY_BOUNDARY_Z = 0.03      
CAMERA_FRAME = 'camera_color_optical_frame'   
 
VISUALIZE_TF = True  

HOME_JOINTS = {
    'xarm5_joint1': 0.0, 'xarm5_joint2': 0.0, 'xarm5_joint3': -1.57,
    'xarm5_joint4': 1.57, 'xarm5_joint5': 0.0
}

# ================= FINE MANIPULATION CONFIG =================
VEL_X_GAIN = 1.0 
VEL_Y_GAIN = 1.0 

# ALIGNMENT SETTINGS
MOVE_SPEED = 0.028      
MIN_PULSE_XY = 0.02     
MAX_PULSE_XY = 0.1     
PULSE_GAIN_XY = 0.0018 
ALIGNMENT_TOL = 45     

# DESCENT SETTINGS
Z_SPEED_DOWN = -0.028 
Z_PULSE_DUR = 0.1     
Z_FORCE_OFFSET = 7.0  
SETTLE_TIME = 0.50      

# VISUAL RECOVERY
SPIRAL_SPEED = 0.025
SPIRAL_BASE_TIME = 0.1
SPIRAL_MAX_LEGS = 20      
VISION_LOST_TIMEOUT = 0.8

# TACTILE / PULSED WIGGLE
WIGGLE_STEP_SPEED = 0.028       
WIGGLE_STEP_TIME = 0.1          
WIGGLE_MAX_STEPS = 3            
WIGGLE_FORCE_THRESH = 3.0       
WIGGLE_SETTLE_TIME = 0.5

# CONTINUOUS LEAD COMPENSATION
Z_MAX_PRESSURE_LIMIT = 0.5      # 👈 More sensitive (Start lifting at 0.5N)
Z_COMP_GAIN = 0.050             # 👈 More aggressive (More speed per Newton)
Z_MIN_LIFT_SPEED = 0.028        # 👈 Lowered to match your hardware's minimum viable speed
Z_MAX_LIFT_SPEED = 0.080    
UNSCREW_STABILITY_TOL = 0.5     
Z_BITE_THRESHOLD = 1.2  
PROGRESSION_TIMEOUT = 5.0 
MIN_PROGRESSION_MM = 1.5        

# SAFETY
XY_FORCE_LIMIT = 2.5 
# =========================================================

class MasterDisassemblyAgent(Node):
    def __init__(self):
        super().__init__('master_disassembly_agent')
        
        # Initialize both backends
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        self.vel_backend = VelocityBackend(self)
        
        # ROS Interfaces
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Shared State Variables
        self.data_lock = threading.Lock()
        self.latest_screw_targets = []
        self.latest_local_data = None
        
        # Force/Torque State
        self.ft_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.baseline_z = self.baseline_x = self.baseline_y = None
        self.threshold_z = None
        self.collision_detected = False
        self.last_vision_time = time.time()
        
        self.get_logger().info("✅ Master Disassembly Agent Ready.")

    # ----------------- VISION & STATE UPDATES -----------------
    def vision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            
            # 1. Parse Global Targets (Code 1)
            objects = data.get("global_view", {}).get("objects", [])
            new_targets = []
            for obj in objects:
                label = obj.get("label", "")
                if "Screw_Zone" in label and "xyz" in obj:
                    new_targets.append({
                        "id": obj.get("id"),
                        "label": label,
                        "xyz": obj.get("xyz"), 
                        "confidence": obj.get("confidence", 0.0)
                    })
            
            # 2. Parse Local View & FT (Code 2)
            ft = data.get("force_torque", {}).get("force", {})
            local_ft = {'x': ft.get("x", 0.0), 'y': ft.get("y", 0.0), 'z': ft.get("z", 0.0)}
            
            with self.data_lock:
                self.latest_screw_targets = new_targets
                self.latest_local_data = data.get("local_view", {})
                self.ft_force = local_ft
                
            # Check collisions dynamically
            if self.threshold_z is not None and self.ft_force['z'] < self.threshold_z:
                self.collision_detected = True
            if self.baseline_x is not None:
                if abs(self.ft_force['x'] - self.baseline_x) > XY_FORCE_LIMIT or abs(self.ft_force['y'] - self.baseline_y) > XY_FORCE_LIMIT:
                    self.collision_detected = True
                    
        except Exception as e:
            self.get_logger().error(f"Error in vision callback: {e}")

    def send_tool_cmd(self, val):
        msg = Int8()
        msg.data = int(val)
        self.tool_pub.publish(msg)

    def publish_debug_tf(self, x, y, z, screw_id):
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

    def get_error_pixels(self):
        with self.data_lock:
            if not self.latest_local_data: return None, None
            screws = self.latest_local_data.get("screw_heads", [])
            tools = self.latest_local_data.get("tool_tips", [])
            
        if not screws or not tools: return None, None
        t = max(tools, key=lambda x: x.get("conf", 0))
        s = max(screws, key=lambda x: x.get("conf", 0))
        tx, ty = t.get("contact_point", [None, None])
        sx, sy = s.get("center", [None, None])
        
        if None in [tx, ty, sx, sy]: return None, None
        return (sx - tx), (sy - ty)

    # ----------------- MAIN PIPELINE LOOP -----------------
    def run_logic(self):
        print("⏳ Checking connection to MoveGroup servers...")
        if not self.moveit_backend._action_client.wait_for_server(timeout_sec=3.0):
            print("❌ ERROR: xArm5 MoveGroup Action Server NOT found!")
            return
        
        self.moveit_backend.reset_robot()

        while rclpy.ok():
            current_targets = []
            with self.data_lock:
                current_targets = copy.deepcopy(self.latest_screw_targets)
            
            if not current_targets:
                time.sleep(1.0)
                continue

            print(f"\n🔎 FOUND {len(current_targets)} SCREW ZONES. PROCESSING...")
            
            for part_data in current_targets:
                if not rclpy.ok(): return
                
                # 1. Calculate Pose
                target_pose = self.calculate_target_pose(part_data)
                if not target_pose: continue
                world_x, world_y, final_z = target_pose
                
                # GATE 1: Go to estimated pose
                print("\n" + "="*60)
                print(f"🔩 TARGET: {part_data['label']} (ID: {part_data['id']})")
                print(f"   🤖 Target Coord: X={world_x:.4f} Y={world_y:.4f} Z={final_z:.4f}")
                
                ui1 = input(f"\n👉 GATE 1: Press ENTER to move to estimated pose of ID {part_data['id']} ('s'=skip, 'q'=quit): ")
                if ui1.lower() == 's': continue
                if ui1.lower() == 'q': sys.exit(0)

                print("   🚀 Moving to coarse position...")
                success = self.moveit_backend.move_to_pose_robust(
                    world_x, world_y, final_z, {}, 
                    link_name="xarm5_link5", frame_id=PLANNING_FRAME, velocity=0.1
                )
                
                if not success:
                    print("   ❌ Coarse move failed! Skipping...")
                    continue

                # GATE 2: Start Alignment
                ui2 = input(f"\n👉 GATE 2: Arrived at ID {part_data['id']}. Press ENTER to start visual alignment & insertion... ('s'=skip): ")
                if ui2.lower() == 's': continue
                
                # 2. Run Fine Manipulation Sub-Function
                self.unscrew_routine(part_data['id'])
                
                # GATE 4: Next Part
                ui4 = input(f"\n👉 GATE 4: Press ENTER to go to the next part location... ('q'=quit): ")
                if ui4.lower() == 'q': sys.exit(0)
                
            print("\n✅ Batch Complete. Waiting 5s for new vision data...")
            time.sleep(5.0) 

    def calculate_target_pose(self, part_data):
        raw_xyz = part_data['xyz']
        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = raw_xyz[0], raw_xyz[1], raw_xyz[2]
        source_pose.orientation.w = 1.0

        target_stamped = None
        for _ in range(5):
            target_stamped = self.moveit_backend.get_transformed_pose(source_pose, CAMERA_FRAME, PLANNING_FRAME)
            if target_stamped: break
            time.sleep(0.2)

        if not target_stamped:
            print(f"❌ TF Error: Could not transform {CAMERA_FRAME} -> {PLANNING_FRAME}")
            return None

        world_x, world_y = target_stamped.pose.position.x, target_stamped.pose.position.y
        base_z  = target_stamped.pose.position.z
        final_z = base_z + TOOL_LENGTH_XARM + SAFETY_BOUNDARY_Z
        
        self.publish_debug_tf(world_x, world_y, final_z, part_data['id'])
        return (world_x, world_y, final_z)

    # ----------------- FINE MANIPULATION & UNSCREW -----------------
    def unscrew_routine(self, part_id):
        print(f"\n🛠️ [ROUTINE] Starting fine manipulation for ID {part_id}...")
        
        # Settle to get accurate baseline force
        time.sleep(1.0) 
        with self.data_lock:
            self.baseline_z, self.baseline_x, self.baseline_y = self.ft_force['z'], self.ft_force['x'], self.ft_force['y']
        self.threshold_z = self.baseline_z - Z_FORCE_OFFSET
        self.collision_detected = False
        self.last_vision_time = time.time()
        
        print(f"   ✅ Hardware Ready. Baseline Z-Force: {self.baseline_z:.1f}N")
        
        if not self.vel_backend.start_velocity_mode(): 
            print("   ❌ Failed to start velocity mode.")
            return

        try:
            while rclpy.ok():
                # 1. Collision / Insertion Check
                if self.collision_detected:
                    self.vel_backend.send_velocity(0,0,0)
                    print(f"\n💥 [DEBUG] Z-Force threshold triggered. Verifying insertion...")
                    
                    if self.verify_insertion():
                        # Insertion verified, unscrewing complete
                        self.post_extraction_retract()
                        print("\n🏆 TASK COMPLETE for this screw.")
                        break # Successfully done, break to move to next part
                    else:
                        self.retract_and_retry()
                        continue

                # 2. Visual Alignment Loop
                err_u, err_v = self.get_error_pixels()
                if err_u is None:
                    if (time.time() - self.last_vision_time) > VISION_LOST_TIMEOUT:
                        if not self.run_visual_spiral():
                            # --- CRITICAL FIX HERE ---
                            # Ask the user instead of automatically breaking and skipping the part
                            self.vel_backend.send_velocity(0,0,0) # Stop movement
                            print("\n❌ [ERROR] Visual Spiral Recovery failed. Target still lost.")
                            ui_retry = input("   👉 Press ENTER to keep trying (wait for vision), or 's' to skip to next part: ")
                            if ui_retry.lower() == 's':
                                print("   ⏭️ Skipping part...")
                                break
                            else:
                                print("   🔄 Continuing to wait for vision on current part...")
                                self.last_vision_time = time.time() # Reset timer so it doesn't instantly spiral again
                                continue
                        else:
                            # Spiral found it!
                            self.last_vision_time = time.time()
                        continue
                    else: 
                        time.sleep(0.05); continue
                
                self.last_vision_time = time.time()
                dist_px = math.sqrt(err_u**2 + err_v**2)

                # 3. Servo XY or Descend Z
                if dist_px > ALIGNMENT_TOL:
                    pulse = max(MIN_PULSE_XY, min(MAX_PULSE_XY, dist_px * PULSE_GAIN_XY))
                    vx_cmd = (MOVE_SPEED if err_v > 0 else -MOVE_SPEED) * VEL_X_GAIN
                    vy_cmd = (MOVE_SPEED if err_u > 0 else -MOVE_SPEED) * VEL_Y_GAIN
                    print(f"   🔭 [ALIGN] Dist: {dist_px:.1f}px. Pulsing XY...", end='\r')
                    self.vel_backend.send_velocity(vx=vx_cmd, vy=vy_cmd)
                    time.sleep(pulse)
                else:
                    print(f"   ⬇️ [DESCEND] Aligned (Dist: {dist_px:.1f}px). Pulsing Z down...", end='\r')
                    self.vel_backend.send_velocity(vz=Z_SPEED_DOWN)
                    time.sleep(Z_PULSE_DUR)
                
                self.vel_backend.send_velocity(0,0,0)
                time.sleep(SETTLE_TIME)
                
        finally:
            self.vel_backend.stop_velocity_mode()

    def verify_insertion(self):
        print(f"\n🔍 [DEBUG] Starting Double 4-Way Wiggle (Pulsed Guarded Move)...")
        
        # --- NEW: SOLUTION 2 - PRE-WIGGLE ROTATIONAL SEATING (TWIST & DROP) ---
        print("   🔄 [ACTION] Pre-seating tool bit into screw slots...")
        # Apply a gentle downward pressure to help the bit drop in
        self.vel_backend.send_velocity(vz=-0.02)  
        # Pulse the screwdriver motor forward to find the slot
        self.send_tool_cmd(1)                     
        time.sleep(0.3)                           
        # Stop the motor but keep pushing down for a moment to bottom out
        self.send_tool_cmd(0)                     
        time.sleep(0.3)                           
        self.vel_backend.send_velocity(0, 0, 0)   
        time.sleep(WIGGLE_SETTLE_TIME)
        
        # Update the baseline_z because the tool likely dropped deeper into the screw
        with self.data_lock:
            self.baseline_z = self.ft_force['z'] 
        print(f"   ✅ Tool seated. New Baseline Z: {self.baseline_z:.1f}N")
        # -----------------------------------------------------------------------

        # Run 2 passes
        for attempt in range(1, 3):
            print(f"   🔄 Wiggle Pass {attempt}/2...")
            
            # Define the axis and direction multiplier
            moves = [
                ('+X', 'vx', 1, 'x'), 
                ('-X', 'vx', -1, 'x'),
                ('+Y', 'vy', 1, 'y'), 
                ('-Y', 'vy', -1, 'y')
            ]
            
            locks = {'+X': False, '-X': False, '+Y': False, '-Y': False}

            for name, cmd_axis, direction, force_axis in moves:
                # 1. Take a FRESH baseline right before this specific direction
                time.sleep(0.1) 
                current_baseline = self.ft_force[force_axis]
                
                max_deviation = 0.0
                peak_raw_force = current_baseline 
                steps_taken = 0
                hit_wall = False
                
                # 2. STEPPING LOOP (Pulse, Stop, Check)
                for step in range(WIGGLE_MAX_STEPS):
                    # Pulse forward
                    self.vel_backend.send_velocity(**{cmd_axis: direction * WIGGLE_STEP_SPEED})
                    time.sleep(WIGGLE_STEP_TIME)
                    
                    # Stop and let forces settle
                    self.vel_backend.send_velocity(0,0,0)
                    time.sleep(WIGGLE_SETTLE_TIME) 
                    steps_taken += 1
                    
                    # Read the stabilized force after the step
                    current_raw = self.ft_force[force_axis]
                    current_deviation = abs(current_raw - current_baseline)
                    
                    if current_deviation > max_deviation:
                        max_deviation = current_deviation
                        peak_raw_force = current_raw
                        
                    # If we hit the threshold, STOP stepping!
                    if max_deviation > WIGGLE_FORCE_THRESH:
                        hit_wall = True
                        break 
                
                # 3. Evaluate the lock
                if hit_wall:
                    locks[name] = True
                    print(f"      ✔️ {name} Locked  (Raw: {peak_raw_force:.1f}N | Base: {current_baseline:.1f}N | Diff: {max_deviation:.1f}N | Steps: {steps_taken})")
                else:
                    print(f"      ❌ {name} Slipped (Raw: {peak_raw_force:.1f}N | Base: {current_baseline:.1f}N | Diff: {max_deviation:.1f}N | Steps: {steps_taken})")
                    
                if self.ft_force['z'] > (self.baseline_z - 0.5):
                    print(f"      ⚠️ Z-Contact lost during {name} wiggle.")
                    
                # 4. Return to center (reverse exactly the number of steps we took)
                self.vel_backend.send_velocity(**{cmd_axis: -direction * WIGGLE_STEP_SPEED})
                time.sleep(steps_taken * WIGGLE_STEP_TIME)
                self.vel_backend.send_velocity(0,0,0)
                time.sleep(0.1)
            
            # Hold for 1 second between pass 1 and 2
            if attempt == 1:
                print("   ⏳ Holding for 1.0s before second pass...")
                time.sleep(1.0)

        # Make decision based on the final pass
        if all(locks.values()):
            print("⚓ [DEBUG] 4-Way Lock Confirmed. Tool is seated inside screw head.")
            
            # GATE 3: Start Unscrewing
            ui3 = input(f"\n👉 GATE 3: Tool head inside screw head. Press ENTER to start unscrewing... ('c'=cancel): ")
            if ui3.lower() == 'c': return False
            
            return self.perform_unscrew_with_compensation()
        else:
            print("💨 [DEBUG] Slip detected after second pass. Not fully seated in all directions.")
            return False

    def perform_unscrew_with_compensation(self):
        print(f"\n🔄 [ACTION] Verifying Tool Seating (Z-Spike)...")
        initial_z_force = self.ft_force['z']
        total_lift_mm = 0.0 
        
        self.send_tool_cmd(-1) 
        time.sleep(1.5) 
        
        current_z_force = self.ft_force['z']
        force_change = current_z_force - initial_z_force
        
        if force_change > -Z_BITE_THRESHOLD:
            print(f"❌ [VERIFY] No Z-Spike detected ({force_change:.2f}N). False Positive Lock.")
            self.send_tool_cmd(0)
            return False

        print(f"✅ [VERIFY] Bite confirmed ({force_change:.2f}N). Streaming Continuous Compensation...")
        
        z_history = []
        start_t = time.time()
        engaged_z_force = current_z_force 
        loop_rate = 0.02 
        
        try:
            while rclpy.ok():
                curr_force = self.ft_force['z']
                z_history.append(curr_force)
                if len(z_history) > 30: z_history.pop(0)

                if (time.time() - start_t) > PROGRESSION_TIMEOUT and total_lift_mm < MIN_PROGRESSION_MM:
                    print(f"\n❌ [VERIFY] Extraction Failed! Pressure exists but no height gain ({total_lift_mm:.1f}mm).")
                    return False

                pressure_threshold = engaged_z_force - Z_MAX_PRESSURE_LIMIT
                vz_cmd = 0.0
                
                if curr_force < pressure_threshold:
                    excess_force = pressure_threshold - curr_force
                    raw_vz = excess_force * Z_COMP_GAIN
                    vz_cmd = max(Z_MIN_LIFT_SPEED, min(Z_MAX_LIFT_SPEED, raw_vz))
                    
                    print(f"🌊 [STREAM] Excess: {excess_force:.1f}N | Cmd_Vz: {vz_cmd:.3f}m/s | Lift: {total_lift_mm:.1f}mm", end='\r')
                    total_lift_mm += (vz_cmd * 1000) * loop_rate
                
                self.vel_backend.send_velocity(vz=vz_cmd)

                if len(z_history) >= 30:
                    variance = max(z_history) - min(z_history)
                    if variance < UNSCREW_STABILITY_TOL and (time.time() - start_t) > PROGRESSION_TIMEOUT:
                        print(f"\n🏆 [DONE] Extraction confirmed ({total_lift_mm:.1f}mm lift). Z-Force Stable.")
                        return True

                if (time.time() - start_t) > 25.0: return False
                time.sleep(loop_rate)
        finally:
            self.send_tool_cmd(0)
            self.vel_backend.send_velocity(0,0,0) 

    # ----------------- RECOVERY & RETRACTION -----------------
    def run_visual_spiral(self):
        print(f"\n🌀 [DEBUG] Target Lost. Visual Spiral Recovery...")
        
        self.collision_detected = False 
        
        # 👈 CHANGED: Sequence now starts with [1, 0] to move BACKWARD first.
        # Pattern: Backward, Sideways, Forward, Sideways
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        
        leg, dur = 0, SPIRAL_BASE_TIME
        
        while leg < SPIRAL_MAX_LEGS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: dur += SPIRAL_BASE_TIME
            
            print(f"   🔄 Spiral Leg {leg+1}/{SPIRAL_MAX_LEGS}...", end='\r')
            
            self.vel_backend.send_velocity(vx=dirs[leg%4][0]*SPIRAL_SPEED, vy=dirs[leg%4][1]*SPIRAL_SPEED)
            time.sleep(dur)
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.5)
            
            if self.get_error_pixels()[0] is not None: 
                print(f"\n   👁️ [DEBUG] Target Reacquired at leg {leg+1}!")
                return True
                
            if self.collision_detected: 
                print(f"\n   ⚠️ [DEBUG] Spiral aborted! Force spike detected (Collision).")
                return False
                
            leg += 1
            
        print("\n   ❌ [DEBUG] Spiral finished all legs. Target not found.")
        return False

    def retract_and_retry(self):
        print(f"\n⬆️  [DEBUG] Slip detected. Safety Retract (1.0cm)...")
        self.vel_backend.send_velocity(0,0,0); time.sleep(0.1)
        r_start = time.time()
        while (time.time() - r_start) < 0.25:
            self.vel_backend.send_velocity(vz=0.04); time.sleep(0.05)
        self.vel_backend.send_velocity(0,0,0); time.sleep(0.5)
        self.collision_detected = False 

    def post_extraction_retract(self):
        target_force = self.baseline_z - 0.5 
        print(f"\n⬆️ [ACTION] Clearing tool. Retracting until Z-Force hits {target_force:.1f}N...")
        
        start_t = time.time()
        while rclpy.ok() and (time.time() - start_t) < 8.0:
            if self.ft_force['z'] >= target_force:
                print(f"✅ [DONE] Tool is completely clear (Force: {self.ft_force['z']:.1f}N).")
                break
                
            self.vel_backend.send_velocity(vz=0.035)
            time.sleep(0.05); self.vel_backend.send_velocity(0,0,0); time.sleep(0.05)
            
        for _ in range(10):
            self.vel_backend.send_velocity(0,0,0); time.sleep(0.05)


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
        node.vel_backend.stop_velocity_mode()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()