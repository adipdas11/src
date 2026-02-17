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

class UnscrewSkill(Node):
    """
    Skill Node for adaptive dual-arm disassembly sequences.
    Can be run standalone or imported into a Master Orchestrator.
    """
    def __init__(self):
        super().__init__('unscrew_skill_node')
        
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
        
        self.get_logger().info("✅ Unscrew Skill Node Ready.")

        # =========================================================
        #                    CONFIGURATION
        # =========================================================
        self.CAMERA_FRAME = 'camera_color_optical_frame'   
        self.PLANNING_FRAME = 'world_world'                
        self.VISUALIZE_TF = True                           

        self.MM_PER_PIXEL = 0.000130   
        self.INVERT_X = False          
        self.INVERT_Y = False          
        self.SWAP_XY = True            

        self.TOOL_LENGTH_XARM = 0.24   
        self.APPROACH_HOVER_Z = 0.010  
        self.APPROACH_SPEED = 30.0     

        self.SUCCESS_RETRACT = 0.050   
        self.RETRY_RETRACT = 0.005     
        self.EMERGENCY_RETRACT = 0.010 

        self.XY_ALIGN_SPEED = 25.0     
        self.Z_STEP_DOWN_DIST = -0.002 
        self.Z_STEP_SPEED = 5.0        
        self.WIGGLE_DIST = 0.002       
        self.UNSCREW_RELIEF_STEP = 0.0005 

        self.APPROACH_COLLISION_THRESH = 3.0 
        self.SURFACE_CONTACT_THRESH = 2.0    
        self.WIGGLE_LOCK_THRESH = 3.5        
        self.UNSCREW_SPIKE_THRESH = 1.5      
        self.UNSCREW_DONE_TIMEOUT = 4.0      

        self.SPIRAL_STEPS = 20         
        self.SPIRAL_GAP = 0.002        
        # =========================================================

    def vision_callback(self, msg):
        try:
            clean_data = msg.data.strip("'")
            data = json.loads(clean_data)
            
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
            
            ft = data.get("force_torque", {}).get("force", {})
            local_ft = {'x': ft.get("x", 0.0), 'y': ft.get("y", 0.0), 'z': ft.get("z", 0.0)}
            
            with self.data_lock:
                self.latest_screw_targets = new_targets
                self.latest_local_data = data.get("local_view", {})
                self.ft_force = local_ft
                    
        except Exception as e:
            pass 

    def send_tool_cmd(self, val):
        msg = Int8()
        msg.data = int(val)
        self.tool_pub.publish(msg)

    def publish_debug_tf(self, x, y, z, screw_id):
        if not self.VISUALIZE_TF: return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.PLANNING_FRAME
        t.child_frame_id = f"screw_target_{screw_id}"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def get_fresh_baselines(self, settle_time=0.2):
        time.sleep(settle_time)
        with self.data_lock:
            return self.ft_force['x'], self.ft_force['y'], self.ft_force['z']

    def get_fresh_vision_error(self):
        with self.data_lock:
            self.latest_local_data = None 
            
        wait_start = time.time()
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
            time.sleep(0.1)
        return None, None

    # 👈 [NEW] Direct TF lookup for current position (Fixes the Empty Pose Bug)
    def get_current_tcp_position(self):
        try:
            if self.moveit_backend.tf_buffer.can_transform(self.PLANNING_FRAME, 'xarm5_link5', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.moveit_backend.tf_buffer.lookup_transform(self.PLANNING_FRAME, 'xarm5_link5', rclpy.time.Time())
                return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            return None
        except Exception as e:
            self.get_logger().error(f"TF Lookup Error: {e}")
            return None

    def calculate_target_pose(self, part_data):
        raw_xyz = part_data['xyz']
        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = raw_xyz[0], raw_xyz[1], raw_xyz[2]
        source_pose.orientation.w = 1.0

        target_stamped = None
        for _ in range(5):
            target_stamped = self.moveit_backend.get_transformed_pose(source_pose, self.CAMERA_FRAME, self.PLANNING_FRAME)
            if target_stamped: break
            time.sleep(0.2)

        if not target_stamped: return None
        
        wx = target_stamped.pose.position.x
        wy = target_stamped.pose.position.y
        wz = target_stamped.pose.position.z + self.TOOL_LENGTH_XARM 
        
        self.publish_debug_tf(wx, wy, wz, part_data['id'])
        return (wx, wy, wz)

    def execute_unscrew_command(self, target_id: int, target_label="", interactive=True):
        """
        Public Skill function to be called by Master Orchestrator.
        """
        print("\n" + "="*70)
        print(f"🛠️  [SKILL START] Executing 'unscrew' on ID: {target_id}")
        
        target_data = None
        
        # Wait up to 5 seconds for the vision target to appear
        wait_start = time.time()
        while rclpy.ok() and (time.time() - wait_start) < 5.0:
            with self.data_lock:
                for part in self.latest_screw_targets:
                    if part['id'] == target_id:
                        if target_label and part['label'] != target_label:
                            print(f"⚠️ [WARNING] ID matched ({target_id}), but label mismatched! Expected '{target_label}', saw '{part['label']}'.")
                        target_data = copy.deepcopy(part)
                        break
            if target_data: break
            time.sleep(0.5)
            print("   Waiting for vision target...", end='\r')
                    
        if not target_data:
            print(f"\n❌ [ERROR] Orchestrator requested ID {target_id}, but it is not currently visible in the camera frame.")
            return False
            
        print(f"\n🎯 [LIBRARY] Target acquired. Fetching fresh XYZ for ID {target_id} ({target_data['label']})...")
        
        return self._unscrew_state_machine(target_data, interactive)

    def _unscrew_state_machine(self, part_data, interactive=True):
        target_pose = self.calculate_target_pose(part_data)
        if not target_pose: 
            return False
            
        world_x, world_y, target_z = target_pose
        hover_z = target_z + self.APPROACH_HOVER_Z

        print(f"   🤖 [DEBUG] Global Coords: X={world_x:.4f} Y={world_y:.4f} Z={target_z:.4f}")
        print(f"   🤖 [DEBUG] Hover Z height calculated at: {hover_z:.4f}")
        
        if interactive:
            ui1 = input(f"\n👉 GATE 1: Press ENTER to begin guarded approach to {self.APPROACH_HOVER_Z*1000}mm hover ('s'=skip, 'q'=quit): ")
            if ui1.lower() == 's': return False
            if ui1.lower() == 'q': sys.exit(0)

        if not self.guarded_approach(world_x, world_y, hover_z):
            return False 

        if interactive:
            input(f"\n👉 GATE 2: Arrived at {self.APPROACH_HOVER_Z*1000}mm. Press ENTER to begin autonomous Infinite Staircase: ")

        max_tactile_retries = 3
        for attempt in range(max_tactile_retries):
            print(f"\n   --- Extraction Attempt {attempt + 1}/{max_tactile_retries} ---")
            
            if not self.staircase_align_and_descend():
                return False 

            if self.verify_seating():
                if interactive:
                    input(f"\n👉 GATE 3: Lock verified! Press ENTER to begin reactive unscrewing: ")
                self.reactive_unscrew()
                return True 
            else:
                print(f"   💨 [DEBUG] Wiggle Slipped! Retracting {self.RETRY_RETRACT*1000}mm and automatically retrying staircase...")
                self.moveit_backend.jog_cartesian_sdk(0, 0, self.RETRY_RETRACT, speed_mm_s=20.0)
                
        print("   ❌ [DEBUG] Max tactile retries reached. Aborting skill.")
        return False

    def guarded_approach(self, x, y, hover_z):
        print("   🚀 [PHASE 0] Guarded Approach to Hover Height...")
        
        # 👈 [FIXED] Using direct TF lookup instead of empty Pose transformation
        curr_pos = self.get_current_tcp_position()
        if not curr_pos: 
            print("❌ Failed to get current TCP position. Aborting approach.")
            return False
        safe_z = curr_pos[2]
        
        print(f"      [DEBUG] Planning MoveIt trajectory to XY ({x:.4f}, {y:.4f})...")
        self.moveit_backend.move_to_pose_robust(x, y, safe_z, {}, link_name="xarm5_link5", velocity=0.1)
        bx, by, bz = self.get_fresh_baselines(0.5)
        
        def approach_collision_check():
            cx, cy, cz = self.ft_force['x'], self.ft_force['y'], self.ft_force['z']
            return math.sqrt((cx-bx)**2 + (cy-by)**2 + (cz-bz)**2) > self.APPROACH_COLLISION_THRESH

        z_drop_dist = safe_z - hover_z
        success = self.moveit_backend.move_linear_z_sdk_with_force_stop(
            distance_down_m=z_drop_dist, 
            speed_mm_s=self.APPROACH_SPEED,
            check_force_callback=approach_collision_check
        )
        
        if approach_collision_check():
            print("\n   💥 [FATAL] Unexpected collision during approach! Emergency Retract.")
            self.moveit_backend.jog_cartesian_sdk(0, 0, self.EMERGENCY_RETRACT, speed_mm_s=20.0)
            return False
        return True

    def staircase_align_and_descend(self):
        print("   🔭⬇️ [PHASE 1 & 2] Starting Infinite Staircase...")
        step_count = 0
        
        while rclpy.ok():
            step_count += 1
            print(f"\n      --- Staircase Step {step_count} ---")
            
            aligned = False
            is_first_align_jump = True 
            
            while not aligned and rclpy.ok():
                err_u, err_v = self.get_fresh_vision_error()
                
                if err_u is None:
                    print("         [DEBUG] Vision lost. Spiraling...")
                    self.run_visual_spiral()
                    is_first_align_jump = True 
                    continue
                    
                dist_px = math.sqrt(err_u**2 + err_v**2)
                print(f"         [VISION] U:{err_u:.1f}px V:{err_v:.1f}px | Dist: {dist_px:.1f}px")
                
                if dist_px < 5.0:
                    print("         ✅ Perfectly Aligned.")
                    aligned = True
                    break
                
                raw_dx = err_v * self.MM_PER_PIXEL if self.SWAP_XY else err_u * self.MM_PER_PIXEL
                raw_dy = err_u * self.MM_PER_PIXEL if self.SWAP_XY else err_v * self.MM_PER_PIXEL
                dx_m = -raw_dx if self.INVERT_X else raw_dx
                dy_m = -raw_dy if self.INVERT_Y else raw_dy
                
                max_step = 0.005 if is_first_align_jump else 0.0005
                move_dist = math.hypot(dx_m, dy_m)
                
                if move_dist > max_step:
                    scale = max_step / move_dist
                    dx_m *= scale
                    dy_m *= scale
                    print(f"         [DEBUG] Move capped to {max_step*1000}mm.")
                
                print(f"         [ACTION] Jogging -> dX: {dx_m*1000:.2f}mm, dY: {dy_m*1000:.2f}mm")
                self.moveit_backend.jog_cartesian_sdk(dx_m, dy_m, 0.0, speed_mm_s=self.XY_ALIGN_SPEED)
                is_first_align_jump = False 

            bx, by, bz = self.get_fresh_baselines(0.2)
            self.moveit_backend.jog_cartesian_sdk(0, 0, self.Z_STEP_DOWN_DIST, speed_mm_s=self.Z_STEP_SPEED)
            cx, cy, cz = self.get_fresh_baselines(0.2)
            spike = abs(cz - bz)
            
            print(f"      [DEBUG] Z-Drop {-self.Z_STEP_DOWN_DIST*1000}mm | Base_Z: {bz:.2f}N | Curr_Z: {cz:.2f}N | Spike: {spike:.2f}N")
            
            if spike > self.SURFACE_CONTACT_THRESH:
                print(f"   ✅ [DEBUG] Surface Contact Detected! (Spike > {self.SURFACE_CONTACT_THRESH}N)")
                return True
                
        return False

    def verify_seating(self):
        print("   🔄 [PHASE 3] Pre-seating twist & Tactile Wiggle Verification...")
        
        self.send_tool_cmd(1); time.sleep(0.3)
        self.send_tool_cmd(0); time.sleep(0.5)
        
        moves = [
            ('X+', self.WIGGLE_DIST, 0.0), 
            ('X-', -self.WIGGLE_DIST, 0.0),
            ('Y+', 0.0, self.WIGGLE_DIST), 
            ('Y-', 0.0, -self.WIGGLE_DIST)
        ]
        
        locks = 0
        for name, dx, dy in moves:
            bx, by, bz = self.get_fresh_baselines(0.2)
            self.moveit_backend.jog_cartesian_sdk(dx, dy, 0.0, speed_mm_s=self.Z_STEP_SPEED)
            cx, cy, cz = self.get_fresh_baselines(0.2)
            force_diff = math.sqrt((cx-bx)**2 + (cy-by)**2)
            
            if force_diff > self.WIGGLE_LOCK_THRESH:
                locks += 1
                print(f"      ✔️ [LOCKED] {name} | Spike: {force_diff:.2f}N")
            else:
                print(f"      ❌ [SLIPPED] {name} | Spike: {force_diff:.2f}N")
                
            self.moveit_backend.jog_cartesian_sdk(-dx, -dy, 0.0, speed_mm_s=self.Z_STEP_SPEED)
            
        total_lock = locks >= 3
        print(f"   📊 [DEBUG] Wiggle Summary: {locks}/4 locked. Result: {'PASS' if total_lock else 'FAIL'}")
        return total_lock

    def reactive_unscrew(self):
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
            
            if spike > self.UNSCREW_SPIKE_THRESH:
                print(f"\n      📈 [ACTION] Z-Spike ({spike:.2f}N) > {self.UNSCREW_SPIKE_THRESH}N limit. Stepping up {self.UNSCREW_RELIEF_STEP*1000}mm to relieve pressure...")
                self.moveit_backend.jog_cartesian_sdk(0, 0, self.UNSCREW_RELIEF_STEP, speed_mm_s=self.Z_STEP_SPEED)
                total_lift_mm += (self.UNSCREW_RELIEF_STEP * 1000)
                bx, by, bz = self.get_fresh_baselines(0.1)
                last_spike_time = time.time()
                
            elif (time.time() - last_spike_time) > self.UNSCREW_DONE_TIMEOUT:
                print(f"\n   🏆 [SUCCESS] No spikes for {self.UNSCREW_DONE_TIMEOUT}s. Threads disengaged. Total Lift: {total_lift_mm:.1f}mm.")
                break
                
            time.sleep(0.05)
            
        self.send_tool_cmd(0)
        print(f"   ⬆️ [ACTION] Retracting {self.SUCCESS_RETRACT*1000}mm to safe height...")
        self.moveit_backend.jog_cartesian_sdk(0, 0, self.SUCCESS_RETRACT, speed_mm_s=20.0)

    def run_visual_spiral(self):
        print(f"\n🌀 [DEBUG] Visual Spiral Recovery Initiated ({self.SPIRAL_STEPS} steps, {self.SPIRAL_GAP*1000}mm gap)...")
        
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0]]
        leg = 0
        current_gap_multiplier = 1
        
        while leg < self.SPIRAL_STEPS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: 
                current_gap_multiplier += 1
                
            step_dist = self.SPIRAL_GAP * current_gap_multiplier
            raw_dx = dirs[leg%4][0] * step_dist
            raw_dy = dirs[leg%4][1] * step_dist
            
            if self.SWAP_XY: raw_dx, raw_dy = raw_dy, raw_dx
            dx_m = -raw_dx if self.INVERT_X else raw_dx
            dy_m = -raw_dy if self.INVERT_Y else raw_dy
            
            print(f"   🔄 [DEBUG] Spiral Leg {leg+1}/{self.SPIRAL_STEPS} | Stepping: dX={dx_m*1000:.1f}mm, dY={dy_m*1000:.1f}mm")
            self.moveit_backend.jog_cartesian_sdk(dx_m, dy_m, 0.0, speed_mm_s=self.XY_ALIGN_SPEED)
            
            err_u, _ = self.get_fresh_vision_error()
            if err_u is not None: 
                print(f"   👁️ [DEBUG] Target Reacquired!")
                return True
            leg += 1
            
        return False

# =====================================================================
# STANDALONE EXECUTION BLOCK (When run directly via 'ros2 run')
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    skill_node = UnscrewSkill()
    
    executor = MultiThreadedExecutor()
    executor.add_node(skill_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # For standalone testing, we just simulate the VLA calling it for every screw
        skill_node.moveit_backend.reset_robot()
        
        print("\n" + "="*70)
        print("   🔍 WAITING FOR VISION DATA TO START STANDALONE TEST...")
        print("="*70)
        
        while rclpy.ok():
            current_targets = []
            with skill_node.data_lock:
                current_targets = copy.deepcopy(skill_node.latest_screw_targets)
            
            if not current_targets:
                time.sleep(1.0); continue

            print(f"\n🔎 FOUND {len(current_targets)} SCREW ZONES.")
            
            for part_data in current_targets:
                if not rclpy.ok(): return
                success = skill_node.execute_unscrew_command(target_id=part_data['id'], target_label=part_data['label'], interactive=True)
                if success:
                    print(f"✅ Successfully processed {part_data['id']}.")
                else:
                    print(f"⏭️ Failed or aborted {part_data['id']}. Moving to next target.")
                
                ui_next = input(f"\n👉 GATE 4: Press ENTER to move to the next part in batch ('q'=quit): ")
                if ui_next.lower() == 'q': sys.exit(0)
                
            print("\n✅ Batch Complete. Waiting 5s for new vision data...")
            time.sleep(5.0) 

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected.")
    finally:
        skill_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()