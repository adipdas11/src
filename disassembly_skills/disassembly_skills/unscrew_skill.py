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
import numpy as np 

from disassembly_skills.motion_backend import MotionBackend

class UnscrewSkill(Node):
    def __init__(self):
        super().__init__('unscrew_skill_node')
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        
        self.vision_sub = self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        self.bin_sub = self.create_subscription(String, '/vision/bin_coordinates', self.bin_callback, 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.data_lock = threading.Lock()
        self.latest_screw_targets = []
        self.latest_local_data = None
        self.latest_bin_coords = {} 
        self.ft_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # 👈 [NEW] Cache for Bin Location
        self.cached_bin_pose = None 
        
        # --- CONFIG ---
        self.WORKSPACE_LIMITS = {'min_x': 0.780, 'max_x': 1.100, 'min_y': -0.150, 'max_y': 0.310}
        self.CAMERA_FRAME = 'camera_color_optical_frame'   
        self.PLANNING_FRAME = 'world_world'
        self.ROBOT_BASE_FRAME = 'xarm5_base_link'           
        self.VISUALIZE_TF = True                           
        self.MM_PER_PIXEL = 0.000130   
        self.INVERT_X = False; self.INVERT_Y = False; self.SWAP_XY = True            
        self.TOOL_LENGTH_XARM = 0.24   
        self.APPROACH_HOVER_Z = 0.015  
        self.APPROACH_SPEED = 30.0
        self.TRANSIT_SAFE_LIFT = 0.030 
        self.SUCCESS_RETRACT = 0.050   
        
        # --- [STEP RETRACT CONFIG] ---
        self.STEP_RETRACT_COUNT = 5     
        self.STEP_RETRACT_DIST = 0.001  
        self.STEP_RETRACT_SPEED = 20.0  
        self.STEP_RETRACT_PAUSE = 0.2   
        
        self.POST_GRAB_RETRACT = 0.020 
        self.RETRY_RETRACT = 0.005
        self.BIN_DROP_Z_OFFSET = 0.030 
        self.XY_ALIGN_SPEED = 20.0; self.Z_STEP_DOWN_DIST = -0.002; self.Z_STEP_SPEED = 5.0        
        self.WIGGLE_DIST = 0.002; self.BLIND_DROP_LIMIT = 1      
        self.UNSCREW_RELIEF_STEP = 0.001; self.UNSCREW_SPIKE_THRESH = 0.3; self.UNSCREW_DONE_TIMEOUT = 3.0   
        self.EXTRACTION_WIGGLE_AMP = 0.0015; self.EXTRACTION_WIGGLE_SPEED = 15.0 
        self.APPROACH_COLLISION_THRESH = 3.0; self.SURFACE_CONTACT_THRESH = 2.0; self.WIGGLE_LOCK_THRESH = 3.5        
        self.SPIRAL_STEPS = 20; self.SPIRAL_GAP = 0.002        

    def bin_callback(self, msg):
        try:
            data = json.loads(msg.data.strip("'"))
            with self.data_lock: 
                self.latest_bin_coords = data
        except Exception: pass

    def vision_callback(self, msg):
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
                ft = data.get("force_torque", {}).get("force", {})
                self.ft_force = {'x': ft.get("x", 0.0), 'y': ft.get("y", 0.0), 'z': ft.get("z", 0.0)}
        except Exception: pass 

    def send_tool_cmd(self, val):
        msg = Int8(); msg.data = int(val); self.tool_pub.publish(msg)

    def get_fresh_baselines(self, settle_time=0.2):
        time.sleep(settle_time)
        with self.data_lock: return self.ft_force['x'], self.ft_force['y'], self.ft_force['z']

    def get_fresh_vision_error(self):
        with self.data_lock: self.latest_local_data = None 
        wait_start = time.time()
        while rclpy.ok() and (time.time() - wait_start) < 2.0:
            with self.data_lock:
                if self.latest_local_data:
                    screws, tools = self.latest_local_data.get("screw_heads", []), self.latest_local_data.get("tool_tips", [])
                    if screws and tools:
                        t, s = max(tools, key=lambda x: x.get("conf", 0)), max(screws, key=lambda x: x.get("conf", 0))
                        tx, ty = t.get("contact_point", [None, None]); sx, sy = s.get("center", [None, None])
                        if None not in [tx, ty, sx, sy]: return (sx - tx), (sy - ty)
            time.sleep(0.1)
        return None, None

    def detect_hole_anomaly(self):
        with self.data_lock:
            if self.latest_local_data:
                screws = self.latest_local_data.get("screw_heads", [])
                holes = self.latest_local_data.get("holes", []) or self.latest_local_data.get("empty_holes", [])
                if holes and not screws: return True
        return False

    def get_current_tcp_position(self):
        try:
            t = self.moveit_backend.tf_buffer.lookup_transform(self.PLANNING_FRAME, 'xarm5_link5', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
        except Exception: return None

    def validate_workspace_bounds(self, x, y):
        if (x < self.WORKSPACE_LIMITS['min_x'] or x > self.WORKSPACE_LIMITS['max_x']): return False, "X Out"
        if (y < self.WORKSPACE_LIMITS['min_y'] or y > self.WORKSPACE_LIMITS['max_y']): return False, "Y Out"
        return True, "Safe"

    def get_averaged_target_pose(self, source_pose, samples=5):
        valid_x, valid_y, valid_z = [], [], []
        for _ in range(samples):
            target_stamped = self.moveit_backend.get_transformed_pose(source_pose, self.CAMERA_FRAME, self.PLANNING_FRAME)
            if target_stamped:
                vx, vy, vz = target_stamped.pose.position.x, target_stamped.pose.position.y, target_stamped.pose.position.z
                if abs(vx) > 0.001: valid_x.append(vx); valid_y.append(vy); valid_z.append(vz)
            time.sleep(0.05) 
        return (np.median(valid_x), np.median(valid_y), np.median(valid_z)) if valid_x else None

    def calculate_bin_pose(self, bin_name="bin_1"):
        """
        [FIXED] Implements caching logic. Only looks for the bin if it hasn't been found yet.
        """
        if self.cached_bin_pose is not None:
            return self.cached_bin_pose

        print(f"   🔍 Scanning for {bin_name} coordinates to cache...")
        with self.data_lock: bin_data = self.latest_bin_coords.get(bin_name)
        if not bin_data: return None

        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = bin_data['xyz']
        source_pose.orientation.w = 1.0
        
        res = self.get_averaged_target_pose(source_pose, samples=10) # 10 samples for high accuracy bin pose
        if res:
            self.cached_bin_pose = res
            print(f"   ✅ Bin Cached at: {res}")
            return res
        return None

    def execute_unscrew_command(self, target_id: int, target_label="", interactive=True):
        print(f"\n🛠️ [START] ID: {target_id}"); target_data = None; wait_start = time.time()
        while rclpy.ok() and (time.time() - wait_start) < 5.0:
            with self.data_lock:
                for part in self.latest_screw_targets:
                    if part['id'] == target_id: target_data = copy.deepcopy(part); break
            if target_data: break
            time.sleep(0.5)
        return self._unscrew_state_machine(target_data, interactive) if target_data else False

    def _unscrew_state_machine(self, part_data, interactive=True):
        self.send_tool_cmd(0); time.sleep(0.2); self.send_tool_cmd(3); time.sleep(0.2)
        
        # 👈 [NEW] Pre-fetch bin coordinates at the start if not already cached
        self.calculate_bin_pose("bin_1")

        source_pose = Pose()
        source_pose.position.x, source_pose.position.y, source_pose.position.z = part_data['xyz']
        source_pose.orientation.w = 1.0
        target_xyz = self.get_averaged_target_pose(source_pose)
        if not target_xyz: return False
        world_x, world_y, screw_surface_z = target_xyz
        is_safe, _ = self.validate_workspace_bounds(world_x, world_y)
        if not is_safe:
            if input("🛑 OUT OF BOUNDS! SKIP? (y/n): ").lower() == 'y': return False
        
        if interactive: input(f"👉 GATE 1: Approach? ")
        if not self.guarded_approach(world_x, world_y, screw_surface_z): return False 
        
        if interactive: input(f"👉 GATE 2: Arrived. Start Staircase? ")
        for _ in range(3):
            if not self.staircase_align_and_descend(interactive=interactive): return False 
            if self.verify_seating():
                self.reactive_unscrew(interactive)
                self.dispose_screw("bin_1")
                return True 
            else: self.moveit_backend.jog_cartesian_sdk(0, 0, self.RETRY_RETRACT, speed_mm_s=20.0)
        return False

    def guarded_approach(self, x, y, screw_surface_z):
        self.moveit_backend.jog_cartesian_sdk(0, 0, self.TRANSIT_SAFE_LIFT, speed_mm_s=50.0)
        target_flange_z = screw_surface_z + self.TOOL_LENGTH_XARM + self.APPROACH_HOVER_Z
        success = self.moveit_backend.move_to_pose_robust(x, y, target_flange_z, {}, link_name="xarm5_link5", velocity=0.2)
        if not success:
            w_pose = Pose(); w_pose.position.x, w_pose.position.y, w_pose.position.z, w_pose.orientation.w = x, y, screw_surface_z, 1.0
            b_pose = self.moveit_backend.get_transformed_pose(w_pose, self.PLANNING_FRAME, self.ROBOT_BASE_FRAME)
            if b_pose:
                bz_flange = b_pose.pose.position.z + self.TOOL_LENGTH_XARM + self.APPROACH_HOVER_Z
                success = self.moveit_backend.move_to_absolute_pose_sdk(b_pose.pose.position.x, b_pose.pose.position.y, bz_flange, speed_mm_s=50.0)
        return success

    def staircase_align_and_descend(self, interactive=True):
        step_count, blind_steps_consecutive = 0, 0 
        while rclpy.ok():
            step_count += 1
            if self.detect_hole_anomaly():
                if interactive:
                    if input(f"      🕳️ [VISION] Hole Detected! skip part? (y/n): ").lower() == 'y': return False 
                else: return False

            aligned = False; is_first = True 
            while not aligned and rclpy.ok():
                err_u, err_v = self.get_fresh_vision_error()
                if err_u is None:
                    if blind_steps_consecutive < self.BLIND_DROP_LIMIT:
                        blind_steps_consecutive += 1; aligned = True; break
                    else:
                        self.run_visual_spiral(); is_first = True 
                        if self.get_fresh_vision_error()[0]: blind_steps_consecutive = 0 
                        continue
                else: blind_steps_consecutive = 0 
                    
                dist_px = math.sqrt(err_u**2 + err_v**2)
                if dist_px < 5.0: aligned = True; break
                
                raw_dx = (err_v if self.SWAP_XY else err_u) * self.MM_PER_PIXEL
                raw_dy = (err_u if self.SWAP_XY else err_v) * self.MM_PER_PIXEL
                dx_m, dy_m = (-raw_dx if self.INVERT_X else raw_dx), (-raw_dy if self.INVERT_Y else raw_dy)
                
                max_step = 0.001 if is_first else 0.0005
                move_dist = math.hypot(dx_m, dy_m)
                if move_dist > max_step:
                    scale = max_step / move_dist
                    dx_m *= scale; dy_m *= scale
                self.moveit_backend.jog_cartesian_sdk(dx_m, dy_m, 0.0, speed_mm_s=self.XY_ALIGN_SPEED)
                is_first = False 

            bx, _, bz = self.get_fresh_baselines(0.2)
            self.moveit_backend.jog_cartesian_sdk(0, 0, self.Z_STEP_DOWN_DIST, speed_mm_s=self.Z_STEP_SPEED)
            if abs(self.get_fresh_baselines(0.2)[2] - bz) > self.SURFACE_CONTACT_THRESH: return True
        return False

    def verify_seating(self):
        print("   🔄 Wiggle Check..."); self.send_tool_cmd(1); time.sleep(0.3); self.send_tool_cmd(0); time.sleep(0.5)
        locks = 0
        for dx, dy in [(self.WIGGLE_DIST, 0), (-self.WIGGLE_DIST*2, 0), (self.WIGGLE_DIST, 0), (0, self.WIGGLE_DIST), (0, -self.WIGGLE_DIST*2), (0, self.WIGGLE_DIST)]:
            if hasattr(self.moveit_backend, 'arm'): self.moveit_backend.arm.clean_error(); self.moveit_backend.arm.motion_enable(True); self.moveit_backend.arm.set_state(0)
            bx, by, _ = self.get_fresh_baselines(0.2)
            self.moveit_backend.jog_cartesian_sdk(dx, dy, 0.0, speed_mm_s=self.Z_STEP_SPEED)
            cx, cy, _ = self.get_fresh_baselines(0.1)
            if math.sqrt((cx-bx)**2 + (cy-by)**2) > self.WIGGLE_LOCK_THRESH: locks += 1
        return locks >= 3

    def reactive_unscrew(self, interactive=True):
        bx, by, bz = self.get_fresh_baselines(0.5); self.send_tool_cmd(-1); last_spike = time.time()
        while rclpy.ok():
            spike = self.ft_force['z'] - bz
            if spike > self.UNSCREW_SPIKE_THRESH:
                self.moveit_backend.jog_cartesian_sdk(0, 0, self.UNSCREW_RELIEF_STEP, speed_mm_s=self.Z_STEP_SPEED)
                bx, by, bz = self.get_fresh_baselines(0.1); last_spike = time.time()
            elif (time.time() - last_spike) > self.UNSCREW_DONE_TIMEOUT: break
            time.sleep(0.05)
            
        self.send_tool_cmd(0)
        print("   🤏 [EXTRACT] Initializing GRAB...")
        self.send_tool_cmd(2)
        time.sleep(0.8)

        print(f"   🪜 [EXTRACT] Starting Step Retract...")
        for i in range(self.STEP_RETRACT_COUNT):
            time.sleep(self.STEP_RETRACT_PAUSE)
            self.moveit_backend.jog_cartesian_sdk(0, 0, self.STEP_RETRACT_DIST, speed_mm_s=self.STEP_RETRACT_SPEED)
        
        if interactive: input(f"\n👉 GATE 4: Step Retract Complete. ENTER to Wiggle: ")
        self.perform_post_grab_wiggle()
        self.moveit_backend.jog_cartesian_sdk(0, 0, self.POST_GRAB_RETRACT, speed_mm_s=20.0)

    def perform_post_grab_wiggle(self):
        amp, spd = self.EXTRACTION_WIGGLE_AMP, self.EXTRACTION_WIGGLE_SPEED
        for dx, dy in [(amp, 0), (-amp*2, 0), (amp, 0), (0, amp), (0, -amp*2), (0, amp)]:
            self.moveit_backend.jog_cartesian_sdk(dx, dy, 0, speed_mm_s=spd)

    def dispose_screw(self, bin_name="bin_1"):
        """
        [FIXED] Uses cached bin coordinate to ensure the robot never fails to find the bin.
        """
        bin_xyz = self.calculate_bin_pose(bin_name)
        if not bin_xyz:
            print(f"   ❌ [ERROR] {bin_name} was never found/cached. Releasing locally.")
            self.send_tool_cmd(3); return False
        
        wx, wy, bz = bin_xyz
        target_flange_z = bz + self.TOOL_LENGTH_XARM + self.BIN_DROP_Z_OFFSET
        
        print(f"   🗑️ [PHASE 5] Disposing in {bin_name} at X:{wx:.3f} Y:{wy:.3f}...")
        
        if not self.moveit_backend.move_to_pose_robust(wx, wy, target_flange_z, {}, link_name="xarm5_link5", velocity=0.2):
            # SDK Fallback with cached coordinates
            w_pose = Pose(); w_pose.position.x, w_pose.position.y, w_pose.position.z, w_pose.orientation.w = wx, wy, bz, 1.0
            b_pose = self.moveit_backend.get_transformed_pose(w_pose, self.PLANNING_FRAME, self.ROBOT_BASE_FRAME)
            if b_pose:
                self.moveit_backend.move_to_absolute_pose_sdk(b_pose.pose.position.x, b_pose.pose.position.y, b_pose.pose.position.z + self.TOOL_LENGTH_XARM + self.BIN_DROP_Z_OFFSET, speed_mm_s=50.0)
        
        self.send_tool_cmd(3); time.sleep(0.5); self.moveit_backend.jog_cartesian_sdk(0, 0, self.SUCCESS_RETRACT, speed_mm_s=20.0)
        return True

    def run_visual_spiral(self):
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0]]; leg, gap_mult = 0, 1
        while leg < self.SPIRAL_STEPS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: gap_mult += 1
            dx, dy = dirs[leg%4][0] * self.SPIRAL_GAP * gap_mult, dirs[leg%4][1] * self.SPIRAL_GAP * gap_mult
            if self.SWAP_XY: dx, dy = dy, dx
            if self.INVERT_X: dx = -dx
            if self.INVERT_Y: dy = -dy
            self.moveit_backend.jog_cartesian_sdk(dx, dy, 0.0, speed_mm_s=self.XY_ALIGN_SPEED)
            if self.get_fresh_vision_error()[0]: return True
            leg += 1
        return False

def main(args=None):
    rclpy.init(args=args); node = UnscrewSkill(); executor = MultiThreadedExecutor(); executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True); spin_thread.start()
    try:
        node.moveit_backend.reset_robot()
        while rclpy.ok():
            with node.data_lock: current_targets = copy.deepcopy(node.latest_screw_targets)
            if not current_targets: time.sleep(1.0); continue
            for part in current_targets:
                if not rclpy.ok(): return
                if node.execute_unscrew_command(part['id'], part['label'], interactive=True):
                    if input(f"\n👉 GATE 5: Part Complete. Next? ('q'=quit): ").lower() == 'q': sys.exit(0)
            time.sleep(5.0) 
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()