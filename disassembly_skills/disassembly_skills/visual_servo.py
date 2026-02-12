#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json, time, threading, math
from disassembly_skills.motion_backend import MotionBackend
from disassembly_skills.velocity_backend import VelocityBackend

# ================= CONFIGURATION =================
START_X, START_Y, START_Z = 0.93449, 0.45711, 1.1877

# AXIS GAINS (Flipped if robot diverges)
VEL_X_GAIN = 1.0 
VEL_Y_GAIN = 1.0 

# ALIGNMENT SETTINGS (TUNED)
MOVE_SPEED = 0.035      
MIN_PULSE_XY = 0.02     
MAX_PULSE_XY = 0.1     
PULSE_GAIN_XY = 0.0018 
ALIGNMENT_TOL = 45     

# DESCENT SETTINGS
Z_SPEED_DOWN = -0.030 
Z_PULSE_DUR = 0.1     
Z_FORCE_OFFSET = 1.0  
SETTLE_TIME = 0.30      

# VISUAL RECOVERY
SPIRAL_SPEED = 0.025
SPIRAL_BASE_TIME = 0.2
SPIRAL_MAX_LEGS = 5
VISION_LOST_TIMEOUT = 0.8

# TACTILE SEARCH (Peck Search)
TACTILE_TIMEOUT = 2.0     
TACTILE_SPEED = 0.035     

# WIGGLE VERIFICATION
WIGGLE_SPEED = 0.025       
WIGGLE_TIME = 0.2
WIGGLE_FORCE_THRESH = 1.5 

# SAFETY
XY_FORCE_LIMIT = 2.5 
# =================================================

class VisualServo(Node):
    def __init__(self):
        super().__init__('visual_servo_master_final')
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        self.vel_backend = VelocityBackend(self)
        self.sub = self.create_subscription(String, '/vision/agent_state', self.vision_cb, 10)
        
        self.latest_data = None
        self.ft_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.baseline_z = self.baseline_x = self.baseline_y = None
        self.threshold_z = None
        self.collision_detected = False
        self.last_vision_time = time.time()

    def vision_cb(self, msg):
        try: 
            self.latest_data = json.loads(msg.data)
            ft = self.latest_data.get("force_torque", {}).get("force", {})
            self.ft_force['x'], self.ft_force['y'], self.ft_force['z'] = ft.get("x", 0.0), ft.get("y", 0.0), ft.get("z", 0.0)
            
            if self.threshold_z is not None and self.ft_force['z'] < self.threshold_z:
                self.collision_detected = True
            
            if self.baseline_x is not None:
                if abs(self.ft_force['x'] - self.baseline_x) > XY_FORCE_LIMIT or abs(self.ft_force['y'] - self.baseline_y) > XY_FORCE_LIMIT:
                    self.collision_detected = True
        except: pass

    def get_error_pixels(self):
        if not self.latest_data: return None, None
        local = self.latest_data.get("local_view", {})
        screws, tools = local.get("screw_heads", []), local.get("tool_tips", [])
        if not screws or not tools: return None, None
        t, s = max(tools, key=lambda x: x.get("conf", 0)), max(screws, key=lambda x: x.get("conf", 0))
        tx, ty = t.get("contact_point", [None, None])
        sx, sy = s.get("center", [None, None])
        if None in [tx, ty, sx, sy]: return None, None
        return (sx - tx), (sy - ty)

    def verify_insertion(self):
        print(f"\n🔍 [DEBUG] Verifying Insertion (Wiggle Test)...")
        bx, by = self.ft_force['x'], self.ft_force['y']
        
        # We test both X and Y because we don't know which axis will lock first
        for axis, speed_cmd in [('vx', WIGGLE_SPEED), ('vy', WIGGLE_SPEED)]:
            print(f"   -> Wiggling {axis}...")
            self.vel_backend.send_velocity(**{axis: speed_cmd})
            time.sleep(WIGGLE_TIME)
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.2)
            
            dx = abs(self.ft_force['x'] - bx)
            dy = abs(self.ft_force['y'] - by)
            print(f"   [DEBUG] Delta Fx: {dx:.2f}N | Delta Fy: {dy:.2f}N")
            
            if dx > WIGGLE_FORCE_THRESH or dy > WIGGLE_FORCE_THRESH:
                print(f"   ✅ SUCCESS: Tool Locked in Hole!")
                return True
            
            # Reset
            self.vel_backend.send_velocity(**{axis: -speed_cmd})
            time.sleep(WIGGLE_TIME)
            self.vel_backend.send_velocity(0,0,0)
            
        print(f"   ❌ FAILURE: Tool is sliding on surface.")
        return False

    def run_tactile_search(self):
        """Timed peck search: Lift -> Shift -> Push."""
        try:
            print(f"\n🤏 [DEBUG] Starting Tactile Search (2.0s Timeout)...")
            start_t = time.time()
            # Start Backward (as requested)
            dirs = [[-1, 0], [0, -1], [1, 0], [0, 1]] 
            idx = 0
            
            while (time.time() - start_t) < TACTILE_TIMEOUT and rclpy.ok():
                elapsed = time.time() - start_t
                print(f"   [TACTILE] Peck {idx+1} | {TACTILE_TIMEOUT - elapsed:.1f}s left...", end='\r')
                
                # 1. Micro-Lift
                self.vel_backend.send_velocity(vz=0.02)
                time.sleep(0.05); self.vel_backend.send_velocity(0,0,0)
                
                # 2. Micro-Shift (Cross-axis mapping applied)
                dx_multiplier, dy_multiplier = dirs[idx%4]
                vx = dx_multiplier * TACTILE_SPEED * VEL_X_GAIN
                vy = dy_multiplier * TACTILE_SPEED * VEL_Y_GAIN
                self.vel_backend.send_velocity(vx=vx, vy=vy)
                time.sleep(0.04); self.vel_backend.send_velocity(0,0,0)
                
                # 3. Push Down
                self.collision_detected = False
                self.vel_backend.send_velocity(vz=Z_SPEED_DOWN)
                
                # Wait for contact or 0.4s max push
                contact = False
                p_start = time.time()
                while (time.time() - p_start) < 0.4:
                    if self.ft_force['z'] < self.threshold_z:
                        contact = True; break
                    time.sleep(0.01)
                self.vel_backend.send_velocity(0,0,0)
                
                if contact:
                    if self.verify_insertion(): return True
                    else: idx += 1
            
            print(f"\n⌛ [DEBUG] Tactile Search Timeout reached.")
            return False
        except Exception as e:
            print(f"❌ Tactile Error: {e}")
            return False

    def retract_and_retry(self):
        print(f"\n⬆️  [DEBUG] Safety Retract (0.5cm).")
        self.vel_backend.send_velocity(0,0,0); time.sleep(0.2)
        self.vel_backend.send_velocity(vz=0.025); time.sleep(0.2)
        self.vel_backend.send_velocity(0,0,0); time.sleep(0.5)
        self.collision_detected = False 

    def run_visual_spiral(self):
        print(f"\n🌀 [DEBUG] Target Lost. Visual Spiral Search...")
        dirs = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        leg, dur = 0, SPIRAL_BASE_TIME
        while leg < SPIRAL_MAX_LEGS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: dur += SPIRAL_BASE_TIME
            idx = leg % 4
            self.vel_backend.send_velocity(vx=dirs[idx][0]*SPIRAL_SPEED, vy=dirs[idx][1]*SPIRAL_SPEED)
            time.sleep(dur); self.vel_backend.send_velocity(0,0,0); time.sleep(0.5)
            if self.get_error_pixels()[0] is not None:
                print(f"   ✨ [DEBUG] Target Found!")
                return True
            if self.collision_detected: return False
            leg += 1
        return False

    def release_pressure(self):
        print(f"🧘 [DEBUG] Success! Normalizing Z-force...")
        target = self.baseline_z - 0.5 
        for _ in range(10): 
            if self.ft_force['z'] >= target: break
            self.vel_backend.send_velocity(vz=0.005); time.sleep(0.1)
            self.vel_backend.send_velocity(0,0,0); time.sleep(0.1)
        print(f"✅ Final Z-Force: {self.ft_force['z']:.2f}N. Task Complete.")

    def run(self):
        print(f"🚀 [DEBUG] Preparing Robot Pose...")
        self.moveit_backend.reset_robot()
        self.moveit_backend.move_to_pose_robust(START_X, START_Y, START_Z, {}, "xarm5_link5", "world_world", 0.1)
        
        print(f"⚖️  [DEBUG] Calibrating FT Sensors...")
        time.sleep(1.0)
        while self.latest_data is None: time.sleep(0.1)
        
        self.baseline_z, self.baseline_x, self.baseline_y = self.ft_force['z'], self.ft_force['x'], self.ft_force['y']
        self.threshold_z = self.baseline_z - Z_FORCE_OFFSET
        print(f"✅ Baseline: {self.baseline_z:.2f}N | Threshold: {self.threshold_z:.2f}N")

        self.vel_backend.start_velocity_mode()
        input("\n👉 Press ENTER to Start...")
        
        try:
            while rclpy.ok():
                if self.collision_detected:
                    self.vel_backend.send_velocity(0,0,0)
                    z_f = self.ft_force['z']
                    print(f"\n💥 [DEBUG] Contact at {z_f:.2f}N")
                    
                    if z_f < self.threshold_z:
                        if self.verify_insertion():
                            self.release_pressure(); break 
                        else:
                            if self.run_tactile_search():
                                self.release_pressure(); break
                            else:
                                self.retract_and_retry(); continue
                    else:
                        print(f"⚠️ Side Impact detected. Retracting.")
                        self.retract_and_retry(); continue

                err_u, err_v = self.get_error_pixels()
                
                if err_u is None:
                    if (time.time() - self.last_vision_time) > VISION_LOST_TIMEOUT:
                        if not self.run_visual_spiral(): break
                        continue
                    else:
                        time.sleep(0.05); continue

                self.last_vision_time = time.time()
                dist_px = math.sqrt(err_u**2 + err_v**2)

                # --- CROSS-AXIS ALIGNMENT BLOCK ---
                if dist_px > ALIGNMENT_TOL:
                    pulse = max(MIN_PULSE_XY, min(MAX_PULSE_XY, dist_px * PULSE_GAIN_XY))
                    # MAPPING: V (image vertical) -> Robot X | U (image horizontal) -> Robot Y
                    vx_cmd = (MOVE_SPEED if err_v > 0 else -MOVE_SPEED) * VEL_X_GAIN
                    vy_cmd = (MOVE_SPEED if err_u > 0 else -MOVE_SPEED) * VEL_Y_GAIN
                    
                    print(f"🎯 Err: {dist_px:.0f}px | Pulse: {pulse:.3f}s | Fz: {self.ft_force['z']:.1f}N", end='\r')
                    self.vel_backend.send_velocity(vx=vx_cmd, vy=vy_cmd)
                    time.sleep(pulse)
                else:
                    # Aligned -> Descent
                    print(f"✅ Aligned ({dist_px:.0f}px). Descending Z...", end='\r')
                    self.vel_backend.send_velocity(vz=Z_SPEED_DOWN)
                    time.sleep(Z_PULSE_DUR)

                self.vel_backend.send_velocity(0,0,0)
                time.sleep(SETTLE_TIME)

        finally:
            self.vel_backend.stop_velocity_mode()

def main(args=None):
    rclpy.init(args=args)
    node = VisualServo()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); t.start()
    try: node.run()
    finally: rclpy.shutdown()

if __name__ == '__main__': main()