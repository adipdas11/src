#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8
import json, time, threading, math
from disassembly_skills.motion_backend import MotionBackend
from disassembly_skills.velocity_backend import VelocityBackend

# ================= CONFIGURATION =================
START_X, START_Y, START_Z = 0.908, 0.127, 1.234

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
Z_FORCE_OFFSET = 1.0  
SETTLE_TIME = 0.30      

# VISUAL RECOVERY
SPIRAL_SPEED = 0.025
SPIRAL_BASE_TIME = 0.1
SPIRAL_MAX_LEGS = 5      
VISION_LOST_TIMEOUT = 0.8

# TACTILE / MICRO-WIGGLE
TACTILE_TIMEOUT = 4.0     
TACTILE_SPEED = 0.035     
WIGGLE_SPEED = 0.035       
WIGGLE_TIME = 0.14              # 0.14s @ 0.035m/s = ~5mm travel to overcome screw slop
WIGGLE_FORCE_THRESH = 0.8       # Lowered to 0.8N to catch softer impacts in the Y-axis

# CONTINUOUS LEAD COMPENSATION
Z_MAX_PRESSURE_LIMIT = 1.5      
Z_COMP_GAIN = 0.015             
Z_MIN_LIFT_SPEED = 0.035        
Z_MAX_LIFT_SPEED = 0.080        
UNSCREW_STABILITY_TOL = 0.5     
Z_BITE_THRESHOLD = 1.2  
PROGRESSION_TIMEOUT = 5.0 
MIN_PROGRESSION_MM = 1.5        

# SAFETY
XY_FORCE_LIMIT = 2.5 
# =================================================

class VisualServo(Node):
    def __init__(self):
        super().__init__('visual_servo_agentic_final')
        self.moveit_backend = MotionBackend(self, "xarm_arm")
        self.vel_backend = VelocityBackend(self)
        
        self.sub = self.create_subscription(String, '/vision/agent_state', self.vision_cb, 10)
        self.tool_pub = self.create_publisher(Int8, '/tool_cmd', 10)
        
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

    def send_tool_cmd(self, val):
        msg = Int8()
        msg.data = int(val)
        self.tool_pub.publish(msg)

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

    def run_visual_spiral(self):
        print(f"\n🌀 [DEBUG] Target Lost. Visual Spiral Recovery...")
        dirs = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        leg, dur = 0, SPIRAL_BASE_TIME
        while leg < SPIRAL_MAX_LEGS and rclpy.ok():
            if leg > 0 and leg % 2 == 0: dur += SPIRAL_BASE_TIME
            self.vel_backend.send_velocity(vx=dirs[leg%4][0]*SPIRAL_SPEED, vy=dirs[leg%4][1]*SPIRAL_SPEED)
            time.sleep(dur); self.vel_backend.send_velocity(0,0,0); time.sleep(0.5)
            if self.get_error_pixels()[0] is not None: return True
            if self.collision_detected: return False
            leg += 1
        return False

    def perform_unscrew_with_compensation(self):
        input("\n🛠️  [GATE] 4-Way Lock confirmed. Press ENTER to verify 'Bite' and unscrew...")
        
        print(f"🔄 [ACTION] Verifying Tool Seating (Z-Spike)...")
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

    def verify_insertion(self):
        print(f"\n🔍 [DEBUG] Starting 4-Way Wiggle (5mm sweep)...")
        bx, by = self.ft_force['x'], self.ft_force['y']
        
        locks = {'+X': False, '-X': False, '+Y': False, '-Y': False}
        
        moves = [
            ('+X', 'vx', WIGGLE_SPEED, 'x', bx),
            ('-X', 'vx', -WIGGLE_SPEED, 'x', bx),
            ('+Y', 'vy', WIGGLE_SPEED, 'y', by),
            ('-Y', 'vy', -WIGGLE_SPEED, 'y', by)
        ]
        
        for name, cmd_axis, speed, force_axis, baseline in moves:
            self.vel_backend.send_velocity(**{cmd_axis: speed})
            time.sleep(WIGGLE_TIME)
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.1) 
            
            spike = abs(self.ft_force[force_axis] - baseline)
            if spike > WIGGLE_FORCE_THRESH:
                locks[name] = True
                print(f"   ✔️ {name} Locked (Spike: {spike:.1f}N)")
            else:
                print(f"   ❌ {name} Slipped (Spike: {spike:.1f}N)")
                
            if self.ft_force['z'] > (self.baseline_z - 0.5):
                print(f"   ⚠️ Z-Contact lost during {name} wiggle.")
                
            self.vel_backend.send_velocity(**{cmd_axis: -speed})
            time.sleep(WIGGLE_TIME)
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.1)

        if all(locks.values()):
            print("⚓ [DEBUG] 4-Way Lock Confirmed.")
            return self.perform_unscrew_with_compensation()
        else:
            print("💨 [DEBUG] Slip detected. Not fully seated in all directions.")
            return False

    def run_tactile_search(self):
        try:
            print(f"\n🤏 [DEBUG] Tactile Search (Expanding Spiral)...")
            start_t = time.time()
            dirs = [[-1, 0], [0, -1], [1, 0], [0, 1]] 
            leg_idx = 0
            while (time.time() - start_t) < TACTILE_TIMEOUT and rclpy.ok():
                step_mult = (leg_idx // 2) + 1
                self.vel_backend.send_velocity(vz=0.035); time.sleep(0.05); self.vel_backend.send_velocity(0,0,0)
                dx_m, dy_m = dirs[leg_idx % 4]
                self.vel_backend.send_velocity(vx=dx_m * TACTILE_SPEED, vy=dy_m * TACTILE_SPEED)
                time.sleep(0.04 * step_mult); self.vel_backend.send_velocity(0,0,0)
                
                self.collision_detected = False
                self.vel_backend.send_velocity(vz=Z_SPEED_DOWN)
                contact = False
                p_start = time.time()
                while (time.time() - p_start) < 0.5:
                    if self.ft_force['z'] < self.threshold_z:
                        contact = True; break
                    time.sleep(0.01)
                self.vel_backend.send_velocity(0,0,0)
                
                if contact:
                    if self.verify_insertion(): return True
                    else: leg_idx += 1
            return False
        except: return False

    def retract_and_retry(self):
        print(f"\n⬆️  [DEBUG] Safety Retract (1.0cm)...")
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
            time.sleep(0.05)
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.05)
            
        for _ in range(10):
            self.vel_backend.send_velocity(0,0,0)
            time.sleep(0.05)

    def run(self):
        self.moveit_backend.reset_robot()
        self.moveit_backend.move_to_pose_robust(START_X, START_Y, START_Z, {}, "xarm5_link5", "world_world", 0.1)
        time.sleep(1.0)
        while self.latest_data is None: time.sleep(0.1)
        self.baseline_z, self.baseline_x, self.baseline_y = self.ft_force['z'], self.ft_force['x'], self.ft_force['y']
        self.threshold_z = self.baseline_z - Z_FORCE_OFFSET
        
        if not self.vel_backend.start_velocity_mode(): return
        print(f"✅ Hardware Ready. Baseline: {self.baseline_z:.1f}N")
        
        input("\n🎯 [GATE] Calibration done. Press ENTER to start Visual Alignment...")

        try:
            while rclpy.ok():
                if self.collision_detected:
                    self.vel_backend.send_velocity(0,0,0)
                    if self.verify_insertion():
                        self.post_extraction_retract()
                        print("🏆 TASK COMPLETE.")
                        break
                    else:
                        self.retract_and_retry(); continue

                err_u, err_v = self.get_error_pixels()
                if err_u is None:
                    if (time.time() - self.last_vision_time) > VISION_LOST_TIMEOUT:
                        if not self.run_visual_spiral(): break
                        continue
                    else: time.sleep(0.05); continue
                
                self.last_vision_time = time.time()
                dist_px = math.sqrt(err_u**2 + err_v**2)

                if dist_px > ALIGNMENT_TOL:
                    pulse = max(MIN_PULSE_XY, min(MAX_PULSE_XY, dist_px * PULSE_GAIN_XY))
                    vx_cmd = (MOVE_SPEED if err_v > 0 else -MOVE_SPEED) * VEL_X_GAIN
                    vy_cmd = (MOVE_SPEED if err_u > 0 else -MOVE_SPEED) * VEL_Y_GAIN
                    self.vel_backend.send_velocity(vx=vx_cmd, vy=vy_cmd); time.sleep(pulse)
                else:
                    self.vel_backend.send_velocity(vz=Z_SPEED_DOWN); time.sleep(Z_PULSE_DUR)
                
                self.vel_backend.send_velocity(0,0,0); time.sleep(SETTLE_TIME)
        finally:
            self.vel_backend.stop_velocity_mode()

def main(args=None):
    rclpy.init(args=args)
    node = VisualServo()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        node.get_logger().info("🛑 Shutting down...")
        node.vel_backend.stop_velocity_mode()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()