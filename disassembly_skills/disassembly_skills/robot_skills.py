#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Pose
import json
import time
import math

# IMPORT YOUR NEW BACKEND
from disassembly_skills.motion_backend import MotionBackend

# Reuse the State classes from before...
class SystemState:
    def __init__(self, node):
        self.node = node
        self.latest_data = {}
        self.sub = node.create_subscription(String, '/vision/agent_state', self.cb, 10)
    
    def cb(self, msg):
        try: self.latest_data = json.loads(msg.data)
        except: pass
        
    def get_force_z(self):
        return self.latest_data.get('force_torque', {}).get('force', {}).get('z', 0.0)
    
    def get_sniper_error(self):
        # ... (Same logic as before) ...
        local = self.latest_data.get('local_view', {})
        tips = local.get('tool_tips', [])
        heads = local.get('screw_heads', [])
        if tips and heads:
            # Simple check
            return (heads[0]['center'][0] - tips[0]['contact_point'][0], 
                    heads[0]['center'][1] - tips[0]['contact_point'][1])
        return None

class RobotSkills:
    def __init__(self, node):
        self.node = node
        self.state = SystemState(node)
        
        # --- NEW MOTION BACKENDS ---
        # Uses your robust IK logic
        self.xarm = MotionBackend(node, "xarm_arm")
        self.uf850 = MotionBackend(node, "uf_arm")
        
        self.tool_pub = self.node.create_publisher(Int8, '/tool_cmd', 10)
        
        # Keep track of last pose for relative moves
        self.last_xarm_pose = Pose() 
        self.last_xarm_pose.position.x = 0.3 # Default safe start
        self.last_xarm_pose.position.z = 0.2
        self.last_xarm_pose.orientation.w = 1.0

    # ==========================================================
    # SKILL 1: ALIGN (Visual Servo)
    # ==========================================================
    def align_tool(self):
        self.node.get_logger().info("SKILL: Align")
        Kp = 0.0001
        
        for i in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            err = self.state.get_sniper_error()
            if not err: return False
            
            dx, dy = err
            if abs(dx) < 5 and abs(dy) < 5: return True
            
            # Update Pose
            # Assuming Camera X aligned with Robot Y
            self.last_xarm_pose.position.x += (-dy * Kp)
            self.last_xarm_pose.position.y += (-dx * Kp)
            
            # Execute using your robust backend
            self.xarm.move_to_pose(self.last_xarm_pose, "xarm5_tool0")
            
        return False

    # ==========================================================
    # SKILL 2: SEEK SURFACE
    # ==========================================================
    def seek_contact(self):
        self.node.get_logger().info("SKILL: Seek Contact")
        step = 0.001
        
        for _ in range(50): # Max 5cm
            rclpy.spin_once(self.node, timeout_sec=0.05)
            fz = self.state.get_force_z()
            if fz < -2.0: return True
            
            # Step Down
            self.last_xarm_pose.position.z -= step
            self.xarm.move_to_pose(self.last_xarm_pose, "xarm5_tool0")
            
        return False

    # ==========================================================
    # SKILL 3: UNSCREW
    # ==========================================================
    def unscrew(self):
        self.node.get_logger().info("SKILL: Unscrew")
        self.tool_pub.publish(Int8(data=-1))
        time.sleep(5.0) # Placeholder for Torque Logic
        self.tool_pub.publish(Int8(data=0))
        return True

    # ==========================================================
    # MASTER: REMOVE SCREW
    # ==========================================================
    def remove_screw(self, start_pose):
        # 1. Move to Hover
        self.last_xarm_pose = start_pose
        self.xarm.move_to_pose(start_pose, "xarm5_tool0")
        
        # 2. Align
        if not self.align_tool(): return "ALIGN_FAIL"
        
        # 3. Seek
        if not self.seek_contact(): return "CONTACT_FAIL"
        
        # 4. Unscrew
        if self.unscrew():
            # Lift
            self.last_xarm_pose.position.z += 0.05
            self.xarm.move_to_pose(self.last_xarm_pose, "xarm5_tool0")
            return "SUCCESS"
        else:
            return "UNSCREW_FAIL"