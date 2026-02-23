#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
import threading, json, math, time
from disassembly_skills.motion_backend import MotionBackend

class ObjectHoldSkill(Node):
    def __init__(self):
        super().__init__('object_hold_skill_node')
        
        # 🦾 MUST MATCH PLANNING GROUP NAMES IN SRDF
        self.uf850 = MotionBackend(self, "uf_arm") 
        self.gripper = MotionBackend(self, "rg6_gripper")
        
        self.latest_vision_data = None
        self.vision_event = threading.Event()
        self.create_subscription(String, '/vision/agent_state', self.vision_callback, 10)
        
        # --- Updated Topics for your Central State Managers ---
        # This triggers the ObjectHoldStateManager
        self.hold_status_pub = self.create_publisher(Bool, '/object_hold_status', 10)
        # This triggers the DisassemblyStateManager
        self.state_update_pub = self.create_publisher(String, '/robot_state/manip_arm/update', 10)
        
        # Configuration
        self.CAMERA_FRAME = 'camera_color_optical_frame'
        self.PLANNING_FRAME = 'world_world'
        self.ROBOT_EE_LINK = "u1_tool0"
        self.TOOL_LENGTH = 0.28
        self.APPROACH_BUFFER = 0.05
        self.HOVER_Z_OFFSET = 0.10
        self.JOINT_GRIPPER = "rg6_l_out"
        self.OPEN_DEG = 35.0
        self.CLOSE_DEG = -35.0
        self.TORQUE_THRESHOLD = 3.0
        
        self.get_logger().info("🚀 Object Hold Skill: Two-Stage Tactile Mode + Central State Sync Active.")
        self.publish_state("IDLE")

    def publish_state(self, s): 
        self.state_update_pub.publish(String(data=s))

    def publish_hold_status(self, h): 
        # This is the 'Trigger' for your ObjectHoldStateManager
        self.hold_status_pub.publish(Bool(data=h))

    def vision_callback(self, msg):
        try:
            self.latest_vision_data = json.loads(msg.data.strip("'"))
            self.vision_event.set()
        except: pass

    def wait_for_gripper(self, target_deg, timeout=5.0):
        """
        Monitors the gripper. 
        Success if goal is reached OR mechanical stall is detected (the grasp).
        """
        target_rad = math.radians(target_deg)
        start_t = time.time(); last_pos = 999.0; stall_timer = 0.0
        is_closing = target_deg < 0 

        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr = self.gripper.current_joint_positions.get(self.JOINT_GRIPPER, 999)
            if curr == 999:
                time.sleep(0.1); continue

            # Case 1: Goal reached (likely empty or opening)
            if abs(curr - target_rad) < 0.05:
                return True

            # Case 2: Stall Detection (Mechanical resistance = Successful Grasp)
            if abs(curr - last_pos) < 0.002:
                stall_timer += 0.1
                if stall_timer >= 0.5:
                    if is_closing:
                        self.get_logger().info(f"📦 Grasp Secured: Stalled at {curr:.3f} rad.")
                        return True
                    else:
                        return False
            else:
                stall_timer = 0.0
            
            last_pos = curr
            time.sleep(0.1)
        return False

    def _run_hold_sequence(self, part_id, target_label, interactive):
        self.vision_event.clear()
        if not self.vision_event.wait(timeout=10.0): return False
        
        objects = self.latest_vision_data.get("global_view", {}).get("objects", [])
        obj = next((o for o in objects if o.get("id") == part_id), None)
        if not obj: return False

        # --- Math & Transforms ---
        p = Pose(); p.position.x, p.position.y, p.position.z = obj["xyz"]; p.orientation.w = 1.0
        t_pose = self.uf850.get_transformed_pose(p, self.CAMERA_FRAME, self.PLANNING_FRAME)
        if not t_pose: return False
        
        wx, wy, wz = t_pose.pose.position.x, t_pose.pose.position.y, t_pose.pose.position.z
        v_rad = math.radians(-obj["angle"]) + (math.pi / 2.0); tilt = math.radians(8.0)
        
        off = (self.TOOL_LENGTH * math.cos(tilt)) + self.APPROACH_BUFFER
        tx, ty = wx - (off * math.cos(v_rad)), wy - (off * math.sin(v_rad))
        hz = wz + (self.TOOL_LENGTH * math.sin(tilt)) + self.HOVER_Z_OFFSET
        
        q = self.uf850._rpy_to_quaternion(math.pi, -math.pi/2 + tilt, v_rad)
        qd = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}

        # --- STEP 1: HOVER ---
        self.publish_state("MOVING")
        if interactive: input(f"🚀 STEP 1: HOVER above {target_label}? [Enter]")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.OPEN_DEG)})
        if not self.uf850.move_to_pose_robust(tx, ty, hz, qd, velocity=0.1): return False

        # --- STEP 2: 1st TACTILE DESCENT & RETRACT 0.01m ---
        if interactive: input("🚀 STEP 2: 1st TACTILE DESCENT (Retract 0.01m)? [Enter]")
        if not self.uf850.move_linear_z_with_torque_stop(0.1, self.TORQUE_THRESHOLD): return False
        time.sleep(0.5)
        if not self.uf850.retract_relative_z(0.01): return False

        # --- STEP 3: SLIDE ---
        if interactive: input("🚀 STEP 3: SLIDE into position? [Enter]")
        try:
            cz = self.uf850.tf_buffer.lookup_transform(self.PLANNING_FRAME, self.ROBOT_EE_LINK, rclpy.time.Time()).transform.translation.z
        except: return False
        
        gx = wx - (self.TOOL_LENGTH * math.cos(tilt) * math.cos(v_rad))
        gy = wy - (self.TOOL_LENGTH * math.cos(tilt) * math.sin(v_rad))
        if not self.uf850.move_to_pose_robust(gx, gy, cz, qd, velocity=0.1): return False

        # --- STEP 4: 2nd TACTILE DESCENT & RETRACT 0.005m ---
        if interactive: input("🚀 STEP 4: 2nd TACTILE DESCENT (Retract 0.005m)? [Enter]")
        if not self.uf850.move_linear_z_with_torque_stop(0.08, self.TORQUE_THRESHOLD): return False
        time.sleep(0.5)
        if not self.uf850.retract_relative_z(0.005): return False

        # --- STEP 5: CLOSE ---
        if interactive: input("🚀 STEP 5: CLOSE Gripper? [Enter]")
        # Set state to HOLDING so the System Manager knows we are grasping
        self.publish_state("HOLDING")
        self.gripper.move_to_joint_positions({self.JOINT_GRIPPER: math.radians(self.CLOSE_DEG)})
        
        # Stall detection logic
        success = self.wait_for_gripper(self.CLOSE_DEG)
        return success

    def execute_hold(self, part_id, target_label, interactive=True):
        # Reset the status manager to false at start
        self.publish_hold_status(False)
        success = False
        try: 
            success = self._run_hold_sequence(part_id, target_label, interactive)
        except Exception as e: 
            self.get_logger().error(f"💥 Crashed: {e}")
        finally:
            # 📢 This single line will now update the ObjectHoldStateManager!
            self.publish_hold_status(success)
            # Update the DisassemblyStateManager
            self.publish_state("HOLDING" if success else "IDLE")
            return success

def main(args=None):
    rclpy.init(args=args); node = ObjectHoldSkill()
    executor = MultiThreadedExecutor(); executor.add_node(node)
    
    # Threading to allow blocking 'input()' calls without freezing the node
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    time.sleep(2.0) # MoveIt warm-up
    
    # Execute and capture final boolean
    is_held = node.execute_hold(part_id=0, target_label="Top_Lid", interactive=True)
    
    # Keep node alive so publishers can reach the state managers
    try:
        while rclpy.ok():
            node.publish_hold_status(is_held)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__': main()