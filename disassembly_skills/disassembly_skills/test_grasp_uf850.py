#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf_transformations
import json
import math
import numpy as np

# --- IMPORT OUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# --- CONFIGURATION ---
ROBOT_BASE_FRAME = 'u1_base_link'       # UF850 Base
CAMERA_FRAME = 'camera_color_optical_frame' 
HDD_THICKNESS = 0.026                   # Meters
TABLE_HEIGHT = 0.0                      
GRIPPER_GROUP = "uf_arm"                

class UF850GraspTester(Node):
    def __init__(self):
        super().__init__('uf850_grasp_tester')
        
        # 1. Vision & TF Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.latest_json = {}
        self.camera_model = None
        
        self.create_subscription(String, '/vision/agent_state', self.json_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.cam_cb, 10)
        
        # 2. Motion Backend (The Clean Way!)
        # We initialize it for the UF850 group
        self.motion = MotionBackend(self, GRIPPER_GROUP)
        
        self.get_logger().info("Waiting for Vision Data...")

    def json_cb(self, msg):
        try: self.latest_json = json.loads(msg.data)
        except: pass

    def cam_cb(self, msg):
        self.camera_model = msg

    def get_target_pose(self):
        """
        Calculates the 3D Grasp Pose from Vision JSON.
        Returns: PoseStamped (or None)
        """
        # A. Find HDD
        objects = self.latest_json.get('global_view', {}).get('objects', [])
        hdd_obj = None
        for obj in objects:
            if "Lid" in obj.get('label', '') or "Hard_Drive" in obj.get('label', ''):
                hdd_obj = obj
                break
        
        if not hdd_obj:
            self.get_logger().warn("Vision: No HDD detected!")
            return None

        # B. Get 2D Data
        box = hdd_obj.get('box', [0,0,0,0])
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        angle_deg = hdd_obj.get('angle', 0.0)
        
        self.get_logger().info(f"Vision Target: ({cx:.1f}, {cy:.1f}) Angle: {angle_deg:.1f}")

        # C. 3D Projection (Ray Casting)
        if not self.camera_model: return None
        
        try:
            trans = self.tf_buffer.lookup_transform(ROBOT_BASE_FRAME, CAMERA_FRAME, rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}")
            return None

        fx = self.camera_model.k[0]
        fy = self.camera_model.k[4]
        u0 = self.camera_model.k[2]
        v0 = self.camera_model.k[5]

        # Ray in Camera Frame
        ray_cam = np.array([(cx - u0) / fx, (cy - v0) / fy, 1.0])
        
        # Ray in World Frame
        q = [trans.transform.rotation.x, trans.transform.rotation.y, 
             trans.transform.rotation.z, trans.transform.rotation.w]
        R_mat = tf_transformations.quaternion_matrix(q)[:3, :3]
        t_vec = np.array([trans.transform.translation.x, 
                          trans.transform.translation.y, 
                          trans.transform.translation.z])

        ray_world = np.dot(R_mat, ray_cam)
        ray_origin = t_vec

        # Plane Intersection
        target_z = TABLE_HEIGHT + HDD_THICKNESS
        if abs(ray_world[2]) < 1e-6: return None
        
        t_param = (target_z - ray_origin[2]) / ray_world[2]
        intersection = ray_origin + t_param * ray_world

        self.get_logger().info(f"3D Point: {intersection}")

        # D. Orientation (Align Gripper to Side)
        # 1. Point Down (Rotate 180 around Y axis relative to Base X-Forward)
        q_down = tf_transformations.quaternion_from_euler(0, math.pi, 0)
        
        # 2. Align Z-rotation to HDD Angle + 90 deg
        angle_rad = math.radians(angle_deg)
        q_rot = tf_transformations.quaternion_from_euler(0, 0, angle_rad + math.pi/2)
        
        q_final = tf_transformations.quaternion_multiply(q_rot, q_down)

        # E. Pose
        ps = PoseStamped()
        ps.header.frame_id = ROBOT_BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = intersection[0]
        ps.pose.position.y = intersection[1]
        ps.pose.position.z = intersection[2] + 0.15 # HOVER 15cm
        ps.pose.orientation = Quaternion(x=q_final[0], y=q_final[1], z=q_final[2], w=q_final[3])
        
        return ps

def main():
    rclpy.init()
    node = UF850GraspTester()
    
    # 1. Wait for Vision
    print("Waiting for Vision Data...")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_json and node.camera_model:
            break
            
    # 2. Compute Target
    target = node.get_target_pose()
    
    if target:
        print(f"\nTarget Found: X={target.pose.position.x:.3f}, Y={target.pose.position.y:.3f}")
        input("Press ENTER to Execute Move (using MotionBackend)...")
        
        # 3. Execute using Backend
        # We assume the gripper tip link name is 'rg6_hand_tcp' (Check your URDF!)
        success = node.motion.move_to_pose(target.pose, "rg6_hand_tcp")
        
        if success:
            print("Motion SUCCESS!")
            print("If aligned, you can implement descent logic next.")
        else:
            print("Motion FAILED (IK or Execution Error)")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()