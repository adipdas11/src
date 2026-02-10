#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformBroadcaster
import tf_transformations
import math

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
MARKER_ID = 0           
MARKER_SIZE = 0.03      # 30mm
Z_OFFSET = 0.05        # 100mm (0.1m) added to the marker height
PUBLISH_RATE = 0.1      # 10Hz (0.1 seconds) for steady TF
ROBOT_BASE_FRAME = 'u1_base_link'
CAMERA_FRAME = 'camera_color_optical_frame'
TARGET_FRAME_NAME = 'aruco_target_world'

GROUP_UF850 = "uf_arm"
LINK_UF850 = "rg6_hand_tcp"
# =================================================

class PersistentArucoBroadcaster(Node):
    def __init__(self):
        super().__init__('persistent_aruco_broadcaster')
        
        # 1. Vision & TF Setup
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 2. ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_world_pose = None 

        # 3. Motion Backend
        self.motion = MotionBackend(self, GROUP_UF850)

        # 4. Pub/Sub & Timers
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.cam_info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
        
        # Continuous Timer to keep TF from disappearing in RViz
        self.tf_timer = self.create_timer(PUBLISH_RATE, self.broadcast_timer_cb)

        self.get_logger().info(f"🚀 Node Active. Publishing ID {MARKER_ID} with +100mm Z-offset.")

    def cam_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def image_cb(self, msg):
        if self.camera_matrix is None: return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and MARKER_ID in ids:
            idx = np.where(ids == MARKER_ID)[0][0]
            obj_points = np.array([[-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                                   [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)
            
            ret, rvec, tvec = cv2.solvePnP(obj_points, corners[idx][0], self.camera_matrix, self.dist_coeffs)
            if ret:
                self.process_pose(rvec, tvec)

    def process_pose(self, rvec, tvec):
        try:
            # Lookup Calibration (Base to Camera)
            trans = self.tf_buffer.lookup_transform(ROBOT_BASE_FRAME, CAMERA_FRAME, rclpy.time.Time())
            
            p_cam = Pose()
            p_cam.position = Point(x=tvec[0][0], y=tvec[1][0], z=tvec[2][0])
            
            rot_mat, _ = cv2.Rodrigues(rvec)
            T = np.eye(4); T[:3, :3] = rot_mat
            q = tf_transformations.quaternion_from_matrix(T)
            p_cam.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            
            # Transform to Robot World Frame
            world_pose = tf2_geometry_msgs.do_transform_pose(p_cam, trans)

            # --- APPLY 100mm Z-OFFSET ---
            world_pose.position.z += Z_OFFSET 

            # Update latest pose and print coordinates
            self.latest_world_pose = world_pose
            print(f"📍 TARGET (+100mm Z): X={world_pose.position.x:.4f}, Y={world_pose.position.y:.4f}, Z={world_pose.position.z:.4f}", end='\r')
            
        except Exception:
            pass

    def broadcast_timer_cb(self):
        """Timer callback that continuously publishes the TF to keep it visible in RViz."""
        if self.latest_world_pose is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg() # Fresh timestamp prevents fading
        t.header.frame_id = ROBOT_BASE_FRAME
        t.child_frame_id = TARGET_FRAME_NAME
        
        t.transform.translation.x = self.latest_world_pose.position.x
        t.transform.translation.y = self.latest_world_pose.position.y
        t.transform.translation.z = self.latest_world_pose.position.z
        t.transform.rotation = self.latest_world_pose.orientation
        
        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = PersistentArucoBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()