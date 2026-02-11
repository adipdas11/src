#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf_transformations
from tf2_ros import TransformBroadcaster

# --- IMPORT YOUR BACKEND ---
from disassembly_skills.motion_backend import MotionBackend

# ================= CONFIGURATION =================
MARKER_ID = 0           
MARKER_SIZE = 0.03      # 30mm
Z_OFFSET = 0.006         # 50mm offset
PUBLISH_RATE = 0.1      # 10Hz
ROBOT_BASE_FRAME = 'u1_base_link'
CAMERA_FRAME = 'camera_color_optical_frame'
TARGET_FRAME_NAME = 'aruco_target_world'

# --- CRITICAL: Use the Aligned Topic ---
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
COLOR_TOPIC = '/camera/camera/color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'

GROUP_UF850 = "uf_arm"
# =================================================

class PersistentArucoBroadcaster(Node):
    def __init__(self):
        super().__init__('persistent_aruco_broadcaster')
        
        # 1. Vision Setup
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_depth_image = None
        
        # 2. Motion Backend
        self.motion = MotionBackend(self, GROUP_UF850)

        # 3. Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_world_pose = None 

        # 4. Pub/Sub
        self.create_subscription(CameraInfo, INFO_TOPIC, self.cam_info_cb, 10)
        self.create_subscription(Image, COLOR_TOPIC, self.image_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        
        self.tf_timer = self.create_timer(PUBLISH_RATE, self.broadcast_timer_cb)

        self.get_logger().info(f"🚀 Node Active. Using Aligned Depth: {DEPTH_TOPIC}")

    def cam_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def depth_cb(self, msg):
        try:
            if msg.encoding == "16UC1":
                raw_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.latest_depth_image = raw_img.astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def image_cb(self, msg):
        if self.camera_matrix is None or self.latest_depth_image is None: return
        
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None and MARKER_ID in ids:
            idx = np.where(ids == MARKER_ID)[0][0]
            marker_corners = corners[idx][0]

            obj_points = np.array([[-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                                   [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)
            ret, rvec, _ = cv2.solvePnP(obj_points, marker_corners, self.camera_matrix, self.dist_coeffs)
            
            if ret:
                cx = int(np.mean(marker_corners[:, 0]))
                cy = int(np.mean(marker_corners[:, 1]))

                h, w = self.latest_depth_image.shape
                if 0 <= cx < w and 0 <= cy < h:
                    z_meters = self.latest_depth_image[cy, cx]
                    if z_meters > 0.1: 
                        self.process_pose_from_depth(rvec, cx, cy, z_meters)

    def process_pose_from_depth(self, rvec, u, v, z_depth):
        # Camera Intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_intrinsics = self.camera_matrix[0, 2]
        cy_intrinsics = self.camera_matrix[1, 2]

        # 1. Deproject Pixel to 3D Point
        x_val = (u - cx_intrinsics) * z_depth / fx
        y_val = (v - cy_intrinsics) * z_depth / fy
        z_val = z_depth

        # --- FIX: FORCE PYTHON FLOAT CONVERSION ---
        # ROS 2 messages crash if you pass numpy types (e.g. numpy.float32)
        p_cam = Pose()
        p_cam.position = Point(
            x=float(x_val), 
            y=float(y_val), 
            z=float(z_val)
        )
        
        rot_mat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3, :3] = rot_mat
        q = tf_transformations.quaternion_from_matrix(T)
        p_cam.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

        # 2. Transform via Backend
        world_pose_stamped = self.motion.get_transformed_pose(
            source_pose=p_cam, 
            source_frame=CAMERA_FRAME, 
            target_frame=ROBOT_BASE_FRAME, 
            z_offset=Z_OFFSET
        )

        # 3. Update State
        if world_pose_stamped:
            self.latest_world_pose = world_pose_stamped.pose
            print(f"📍 Depth Target: X={self.latest_world_pose.position.x:.3f}, Y={self.latest_world_pose.position.y:.3f}, Z={self.latest_world_pose.position.z:.3f}", end='\r')

    def broadcast_timer_cb(self):
        if self.latest_world_pose is None: return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
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