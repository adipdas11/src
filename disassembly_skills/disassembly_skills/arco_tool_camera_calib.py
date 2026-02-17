#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class ArucoCalibrator(Node):
    def __init__(self):
        super().__init__('aruco_calibrator')
        
        # Subscribe to your specific tool camera topic
        self.subscription = self.create_subscription(
            Image,
            '/tool_cam/image_raw',
            self.image_callback,
            10)
            
        self.bridge = CvBridge()
        
        # Target Specifications
        self.target_id = 0
        self.marker_size_mm = 19.0
        
        # OpenCV 4.x ArUco Setup (Compatible with ROS 2 Humble)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            # OpenCV 4.7+
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            self.use_new_api = True
        except AttributeError:
            # OpenCV 4.5/4.6 (Default in Ubuntu 22.04 / Humble)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        self.get_logger().info(f"📷 ArUco Calibrator Started.")
        self.get_logger().info(f"Looking for 4x4 ID:{self.target_id} ({self.marker_size_mm}mm) on /tool_cam/image_raw")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 1. Detect the Markers
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.dictionary, parameters=self.parameters)

        # 2. Process Detections
        if ids is not None:
            # Draw bounding boxes around all detected markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            for i in range(len(ids)):
                if ids[i][0] == self.target_id:
                    # Extract the 4 corners of our specific target marker
                    # c[0]=top-left, c[1]=top-right, c[2]=bottom-right, c[3]=bottom-left
                    c = corners[i][0]
                    
                    # Calculate pixel width and height using Euclidean distance
                    width_px = math.dist(c[0], c[1])
                    height_px = math.dist(c[1], c[2])
                    
                    # Average the width and height for a more stable reading
                    avg_px = (width_px + height_px) / 2.0

                    if avg_px > 0:
                        # Calculate the ratios
                        mm_per_pixel = self.marker_size_mm / avg_px
                        m_per_pixel = (self.marker_size_mm / 1000.0) / avg_px

                        # Overlay the math on the camera feed
                        text1 = f"ID: {self.target_id} Size: {avg_px:.1f}px"
                        text2 = f"MM_PER_PIXEL = {m_per_pixel:.6f}"
                        
                        cv2.putText(cv_image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(cv_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # Print to terminal once per second so it doesn't flood the console
                        self.get_logger().info(f"Update your script -> MM_PER_PIXEL = {m_per_pixel:.6f}", throttle_duration_sec=1.0)

        # 3. Display the live feed window
        cv2.imshow("ArUco Calibration - Press CTRL+C in terminal to exit", cv_image)
        cv2.waitKey(1) 

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down calibrator...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()