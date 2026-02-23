#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Int8, Bool
from cv_bridge import CvBridge
import h5py
import numpy as np
import os

class ILDataRecorder(Node):
    def __init__(self):
        super().__init__('il_data_recorder')
        
        self.bridge = CvBridge()
        
        # --- Recording Setup ---
        self.is_recording = False
        self.episode_idx = 0
        self.save_dir = "./il_dataset"
        os.makedirs(self.save_dir, exist_ok=True)
        self.clear_buffers()
        
        # --- Sensor Caches ---
        self.latest_image = None
        self.latest_joints = None
        self.latest_ft = None
        self.current_tool_action = 0
        
        # --- Subscriptions (Listening to Hardware + Teleop Trigger) ---
        self.create_subscription(Image, '/tool_camera/image_raw', self.image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(WrenchStamped, '/ft300_force_torque', self.ft_callback, 10) 
        self.create_subscription(Int8, '/tool_cmd', self.tool_callback, 10)
        
        # Listens to the gamepad script!
        self.create_subscription(Bool, '/il_record_trigger', self.trigger_callback, 10)
        
        # 20Hz Capture Loop
        self.create_timer(0.05, self.record_timestep)
        
        self.get_logger().info("💾 IL Data Recorder active. Waiting for signal from gamepad...")

    def clear_buffers(self):
        self.buf_images = []; self.buf_qpos = []; self.buf_ft = []
        self.buf_action_qpos = []; self.buf_action_tool = []

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = cv_image.shape[:2]
        cy, cx = h//2, w//2 
        if h >= 256 and w >= 256:
            self.latest_image = cv_image[cy-128:cy+128, cx-128:cx+128]
        else:
            self.latest_image = cv_image 

    def joint_callback(self, msg):
        # We only care about recording the xArm5 (Unscrewing arm) for this policy
        self.latest_joints = np.array(msg.position[:5])

    def ft_callback(self, msg):
        self.latest_ft = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def tool_callback(self, msg):
        self.current_tool_action = msg.data

    def trigger_callback(self, msg):
        # This catches the signal from the teleop script
        if msg.data and not self.is_recording:
            self.is_recording = True
            self.get_logger().info(f"▶️ Capturing data for Episode {self.episode_idx}...")
        elif not msg.data and self.is_recording:
            self.is_recording = False
            self.get_logger().info(f"⏸️ Saving Episode {self.episode_idx} to HDF5...")
            self.save_episode()
            self.episode_idx += 1
            self.clear_buffers()

    def record_timestep(self):
        if not self.is_recording: return
        
        if self.latest_image is None or self.latest_joints is None or self.latest_ft is None:
            return 
            
        self.buf_images.append(self.latest_image.copy())
        self.buf_qpos.append(self.latest_joints.copy())
        self.buf_ft.append(self.latest_ft.copy())
        self.buf_action_qpos.append(self.latest_joints.copy()) 
        self.buf_action_tool.append(self.current_tool_action)

    def save_episode(self):
        filename = os.path.join(self.save_dir, f"episode_{self.episode_idx}.hdf5")
        with h5py.File(filename, 'w') as root:
            obs = root.create_group('observations')
            obs.create_dataset('image', data=np.array(self.buf_images))
            obs.create_dataset('qpos', data=np.array(self.buf_qpos))
            obs.create_dataset('force_torque', data=np.array(self.buf_ft))
            
            action = root.create_group('action')
            action.create_dataset('qpos', data=np.array(self.buf_action_qpos))
            action.create_dataset('tool', data=np.array(self.buf_action_tool))
            
        self.get_logger().info(f"✅ Data Saved: {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = ILDataRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()