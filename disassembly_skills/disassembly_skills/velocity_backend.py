#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool 
import time

class VelocityBackend:
    def __init__(self, node: Node):
        self.node = node
        # Queue size 1 is good for low latency
        self.vel_pub = self.node.create_publisher(Twist, '/xarm/velo_cmd', 1)
        self.set_mode_client = self.node.create_client(SetBool, '/xarm/set_velocity_mode')

    def start_velocity_mode(self):
        self.node.get_logger().info("🔗 Requesting Mode 5 (Velocity)...")
        if not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("❌ Mode Service not found!")
            return False
            
        future = self.set_mode_client.call_async(SetBool.Request(data=True))
        
        # Wait for the service to return
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
            
        if future.result() is not None and future.result().success:
            self.node.get_logger().info("✅ Service Success. Waiting 2s for Hardware Sync...")
            # --- CRITICAL FIX ---
            # The SDK warns Mode 5 (0) because it needs time to settle.
            # Without this, the first few pulses return SDK ERROR code=1.
            time.sleep(2.0) 
            return True
        return False

    def stop_velocity_mode(self):
        self.node.get_logger().info("🛑 Stopping Velocity Mode...")
        # 1. Send explicit zero burst to clear the hardware buffer
        for _ in range(10):
            self.send_velocity(0.0, 0.0, 0.0)
            time.sleep(0.02)
            
        # 2. Switch back to Position Mode (Mode 1 or 0)
        if self.set_mode_client.service_is_ready():
            self.set_mode_client.call_async(SetBool.Request(data=False))
            time.sleep(0.5) # Let it settle

    def send_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw=0.0):
        msg = Twist()
        # Ensure we are using SI units (meters/sec)
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(yaw)
        self.vel_pub.publish(msg)