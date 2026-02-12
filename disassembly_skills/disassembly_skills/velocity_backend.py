#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool 
import time

class VelocityBackend:
    def __init__(self, node: Node):
        self.node = node
        self.vel_pub = self.node.create_publisher(Twist, '/xarm/velo_cmd', 1)
        self.set_mode_client = self.node.create_client(SetBool, '/xarm/set_velocity_mode')

    def start_velocity_mode(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=2.0):
            return False
        future = self.set_mode_client.call_async(SetBool.Request(data=True))
        # Thread-safe wait for the service
        while not future.done():
            time.sleep(0.01)
        return future.result().success

    def stop_velocity_mode(self):
        self.send_velocity(0.0, 0.0, 0.0)
        if self.set_mode_client.service_is_ready():
            self.set_mode_client.call_async(SetBool.Request(data=False))

    def send_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw=0.0):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        msg.angular.z = float(yaw)
        self.vel_pub.publish(msg)