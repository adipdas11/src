#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import JointState
import serial
import time
import math
import threading

class ToolCommander(Node):
    def __init__(self):
        super().__init__('tool_commander')

        # --- Parameters ---
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('fake_step', 0.15) # Speed of rotation in RViz
        
        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value
        self.step = self.get_parameter('fake_step').value

        # --- Hardware Connection ---
        self.ser = None
        self.fake = False
        
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self.get_logger().info(f"Connected to Tool @ {self.port}")
        except (serial.SerialException, OSError):
            self.fake = True
            self.get_logger().warn(f"Could not open {self.port}, running in FAKE mode")

        # --- Subscribers ---
        # 1 = Screw (CW), -1 = Unscrew (CCW), 0 = Stop
        self.create_subscription(Int8, 'tool_cmd', self.handle_cmd, 10)

        # --- RViz Visualization Publisher ---
        # We publish to /joint_states so RViz sees the tool spinning
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Internal state for simulation
        self.current_cmd = 0
        self.tool_angle = 0.0
        
        # Timer to update visualization (30Hz)
        self.create_timer(0.033, self.update_visualization)

    def handle_cmd(self, msg: Int8):
        cmd = msg.data
        if cmd not in [1, 0, -1]:
            return

        self.current_cmd = cmd
        
        # Send to Hardware
        if not self.fake and self.ser:
            try:
                # Protocol: just send the number as string with newline
                payload = f"{cmd}\n".encode('utf-8')
                self.ser.write(payload)
                self.get_logger().info(f"Sent command: {cmd}")
            except Exception as e:
                self.get_logger().error(f"Serial Error: {e}")

    def update_visualization(self):
        # If stopped, do nothing
        if self.current_cmd == 0:
            return

        # Update angle based on command
        # 1 = Increase angle (CW), -1 = Decrease (CCW)
        if self.current_cmd == 1:
            self.tool_angle += self.step
        elif self.current_cmd == -1:
            self.tool_angle -= self.step
            
        # Keep angle within 0 to 2pi (optional, but cleaner)
        self.tool_angle %= (2 * math.pi)

        # Publish JointState
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # IMPORTANT: Replace 'screwdriver_joint' with the ACTUAL joint name from your URDF
        msg.name = ['screwdriver_joint'] 
        msg.position = [self.tool_angle]
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ToolCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser: node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()