#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
# from sensor_msgs.msg import JointState  <-- No longer needed
import serial
import time

class ToolCommander(Node):
    def __init__(self):
        super().__init__('tool_commander')

        # --- Parameters ---
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        
        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value

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

        self.current_cmd = 0
        
        # ❌ DISABLED RVIZ VISUALIZATION TO PREVENT MOVEIT CRASHES ❌
        # self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        # self.create_timer(0.033, self.update_visualization)

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

    # ❌ DISABLED: update_visualization(self) function removed to keep code clean ❌

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