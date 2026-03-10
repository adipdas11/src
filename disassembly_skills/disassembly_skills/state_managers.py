#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String
import json

class DisassemblyStateManager(Node):
    def __init__(self):
        super().__init__('disassembly_state_manager')

        self.VALID_STATES = [
            "IDLE", 
            "MOVING", 
            "UNSCREWING", 
            "UNSCREWING_COMPLETE", 
            "UNSCREWING_RETRY", 
            "HOLDING", 
            "FLIPPING",
            "DROPPING",
            "ERROR"
        ]

        self.state_pub = self.create_publisher(String, '/robot_states', 10)
        self.create_subscription(String, '/robot_state/tool_arm/update', self.tool_update_cb, 10)
        self.create_subscription(String, '/robot_state/manip_arm/update', self.manip_update_cb, 10)

        self.states = {
            "tool_arm": "IDLE",
            "manip_arm": "IDLE"
        }

        self.timer = self.create_timer(0.1, self.publish_combined_state)
        self.get_logger().info("✅ Dual-Arm State Manager Online (Topic: /robot_states)")

    def tool_update_cb(self, msg):
        new_state = msg.data.upper()
        if new_state in self.VALID_STATES:
            self.states["tool_arm"] = new_state

    def manip_update_cb(self, msg):
        new_state = msg.data.upper()
        if new_state in self.VALID_STATES:
            self.states["manip_arm"] = new_state

    def publish_combined_state(self):
        msg = String()
        msg.data = json.dumps(self.states)
        self.state_pub.publish(msg)

class ObjectHoldStateManager(Node):
    def __init__(self):
        super().__init__('object_hold_state_manager')
        self.is_holding = False
        self.state_string = "EMPTY"
        
        self.create_subscription(Bool, '/object_hold_status', self.hold_status_callback, 10)
        self.bool_pub = self.create_publisher(Bool, '/object_hold_state/is_held', 10)
        self.string_pub = self.create_publisher(String, '/object_hold_state/update', 10)
        self.create_timer(0.5, self.broadcast_state)
        
        self.get_logger().info("✅ Object Hold State Manager Initialized. Current State: EMPTY")

    def hold_status_callback(self, msg: Bool):
        if self.is_holding != msg.data:
            self.is_holding = msg.data
            self.state_string = "HOLDING" if self.is_holding else "EMPTY"

    def broadcast_state(self):
        bool_msg = Bool()
        bool_msg.data = self.is_holding
        self.bool_pub.publish(bool_msg)
        str_msg = String()
        str_msg.data = self.state_string
        self.string_pub.publish(str_msg)

def main(args=None):
    rclpy.init(args=args)
    exec = MultiThreadedExecutor()
    dsm = DisassemblyStateManager()
    ohm = ObjectHoldStateManager()
    exec.add_node(dsm)
    exec.add_node(ohm)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
