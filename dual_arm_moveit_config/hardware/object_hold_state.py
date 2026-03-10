#!/usr/bin/env python3
"""
Object Hold State Manager — Hardware Layer

Subscribes to /object_hold_status (Bool) published by ObjectHoldSkill
and broadcasts the hold state at 2 Hz on:
  - /object_hold_state/is_held  (std_msgs/Bool)
  - /object_hold_state/update   (std_msgs/String)  e.g. "HOLDING" | "EMPTY"

Mirrors the devel_agent_v4 hardware directory pattern.
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String


class ObjectHoldState(Node):
    def __init__(self):
        super().__init__('object_hold_state_node')

        self.is_holding = False
        self.state_string = "EMPTY"

        self.create_subscription(Bool, '/object_hold_status', self._hold_status_cb, 10)

        self.bool_pub = self.create_publisher(Bool, '/object_hold_state/is_held', 10)
        self.string_pub = self.create_publisher(String, '/object_hold_state/update', 10)

        # Broadcast at 2 Hz
        self.create_timer(0.5, self._broadcast)

        self.get_logger().info(
            "✅ Object Hold State Node Online. "
            "Listening: /object_hold_status → "
            "Broadcasting: /object_hold_state/is_held + /object_hold_state/update"
        )

    def _hold_status_cb(self, msg: Bool):
        if self.is_holding != msg.data:
            self.is_holding = msg.data
            self.state_string = "HOLDING" if self.is_holding else "EMPTY"
            self.get_logger().info(f"🔄 Hold State → {self.state_string}")

    def _broadcast(self):
        bool_msg = Bool()
        bool_msg.data = self.is_holding
        self.bool_pub.publish(bool_msg)

        str_msg = String()
        str_msg.data = self.state_string
        self.string_pub.publish(str_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectHoldState()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
