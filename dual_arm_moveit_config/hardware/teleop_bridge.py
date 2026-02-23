#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration

class TeleopBridge(Node):
    def __init__(self):
        super().__init__('teleop_bridge_direct')

        self.active_arm = 'xarm' 
        self.planning_frame = 'sd1_base_link' 
        
        # ⚙️ Scales & Deadbands
        self.linear_scale = 0.5   
        self.angular_scale = 0.8  
        self.min_velocity_boost = 0.15 
        self.joystick_deadband = 0.05

        # 🧈 THE MAGIC SMOOTHER: Exponential Moving Average (EMA)
        # Lower = Smoother & heavier feel (0.1 to 0.3 is best)
        # Higher = Sharper & more responsive
        self.joystick_alpha = 0.15  

        self.target_linear = [0.0, 0.0, 0.0]
        self.target_angular = [0.0, 0.0, 0.0]
        self.current_linear = [0.0, 0.0, 0.0]
        self.current_angular = [0.0, 0.0, 0.0]

        # --- Publishers ---
        self.xarm_pub = self.create_publisher(TwistStamped, '/xarm_servo_node/delta_twist_cmds', 10)
        self.uf_pub = self.create_publisher(TwistStamped, '/uf_servo_node/delta_twist_cmds', 10)
        self.gripper_traj_pub = self.create_publisher(JointTrajectory, '/rg6_controller/joint_trajectory', 10)
        self.xarm_traj_pub = self.create_publisher(JointTrajectory, '/xarm_controller/joint_trajectory', 10)
        self.uf_traj_pub = self.create_publisher(JointTrajectory, '/uf_controller/joint_trajectory', 10)
        self.tool_pub = self.create_publisher(Int8, 'tool_cmd', 10)

        # --- Servo Clients ---
        self.xarm_start_cli = self.create_client(Trigger, '/xarm_servo_node/start_servo')
        self.uf_start_cli = self.create_client(Trigger, '/uf_servo_node/start_servo')
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.last_buttons = [0] * 12
        self.gripper_is_closed = False
        self.unscrew_active = False
        self.grab_active = False
        self.deadman_active = False
        
        # 🔄 Timers
        self.create_timer(0.5, self.force_sync)
        # 🚀 50Hz Streamer with built-in EMA Smoothing
        self.create_timer(0.02, self.continuous_twist_publisher)

        self.get_logger().info("🧈 TELEOP BRIDGE: Cinematic Smoothing Active")

    def force_sync(self):
        req = Trigger.Request()
        if not self.deadman_active:
            if self.xarm_start_cli.wait_for_service(timeout_sec=0.1): self.xarm_start_cli.call_async(req)
            if self.uf_start_cli.wait_for_service(timeout_sec=0.1): self.uf_start_cli.call_async(req)

    def send_traj_goal(self, publisher, joint_names, positions, time_sec=1.5):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=0, nanosec=int(time_sec * 1e9))
        msg.points.append(point)
        publisher.publish(msg)

    def apply_minimum_velocity(self, val):
        if abs(val) < self.joystick_deadband: return 0.0
        if abs(val) < self.min_velocity_boost:
            return self.min_velocity_boost if val > 0 else -self.min_velocity_boost
        return val

    def joy_callback(self, msg):
        try:
            btns = list(msg.buttons) + [0] * max(0, 12 - len(msg.buttons))
            axes = list(msg.axes) + [0.0] * max(0, 8 - len(msg.axes))

            if btns[3] == 1 and self.last_buttons[3] == 0:
                self.send_traj_goal(self.xarm_traj_pub, 
                    ['xarm5_joint1', 'xarm5_joint2', 'xarm5_joint3', 'xarm5_joint4', 'xarm5_joint5'],
                    [0.0, 0.0, -1.5708, 1.5708, 0.0])
                self.send_traj_goal(self.uf_traj_pub,
                    ['u1_joint1', 'u1_joint2', 'u1_joint3', 'u1_joint4', 'u1_joint5', 'u1_joint6'],
                    [0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0])

            if btns[5] == 1 and self.last_buttons[5] == 0:
                self.gripper_is_closed = not self.gripper_is_closed
                self.send_traj_goal(self.gripper_traj_pub, ['rg6_l_out'], [-0.6 if self.gripper_is_closed else 0.6], time_sec=0.5)

            if btns[4] == 1 and self.last_buttons[4] == 0:
                self.active_arm = 'uf' if self.active_arm == 'xarm' else 'xarm'
                self.get_logger().info(f"🦾 Active Arm Swapped to: {self.active_arm.upper()}")

            # 🎯 Capture the raw TARGET speed based on the joystick
            if btns[0] == 1:
                self.deadman_active = True
                self.target_linear[0] = self.apply_minimum_velocity(axes[1]) * self.linear_scale
                self.target_linear[1] = self.apply_minimum_velocity(axes[0]) * self.linear_scale
                self.target_linear[2] = self.apply_minimum_velocity(axes[4]) * self.linear_scale
                self.target_angular[2] = self.apply_minimum_velocity(axes[3]) * self.angular_scale
                
                if self.active_arm == 'uf':
                    self.target_angular[1] = self.apply_minimum_velocity(axes[7]) * self.angular_scale 
                    self.target_angular[0] = self.apply_minimum_velocity(axes[6]) * self.angular_scale 
                else:
                    self.target_angular[1] = 0.0
                    self.target_angular[0] = 0.0
            else:
                self.deadman_active = False
                # If button is released, target speed drops to exactly zero
                self.target_linear = [0.0, 0.0, 0.0]
                self.target_angular = [0.0, 0.0, 0.0]

            self.last_buttons = list(btns)
            
        except Exception as e:
            self.get_logger().error(f"❌ JOY CALLBACK CRASHED: {e}")

    def continuous_twist_publisher(self):
        """Runs cleanly at 50Hz, slowly ramping current speed towards target speed."""
        
        # Apply the Exponential Moving Average Math
        for i in range(3):
            self.current_linear[i] = (self.joystick_alpha * self.target_linear[i]) + ((1.0 - self.joystick_alpha) * self.current_linear[i])
            self.current_angular[i] = (self.joystick_alpha * self.target_angular[i]) + ((1.0 - self.joystick_alpha) * self.current_angular[i])

        # If we are moving even microscopically, publish the Twist
        # (This allows the robot to "coast" to a smooth stop after you let go)
        if self.deadman_active or any(abs(v) > 0.001 for v in self.current_linear + self.current_angular):
            tw = TwistStamped()
            tw.header.stamp = self.get_clock().now().to_msg()
            tw.header.frame_id = self.planning_frame
            tw.twist.linear.x = self.current_linear[0]
            tw.twist.linear.y = self.current_linear[1]
            tw.twist.linear.z = self.current_linear[2]
            tw.twist.angular.x = self.current_angular[0]
            tw.twist.angular.y = self.current_angular[1]
            tw.twist.angular.z = self.current_angular[2]

            if self.active_arm == 'xarm':
                self.xarm_pub.publish(tw)
            else:
                self.uf_pub.publish(tw)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TeleopBridge())
    rclpy.shutdown()

if __name__ == '__main__':
    main()