#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController
import time

class TeleopBridge(Node):
    def __init__(self):
        super().__init__('teleop_bridge_final')

        self.active_arm = 'xarm' 
        self.planning_frame = 'world_world' 
        
        # ⚙️ Scales & Filters
        self.linear_scale = 0.3   # m/s max (speed_units mode)
        self.angular_scale = 0.3  # rad/s max (speed_units mode)
        self.joystick_alpha = 0.12 

        # 📐 6-Axis State
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

        # --- Service Clients ---
        self.cm_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.srv_clients = {
            'xarm_start': self.create_client(Trigger, '/xarm_servo_node/start_servo'),
            'uf_start':   self.create_client(Trigger, '/uf_servo_node/start_servo'),
        }

        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.last_buttons = [0] * 12
        
        # --- Internal States ---
        self.gripper_is_closed = False
        self.unscrew_active = False
        self.grab_active = False
        self.deadman_active = False
        
        self.create_timer(0.033, self.continuous_twist_publisher)  # Match servo publish rate
        self.sync_timer = self.create_timer(2.0, self.initial_sync_timer_callback)

        self.get_logger().info("🦾 DISASSEMBLY MASTER BRIDGE: R1 Gripper Mapping Online")

    def initial_sync_timer_callback(self):
        self.manage_servo_nodes()
        self.sync_timer.cancel()

    def manage_servo_nodes(self):
        """Switches hardware controllers intelligently to avoid empty request warnings."""
        if not self.cm_client.wait_for_service(timeout_sec=1.0): return

        # 🦾 SMART ACTIVATE: Only ask to activate what is currently INACTIVE
        # This stops the "Controller is not inactive" warnings
        try:
            import subprocess
            res = subprocess.run(['ros2', 'control', 'list_controllers'], capture_output=True, text=True)
            controller_status = res.stdout
        except Exception:
            controller_status = ""

        all_needed = ['xarm_controller', 'uf_controller', 'slider_controller', 'rg6_controller']
        to_activate = [c for c in all_needed if f"{c:20} [inactive]" in controller_status]

        if to_activate:
            sw_req = SwitchController.Request()
            sw_req.activate_controllers = to_activate
            sw_req.deactivate_controllers = [] 
            sw_req.strictness = SwitchController.Request.BEST_EFFORT
            self.cm_client.call_async(sw_req)
            self.get_logger().info(f"⚡ Activating inactive controllers: {to_activate}")

        # --- Handle Servo Node Start ---
        if self.active_arm == 'xarm':
            print("\n🚀 [Control] Mode: XARM Servo (link5 focus)")
            self.srv_clients['xarm_start'].call_async(Trigger.Request())
        else:
            print("\n🚀 [Control] Mode: UF850 Servo (tool0 focus)")
            self.srv_clients['uf_start'].call_async(Trigger.Request())

    def joy_callback(self, msg):
        try:
            btns = list(msg.buttons) + [0] * (12 - len(msg.buttons))
            axes = list(msg.axes) + [0.0] * (8 - len(msg.axes))

            # --- 🏠 BTN 3 (Y): GLOBAL HOMING ---
            if btns[3] == 1 and self.last_buttons[3] == 0:
                print("🏠 [Command] GLOBAL HOME: Forcing all controllers active...")
                
                # 1. Force a controller sync before moving
                self.manage_servo_nodes()
                
                # 2. Small sleep to let the controllers settle
                time.sleep(0.2)

                # 3. Now send the joint trajectories
                # XARM Home Pose
                self.send_traj_goal(self.xarm_traj_pub, 
                    ['xarm5_joint1', 'xarm5_joint2', 'xarm5_joint3', 'xarm5_joint4', 'xarm5_joint5'], 
                    [0.0, 0.0, -1.57, 1.57, 0.0])
            
                # UF850 Home Pose
                self.send_traj_goal(self.uf_traj_pub, 
                    ['u1_joint1', 'u1_joint2', 'u1_joint3', 'u1_joint4', 'u1_joint5', 'u1_joint6'], 
                    [0.0, 0.0, -1.57, 0.0, -1.57, 0.0])

            # --- 📦 BTN 1 (B): GRAB / RELEASE (2/3) ---
            if btns[1] == 1 and self.last_buttons[1] == 0:
                self.grab_active = not self.grab_active
                cmd = 2 if self.grab_active else 3
                self.tool_pub.publish(Int8(data=cmd))
                print(f"🗜️ [Tool] {'GRAB (2)' if self.grab_active else 'RELEASE (3)'}")

            # --- 🔩 BTN 2 (X): UNSCREW / STOP (-1/0) ---
            if btns[2] == 1 and self.last_buttons[2] == 0:
                self.unscrew_active = not self.unscrew_active
                cmd = -1 if self.unscrew_active else 0
                self.tool_pub.publish(Int8(data=cmd))
                print(f"⚙️ [Tool] {'UNSCREW (-1)' if self.unscrew_active else 'STOP (0)'}")

            # --- 🔄 BTN 4 (L1): SWAP ROBOT ---
            if btns[4] == 1 and self.last_buttons[4] == 0:
                self.active_arm = 'uf' if self.active_arm == 'xarm' else 'xarm'
                self.manage_servo_nodes()

            # --- 🛠️ BTN 5 (R1): GRIPPER OPEN / CLOSE ---
            if btns[5] == 1 and self.last_buttons[5] == 0:
                self.gripper_is_closed = not self.gripper_is_closed
                pos = -0.6 if self.gripper_is_closed else 0.6
                print(f"🧤 [Gripper] {'CLOSING (-0.6)' if self.gripper_is_closed else 'OPENING (0.6)'}")
                self.send_traj_goal(self.gripper_traj_pub, ['rg6_l_out'], [pos], duration=0.6)

            # --- 🕹️ BTN 0 (A): DEADMAN SWITCH ---
            if btns[0] == 1:
                self.deadman_active = True
                self.target_linear = [axes[1] * self.linear_scale, axes[0] * self.linear_scale, axes[4] * self.linear_scale]
                self.target_angular[2] = axes[3] * self.angular_scale
                if self.active_arm == 'uf':
                    # D-pad controls for UF850 Pitch/Roll
                    self.target_angular[1] = axes[7] * self.angular_scale
                    self.target_angular[0] = axes[6] * self.angular_scale
            else:
                self.deadman_active = False
                self.target_linear, self.target_angular = [0.0]*3, [0.0]*3

            self.last_buttons = list(btns)
        except Exception as e: self.get_logger().error(f"Joy Error: {e}")

    def continuous_twist_publisher(self):
        for i in range(3):
            self.current_linear[i] = (self.joystick_alpha * self.target_linear[i]) + ((1.0 - self.joystick_alpha) * self.current_linear[i])
            self.current_angular[i] = (self.joystick_alpha * self.target_angular[i]) + ((1.0 - self.joystick_alpha) * self.current_angular[i])

        if self.deadman_active or any(abs(v) > 0.001 for v in self.current_linear + self.current_angular):
            tw = TwistStamped()
            tw.header.stamp, tw.header.frame_id = self.get_clock().now().to_msg(), self.planning_frame
            tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = self.current_linear
            tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z = self.current_angular
            (self.xarm_pub if self.active_arm == 'xarm' else self.uf_pub).publish(tw)

    def send_traj_goal(self, publisher, names, positions, duration=4.0):
        msg = JointTrajectory()
        msg.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        msg.points.append(point)
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TeleopBridge())
    rclpy.shutdown()

if __name__ == '__main__': main()