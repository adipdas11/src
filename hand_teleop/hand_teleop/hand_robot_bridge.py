#!/usr/bin/env python3
"""
Hand-to-Robot Bridge Node for Dual-Arm Teleoperation.

Subscribes to hand tracking topics from hand_tracker_node and converts
hand wrist movements into MoveIt Servo delta-twist commands.

Control model: displacement-from-center (virtual joystick).
  - Close fist to enable (deadman) and calibrate center
  - Move hand away from center -> proportional velocity
  - Return hand to center -> robot stops
  - Right pinch -> toggle gripper
  - Left pinch -> recenter left arm

Publishes to the same servo topics as teleop_bridge.py.
"""

import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3Stamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool, String
import json
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration


class ArmState:
    """Per-arm tracking state."""

    __slots__ = [
        'center', 'center_set', 'current_pos', 'fist_active',
        'pinch_active', 'pinch_prev', 'last_msg_time',
        'filtered_linear', 'halt_count',
        'mode', 'angular_vel', 'filtered_angular',
        'arm_orientation', 'ref_orientation',
    ]

    def __init__(self):
        self.center = [0.5, 0.5, 0.0]   # Normalized center point
        self.center_set = False
        self.current_pos = [0.5, 0.5, 0.0]
        self.fist_active = False
        self.pinch_active = False
        self.pinch_prev = False
        self.last_msg_time = 0.0
        self.filtered_linear = [0.0, 0.0, 0.0]
        self.halt_count = 99  # Start high so no spurious commands
        self.mode = 'position'            # 'position' or 'orientation'
        self.angular_vel = [0.0, 0.0, 0.0]   # Latest angular vel from tracker
        self.filtered_angular = [0.0, 0.0, 0.0]
        self.arm_orientation = None       # Current arm quaternion [x,y,z,w]
        self.ref_orientation = None       # Reference quat (captured on mode switch)


class HandRobotBridge(Node):
    """Bridge between hand tracking and MoveIt Servo."""

    def __init__(self):
        super().__init__('hand_robot_bridge')

        # --- Parameters ---
        # --- Position mode params ---
        self.declare_parameter('linear_scale', 1.2)
        self.declare_parameter('max_linear_vel', 0.15)
        self.declare_parameter('deadzone', 0.03)
        self.declare_parameter('smoothing_alpha', 0.12)
        self.declare_parameter('publish_rate', 33.0)
        self.declare_parameter('halt_cycles', 4)
        self.declare_parameter('axis_map_x', '-y')
        self.declare_parameter('axis_map_y', '-z')
        self.declare_parameter('axis_map_z', 'x')
        self.declare_parameter('enable_gripper', True)
        self.declare_parameter('gripper_open_pos', 0.6)
        self.declare_parameter('gripper_close_pos', -0.6)
        self.declare_parameter('auto_center', True)
        self.declare_parameter('depth_multiplier', 5.0)
        self.declare_parameter('control_radius', 0.25)
        # --- Orientation mode params (UF850 only) ---
        self.declare_parameter('angular_scale', 1.5)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('angular_deadzone', 0.05)
        self.declare_parameter('angular_alpha', 0.2)
        self.declare_parameter('axis_map_ang_x', 'z')
        self.declare_parameter('axis_map_ang_y', '-x')
        self.declare_parameter('axis_map_ang_z', 'y')

        self.linear_scale = self.get_parameter('linear_scale').value
        self.max_linear = self.get_parameter('max_linear_vel').value
        self.deadzone = self.get_parameter('deadzone').value
        self.alpha = self.get_parameter('smoothing_alpha').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.halt_cycles = self.get_parameter('halt_cycles').value
        self.enable_gripper = self.get_parameter('enable_gripper').value
        self.gripper_open = self.get_parameter('gripper_open_pos').value
        self.gripper_close = self.get_parameter('gripper_close_pos').value
        self.auto_center = self.get_parameter('auto_center').value
        self.depth_mult = self.get_parameter('depth_multiplier').value
        self.control_radius = self.get_parameter('control_radius').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.max_angular = self.get_parameter('max_angular_vel').value
        self.angular_deadzone = self.get_parameter('angular_deadzone').value
        self.angular_alpha = self.get_parameter('angular_alpha').value
        self.angular_axis_map = self._parse_axis_map([
            self.get_parameter('axis_map_ang_x').value,
            self.get_parameter('axis_map_ang_y').value,
            self.get_parameter('axis_map_ang_z').value,
        ])

        # Parse axis map
        self.axis_map = self._parse_axis_map([
            self.get_parameter('axis_map_x').value,
            self.get_parameter('axis_map_y').value,
            self.get_parameter('axis_map_z').value,
        ])

        # --- Per-arm state ---
        self.left = ArmState()   # Left hand -> xArm5
        self.right = ArmState()  # Right hand -> UF850

        # Gripper state
        self.gripper_closed = False

        # Debounce: cooldown after each pinch toggle (seconds)
        self._left_pinch_cooldown = 0.0
        self._right_pinch_cooldown = 0.0
        self._pinch_cooldown_sec = 0.8  # Ignore pinches for this long after toggle

        # --- Publishers ---
        twist_qos = QoSProfile(depth=10)

        self.xarm_pub = self.create_publisher(
            TwistStamped,
            '/xarm_servo_node/delta_twist_cmds',
            twist_qos)
        self.uf_pub = self.create_publisher(
            TwistStamped,
            '/uf_servo_node/delta_twist_cmds',
            twist_qos)
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/rg6_controller/joint_trajectory',
            twist_qos)

        # --- Subscribers (BEST_EFFORT to match hand_tracker) ---
        sub_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            PoseStamped, '/hand_teleop/left/wrist_pose',
            self._left_wrist_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, '/hand_teleop/right/wrist_pose',
            self._right_wrist_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Bool, '/hand_teleop/left/fist',
            self._left_fist_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Bool, '/hand_teleop/right/fist',
            self._right_fist_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Bool, '/hand_teleop/left/pinch',
            self._left_pinch_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Bool, '/hand_teleop/right/pinch',
            self._right_pinch_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Vector3Stamped, '/hand_teleop/left/angular_vel',
            self._left_angular_vel_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            Vector3Stamped, '/hand_teleop/right/angular_vel',
            self._right_angular_vel_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, '/hand_teleop/left/arm_orientation',
            self._left_arm_orient_cb, sub_qos, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, '/hand_teleop/right/arm_orientation',
            self._right_arm_orient_cb, sub_qos, callback_group=cb_group)

        # --- Service clients for servo startup ---
        self.xarm_servo_client = self.create_client(
            Trigger, '/xarm_servo_node/start_servo')
        self.uf_servo_client = self.create_client(
            Trigger, '/uf_servo_node/start_servo')

        # --- Debug publisher (bridge status as JSON for visualization) ---
        self.pub_debug = self.create_publisher(
            String, '/hand_teleop/bridge_status', twist_qos)

        # --- Control loop timer ---
        period = 1.0 / self.publish_rate
        self.control_timer = self.create_timer(period, self._control_loop)

        # --- Status: log every 2s, publish JSON at 10Hz for visualization ---
        self.log_timer = self.create_timer(2.0, self._log_status)
        self.status_timer = self.create_timer(0.1, self._publish_status)
        self._xarm_cmd_count = 0
        self._uf_cmd_count = 0
        self._receiving_data = False

        # --- Servo startup (delayed 3s to wait for servo nodes) ---
        self.startup_timer = self.create_timer(
            3.0, self._start_servos, callback_group=cb_group)

        self.get_logger().info('Hand-Robot Bridge initialized')
        self.get_logger().info(
            f'  Linear scale: {self.linear_scale}, '
            f'Max vel: {self.max_linear} m/s, '
            f'Deadzone: {self.deadzone}')
        self.get_logger().info(
            f'  Axis map: cam_X->{self.axis_map[0]}, '
            f'cam_Y->{self.axis_map[1]}, cam_Z->{self.axis_map[2]}')
        self.get_logger().info(
            '  Left fist = enable xArm5, Right fist = enable UF850')
        self.get_logger().info(
            '  Right pinch = toggle gripper')
        self.get_logger().info(
            '  Left pinch = toggle UF850 position/orientation mode')

    # --- Axis mapping ---

    def _parse_axis_map(self, map_strs):
        """Parse axis map strings like 'y', '-z', 'x', '' into tuples.

        Returns list of (axis_index, sign) or None for disabled axes.
        axis_index: 0=x, 1=y, 2=z in robot twist.
        """
        result = []
        axis_to_idx = {'x': 0, 'y': 1, 'z': 2}
        for s in map_strs:
            s = s.strip()
            if not s:
                result.append(None)
                continue
            sign = 1.0
            if s.startswith('-'):
                sign = -1.0
                s = s[1:]
            if s in axis_to_idx:
                result.append((axis_to_idx[s], sign))
            else:
                self.get_logger().warn(f'Invalid axis_map entry: {s}')
                result.append(None)
        return result

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions [x, y, z, w]."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    @staticmethod
    def _quat_inverse(q):
        """Inverse of unit quaternion [x, y, z, w]."""
        return [-q[0], -q[1], -q[2], q[3]]

    @staticmethod
    def _quat_to_rotvec(q):
        """Convert quaternion [x, y, z, w] to rotation vector (axis * angle)."""
        x, y, z, w = q
        # Clamp w to avoid acos domain errors
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(abs(w))
        if angle < 1e-6:
            return [0.0, 0.0, 0.0]
        s = math.sin(angle / 2.0)
        if s < 1e-8:
            return [0.0, 0.0, 0.0]
        # Handle negative w (ensure shortest rotation path)
        sign = 1.0 if w >= 0 else -1.0
        return [sign * x / s * angle,
                sign * y / s * angle,
                sign * z / s * angle]

    def _map_displacement_to_twist(self, displacement, axis_map=None):
        """Convert camera-frame vector to robot-frame vector via axis mapping.

        Args:
            displacement: [dx, dy, dz] in camera frame
            axis_map: list of (robot_axis_idx, sign) or None entries.
                      Defaults to self.axis_map (linear).

        Returns:
            [vx, vy, vz] in robot frame
        """
        if axis_map is None:
            axis_map = self.axis_map
        twist = [0.0, 0.0, 0.0]
        for cam_axis in range(3):
            mapping = axis_map[cam_axis]
            if mapping is None:
                continue
            robot_axis, sign = mapping
            twist[robot_axis] += sign * displacement[cam_axis]
        return twist

    # --- Callbacks ---

    def _left_wrist_cb(self, msg):
        self.left.current_pos = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        self.left.last_msg_time = time.monotonic()

    def _right_wrist_cb(self, msg):
        self.right.current_pos = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        self.right.last_msg_time = time.monotonic()

    def _left_fist_cb(self, msg):
        prev = self.left.fist_active
        self.left.fist_active = msg.data
        if msg.data and not prev:
            self.get_logger().info('xArm5 ENABLED (left fist closed)')
        elif not msg.data and prev:
            self.get_logger().info('xArm5 DISABLED (left fist opened)')

    def _right_fist_cb(self, msg):
        prev = self.right.fist_active
        self.right.fist_active = msg.data
        if msg.data and not prev:
            self.get_logger().info('UF850 ENABLED (right fist closed)')
        elif not msg.data and prev:
            self.get_logger().info('UF850 DISABLED (right fist opened)')

    def _left_pinch_cb(self, msg):
        prev = self.left.pinch_prev
        current = msg.data
        self.left.pinch_prev = current

        now = time.monotonic()
        # Rising edge + cooldown -> toggle UF850 mode
        if current and not prev and (now - self._left_pinch_cooldown) > self._pinch_cooldown_sec:
            self._left_pinch_cooldown = now
            if self.right.mode == 'position':
                self.right.mode = 'orientation'
                self.right.filtered_angular = [0.0, 0.0, 0.0]
                # Capture current arm orientation as reference
                if self.right.arm_orientation is not None:
                    self.right.ref_orientation = list(
                        self.right.arm_orientation)
                    self.get_logger().info(
                        'UF850 -> ORIENTATION MODE (ref captured)')
                else:
                    # Fallback: recenter for displacement-based control
                    self.right.center = list(self.right.current_pos)
                    self.get_logger().info(
                        'UF850 -> ORIENTATION MODE (no pose, recentered)')
            else:
                self.right.mode = 'position'
                self.right.filtered_linear = [0.0, 0.0, 0.0]
                self.right.center = list(self.right.current_pos)
                self.get_logger().info(
                    'UF850 -> POSITION MODE (recentered)')

    def _right_pinch_cb(self, msg):
        prev = self.right.pinch_prev
        current = msg.data
        self.right.pinch_prev = current

        now = time.monotonic()
        # Rising edge + cooldown -> toggle gripper
        if (current and not prev and self.enable_gripper
                and (now - self._right_pinch_cooldown) > self._pinch_cooldown_sec):
            self._right_pinch_cooldown = now
            self.gripper_closed = not self.gripper_closed
            self._send_gripper(self.gripper_closed)

    def _left_angular_vel_cb(self, msg):
        self.left.angular_vel = [
            msg.vector.x, msg.vector.y, msg.vector.z]

    def _right_angular_vel_cb(self, msg):
        self.right.angular_vel = [
            msg.vector.x, msg.vector.y, msg.vector.z]

    def _left_arm_orient_cb(self, msg):
        q = msg.pose.orientation
        self.left.arm_orientation = [q.x, q.y, q.z, q.w]

    def _right_arm_orient_cb(self, msg):
        q = msg.pose.orientation
        self.right.arm_orientation = [q.x, q.y, q.z, q.w]

    # --- Gripper control ---

    def _send_gripper(self, close):
        """Send gripper open/close trajectory."""
        pos = self.gripper_close if close else self.gripper_open
        traj = JointTrajectory()
        traj.joint_names = ['rg6_l_out']
        point = JointTrajectoryPoint()
        point.positions = [pos]
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)
        traj.points = [point]
        self.gripper_pub.publish(traj)
        state_str = 'CLOSED' if close else 'OPEN'
        self.get_logger().info(f'Gripper -> {state_str} ({pos:.1f} rad)')

    # --- Servo startup ---

    def _start_servos(self):
        """Call start_servo for both MoveIt Servo nodes."""
        self.startup_timer.cancel()

        for name, client in [
            ('xArm5', self.xarm_servo_client),
            ('UF850', self.uf_servo_client),
        ]:
            if client.service_is_ready():
                future = client.call_async(Trigger.Request())
                future.add_done_callback(
                    lambda f, n=name: self._servo_started(f, n))
            else:
                self.get_logger().warn(
                    f'{name} servo service not available — '
                    f'make sure demo.launch.py is running')

    def _servo_started(self, future, name):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'{name} servo started')
            else:
                self.get_logger().warn(
                    f'{name} servo start failed: {result.message}')
        except Exception as e:
            self.get_logger().warn(f'{name} servo start error: {e}')

    # --- Core control loop ---

    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def _make_twist(self, linear=None, angular=None):
        """Build a TwistStamped in world_world frame."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world_world'
        if linear is not None:
            msg.twist.linear.x = float(linear[0])
            msg.twist.linear.y = float(linear[1])
            msg.twist.linear.z = float(linear[2])
        if angular is not None:
            msg.twist.angular.x = float(angular[0])
            msg.twist.angular.y = float(angular[1])
            msg.twist.angular.z = float(angular[2])
        return msg

    def _make_zero_twist(self):
        return self._make_twist([0.0, 0.0, 0.0])

    def _process_arm(self, state, publisher, arm_name):
        """Process one arm's control for this tick."""
        now = time.monotonic()

        # Watchdog: no data for 0.5s
        if state.last_msg_time > 0 and (now - state.last_msg_time) > 0.5:
            if state.halt_count < self.halt_cycles:
                publisher.publish(self._make_zero_twist())
                state.halt_count += 1
            state.filtered_linear = [0.0, 0.0, 0.0]
            state.filtered_angular = [0.0, 0.0, 0.0]
            return

        # Deadman: fist not active
        if not state.fist_active:
            if state.halt_count < self.halt_cycles:
                publisher.publish(self._make_zero_twist())
                state.halt_count += 1
            state.filtered_linear = [0.0, 0.0, 0.0]
            state.filtered_angular = [0.0, 0.0, 0.0]
            return

        # Auto-calibrate center on first fist activation
        if not state.center_set and self.auto_center:
            state.center = list(state.current_pos)
            state.center_set = True
            self.get_logger().info(
                f'{arm_name} center calibrated at '
                f'({state.center[0]:.2f}, {state.center[1]:.2f})')

        # ---- ORIENTATION MODE (UF850 only) ----
        # Proportional pose-matching control: robot gripper tracks human arm.
        # error = current_arm_quat * ref_quat.inv() -> axis*angle -> angular_vel
        # Fallback: angular-velocity-based if no quaternion data.
        if state.mode == 'orientation':
            if (state.arm_orientation is not None
                    and state.ref_orientation is not None):
                # Quaternion error: how far current arm has rotated from ref
                ref_inv = self._quat_inverse(state.ref_orientation)
                error_q = self._quat_multiply(
                    state.arm_orientation, ref_inv)
                error_rotvec = self._quat_to_rotvec(error_q)

                # Map camera-frame rotation to robot frame
                mapped = self._map_displacement_to_twist(
                    error_rotvec, self.angular_axis_map)
            else:
                # Fallback: use angular velocity from tracker
                raw_ang = list(state.angular_vel)
                mapped = self._map_displacement_to_twist(
                    raw_ang, self.angular_axis_map)

            # Deadzone: ignore small angular jitter
            for i in range(3):
                if abs(mapped[i]) < self.angular_deadzone:
                    mapped[i] = 0.0

            # Scale (proportional gain) and clamp
            for i in range(3):
                mapped[i] *= self.angular_scale
                mapped[i] = self._clamp(mapped[i], self.max_angular)

            # Exponential smoothing
            for i in range(3):
                state.filtered_angular[i] = (
                    self.angular_alpha * mapped[i] +
                    (1.0 - self.angular_alpha) * state.filtered_angular[i])

            publisher.publish(self._make_twist(angular=state.filtered_angular))
            state.halt_count = 0
            if publisher == self.xarm_pub:
                self._xarm_cmd_count += 1
            else:
                self._uf_cmd_count += 1
            return

        # ---- POSITION MODE ----
        disp = [
            state.current_pos[i] - state.center[i]
            for i in range(3)
        ]

        # Amplify Z (depth) — hand scale delta is smaller than X/Y displacement
        disp[2] *= self.depth_mult

        # Deadzone
        for i in range(3):
            if abs(disp[i]) < self.deadzone:
                disp[i] = 0.0
            else:
                sign = 1.0 if disp[i] > 0 else -1.0
                disp[i] = sign * (abs(disp[i]) - self.deadzone)

        # Normalize by control_radius: hand at radius edge = 1.0, clamp beyond
        for i in range(3):
            disp[i] = self._clamp(disp[i] / self.control_radius, 1.0)

        # Quadratic curve: fine near center, fast at edge
        for i in range(3):
            disp[i] = disp[i] * abs(disp[i])

        # Map camera axes to robot axes
        raw_twist = self._map_displacement_to_twist(disp)

        # Scale to max velocity
        for i in range(3):
            raw_twist[i] *= self.max_linear
            raw_twist[i] = self._clamp(raw_twist[i], self.max_linear)

        # Exponential smoothing
        for i in range(3):
            state.filtered_linear[i] = (
                self.alpha * raw_twist[i] +
                (1.0 - self.alpha) * state.filtered_linear[i])

        publisher.publish(self._make_twist(linear=state.filtered_linear))
        state.halt_count = 0

        if publisher == self.xarm_pub:
            self._xarm_cmd_count += 1
        else:
            self._uf_cmd_count += 1

    def _log_status(self):
        """Periodic status logging to help debug."""
        now = time.monotonic()
        left_age = (now - self.left.last_msg_time
                    if self.left.last_msg_time > 0 else -1)
        right_age = (now - self.right.last_msg_time
                     if self.right.last_msg_time > 0 else -1)

        receiving = left_age >= 0 or right_age >= 0
        if receiving and not self._receiving_data:
            self.get_logger().info('Receiving hand tracking data')
            self._receiving_data = True
        elif not receiving and self._receiving_data:
            self.get_logger().warn('Lost hand tracking data')
            self._receiving_data = False

        if not receiving:
            self.get_logger().info(
                'Waiting for hand tracker... '
                '(is hand_tracker node running?)',
                throttle_duration_sec=5.0)
            return

        # Build status line
        def arm_str(state, name, cmd_count):
            if state.last_msg_time <= 0:
                return f'{name}: no data'
            fist = 'FIST' if state.fist_active else 'open'
            centered = 'centered' if state.center_set else 'not centered'
            if state.mode == 'orientation':
                v = state.filtered_angular
                vel_str = f'ang=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) [ORI]'
            else:
                v = state.filtered_linear
                vel_str = f'vel=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}) [POS]'
            return f'{name}: {fist} {centered} {vel_str} cmds={cmd_count}'

        self.get_logger().info(
            f'{arm_str(self.left, "xArm5", self._xarm_cmd_count)} | '
            f'{arm_str(self.right, "UF850", self._uf_cmd_count)}')

    def _publish_status(self):
        """Publish bridge status JSON at 10Hz for tracker visualization."""
        status = {
            'xarm5': {
                'fist': self.left.fist_active,
                'centered': self.left.center_set,
                'mode': self.left.mode,
                'active': self.left.fist_active and self.left.last_msg_time > 0,
            },
            'uf850': {
                'fist': self.right.fist_active,
                'centered': self.right.center_set,
                'mode': self.right.mode,
                'active': self.right.fist_active and self.right.last_msg_time > 0,
            },
            'gripper_closed': self.gripper_closed,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_debug.publish(msg)

    def _control_loop(self):
        """Main control loop at publish_rate Hz."""
        self._process_arm(self.left, self.xarm_pub, 'xArm5')
        self._process_arm(self.right, self.uf_pub, 'UF850')

    def destroy_node(self):
        """Publish zero twist on shutdown."""
        self.xarm_pub.publish(self._make_zero_twist())
        self.uf_pub.publish(self._make_zero_twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandRobotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
