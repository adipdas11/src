#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_ros import Buffer, TransformListener
import threading
import math
import time
import copy
import std_srvs.srv

from xarm.wrapper import XArmAPI

# ==========================================
# DYNAMIC SDK IPs
XARM_IP = '192.168.1.239'
UF850_IP = '192.168.1.195'
# ==========================================

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        self.is_xarm5 = "xarm" in group_name.lower()

        # Determine prefixes for joint filtering
        if self.is_xarm5:
            self._prefixes = ['xarm5', 'slider']
        elif 'rg6' in self.group_name:
            self._prefixes = ['rg6']
        else:
            self._prefixes = ['u1']

        self._target_link = "xarm5_link5" if self.is_xarm5 else "u1_tool0"

        # --- ROS 2 MoveIt Interface ---
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self._cartesian_client = self.node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._stop_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/stop_robot')
        self._reset_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/reset_robot')

        # --- Direct Hardware Connection (SDK) ---
        target_ip = None
        if "uf" in self.group_name.lower():
            target_ip = UF850_IP
        elif "xarm" in self.group_name.lower():
            target_ip = XARM_IP

        self._sdk_mode = -1  # Track current SDK mode to avoid redundant switches

        if target_ip:
            try:
                self.node.get_logger().info(f"Connecting SDK to {target_ip} for '{self.group_name}'...")
                self.arm = XArmAPI(target_ip)
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(0)
                self.arm.set_state(state=0)
                self.arm.set_tcp_offset([0, 0, 0, 0, 0, 0])
                self._sdk_mode = 0
                self.node.get_logger().info(f"SDK Connected ({target_ip}) & TCP Offset Cleared!")
            except Exception as e:
                self.node.get_logger().error(f"Failed to connect SDK at {target_ip}: {e}")
        else:
            self.node.get_logger().info(f"No SDK connection needed for '{self.group_name}'.")

        self._current_goal_handle = None
        self.current_joint_msg = None
        self.current_joint_positions = {}
        self.current_joint_efforts = {}
        self.state_received = threading.Event()

        # Subscribers & TF
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    def _joint_state_callback(self, msg):
        self.current_joint_msg = msg
        for i, name in enumerate(msg.name):
            self.current_joint_positions[name] = msg.position[i]
            if len(msg.effort) > i:
                self.current_joint_efforts[name] = msg.effort[i]
        self.state_received.set()

    def _get_full_robot_state(self):
        state = RobotState()
        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.name = list(self.current_joint_positions.keys())
        js.position = list(self.current_joint_positions.values())
        state.joint_state = js
        return state

    def _rpy_to_quaternion(self, roll, pitch, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return Quaternion(w=cr*cp*cy + sr*sp*sy, x=sr*cp*cy - cr*sp*sy, y=cr*sp*cy + sr*cp*sy, z=cr*cp*sy - sr*sp*cy)

    def _ensure_sdk_mode(self, mode=0):
        """Switch SDK mode. We must unconditionally enforce this because MoveIt/ros2_control steals the mode dynamically."""
        if not hasattr(self, 'arm'):
            return False
        
        _, state = self.arm.get_state()
        if state == 4:
            self.arm.clean_error()
            self.arm.motion_enable(enable=True)
            
        self.arm.set_mode(mode)
        self.arm.set_state(state=0)
        self._sdk_mode = mode
        time.sleep(0.05)
        
        return True

    def _call_ik_sync(self, req, timeout=10.0):
        future = self._ik_client.call_async(req)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                self.node.get_logger().error("IK service timeout.")
                return type('FakeResult', (), {'error_code': type('EC', (), {'val': -1})()})()
            time.sleep(0.01)
        return future.result()

    def _execute_joint_goal(self, joint_state, velocity):
        """Send a MoveGroup joint goal and wait for completion with timeouts."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = velocity

        constraints = Constraints()
        found_any = False
        for n, p in zip(joint_state.name, joint_state.position):
            if any(n.startswith(pfx) for pfx in self._prefixes):
                jc = JointConstraint()
                jc.joint_name, jc.position, jc.weight = n, p, 1.0
                jc.tolerance_above = jc.tolerance_below = 0.01
                constraints.joint_constraints.append(jc)
                found_any = True
        if not found_any:
            return False

        goal.request.goal_constraints.append(constraints)

        future = self._action_client.send_goal_async(goal)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 30.0:
                self.node.get_logger().error("Goal acceptance timeout (30s).")
                return False
            time.sleep(0.01)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        t0 = time.time()
        while not result_future.done():
            if time.time() - t0 > 60.0:
                self.node.get_logger().error("Goal execution timeout (60s).")
                return False
            time.sleep(0.01)

        err_code = result_future.result().result.error_code.val
        if err_code not in [1, -4]:
            self.node.get_logger().warn(f"MoveIt executed with non-success code. Expected 1 or -4, got: {err_code}")
        return err_code in [1, -4]  # -4 typically means PREEMPTED which happens occasionally even when physically arriving 

    # =====================================================================
    # HARDWARE MANAGEMENT
    # =====================================================================

    def reset_robot(self):
        """Resets errors on both hardware SDK and ROS driver side."""
        if hasattr(self, 'arm'):
            self.arm.clean_error()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            self.arm.set_tcp_offset([0, 0, 0, 0, 0, 0])
            self._sdk_mode = 0

        if self._reset_srv.wait_for_service(timeout_sec=1.0):
            self._reset_srv.call_async(std_srvs.srv.Empty.Request())
            self.node.get_logger().info("Robot Reset & Enabled")

    def stop_immediately(self):
        """Triggers emergency stop on hardware and cancels ROS goals."""
        if self._current_goal_handle:
            self._current_goal_handle.cancel_goal_async()
        if hasattr(self, 'arm'):
            self.arm.set_state(state=4)
            self._sdk_mode = -1
        if self._stop_srv.wait_for_service(timeout_sec=0.1):
            self._stop_srv.call_async(std_srvs.srv.Empty.Request())
            self.node.get_logger().error("!!! HARDWARE STOP SENT !!!")

    # =====================================================================
    # TF & POSE UTILITIES
    # =====================================================================

    def get_transformed_pose(self, source_pose, source_frame: str, target_frame: str, z_offset=0.0):
        try:
            import tf2_geometry_msgs
            real_pose = source_pose.pose if hasattr(source_pose, 'pose') else source_pose
            p = PoseStamped()
            p.header.frame_id = source_frame
            p.header.stamp = rclpy.time.Time().to_msg()
            p.pose = real_pose

            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=1.0)):
                return None
            t = self.tf_buffer.transform(p, target_frame)
            t.pose.position.z += z_offset
            return t
        except Exception as e:
            self.node.get_logger().error(f"TF Error: {e}")
            return None

    def get_current_tcp_position(self, frame_id='world_world'):
        """Returns (x, y, z) of the current TCP in the given frame, or None."""
        try:
            if self.tf_buffer.can_transform(frame_id, self._target_link, rclpy.time.Time(),
                                             timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.tf_buffer.lookup_transform(frame_id, self._target_link, rclpy.time.Time())
                return (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        except Exception:
            pass
        return None

    def get_current_tcp_pose(self, frame_id='world_world'):
        """Returns ((x, y, z), (qx, qy, qz, qw)) of current TCP, or (None, None)."""
        try:
            if self.tf_buffer.can_transform(frame_id, self._target_link, rclpy.time.Time(),
                                             timeout=rclpy.duration.Duration(seconds=2.0)):
                t = self.tf_buffer.lookup_transform(frame_id, self._target_link, rclpy.time.Time())
                pos = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                quat = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                return pos, quat
        except Exception:
            pass
        return None, None

    # =====================================================================
    # SETTLING & MONITORING
    # =====================================================================

    def wait_for_arm_settled(self, timeout=20.0, noise_tolerance=0.006, settle_duration=0.4):
        """Monitors joint states, proceeds when all joints stop moving."""
        time.sleep(0.2)
        start_t = time.time()
        settle_timer = 0.0
        last_positions = {}

        while rclpy.ok() and (time.time() - start_t) < timeout:
            curr_positions = self.current_joint_positions.copy()
            if not curr_positions:
                time.sleep(0.1)
                continue

            if last_positions:
                max_delta = 0.0
                for j_name, j_pos in curr_positions.items():
                    if j_name in last_positions:
                        delta = abs(j_pos - last_positions[j_name])
                        if delta > max_delta:
                            max_delta = delta

                if max_delta <= noise_tolerance:
                    settle_timer += 0.1
                    if settle_timer >= settle_duration:
                        return True
                else:
                    settle_timer = 0.0

            last_positions = curr_positions
            time.sleep(0.1)

        return True  # Proceed anyway on timeout

    # =====================================================================
    # MOVEIT 2 PLANNED MOTION
    # =====================================================================

    def move_to_pose_robust(self, x, y, z, q_dict=None, link_name=None, frame_id='world_world', velocity=0.1):
        """IK + yaw sweep fallback. Accepts q_dict=None or {} for auto vertical orientation."""
        if link_name is None:
            link_name = self._target_link

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = link_name
        req.ik_request.avoid_collisions = True
        target_pose = PoseStamped()
        target_pose.header.frame_id = frame_id
        target_pose.header.stamp = self.node.get_clock().now().to_msg()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z

        # Try explicit orientation first
        if q_dict:
            target_pose.pose.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
            req.ik_request.pose_stamped = target_pose
            req.ik_request.robot_state = self._get_full_robot_state()
            res = self._call_ik_sync(req)
            if res.error_code.val == 1:
                return self._execute_joint_goal(res.solution.joint_state, velocity)

        # Yaw sweep fallback (vertical orientation with varying yaw)
        for yaw in [0.0, 0.4, -0.4, 0.8, -0.8, 1.57, -1.57, 3.14]:
            target_pose.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw)
            req.ik_request.pose_stamped = target_pose
            req.ik_request.robot_state = self._get_full_robot_state()
            res = self._call_ik_sync(req)
            if res.error_code.val == 1:
                return self._execute_joint_goal(res.solution.joint_state, velocity)
        return False

    def move_cartesian_to_pose(self, x, y, z, q_dict=None, velocity=0.1, frame_id='world_world'):
        """Straight-line Cartesian path planning. Solves IK incrementally — works where single-shot fails on 5-DOF."""
        target = Pose()
        target.position.x, target.position.y, target.position.z = x, y, z

        if q_dict:
            target.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
        else:
            try:
                current_tf = self.tf_buffer.lookup_transform(frame_id, self._target_link, rclpy.time.Time())
                q = current_tf.transform.rotation
                target.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
            except Exception:
                target.orientation = self._rpy_to_quaternion(math.pi, 0.0, 0.0)

        req = GetCartesianPath.Request()
        req.header.frame_id = frame_id
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.start_state = self._get_full_robot_state()
        req.group_name = self.group_name
        req.link_name = self._target_link
        req.waypoints = [target]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        future = self._cartesian_client.call_async(req)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 15.0:
                return False
            time.sleep(0.01)

        result = future.result()
        if result.fraction < 0.90:
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = result.solution

        future = self._execute_client.send_goal_async(exec_goal)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 30.0:
                return False
            time.sleep(0.01)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        t0 = time.time()
        while not result_future.done():
            if time.time() - t0 > 60.0:
                return False
            time.sleep(0.01)

        return result_future.result().result.error_code.val == 1

    def move_to_joint_positions(self, target_joints, filter_prefix=None, velocity=0.1):
        """Move to joint positions. filter_prefix auto-detected if not provided."""
        if not self.state_received.wait(2.0):
            return False

        if filter_prefix is None:
            filter_prefix = self._prefixes[0]

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = velocity
        constraints = Constraints()
        found_any = False
        for name, pos in target_joints.items():
            if any(name.startswith(pfx) for pfx in (self._prefixes if filter_prefix is None else [filter_prefix])):
                jc = JointConstraint()
                jc.joint_name, jc.position, jc.weight = name, float(pos), 1.0
                jc.tolerance_above = jc.tolerance_below = 0.01
                constraints.joint_constraints.append(jc)
                found_any = True

        if not found_any:
            return False

        goal.request.goal_constraints.append(constraints)

        future = self._action_client.send_goal_async(goal)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 30.0:
                return False
            time.sleep(0.01)

        gh = future.result()
        if not gh.accepted:
            return False

        rf = gh.get_result_async()
        t0 = time.time()
        while not rf.done():
            if time.time() - t0 > 60.0:
                return False
            time.sleep(0.01)

        return rf.result().result.error_code.val == 1

    def retract_relative_z(self, distance, velocity=0.05):
        """Planned relative lift along the Z-axis using MoveIt."""
        try:
            current_tf = self.tf_buffer.lookup_transform('world_world', self._target_link, rclpy.time.Time())
            tx = current_tf.transform.translation.x
            ty = current_tf.transform.translation.y
            tz = current_tf.transform.translation.z + distance
            q = current_tf.transform.rotation
            q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            return self.move_to_pose_robust(tx, ty, tz, q_dict, velocity=velocity)
        except Exception as e:
            self.node.get_logger().error(f"Retract TF Error: {e}")
            return False

    def move_linear_z_with_force_stop(self, distance_down, velocity_scaling, check_force_callback):
        """MoveIt planned Z descent with force monitoring during execution."""
        if not self._execute_client.wait_for_server(timeout_sec=2.0):
            return False
        s = Pose()
        s.orientation.w = 1.0
        start_pose = self.get_transformed_pose(s, self._target_link, 'world_world')
        if not start_pose:
            return False

        target_pose = copy.deepcopy(start_pose)
        target_pose.pose.position.z -= abs(distance_down)

        ik_solution = None
        for yaw_offset in [0.0, 0.1, -0.1, 0.2, -0.2]:
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.group_name
            req.ik_request.ik_link_name = self._target_link
            req.ik_request.avoid_collisions = True
            ik_pose = PoseStamped()
            ik_pose.header.frame_id = 'world_world'
            ik_pose.pose = target_pose.pose
            ik_pose.pose.orientation = start_pose.pose.orientation
            req.ik_request.pose_stamped = ik_pose
            req.ik_request.robot_state = self._get_full_robot_state()
            res = self._call_ik_sync(req)
            if res.error_code.val == 1:
                ik_solution = res.solution.joint_state
                break

        if not ik_solution:
            return False

        goal_joints = {name: pos for name, pos in zip(ik_solution.name, ik_solution.position)}
        mg_goal = MoveGroup.Goal()
        mg_goal.request.group_name = self.group_name
        mg_goal.request.max_velocity_scaling_factor = velocity_scaling
        mg_goal.request.max_acceleration_scaling_factor = 0.05
        mg_goal.request.allowed_planning_time = 2.0
        constraints = Constraints()
        prefix = "xarm" if self.is_xarm5 else "u1"
        for name, pos in goal_joints.items():
            if prefix in name:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = pos
                jc.weight = 1.0
                jc.tolerance_above = 0.001
                jc.tolerance_below = 0.001
                constraints.joint_constraints.append(jc)
        mg_goal.request.goal_constraints.append(constraints)
        mg_goal.planning_options.plan_only = True

        plan_future = self._action_client.send_goal_async(mg_goal)
        while not plan_future.done():
            time.sleep(0.01)
        plan_handle = plan_future.result()
        res_future = plan_handle.get_result_async()
        while not res_future.done():
            time.sleep(0.01)
        plan_result = res_future.result().result

        if plan_result.error_code.val != 1:
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan_result.planned_trajectory
        gf = self._execute_client.send_goal_async(goal)
        while not gf.done():
            time.sleep(0.01)
        self._current_goal_handle = gf.result()
        rf = self._current_goal_handle.get_result_async()

        start_t = time.time()
        while not rf.done():
            if (time.time() - start_t) > 0.5 and check_force_callback():
                self.stop_immediately()
                return True
            time.sleep(0.005)
        return True

    # =====================================================================
    # DIRECT PYTHON SDK CONTROLLERS (Position Mode 0)
    # =====================================================================

    def jog_cartesian_sdk(self, dx_m, dy_m, dz_m, speed_mm_s=35.0):
        """Absolute calculated Cartesian jog via SDK. Blocks until move complete."""
        if not self._ensure_sdk_mode(0):
            return False
            
        code, curr_pos = self.arm.get_position(is_radian=False)
        if code != 0 or not curr_pos:
            return False
            
        target_x = curr_pos[0] + (dx_m * 1000.0)
        target_y = curr_pos[1] + (dy_m * 1000.0)
        target_z = curr_pos[2] + (dz_m * 1000.0)
        
        code = self.arm.set_position(x=target_x, y=target_y, z=target_z, 
                                     roll=curr_pos[3], pitch=curr_pos[4], yaw=curr_pos[5],
                                     speed=speed_mm_s, is_radian=False, wait=True)
        return code == 0

    def move_to_absolute_pose_sdk(self, x_m, y_m, z_m, speed_mm_s=50.0):
        """Absolute Cartesian move via SDK, preserving current orientation."""
        if not self._ensure_sdk_mode(0):
            return False
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        z_mm = z_m * 1000.0

        code, curr_pos = self.arm.get_position(is_radian=False)
        if code != 0 or not curr_pos:
            return False
        curr_r, curr_p, curr_y = curr_pos[3], curr_pos[4], curr_pos[5]

        ret = self.arm.set_position(x=x_mm, y=y_mm, z=z_mm, roll=curr_r, pitch=curr_p, yaw=curr_y,
                                     speed=speed_mm_s, is_radian=False, wait=True)
        return ret == 0

    def move_linear_z_sdk_with_force_stop(self, distance_down_m, speed_mm_s, check_force_callback):
        """SDK Z descent using discrete absolute steps (Stepwise Position Control)."""
        if not self._ensure_sdk_mode(0):
            return False

        code, curr_pos = self.arm.get_position(is_radian=False)
        if code != 0 or not curr_pos:
            return False

        # Step 2mm at a time downwards
        step_mm = 2.0
        total_dist_mm = abs(distance_down_m * 1000.0)
        steps = int(total_dist_mm / step_mm)

        for _ in range(steps):
            if not rclpy.ok(): break
            
            # Check for force before moving
            if check_force_callback():
                self.arm.set_state(state=4)
                self._sdk_mode = -1
                return True
                
            curr_pos[2] -= step_mm
            # Block and wait for step to finish
            self.arm.set_position(x=curr_pos[0], y=curr_pos[1], z=curr_pos[2],
                                  roll=curr_pos[3], pitch=curr_pos[4], yaw=curr_pos[5],
                                  speed=speed_mm_s, is_radian=False, wait=True)
                                  
        return True # Finished distance max limit

    def retract_sdk_z_closed_loop(self, distance, speed_mm_s=35.0, timeout=30.0):
        """SDK-based clear Z retract."""
        return self.jog_cartesian_sdk(0, 0, distance, speed_mm_s)
