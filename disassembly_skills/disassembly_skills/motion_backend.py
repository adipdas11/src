#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from sensor_msgs.msg import JointState

# --- TF IMPORTS ---
import tf2_ros
# This import is CRITICAL. It registers PoseStamped support for buffer.transform()
import tf2_geometry_msgs 
from tf2_ros import Buffer, TransformListener

import threading
import math
import time

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        
        self.current_joint_msg = None
        self.current_joint_positions = {}
        self.state_received = threading.Event()
        
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        # --- Internal TF Buffer ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def _joint_state_callback(self, msg):
        self.current_joint_msg = msg
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_positions[name] = pos
        self.state_received.set()

    def _rpy_to_quaternion(self, roll, pitch, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return Quaternion(w=cr*cp*cy + sr*sp*sy, x=sr*cp*cy - cr*sp*sy, y=cr*sp*cy + sr*cp*sy, z=cr*cp*sy - sr*sp*cy)

    # ==========================================================
    # IMPROVED: Coordinate Transformation
    # ==========================================================
    def get_transformed_pose(self, source_pose, source_frame: str, target_frame: str, z_offset=0.0):
        """
        Robustly transforms a pose from source_frame to target_frame.
        Handles both Pose and PoseStamped inputs to prevent AttributeErrors.
        """
        try:
            # 1. sanitize Input: Ensure we have a pure Pose object
            real_pose = source_pose
            if hasattr(source_pose, 'pose'): # If user passed a PoseStamped by mistake
                real_pose = source_pose.pose
            
            # 2. Create the wrapper PoseStamped for TF
            p_stamped = PoseStamped()
            p_stamped.header.frame_id = source_frame
            # Use 0 time to get the latest available transform (most robust)
            p_stamped.header.stamp = rclpy.time.Time().to_msg()
            p_stamped.pose = real_pose

            # 3. Check Transform Availability
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                self.node.get_logger().warn(f"Transform missing: {source_frame} -> {target_frame}")
                return None

            # 4. Perform Transform (Using buffer.transform is safer than manual do_transform)
            target_pose_stamped = self.tf_buffer.transform(p_stamped, target_frame)

            # 5. Apply Offset
            target_pose_stamped.pose.position.z += z_offset

            return target_pose_stamped

        except Exception as e:
            self.node.get_logger().error(f"TF Error in get_transformed_pose: {e}")
            return None

    # ==========================================================
    # EXISTING MOVEMENT LOGIC
    # ==========================================================
    def move_to_joint_positions(self, target_joints: dict, filter_prefix: str = "", velocity=0.1, acceleration=0.1):
        if not self.state_received.wait(timeout=2.0): return False
        full_goal_map = self.current_joint_positions.copy()
        full_goal_map.update(target_joints)

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.group_name
        
        constraints = Constraints()
        for name, pos in full_goal_map.items():
            if filter_prefix in name:
                jc = JointConstraint()
                jc.joint_name, jc.position, jc.weight = name, pos, 1.0
                if "rg6" in name:
                    jc.tolerance_above = jc.tolerance_below = 0.1 
                else:
                    jc.tolerance_above = jc.tolerance_below = 0.02
                constraints.joint_constraints.append(jc)
        
        goal_msg.request.goal_constraints.append(constraints)
        return self._send_goal(goal_msg, velocity, acceleration)

    def move_to_pose_robust(self, x, y, z, q_dict, link_name, frame_id='world_world', velocity=0.1, acceleration=0.1):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = link_name
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 2 
        
        req.ik_request.robot_state = RobotState()
        if self.current_joint_msg:
            req.ik_request.robot_state.joint_state = self.current_joint_msg
        req.ik_request.robot_state.is_diff = True
        
        target_pose = PoseStamped()
        target_pose.header.frame_id = frame_id 
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z

        if q_dict and all(k in q_dict for k in ['qx', 'qy', 'qz', 'qw']):
            target_pose.pose.orientation.x = q_dict['qx']
            target_pose.pose.orientation.y = q_dict['qy']
            target_pose.pose.orientation.z = q_dict['qz']
            target_pose.pose.orientation.w = q_dict['qw']
            req.ik_request.pose_stamped = target_pose
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: 
                return self._process_ik_result(res, velocity, acceleration)

        target_pose.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, 0.0)
        req.ik_request.pose_stamped = target_pose
        res = self._call_ik_sync(req)
        if res.error_code.val == 1: 
            return self._process_ik_result(res, velocity, acceleration)

        for yaw in [0.4, -0.4, 0.8, -0.8, 1.57, -1.57]:
            target_pose.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw)
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: 
                return self._process_ik_result(res, velocity, acceleration)

        return False

    def _call_ik_sync(self, req):
        future = self._ik_client.call_async(req)
        while not future.done(): time.sleep(0.01)
        return future.result()

    def _process_ik_result(self, response, velocity, acceleration):
        ik_joints = {name: pos for name, pos in zip(response.solution.joint_state.name, response.solution.joint_state.position)}
        prefix = "xarm5" if "xarm" in self.group_name else "u1"
        return self.move_to_joint_positions(ik_joints, filter_prefix=prefix, velocity=velocity, acceleration=acceleration)

    def _send_goal(self, goal_msg, velocity, acceleration):
        if not self._action_client.wait_for_server(timeout_sec=5.0): return False
        goal_msg.request.max_velocity_scaling_factor = velocity
        goal_msg.request.max_acceleration_scaling_factor = acceleration
        goal_msg.request.allowed_planning_time = 5.0
        future = self._action_client.send_goal_async(goal_msg)
        while not future.done(): time.sleep(0.1)
        handle = future.result()
        if not handle.accepted: return False
        res_future = handle.get_result_async()
        while not res_future.done(): time.sleep(0.1)
        return res_future.result().result.error_code.val == 1