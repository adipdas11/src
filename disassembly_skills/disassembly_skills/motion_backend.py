#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
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
                
                # --- FIX: Relaxed tolerance for Gripper ---
                if "rg6" in name:
                    jc.tolerance_above = jc.tolerance_below = 0.1 
                else:
                    jc.tolerance_above = jc.tolerance_below = 0.02
                # ------------------------------------------
                
                constraints.joint_constraints.append(jc)
        
        goal_msg.request.goal_constraints.append(constraints)
        return self._send_goal(goal_msg, velocity, acceleration)

    def move_to_pose_robust(self, x, y, z, q_dict, link_name, frame_id='world_world', velocity=0.1, acceleration=0.1):
        """Modified to accept velocity/acceleration and pass to _send_goal."""
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

        # --- BRANCH 1: Strict Orientation ---
        if q_dict and all(k in q_dict for k in ['qx', 'qy', 'qz', 'qw']):
            target_pose.pose.orientation.x = q_dict['qx']
            target_pose.pose.orientation.y = q_dict['qy']
            target_pose.pose.orientation.z = q_dict['qz']
            target_pose.pose.orientation.w = q_dict['qw']
            req.ik_request.pose_stamped = target_pose
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: 
                return self._process_ik_result(res, velocity, acceleration)

        # --- BRANCH 2: Relaxation Pipeline ---
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
        """Passes velocity values through to the joint movement."""
        ik_joints = {name: pos for name, pos in zip(response.solution.joint_state.name, response.solution.joint_state.position)}
        prefix = "xarm5" if "xarm" in self.group_name else "u1"
        return self.move_to_joint_positions(ik_joints, filter_prefix=prefix, velocity=velocity, acceleration=acceleration)

    def _send_goal(self, goal_msg, velocity, acceleration):
        """The final point where MoveGroup parameters are applied."""
        if not self._action_client.wait_for_server(timeout_sec=5.0): return False

        # Apply the scaling factors to the request
        goal_msg.request.max_velocity_scaling_factor = velocity
        goal_msg.request.max_acceleration_scaling_factor = acceleration

        future = self._action_client.send_goal_async(goal_msg)
        while not future.done(): time.sleep(0.1)
        handle = future.result()
        if not handle.accepted: return False
        res_future = handle.get_result_async()
        while not res_future.done(): time.sleep(0.1)
        return res_future.result().result.error_code.val == 1