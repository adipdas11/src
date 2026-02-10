#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
import math

class MotionBackend:
    def __init__(self, node, group_name):
        self.node = node
        self.group_name = group_name
        
        # Action Client for Motion Execution
        self._action_client = ActionClient(node, MoveGroup, 'move_action')
        # Service Client for IK (The Robustness Engine)
        self._ik_client = node.create_client(GetPositionIK, 'compute_ik')
        
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().warn(f"Action Server not found for {group_name}")

    def rpy_to_quaternion(self, roll, pitch, yaw):
        """ 
        Stable Euler to Quaternion conversion from your working code.
        Ensures consistent orientation for 5-DOF IK.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return Quaternion(
            w=cr*cp*cy + sr*sp*sy, 
            x=sr*cp*cy - cr*sp*sy, 
            y=cr*sp*cy + sr*cp*sy, 
            z=cr*cp*sy - sr*sp*cy
        )

    def get_ik(self, target_pose, link_name, frame_id="world"):
        """Solves IK for a specific pose in a specific frame."""
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = True
        req.ik_request.ik_link_name = link_name
        req.ik_request.timeout.sec = 2

        ps = PoseStamped()
        ps.header.frame_id = frame_id 
        ps.pose = target_pose
        req.ik_request.pose_stamped = ps

        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.error_code.val == 1:
            return response.solution.joint_state
        return None

    def move_to_pose_rpy(self, x, y, z, roll, pitch, yaw, link_name, frame_id="world"):
        """
        New Method: Uses RPY logic from your old working code to stabilize xArm5 IK.
        Combines global coordinates with stable orientation.
        """
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.orientation = self.rpy_to_quaternion(roll, pitch, yaw)
        
        return self.move_to_pose(target, link_name, frame_id)

    def move_to_pose(self, target_pose, link_name, frame_id="world"):
        """Standard pose move using explicit joint constraints."""
        joints = self.get_ik(target_pose, link_name, frame_id)
        if not joints:
            self.node.get_logger().error(f"IK Failed for {self.group_name} in frame {frame_id}")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.allowed_planning_time = 10.0 
        
        c = Constraints()
        c.name = "Stable IK Target"
        
        # Filter for appropriate robot prefix
        prefix = "xarm5" if "xarm" in self.group_name else "u1"
        
        for name, pos in zip(joints.name, joints.position):
            if prefix in name:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = pos
                jc.tolerance_above = 0.01; jc.tolerance_below = 0.01; jc.weight = 1.0
                c.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(c)
        return self._send_goal(goal)

    def move_to_named_target(self, joint_names, joint_values):
        """Moves to predefined poses using joint constraints."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        
        c = Constraints()
        for name, pos in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01; jc.tolerance_below = 0.01; jc.weight = 1.0
            c.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(c)
        return self._send_goal(goal)

    def _send_goal(self, goal):
        """Helper to manage the MoveGroup action lifecycle."""
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        handle = future.result()
        if not handle or not handle.accepted: return False

        res_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, res_future)
        return res_future.result().result.error_code.val == 1