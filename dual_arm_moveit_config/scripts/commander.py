#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
import math

# ==========================================
# HELPER: Euler -> Quaternion
# ==========================================
def rpy_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return Quaternion(w=cr*cp*cy + sr*sp*sy, x=sr*cp*cy - cr*sp*sy, y=cr*sp*cy + sr*cp*sy, z=cr*cp*sy - sr*sp*cy)

class UniversalCommander(Node):
    def __init__(self):
        super().__init__('universal_commander')
        
        # 1. Action Client for Moving
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._action_client.wait_for_server()
        
        # 2. Service Client for IK (The Robustness Engine)
        self._ik_client = self.create_client(GetPositionIK, 'compute_ik')
        while not self._ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')
            
        self.get_logger().info("Universal Commander Ready!")

    def get_ik_solution(self, group_name, target_dict):
        """ 
        Asks MoveIt: 'Can you physically reach this pose?'
        Returns joint angles if yes, None if no.
        """
        request = GetPositionIK.Request()
        request.ik_request.group_name = group_name
        request.ik_request.robot_state.is_diff = True
        request.ik_request.avoid_collisions = True
        request.ik_request.ik_link_name = target_dict['link']
        request.ik_request.timeout.sec = 1
        
        # Build Pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = "world_world"
        target_pose.pose.position.x = target_dict['x']
        target_pose.pose.position.y = target_dict['y']
        target_pose.pose.position.z = target_dict['z']
        target_pose.pose.orientation = rpy_to_quaternion(target_dict['r'], target_dict['p'], target_dict['yaw'])
        request.ik_request.pose_stamped = target_pose

        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response.error_code.val == 1:
            return response.solution.joint_state
        else:
            self.get_logger().error(f"[{group_name}] IK Failed (Error {response.error_code.val}). Target unreachable.")
            return None

    def execute_joint_trajectory(self, group_to_move, constraints):
        """ Sends the solved joint configuration to the controller """
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = group_to_move
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5
        
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info(f"Sending trajectory to {group_to_move}...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def create_constraints_from_joints(self, joint_state, filter_prefix):
        """ Converts a full robot state into constraints for a specific arm """
        constraints = Constraints()
        constraints.name = "IK Joint Target"
        
        for name, pos in zip(joint_state.name, joint_state.position):
            if filter_prefix in name:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = pos
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)
        return constraints

    # ==========================================
    # USER METHODS
    # ==========================================

    def move_single_arm(self, group_name, target):
        self.get_logger().info(f"--- Moving Single Arm: {group_name} ---")
        
        # 1. Solve IK
        joints = self.get_ik_solution(group_name, target)
        if not joints: return

        # 2. Convert to Constraints
        # Detect prefix (xarm5 or u1)
        prefix = "xarm5" if "xarm" in group_name else "u1"
        constraints = self.create_constraints_from_joints(joints, prefix)
        
        # 3. Execute
        self.execute_joint_trajectory(group_name, constraints)

    def move_dual_arms(self, xarm_target, uf_target):
        self.get_logger().info("--- Moving Dual Arms ---")
        
        # 1. Solve xArm IK
        xarm_joints = self.get_ik_solution("xarm_arm", xarm_target)
        if not xarm_joints: return

        # 2. Solve UF850 IK
        uf_joints = self.get_ik_solution("uf_arm", uf_target)
        if not uf_joints: return

        # 3. Combine Constraints
        constraints = Constraints()
        constraints.name = "Dual Target"
        
        # Add xArm joints
        c1 = self.create_constraints_from_joints(xarm_joints, "xarm5")
        constraints.joint_constraints.extend(c1.joint_constraints)
        
        # Add UF850 joints
        c2 = self.create_constraints_from_joints(uf_joints, "u1")
        constraints.joint_constraints.extend(c2.joint_constraints)
        
        # 4. Execute on 'dual_arms' group
        self.execute_joint_trajectory("dual_arms", constraints)

    # ==========================================
    # CALLBACKS
    # ==========================================
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return
        self.get_logger().info('Goal accepted! Executing...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('SUCCESS: Motion Complete!')
        else:
            self.get_logger().error(f'FAILED: Error Code {result.error_code.val}')

def main(args=None):
    rclpy.init(args=args)
    commander = UniversalCommander()

    # ==========================================
    # 1. DEFINE TARGETS
    # ==========================================
    
    # Target for xArm5 (From your image)
    # 5-DOF Tip: Always try to keep Roll=3.14 (180), Pitch=0 to point down.
    xarm_goal = {
        'x': 0.91008, 'y': 0.13151, 'z': 1.3984,
        'r': 3.14159, 'p': 0.0, 'yaw': 0.0,
        'link': "xarm5_tool0"
    }

    # Target for UF850 (Safe pose)
    uf_goal = {
        'x': 0.8, 'y': -0.3, 'z': 1.1,
        'r': 3.14, 'p': 0.0, 'yaw': 1.57,
        'link': "rg6_hand_tcp"
    }

    # ==========================================
    # 2. SELECT ACTION (Uncomment one)
    # ==========================================

    # MODE A: Move ONLY xArm5
    # commander.move_single_arm("xarm_arm", xarm_goal)

    # MODE B: Move ONLY UF850
    # commander.move_single_arm("uf_arm", uf_goal)

    # MODE C: Move BOTH Simultaneous
    commander.move_dual_arms(xarm_goal, uf_goal)

    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()