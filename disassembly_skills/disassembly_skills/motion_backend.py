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

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self._cartesian_client = self.node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._stop_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/stop_robot')
        self._reset_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/reset_robot')

        self._current_goal_handle = None
        self.current_joint_msg = None
        self.current_joint_positions = {}
        self.state_received = threading.Event()
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
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

    def reset_robot(self):
        if self._reset_srv.wait_for_service(timeout_sec=1.0):
            self._reset_srv.call_async(std_srvs.srv.Empty.Request())
            self.node.get_logger().info("✅ Robot Reset & Enabled")

    def stop_immediately(self):
        if self._current_goal_handle: self._current_goal_handle.cancel_goal_async()
        if self._stop_srv.wait_for_service(timeout_sec=0.1):
            self._stop_srv.call_async(std_srvs.srv.Empty.Request())
            self.node.get_logger().error("!!! HARDWARE STOP SENT !!!")

    def get_transformed_pose(self, source_pose, source_frame: str, target_frame: str, z_offset=0.0):
        try:
            import tf2_geometry_msgs 
            real_pose = source_pose.pose if hasattr(source_pose, 'pose') else source_pose
            p = PoseStamped(); p.header.frame_id = source_frame
            p.header.stamp = rclpy.time.Time().to_msg(); p.pose = real_pose

            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)): return None
            t = self.tf_buffer.transform(p, target_frame)
            t.pose.position.z += z_offset
            return t
        except Exception as e:
            self.node.get_logger().error(f"TF Error: {e}")
            return None

    # ==========================================================
    # UPDATED DEBUGGING LINEAR MOTION
    # ==========================================================
    def move_linear_z_with_force_stop(self, distance_down, velocity_scaling, check_force_callback):
        if not self._execute_client.wait_for_server(timeout_sec=2.0): return False
        
        # 1. Get Start Pose
        s = Pose(); s.orientation.w = 1.0
        start_pose = self.get_transformed_pose(s, 'xarm5_link5', 'world_world') 
        if not start_pose: return False

        # 2. Calculate Target Pose (Bottom of descent)
        target_pose = copy.deepcopy(start_pose)
        # We know SUBTRACTION is correct for your frame
        target_pose.pose.position.z -= abs(distance_down) 

        print(f"\n🔮 ROBUST DESCENT PLAN:")
        print(f"   Start Z:  {start_pose.pose.position.z:.4f}")
        print(f"   Target Z: {target_pose.pose.position.z:.4f}")

        # 3. Calculate IK for the Bottom Position (The "Relaxed" Target)
        # We try multiple yaw angles to find a valid bottom configuration
        ik_solution = None
        for yaw_offset in [0.0, 0.1, -0.1, 0.2, -0.2]: # Small variations allowed
            # Keep original orientation but allow small yaw adjustments if needed
            # For now, let's try strict original orientation first
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.group_name
            req.ik_request.ik_link_name = "xarm5_link5"
            req.ik_request.avoid_collisions = True
            
            # Construct IK Request
            ik_pose = PoseStamped()
            ik_pose.header.frame_id = 'world_world'
            ik_pose.pose = target_pose.pose
            
            # Try to match start orientation exactly
            ik_pose.pose.orientation = start_pose.pose.orientation
            
            req.ik_request.pose_stamped = ik_pose
            req.ik_request.robot_state = RobotState()
            req.ik_request.robot_state.joint_state = self.current_joint_msg
            
            res = self._call_ik_sync(req)
            if res.error_code.val == 1:
                ik_solution = res.solution.joint_state
                break
        
        if not ik_solution:
            self.node.get_logger().error("❌ IK Failed: Cannot reach bottom position.")
            return False

        # 4. Plan Joint Trajectory to that IK Solution
        # This creates a "PTP" move which is robust for 5-DOF
        goal_joints = {name: pos for name, pos in zip(ik_solution.name, ik_solution.position)}
        
        # Create a standard MoveGroup plan to these joints
        # We manually construct a trajectory to execute it with monitoring
        
        # NOTE: To keep the "Force Monitor" logic, we need to Execute a Trajectory.
        # So we ask MoveIt to Plan a trajectory to these joints.
        
        mg_goal = MoveGroup.Goal()
        mg_goal.request.group_name = self.group_name
        mg_goal.request.max_velocity_scaling_factor = velocity_scaling
        mg_goal.request.max_acceleration_scaling_factor = 0.05
        mg_goal.request.allowed_planning_time = 2.0
        
        # Add Joint Constraints
        constraints = Constraints()
        for name, pos in goal_joints.items():
            if "xarm" in name:
                jc = JointConstraint()
                jc.joint_name = name; jc.position = pos; jc.weight = 1.0
                jc.tolerance_above = 0.001; jc.tolerance_below = 0.001
                constraints.joint_constraints.append(jc)
        mg_goal.request.goal_constraints.append(constraints)
        
        # Send PLAN Request (Not Execute yet)
        mg_goal.planning_options.plan_only = True
        
        self.node.get_logger().info("   Planning Robust Descent Path...")
        plan_future = self._action_client.send_goal_async(mg_goal)
        while not plan_future.done(): time.sleep(0.01)
        plan_handle = plan_future.result()
        res_future = plan_handle.get_result_async()
        while not res_future.done(): time.sleep(0.01)
        plan_result = res_future.result().result
        
        if plan_result.error_code.val != 1:
             self.node.get_logger().error(f"❌ Planning Failed Code: {plan_result.error_code.val}")
             return False

        # 5. EXECUTE THE PLAN (With Force Monitoring)
        planned_traj = plan_result.planned_trajectory
        
        # Scale speed manually again to be safe
        # (MoveGroup scaling is sometimes approximate)
        # We won't re-scale here to avoid double-slowing, rely on velocity_scaling passed above

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = planned_traj
        
        gf = self._execute_client.send_goal_async(goal)
        while not gf.done(): time.sleep(0.01)
        self._current_goal_handle = gf.result()
        
        rf = self._current_goal_handle.get_result_async()
        
        # 6. MONITOR LOOP
        print("\n📉 EXECUTING ROBUST DESCENT...")
        start_t = time.time()
        loop_cnt = 0
        
        while not rf.done():
            if (time.time() - start_t) > 0.5:
                if check_force_callback():
                    self.stop_immediately()
                    return True
            
            # Debug Print
            if loop_cnt % 20 == 0:
                current_live = self.get_transformed_pose(Pose(), 'xarm5_link5', 'world_world')
                if current_live:
                    act_z = current_live.pose.position.z
                    tgt_z = target_pose.pose.position.z
                    diff = tgt_z - act_z
                    print(f"   [MONITOR] Tgt: {tgt_z:.4f} | Act: {act_z:.4f} | Δ: {diff:+.4f}", end='\r')
            loop_cnt += 1
            time.sleep(0.005) 
            
        return True

    def move_to_pose_robust(self, x, y, z, q_dict, link_name, frame_id='world_world', velocity=0.1):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name; req.ik_request.ik_link_name = link_name
        req.ik_request.avoid_collisions = True
        
        target_pose = PoseStamped(); target_pose.header.frame_id = frame_id 
        target_pose.pose.position.x = x; target_pose.pose.position.y = y; target_pose.pose.position.z = z
        if q_dict:
            target_pose.pose.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
            req.ik_request.pose_stamped = target_pose
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: return self._process_ik_result(res, velocity, 0.1)
        
        for yaw in [0.0, 0.4, -0.4, 0.8, -0.8, 1.57, -1.57, 3.14]:
            target_pose.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw)
            req.ik_request.pose_stamped = target_pose
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: return self._process_ik_result(res, velocity, 0.1)
        return False

    def _call_ik_sync(self, req):
        future = self._ik_client.call_async(req)
        while not future.done(): time.sleep(0.01)
        return future.result()

    def _process_ik_result(self, response, velocity, acceleration):
        ik_joints = {name: pos for name, pos in zip(response.solution.joint_state.name, response.solution.joint_state.position)}
        prefix = "xarm5" if "xarm" in self.group_name else "u1"
        return self.move_to_joint_positions(ik_joints, filter_prefix=prefix, velocity=velocity)

    def move_to_joint_positions(self, target_joints, filter_prefix="", velocity=0.1):
        if not self.state_received.wait(2.0): return False
        goal = MoveGroup.Goal(); goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = velocity
        constraints = Constraints()
        for name, pos in target_joints.items():
            if filter_prefix in name:
                jc = JointConstraint(); jc.joint_name = name; jc.position = pos; jc.weight = 1.0; jc.tolerance_above = 0.01; jc.tolerance_below = 0.01
                constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        f = self._action_client.send_goal_async(goal)
        while not f.done(): time.sleep(0.1)
        gh = f.result()
        if not gh.accepted: return False
        rf = gh.get_result_async()
        while not rf.done(): time.sleep(0.1)
        return rf.result().result.error_code.val == 1