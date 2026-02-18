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

# Use the native Python SDK directly
from xarm.wrapper import XArmAPI

# ==========================================
ROBOT_IP = '192.168.1.239' 
# ==========================================

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        
        # --- ROS 2 MoveIt Interface ---
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self._cartesian_client = self.node.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._stop_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/stop_robot')
        self._reset_srv = self.node.create_client(std_srvs.srv.Empty, '/xarm/reset_robot')
        
        # --- Direct Hardware Connection (SDK) ---
        try:
            self.node.get_logger().info(f"🔗 Connecting to native xArm SDK at {ROBOT_IP}...")
            self.arm = XArmAPI(ROBOT_IP)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            
            # 👈 [CRITICAL FIX] Force Controller to ignore internal TCP offset
            # We calculate tool length manually in Python, so Controller must be 0
            self.arm.set_tcp_offset([0, 0, 0, 0, 0, 0])
            self.node.get_logger().info("✅ Native SDK Connected & TCP Offset Cleared to [0,0,0]!")
            
        except Exception as e:
            self.node.get_logger().error(f"❌ Failed to connect to xArm SDK: {e}")

        self._current_goal_handle = None
        self.current_joint_msg = None
        self.current_joint_positions = {}
        self.state_received = threading.Event()
        
        # Subscribers & TF
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def _joint_state_callback(self, msg):
        self.current_joint_msg = msg
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_positions[name] = pos
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

    def reset_robot(self):
        """Resets errors on both hardware SDK and ROS driver side."""
        if hasattr(self, 'arm'):
            self.arm.clean_error()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            # Ensure TCP is cleared on reset too
            self.arm.set_tcp_offset([0, 0, 0, 0, 0, 0])
            
        if self._reset_srv.wait_for_service(timeout_sec=1.0):
            self._reset_srv.call_async(std_srvs.srv.Empty.Request())
            self.node.get_logger().info("✅ Robot Reset & Enabled")

    def stop_immediately(self):
        """Triggers emergency stop on hardware and cancels ROS goals."""
        if self._current_goal_handle: self._current_goal_handle.cancel_goal_async()
        if hasattr(self, 'arm'):
            self.arm.set_state(state=4) 
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

    # --- INTACT MOVEIT 2 FUNCTIONS ---
    def move_linear_z_with_force_stop(self, distance_down, velocity_scaling, check_force_callback):
        if not self._execute_client.wait_for_server(timeout_sec=2.0): return False
        s = Pose(); s.orientation.w = 1.0
        start_pose = self.get_transformed_pose(s, 'xarm5_link5', 'world_world') 
        if not start_pose: return False

        target_pose = copy.deepcopy(start_pose)
        target_pose.pose.position.z -= abs(distance_down) 

        ik_solution = None
        for yaw_offset in [0.0, 0.1, -0.1, 0.2, -0.2]:
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.group_name
            req.ik_request.ik_link_name = "xarm5_link5"
            req.ik_request.avoid_collisions = True
            ik_pose = PoseStamped(); ik_pose.header.frame_id = 'world_world'
            ik_pose.pose = target_pose.pose; ik_pose.pose.orientation = start_pose.pose.orientation
            req.ik_request.pose_stamped = ik_pose; req.ik_request.robot_state = self._get_full_robot_state()
            res = self._call_ik_sync(req)
            if res.error_code.val == 1:
                ik_solution = res.solution.joint_state; break
        
        if not ik_solution: return False

        goal_joints = {name: pos for name, pos in zip(ik_solution.name, ik_solution.position)}
        mg_goal = MoveGroup.Goal()
        mg_goal.request.group_name = self.group_name
        mg_goal.request.max_velocity_scaling_factor = velocity_scaling
        mg_goal.request.max_acceleration_scaling_factor = 0.05
        mg_goal.request.allowed_planning_time = 2.0
        constraints = Constraints()
        for name, pos in goal_joints.items():
            if "xarm" in name:
                jc = JointConstraint(); jc.joint_name = name; jc.position = pos; jc.weight = 1.0
                jc.tolerance_above = 0.001; jc.tolerance_below = 0.001
                constraints.joint_constraints.append(jc)
        mg_goal.request.goal_constraints.append(constraints)
        mg_goal.planning_options.plan_only = True
        
        plan_future = self._action_client.send_goal_async(mg_goal)
        while not plan_future.done(): time.sleep(0.01)
        plan_handle = plan_future.result()
        res_future = plan_handle.get_result_async()
        while not res_future.done(): time.sleep(0.01)
        plan_result = res_future.result().result
        
        if plan_result.error_code.val != 1: return False

        goal = ExecuteTrajectory.Goal(); goal.trajectory = plan_result.planned_trajectory
        gf = self._execute_client.send_goal_async(goal)
        while not gf.done(): time.sleep(0.01)
        self._current_goal_handle = gf.result()
        rf = self._current_goal_handle.get_result_async()
        
        start_t = time.time(); loop_cnt = 0
        while not rf.done():
            if (time.time() - start_t) > 0.5 and check_force_callback():
                self.stop_immediately(); return True
            loop_cnt += 1; time.sleep(0.005) 
        return True

    def move_to_pose_robust(self, x, y, z, q_dict, link_name, frame_id='world_world', velocity=0.1):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name; req.ik_request.ik_link_name = link_name
        req.ik_request.avoid_collisions = True
        target_pose = PoseStamped(); target_pose.header.frame_id = frame_id 
        target_pose.pose.position.x = x; target_pose.pose.position.y = y; target_pose.pose.position.z = z
        if q_dict:
            target_pose.pose.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
            req.ik_request.pose_stamped = target_pose; req.ik_request.robot_state = self._get_full_robot_state()
            res = self._call_ik_sync(req)
            if res.error_code.val == 1: return self._process_ik_result(res, velocity, 0.1)
        for yaw in [0.0, 0.4, -0.4, 0.8, -0.8, 1.57, -1.57, 3.14]:
            target_pose.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw)
            req.ik_request.pose_stamped = target_pose; req.ik_request.robot_state = self._get_full_robot_state()
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

    # =========================================================================
    # ⚡ DIRECT PYTHON SDK CONTROLLERS (Position Mode 0)
    # =========================================================================

    def move_to_absolute_pose_sdk(self, x_m, y_m, z_m, speed_mm_s=50.0):
        if not hasattr(self, 'arm'): return False
        _, state = self.arm.get_state()
        if state == 4:
            self.node.get_logger().warn("⚠️ Clearing State 4 Error...")
            self.arm.clean_error(); self.arm.motion_enable(enable=True); self.arm.set_mode(0); self.arm.set_state(state=0)
            time.sleep(0.5)

        x_mm = x_m * 1000.0; y_mm = y_m * 1000.0; z_mm = z_m * 1000.0
        self.arm.set_mode(0); self.arm.set_state(state=0)
        
        code, curr_pos = self.arm.get_position(is_radian=False)
        if code != 0 or not curr_pos: return False
        curr_r, curr_p, curr_y = curr_pos[3], curr_pos[4], curr_pos[5]

        print(f"   🌊 SDK ABS MOVE -> X:{x_mm:.1f} Y:{y_mm:.1f} Z:{z_mm:.1f} (R:{curr_r:.1f} P:{curr_p:.1f} Y:{curr_y:.1f})")
        ret = self.arm.set_position(x=x_mm, y=y_mm, z=z_mm, roll=curr_r, pitch=curr_p, yaw=curr_y, speed=speed_mm_s, relative=False, wait=True)
        return ret == 0

    def move_linear_z_sdk_with_force_stop(self, distance_down_m, speed_mm_s, check_force_callback):
        if not hasattr(self, 'arm'): return False
        _, state = self.arm.get_state()
        if state == 4:
            self.arm.clean_error(); self.arm.motion_enable(enable=True); self.arm.set_mode(0); self.arm.set_state(state=0)

        dz_mm = -abs(distance_down_m * 1000.0)
        self.arm.set_mode(0); self.arm.set_state(state=0)
        self.arm.set_position(x=0, y=0, z=dz_mm, roll=0, pitch=0, yaw=0, speed=speed_mm_s, relative=True, wait=False)
        start_t = time.time(); time.sleep(0.1)
        while rclpy.ok():
            _, state = self.arm.get_state()
            if state != 1: break
            if (time.time() - start_t) > 0.5 and check_force_callback():
                self.arm.set_state(state=4); self.stop_immediately(); return True
            time.sleep(0.005) 
        return True

    def jog_cartesian_sdk(self, dx_m, dy_m, dz_m, speed_mm_s=20.0):
        if not hasattr(self, 'arm'): return False
        _, state = self.arm.get_state()
        if state == 4:
            self.arm.clean_error(); self.arm.motion_enable(enable=True); self.arm.set_mode(0); self.arm.set_state(state=0)

        dx_mm = dx_m * 1000.0; dy_mm = dy_m * 1000.0; dz_mm = dz_m * 1000.0
        self.arm.set_mode(0); self.arm.set_state(state=0)
        code = self.arm.set_position(x=dx_mm, y=dy_mm, z=dz_mm, roll=0, pitch=0, yaw=0, speed=speed_mm_s, relative=True, wait=True)
        return code == 0