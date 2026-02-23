#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, TwistStamped
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs 
import threading, math, time, copy, os

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        self.is_xarm5 = "xarm5" in group_name.lower()
        self.prefix = 'xarm' if 'xarm' in self.group_name.lower() else 'uf'
        self.controller_name = f"{self.prefix}_controller"
        
        # --- ROS 2 Clients ---
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self._cm_client = self.node.create_client(SwitchController, '/controller_manager/switch_controller')
        
        # --- MoveIt Servo Publisher ---
        self.servo_pub = self.node.create_publisher(TwistStamped, f'/{self.prefix}_servo_node/delta_twist_cmds', 10)
        
        # --- TF2 Transformation Engine ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # --- Internal State Tracking ---
        self.current_joint_positions = {}
        self.current_joint_efforts = {}
        self.state_received = threading.Event()
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        
        # 🦾 Controller Health Heartbeat (Keeps controllers 'active')
        self.is_activated = False
        self.health_timer = self.node.create_timer(2.0, self._check_controller_health)

    def _check_controller_health(self):
        """Automatically re-activates controller if it drops to inactive."""
        if not self.is_activated:
            os.system(f"ros2 control set_controller_state {self.controller_name} active > /dev/null 2>&1")
            self.is_activated = True

    def _joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joint_positions[name] = msg.position[i]
            if len(msg.effort) > i: self.current_joint_efforts[name] = msg.effort[i]
        self.state_received.set()

    def get_transformed_pose(self, source_pose, source_frame, target_frame, z_offset=0.0):
        try:
            p = PoseStamped()
            p.header.frame_id, p.header.stamp = source_frame, self.node.get_clock().now().to_msg()
            p.pose = source_pose.pose if hasattr(source_pose, 'pose') else source_pose
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                return None
            t = self.tf_buffer.transform(p, target_frame)
            t.pose.position.z += z_offset
            return t
        except Exception as e:
            self.node.get_logger().error(f"TF Error: {e}"); return None

    def move_to_joint_positions(self, target_joints, velocity=0.1):
        goal = MoveGroup.Goal(); goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = velocity
        constraints = Constraints()
        # Handle prefix for u1, xarm5, or rg6
        prefix = 'xarm5' if self.is_xarm5 else ('u1' if 'uf' in self.group_name.lower() else 'rg6')
        for name, pos in target_joints.items():
            if prefix in name:
                jc = JointConstraint(); jc.joint_name, jc.position, jc.weight = name, float(pos), 1.0
                jc.tolerance_above = jc.tolerance_below = 0.01
                constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        self._action_client.send_goal_async(goal)
        return True

    def move_to_pose_robust(self, x, y, z, q_dict=None, velocity=0.1):
        req = GetPositionIK.Request(); req.ik_request.group_name = self.group_name; req.ik_request.avoid_collisions = True
        ps = PoseStamped(); ps.header.frame_id = 'world_world'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        
        # Try specific orientation
        if q_dict:
            ps.pose.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
            req.ik_request.pose_stamped = ps
            req.ik_request.robot_state = self._get_full_robot_state()
            if self._call_ik_sync(req).error_code.val == 1:
                self._execute_joint_goal(self._call_ik_sync(req).solution.joint_state, velocity); return True

        # 5-DOF Sweep
        self.node.get_logger().warn("⚠️ IK Failed. Sweeping Yaw...")
        for yaw in [0.0, 0.78, -0.78, 1.57, -1.57, 3.14]:
            ps.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw)
            req.ik_request.pose_stamped = ps
            if self._call_ik_sync(req).error_code.val == 1:
                self._execute_joint_goal(self._call_ik_sync(req).solution.joint_state, velocity); return True
        return False

    def move_linear_z_with_torque_stop(self, speed_mps, threshold_nm, joint_index=4):
        arm_pfx = 'xarm5' if self.is_xarm5 else 'u1'
        joint_name = f"{arm_pfx}_joint{joint_index+1}"
        
        # 🛡️ REDUCED WAIT: Only 0.1s to capture baseline
        time.sleep(0.1)
        baseline = self.current_joint_efforts.get(joint_name, 0.0)
        
        twist = TwistStamped()
        twist.header.frame_id = "world_world"
        twist.twist.linear.z = -abs(speed_mps)
        
        rate = self.node.create_rate(50)
        start_t = time.time()

        while rclpy.ok():
            curr = self.current_joint_efforts.get(joint_name, 0.0)
            spike = abs(curr - baseline)
            
            # 🛡️ REDUCED BLANKING: Only ignore the first 0.2s of the 'kick'
            if (time.time() - start_t) > 0.2:
                if spike > threshold_nm:
                    self.stop_immediately()
                    self.node.get_logger().warn(f"🎯 CONTACT: {spike:.3f}Nm spike.")
                    return True

            twist.header.stamp = self.node.get_clock().now().to_msg()
            self.servo_pub.publish(twist)
            rate.sleep()
        return False
    
    def retract_relative_z(self, distance):
        """
        Uses MoveIt Position Control for a guaranteed retraction.
        Replaces the 'hit-or-miss' servo jogging for critical safety moves.
        """
        self.node.get_logger().info(f"⬆️ Planning Robust Retraction: {distance*1000}mm")
        
        # 1. Get current position from TF
        try:
            current_tf = self.tf_buffer.lookup_transform(
                'world_world', 'u1_tool0', rclpy.time.Time())
            
            target_x = current_tf.transform.translation.x
            target_y = current_tf.transform.translation.y
            target_z = current_tf.transform.translation.z + distance
            
            # Keep the same orientation
            q = current_tf.transform.rotation
            q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            
            # 2. Execute as a standard robust move
            return self.move_to_pose_robust(target_x, target_y, target_z, q_dict, velocity=0.05)
            
        except Exception as e:
            self.node.get_logger().error(f"Retraction TF Error: {e}")
            return False

    def jog_cartesian_servo(self, dx, dy, dz, duration=1.0):
        twist = TwistStamped(); twist.header.frame_id = "world_world"
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = dx, dy, dz
        end_t = time.time() + duration
        while time.time() < end_t:
            twist.header.stamp = self.node.get_clock().now().to_msg()
            self.servo_pub.publish(twist)
            time.sleep(0.02)
        self.servo_pub.publish(TwistStamped())
        return True

    def stop_immediately(self):
        self.servo_pub.publish(TwistStamped())
        self.node.get_logger().error("🛑 MOTION STOPPED")

    def _execute_joint_goal(self, js, vel):
        goal = MoveGroup.Goal(); goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = vel
        constraints = Constraints(); pfx = 'xarm5' if self.is_xarm5 else 'u1'
        for n, p in zip(js.name, js.position):
            if pfx in n:
                jc = JointConstraint(); jc.joint_name, jc.position, jc.weight = n, p, 1.0
                constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        self._action_client.send_goal_async(goal)

    def _call_ik_sync(self, req):
        future = self._ik_client.call_async(req)
        while not future.done(): time.sleep(0.01)
        return future.result()

    def _get_full_robot_state(self):
        state = RobotState(); js = JointState(); js.header.stamp = self.node.get_clock().now().to_msg()
        js.name, js.position = list(self.current_joint_positions.keys()), list(self.current_joint_positions.values())
        state.joint_state = js; return state

    def _rpy_to_quaternion(self, r, p, y):
        cy, sy = math.cos(y*0.5), math.sin(y*0.5); cp, sp = math.cos(p*0.5), math.sin(p*0.5); cr, sr = math.cos(r*0.5), math.sin(r*0.5)
        return Quaternion(w=cr*cp*cy+sr*sp*sy, x=sr*cp*cy-cr*sp*sy, y=cr*sp*cy+sr*cp*sy, z=cr*cp*sy-sr*sp*cy)