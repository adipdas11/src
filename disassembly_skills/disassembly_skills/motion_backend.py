#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, TwistStamped
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs 
import threading, math, time, os, subprocess

class MotionBackend:
    def __init__(self, node: Node, group_name: str):
        self.node = node
        self.group_name = group_name
        
        # 🦾 THE CRITICAL FIX: Ensure any mention of 'xarm' sets this to True
        self.is_xarm5 = "xarm" in group_name.lower()
        
        # --- LOGGING FOR VERIFICATION ---
        if self.is_xarm5:
            self.node.get_logger().info("✅ MotionBackend: Initialized for XARM5 group.")
        else:
            self.node.get_logger().info("✅ MotionBackend: Initialized for UF850 group.")

        self.prefix = 'xarm' if self.is_xarm5 else 'uf'
        self.controller_name = f"{self.prefix}_controller"
        
        # --- ROS 2 Interfaces ---
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')
        self._ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        
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
        
        self.is_activated = False

    def _joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joint_positions[name] = msg.position[i]
            if len(msg.effort) > i: 
                self.current_joint_efforts[name] = msg.effort[i]
        self.state_received.set()

    # --- Hardware Management ---
    def reset_robot(self):
        """Asynchronously clears hardware errors and activates controllers."""
        self.node.get_logger().info(f"🔄 Resetting {self.group_name}...")

        def call_service_cmd(cmd):
            try:
                subprocess.run(cmd, shell=True, timeout=3.0, capture_output=True)
                return True
            except Exception:
                return False

        # Clear Errors, Set Mode 0 (Position), Set State 0 (Ready)
        call_service_cmd(f"ros2 service call /{self.prefix}/clear_err std_srvs/srv/Empty {{}}")
        call_service_cmd(f"ros2 service call /{self.prefix}/set_mode xarm_msgs/srv/SetInt16 \"{{data: 0}}\"")
        call_service_cmd(f"ros2 service call /{self.prefix}/set_state xarm_msgs/srv/SetInt16 \"{{data: 0}}\"")
        
        # Re-activate the primary controller
        call_service_cmd(f"ros2 control set_controller_state {self.controller_name} active")
        
        self.is_activated = True
        self.node.get_logger().info(f"✅ Reset Sequence Dispatched for {self.group_name}.")

    def stop_immediately(self):
        """Immediately stops all Servo and MoveIt motion."""
        self.servo_pub.publish(TwistStamped())
        self.node.get_logger().error("🛑 MOTION STOPPED")

    # --- Core Motion Logic ---
    def move_to_pose_robust(self, x, y, z, q_dict=None, velocity=0.1, frame_id='world_world'):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.avoid_collisions = True
        
        # 🦾 THE FIX: Strictly align the tip frame with what MoveIt expects
        # Your log says only [xarm5_link5] is available for this group.
        if self.is_xarm5:
            req.ik_request.ik_link_name = "xarm5_link5"
        else:
            req.ik_request.ik_link_name = "u1_tool0"
        
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = frame_id, self.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        
        # 5-DOF arms usually need Roll=PI (pointed down)
        if q_dict:
            ps.pose.orientation = Quaternion(x=q_dict['qx'], y=q_dict['qy'], z=q_dict['qz'], w=q_dict['qw'])
        else:
            ps.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, 0.0)

        req.ik_request.pose_stamped = ps
        req.ik_request.robot_state = self._get_full_robot_state()
        
        # Primary Attempt
        ik_res = self._call_ik_sync(req)
        if ik_res.error_code.val == 1:
            self._execute_joint_goal(ik_res.solution.joint_state, velocity)
            return True

        # 5-DOF Yaw Sweep Fallback (xarm5_link5)
        if self.is_xarm5:
            self.node.get_logger().warn("⚠️ IK Failed for xarm5_link5. Attempting Yaw Sweep...")
            for yaw_deg in [15, -15, 30, -30, 45, -45, 90, -90, 180]:
                yaw_rad = math.radians(yaw_deg)
                ps.pose.orientation = self._rpy_to_quaternion(math.pi, 0.0, yaw_rad)
                req.ik_request.pose_stamped = ps
                req.ik_request.robot_state = self._get_full_robot_state()
                ik_res = self._call_ik_sync(req)
                if ik_res.error_code.val == 1:
                    self.node.get_logger().info(f"✅ IK Found at Yaw: {yaw_deg}°")
                    self._execute_joint_goal(ik_res.solution.joint_state, velocity)
                    return True
        
        return False
    
    def move_to_joint_positions(self, target_joints, velocity=0.2):
        if not self.state_received.wait(timeout=2.0): return False

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = velocity
        
        constraints = Constraints()
        
        # 🦾 DIRECT CHECK: Use the instance-specific flag set in __init__
        if self.is_xarm5:
            prefixes = ['xarm5', 'slider']
        else:
            prefixes = ['u1', 'rg6']
        
        found_joints = False
        for name, pos in target_joints.items():
            # Check if current joint in target_joints matches the arm we are controlling
            if any(name.startswith(pfx) for pfx in prefixes):
                jc = JointConstraint()
                jc.joint_name, jc.position, jc.weight = name, float(pos), 1.0
                jc.tolerance_above = jc.tolerance_below = 0.01
                constraints.joint_constraints.append(jc)
                found_joints = True
        
        if not found_joints:
            # This is where your error was triggered. 
            # It means the 'prefixes' variable was ['u1', 'rg6'] even for the xArm.
            self.node.get_logger().error(f"❌ Prefix mismatch: {prefixes} not found in target_joints.")
            return False

        goal.request.goal_constraints.append(constraints)
        
        self.node.get_logger().info(f"🦾 Sending Joint Goal for {self.group_name}...")
        
        # 2. Send Goal and wait for result
        future = self._action_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("❌ Joint Goal Rejected by MoveIt.")
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
            
        success = result_future.result().result.error_code.val == 1
        if success:
            self.node.get_logger().info("✅ Joint Move Complete.")
        return success

    def move_linear_z_with_torque_stop(self, speed_mps, threshold_nm, joint_index=4):
        """Tactile descent using MoveIt Servo and baseline-subtraction monitoring."""
        arm_pfx = 'xarm5' if self.is_xarm5 else 'u1'
        joint_name = f"{arm_pfx}_joint{joint_index+1}"
        
        time.sleep(0.1) # Stabilization
        baseline = self.current_joint_efforts.get(joint_name, 0.0)
        
        twist = TwistStamped()
        twist.header.frame_id = "world_world"
        twist.twist.linear.z = -abs(speed_mps)
        
        rate = self.node.create_rate(50)
        start_t = time.time()

        while rclpy.ok():
            curr = self.current_joint_efforts.get(joint_name, 0.0)
            spike = abs(curr - baseline)
            
            # Blanking period (0.2s) to ignore initial jerk
            if (time.time() - start_t) > 0.2:
                if spike > threshold_nm:
                    self.stop_immediately()
                    self.node.get_logger().warn(f"🎯 CONTACT DETECTED: {spike:.3f}Nm spike.")
                    return True

            twist.header.stamp = self.node.get_clock().now().to_msg()
            self.servo_pub.publish(twist)
            rate.sleep()
        return False
    
    def retract_relative_z(self, distance):
        """Planned relative lift along the Z-axis."""
        target_link = "xarm5_link5" if self.is_xarm5 else "u1_tool0"
        try:
            current_tf = self.tf_buffer.lookup_transform('world_world', target_link, rclpy.time.Time())
            tx = current_tf.transform.translation.x
            ty = current_tf.transform.translation.y
            tz = current_tf.transform.translation.z + distance
            q = current_tf.transform.rotation
            q_dict = {'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            return self.move_to_pose_robust(tx, ty, tz, q_dict, velocity=0.05)
        except Exception as e:
            self.node.get_logger().error(f"Retract TF Error: {e}"); return False

    def jog_cartesian_servo(self, dx, dy, dz, duration=1.0):
        """Fine-grained cartesian jogging via MoveIt Servo."""
        twist = TwistStamped(); twist.header.frame_id = "world_world"
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = dx, dy, dz
        end_t = time.time() + duration
        while rclpy.ok() and time.time() < end_t:
            twist.header.stamp = self.node.get_clock().now().to_msg()
            self.servo_pub.publish(twist)
            time.sleep(0.02)
        self.servo_pub.publish(TwistStamped())
        return True
            
    def get_transformed_pose(self, source_pose, source_frame, target_frame, z_offset=0.0):
        try:
            p = PoseStamped()
            p.header.frame_id = source_frame
            # 🦾 FIX: Use Time() (zero) to get the latest available transform
            p.header.stamp = rclpy.time.Time().to_msg() 
            
            p.pose = source_pose.pose if hasattr(source_pose, 'pose') else source_pose
            
            # Check if transform is possible before attempting
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), 
                                              timeout=rclpy.duration.Duration(seconds=1.0)):
                return None
                
            t = self.tf_buffer.transform(p, target_frame)
            t.pose.position.z += z_offset
            return t
        except Exception as e:
            self.node.get_logger().error(f"TF Error: {e}")
            return None

    def _execute_joint_goal(self, js, vel):
        """Standardizes joint execution across arms, grippers, and sliders."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.max_velocity_scaling_factor = vel
        
        constraints = Constraints()
        # 🦾 IMPROVED PREFIX LOGIC
        # Logic: If it's an xarm task, take xarm+slider. If it's a UF task, take u1+rg6.
        if self.is_xarm5:
            prefixes = ['xarm5', 'slider']
        else:
            prefixes = ['u1', 'rg6']

        found_any = False
        for n, p in zip(js.name, js.position):
            # Check if the joint name starts with any of our valid prefixes
            if any(n.startswith(pfx) for pfx in prefixes):
                jc = JointConstraint()
                jc.joint_name, jc.position, jc.weight = n, p, 1.0
                constraints.joint_constraints.append(jc)
                found_any = True
        
        if not found_any:
            self.node.get_logger().error(f"❌ No joints matching {prefixes} found in message!")
            return

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
        cy, sy = math.cos(y*0.5), math.sin(y*0.5)
        cp, sp = math.cos(p*0.5), math.sin(p*0.5)
        cr, sr = math.cos(r*0.5), math.sin(r*0.5)
        return Quaternion(w=cr*cp*cy+sr*sp*sy, x=sr*cp*cy-cr*sp*sy, y=cr*sp*cy+sr*cp*sy, z=cr*cp*sy-sr*sp*cy)