#!/usr/bin/env python3
"""
Hand Tracking Node for Dual-Arm Teleoperation.

Uses MediaPipe HandLandmarker (Tasks API) to track both hands from a laptop webcam.
Left hand  -> xArm5 (tool arm)
Right hand -> UF850 (manip arm) + RG6 gripper

Identity is maintained via spatial tracking (nearest-neighbor matching)
rather than relying solely on MediaPipe's handedness classifier,
which can flip when hands are close together.
"""

import os
import sys

# Inject shared AI venv (same as vision_agent/agent_node_v2.py)
VENV_PATH = '/home/adip/workspaces/disassembly_ws/src/vision_training/train_vision_model/.venv/lib/python3.10/site-packages'
sys.path.insert(0, VENV_PATH)

import cv2
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as ScipyRotation
import torch
from PIL import Image as PILImage

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Vector3Stamped


# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# MediaPipe pose landmark indices (body tracking)
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_MIN_VISIBILITY = 0.5  # Minimum visibility to trust a landmark

# Hand connections for drawing
HAND_CONNECTIONS = list(HandLandmarksConnections.HAND_CONNECTIONS)


class OneEuroFilter:
    """1-Euro filter for low-latency smoothing with adaptive cutoff.

    - At rest (low speed): heavy smoothing (removes jitter/flutter)
    - During motion (high speed): light smoothing (low latency)
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.array(x, dtype=float)
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.tolist()

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.tolist()

        # Derivative (speed)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff: higher speed -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filter each dimension
        result = np.empty_like(x)
        for i in range(len(x)):
            a = self._alpha(cutoff[i], dt)
            result[i] = a * x[i] + (1.0 - a) * self.x_prev[i]

        self.x_prev = result
        self.dx_prev = dx_hat
        self.t_prev = t
        return result.tolist()


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2 on GPU.

    Runs inference every N frames and caches the depth map. Depth values
    at specific pixel locations can be sampled between inference frames.
    """

    def __init__(self, logger, run_every_n=3, infer_size=(320, 240)):
        self.logger = logger
        self.run_every_n = max(1, run_every_n)
        self.infer_size = infer_size
        self._frame_count = 0
        self._cached_depth = None  # numpy array (H, W), float32
        self._pipe = None
        self.ready = False

    def init_model(self):
        """Load Depth Anything V2 Small on GPU."""
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                'depth-estimation',
                model='depth-anything/Depth-Anything-V2-Small-hf',
                device='cuda',
            )
            self.ready = True
            self.logger.info('Depth Anything V2 Small loaded on GPU')
        except Exception as e:
            self.logger.error(f'Failed to load depth model: {e}')
            self.ready = False

    def update(self, frame_rgb):
        """Run depth inference on frame (every N frames).

        Args:
            frame_rgb: RGB numpy array (H, W, 3)
        """
        if not self.ready:
            return
        self._frame_count += 1
        if self._cached_depth is not None and \
           self._frame_count % self.run_every_n != 0:
            return
        small = cv2.resize(frame_rgb, self.infer_size)
        pil_img = PILImage.fromarray(small)
        result = self._pipe(pil_img)
        # predicted_depth is a torch.Tensor — convert to numpy
        raw = result['predicted_depth']
        if isinstance(raw, torch.Tensor):
            self._cached_depth = raw.squeeze().cpu().numpy().astype(np.float32)
        else:
            self._cached_depth = np.array(result['depth'],
                                          dtype=np.float32)

    def get_depth_at(self, x_norm, y_norm):
        """Sample depth at normalized image coordinates.

        Args:
            x_norm, y_norm: [0, 1] coordinates in image frame

        Returns:
            Float depth value (higher = closer to camera), or None.
        """
        if self._cached_depth is None:
            return None
        dh, dw = self._cached_depth.shape
        px = int(np.clip(x_norm * dw, 0, dw - 1))
        py = int(np.clip(y_norm * dh, 0, dh - 1))
        return float(self._cached_depth[py, px])

    def get_depth_map_for_viz(self):
        """Return normalized depth map for visualization overlay."""
        if self._cached_depth is None:
            return None
        d = self._cached_depth
        d_min, d_max = d.min(), d.max()
        if d_max - d_min < 1e-6:
            return None
        return ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)


class HandTrackerNode(Node):
    """ROS 2 node for MediaPipe hand tracking with dual-arm mapping."""

    def __init__(self):
        super().__init__('hand_tracker')

        # --- Parameters ---
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_width', 1280)
        self.declare_parameter('camera_height', 720)
        self.declare_parameter('max_hands', 2)
        self.declare_parameter('detection_confidence', 0.5)
        self.declare_parameter('tracking_confidence', 0.4)
        self.declare_parameter('persistence_frames', 8)
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('flip_horizontal', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('enable_pose', True)
        self.declare_parameter('pose_model_path', '')
        self.declare_parameter('enable_depth_model', True)
        self.declare_parameter('depth_run_every_n', 3)

        self.camera_id = self.get_parameter('camera_id').value
        self.cam_w = self.get_parameter('camera_width').value
        self.cam_h = self.get_parameter('camera_height').value
        self.max_hands = self.get_parameter('max_hands').value
        self.det_conf = self.get_parameter('detection_confidence').value
        self.track_conf = self.get_parameter('tracking_confidence').value
        self.show_viz = self.get_parameter('show_visualization').value
        self.flip_h = self.get_parameter('flip_horizontal').value
        model_path_param = self.get_parameter('model_path').value
        self.enable_pose = self.get_parameter('enable_pose').value
        pose_model_param = self.get_parameter('pose_model_path').value
        self.enable_depth_model = self.get_parameter('enable_depth_model').value
        self.depth_run_every_n = self.get_parameter('depth_run_every_n').value

        # --- Resolve model path ---
        if model_path_param:
            self.model_path = model_path_param
        else:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(pkg_dir, '..', 'config', 'hand_landmarker.task'),
                os.path.join(pkg_dir, 'config', 'hand_landmarker.task'),
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.dirname(pkg_dir)))),
                    'src', 'hand_teleop', 'config', 'hand_landmarker.task'),
                '/home/adip/workspaces/disassembly_ws/src/hand_teleop/config/hand_landmarker.task',
            ]
            self.model_path = None
            for c in candidates:
                resolved = os.path.realpath(c)
                if os.path.isfile(resolved):
                    self.model_path = resolved
                    break
            if self.model_path is None:
                self.get_logger().fatal(
                    'hand_landmarker.task not found. Set model_path parameter '
                    'or place model in src/hand_teleop/config/')
                raise FileNotFoundError('hand_landmarker.task not found')

        self.get_logger().info(f'Using model: {self.model_path}')

        # --- Publishers ---
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub_landmarks = self.create_publisher(
            String, '/hand_teleop/landmarks', qos)
        self.pub_left_wrist = self.create_publisher(
            PoseStamped, '/hand_teleop/left/wrist_pose', qos)
        self.pub_right_wrist = self.create_publisher(
            PoseStamped, '/hand_teleop/right/wrist_pose', qos)
        self.pub_left_pinch = self.create_publisher(
            Bool, '/hand_teleop/left/pinch', qos)
        self.pub_right_pinch = self.create_publisher(
            Bool, '/hand_teleop/right/pinch', qos)
        self.pub_left_fist = self.create_publisher(
            Bool, '/hand_teleop/left/fist', qos)
        self.pub_right_fist = self.create_publisher(
            Bool, '/hand_teleop/right/fist', qos)
        self.pub_left_angular_vel = self.create_publisher(
            Vector3Stamped, '/hand_teleop/left/angular_vel', qos)
        self.pub_right_angular_vel = self.create_publisher(
            Vector3Stamped, '/hand_teleop/right/angular_vel', qos)
        # Absolute arm orientation quaternions (for pose-matching control)
        self.pub_left_arm_orient = self.create_publisher(
            PoseStamped, '/hand_teleop/left/arm_orientation', qos)
        self.pub_right_arm_orient = self.create_publisher(
            PoseStamped, '/hand_teleop/right/arm_orientation', qos)
        self.pub_status = self.create_publisher(
            String, '/hand_teleop/status', qos)

        # Subscribe to bridge status for visualization overlay
        self.bridge_status = {
            'xarm5': {'fist': False, 'centered': False,
                      'mode': 'position', 'active': False},
            'uf850': {'fist': False, 'centered': False,
                      'mode': 'position', 'active': False},
            'gripper_closed': False,
        }
        self.create_subscription(
            String, '/hand_teleop/bridge_status',
            self._bridge_status_cb, qos)

        # --- MediaPipe HandLandmarker (VIDEO mode for tracking) ---
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.det_conf,
            min_hand_presence_confidence=self.track_conf,
            min_tracking_confidence=self.track_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.get_logger().info('MediaPipe running mode: VIDEO (inter-frame tracking)')

        # --- MediaPipe PoseLandmarker for body tracking ---
        self.pose_landmarker = None
        if self.enable_pose:
            pose_model = self._resolve_pose_model(pose_model_param)
            if pose_model:
                pose_options = PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=pose_model,
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.4,
                )
                self.pose_landmarker = PoseLandmarker.create_from_options(
                    pose_options)
                self.get_logger().info(f'Pose landmarker loaded: {pose_model}')
            else:
                self.get_logger().warn(
                    'Pose model not found — body tracking disabled')
                self.enable_pose = False

        # --- Depth Anything V2 for monocular depth estimation ---
        self.depth_estimator = None
        if self.enable_depth_model:
            self.get_logger().info('Loading Depth Anything V2 on GPU...')
            self.depth_estimator = DepthEstimator(
                self.get_logger(),
                run_every_n=self.depth_run_every_n)
            self.depth_estimator.init_model()
            if not self.depth_estimator.ready:
                self.depth_estimator = None
                self.enable_depth_model = False

        self._frame_ts_ms = 0

        # --- Camera ---
        self.cap = None
        self.running = False

        # --- Visualization colors ---
        self.COLOR_LEFT = (0, 200, 0)      # Green for left hand (xArm5)
        self.COLOR_RIGHT = (255, 128, 0)   # Orange for right hand (UF850)
        self.COLOR_PINCH = (0, 0, 255)     # Red for pinch active
        self.COLOR_FIST = (255, 0, 255)    # Magenta for fist active
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BG = (40, 40, 40)

        # --- 1-Euro filters for each hand (x, y, z) ---
        # min_cutoff: lower = more smoothing at rest (reduces jitter)
        # beta: higher = faster response during motion
        self.filter_left = OneEuroFilter(min_cutoff=1.5, beta=0.01)
        self.filter_right = OneEuroFilter(min_cutoff=1.5, beta=0.01)

        # --- Spatial identity tracking ---
        # Instead of trusting MediaPipe's handedness label (which flips),
        # we track hands by position: assign each detection to the nearest
        # previously-known hand position.
        self.tracked_left_pos = None   # (x, y) normalized, last known
        self.tracked_right_pos = None
        self.identity_initialized = False
        # Max distance (normalized) to accept a position match
        self.max_match_dist = 0.25

        # Pinch threshold (normalized distance)
        self.pinch_threshold = 0.04  # Thumb-pinky close distance

        # --- Center reference for visualization ---
        # Records wrist position when fist first closes (same as bridge center)
        self.left_center_pos = None   # (x, y) normalized
        self.right_center_pos = None
        self.left_active = False      # Fist active = arm enabled
        self.right_active = False

        # --- Hand scale baseline for depth (Z) control ---
        # Uses wrist-to-middle-MCP distance as apparent hand size.
        # Bigger hand in frame = closer to camera = forward motion.
        self.left_baseline_scale = None
        self.right_baseline_scale = None

        # --- Arm orientation tracking from body pose ---
        # Forearm orientation (shoulder/elbow/wrist chain) for each arm
        self.prev_left_arm_rot = None
        self.prev_left_arm_time = 0.0
        self.prev_right_arm_rot = None
        self.prev_right_arm_time = 0.0
        self.filter_left_arm_angular = OneEuroFilter(min_cutoff=2.0, beta=0.02)
        self.filter_right_arm_angular = OneEuroFilter(min_cutoff=2.0, beta=0.02)
        # Fallback: hand-rotation-based tracking (when pose unavailable)
        self.prev_rot_right = None
        self.prev_rot_time_right = 0.0
        self.filter_right_angular = OneEuroFilter(min_cutoff=2.0, beta=0.02)
        # Cached pose landmarks for visualization
        self.cached_pose_landmarks = None

        # --- Detection persistence ---
        self.persistence_frames = self.get_parameter('persistence_frames').value
        self.left_miss_count = 0
        self.right_miss_count = 0
        self.cached_left = None
        self.cached_right = None

        # --- FPS tracking ---
        self._fps_count = 0
        self._fps_time = 0.0
        self._fps_display = 0.0

        # --- Start tracking ---
        self._open_camera()
        self.timer = self.create_timer(1.0 / 30.0, self._process_frame)

        self.get_logger().info(
            f'Hand tracker started -- camera {self.camera_id} '
            f'({self.cam_w}x{self.cam_h}), max_hands={self.max_hands}'
        )
        self.get_logger().info(
            'Left hand -> xArm5 (tool arm) | Right hand -> UF850 (manip arm)'
        )

    def _open_camera(self):
        """Open the webcam."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {self.camera_id}')
            return

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cam_w = actual_w
        self.cam_h = actual_h
        self.running = True
        self.get_logger().info(f'Camera opened: {actual_w}x{actual_h}')

    def _calc_pinch(self, landmarks):
        """Check if thumb and pinky finger tips are close (pinch gesture)."""
        thumb = landmarks[THUMB_TIP]
        pinky = landmarks[PINKY_TIP]
        dist = np.sqrt(
            (thumb.x - pinky.x) ** 2 +
            (thumb.y - pinky.y) ** 2 +
            (thumb.z - pinky.z) ** 2
        )
        return bool(dist < self.pinch_threshold), float(dist)

    def _calc_fist(self, landmarks):
        """Check if all fingertips are below their MCP joints (fist)."""
        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        curled = 0
        for tip, mcp in zip(tips, mcps):
            if landmarks[tip].y > landmarks[mcp].y:
                curled += 1
        return bool(curled >= 3)

    def _bridge_status_cb(self, msg):
        try:
            self.bridge_status = json.loads(msg.data)
        except Exception:
            pass

    def _calc_hand_scale(self, landmarks):
        """Compute apparent hand size as depth proxy.

        Returns wrist-to-middle-MCP distance in normalized image coords.
        Larger value = hand closer to camera.
        """
        wrist = landmarks[WRIST]
        mid_mcp = landmarks[MIDDLE_MCP]
        return float(np.sqrt(
            (wrist.x - mid_mcp.x) ** 2 +
            (wrist.y - mid_mcp.y) ** 2
        ))

    def _calc_hand_orientation(self, landmarks):
        """Compute hand orientation as a ScipyRotation in camera frame.

        Builds a right-handed coordinate frame from palm landmarks:
          y-axis: wrist -> middle MCP (finger pointing direction)
          z-axis: palm normal (cross of index-pinky vectors, points out of palm)
          x-axis: cross(y, z) (roughly thumb direction)

        Returns ScipyRotation or None if degenerate.
        """
        def lm(idx):
            l = landmarks[idx]
            return np.array([l.x, l.y, l.z], dtype=float)

        wrist = lm(WRIST)
        mid_mcp = lm(MIDDLE_MCP)
        index_mcp = lm(INDEX_MCP)
        pinky_mcp = lm(PINKY_MCP)

        y_axis = mid_mcp - wrist
        n = np.linalg.norm(y_axis)
        if n < 1e-6:
            return None
        y_axis /= n

        z_axis = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        n = np.linalg.norm(z_axis)
        if n < 1e-6:
            return None
        z_axis /= n

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= max(np.linalg.norm(x_axis), 1e-6)
        # Reorthogonalize
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(np.linalg.norm(z_axis), 1e-6)

        R = np.column_stack([x_axis, y_axis, z_axis])
        try:
            return ScipyRotation.from_matrix(R)
        except Exception:
            return None

    def _resolve_pose_model(self, param_path=''):
        """Find or download pose landmarker model."""
        if param_path and os.path.isfile(param_path):
            return param_path

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(pkg_dir, '..', 'config',
                         'pose_landmarker_lite.task'),
            '/home/adip/workspaces/disassembly_ws/src/hand_teleop/config/'
            'pose_landmarker_lite.task',
        ]
        for c in candidates:
            resolved = os.path.realpath(c)
            if os.path.isfile(resolved):
                return resolved

        # Auto-download
        url = ('https://storage.googleapis.com/mediapipe-models/'
               'pose_landmarker/pose_landmarker_lite/float16/latest/'
               'pose_landmarker_lite.task')
        dest = candidates[-1]
        self.get_logger().info(f'Downloading pose model to {dest}...')
        try:
            import urllib.request
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            self.get_logger().info('Pose model downloaded')
            return dest
        except Exception as e:
            self.get_logger().error(f'Failed to download pose model: {e}')
            return None

    def _compute_arm_rotation(self, shoulder, elbow, wrist):
        """Compute forearm orientation from body landmarks.

        Builds a right-handed coordinate frame from the arm chain:
          z-axis: forearm direction (elbow -> wrist, pointing direction)
          y-axis: arm plane normal (perpendicular to shoulder-elbow-wrist plane)
          x-axis: cross(y, z) completes the frame

        Returns ScipyRotation or None if degenerate.
        """
        forearm = wrist - elbow
        fn = np.linalg.norm(forearm)
        if fn < 1e-6:
            return None
        z_axis = forearm / fn

        upper = elbow - shoulder
        normal = np.cross(upper, forearm)
        nn = np.linalg.norm(normal)
        if nn < 1e-6:
            return None
        y_axis = normal / nn

        x_axis = np.cross(y_axis, z_axis)
        xn = np.linalg.norm(x_axis)
        if xn < 1e-6:
            return None
        x_axis /= xn

        R = np.column_stack([x_axis, y_axis, z_axis])
        try:
            return ScipyRotation.from_matrix(R)
        except Exception:
            return None

    def _process_pose(self, pose_landmarks, now):
        """Process body pose landmarks to compute arm orientations.

        Computes forearm orientation from shoulder/elbow/wrist chain for
        each arm. Returns both:
          - Angular velocity (frame-to-frame delta / dt)
          - Absolute orientation quaternion [x, y, z, w]

        Returns dict with keys:
          'left_ang_vel', 'right_ang_vel': 3-element lists
          'left_quat', 'right_quat': 4-element [x,y,z,w] or None
        """
        result = {
            'left_ang_vel': [0.0, 0.0, 0.0],
            'right_ang_vel': [0.0, 0.0, 0.0],
            'left_quat': None,
            'right_quat': None,
        }

        lms = pose_landmarks[0]  # First (only) person

        def lm_xyz(idx):
            l = lms[idx]
            return np.array([l.x, l.y, l.z], dtype=float), l.visibility

        # --- Left arm (user's left = xArm5) ---
        l_shoulder, l_s_vis = lm_xyz(POSE_LEFT_SHOULDER)
        l_elbow, l_e_vis = lm_xyz(POSE_LEFT_ELBOW)
        l_wrist, l_w_vis = lm_xyz(POSE_LEFT_WRIST)

        if all(v > POSE_MIN_VISIBILITY for v in [l_s_vis, l_e_vis, l_w_vis]):
            rot = self._compute_arm_rotation(l_shoulder, l_elbow, l_wrist)
            if rot is not None:
                q = rot.as_quat()  # [x, y, z, w]
                result['left_quat'] = q.tolist()
                if self.prev_left_arm_rot is not None:
                    dt = now - self.prev_left_arm_time
                    if 0 < dt < 0.5:
                        delta = rot * self.prev_left_arm_rot.inv()
                        raw_ang = delta.as_rotvec() / dt
                        result['left_ang_vel'] = self.filter_left_arm_angular(
                            raw_ang.tolist(), now)
                self.prev_left_arm_rot = rot
                self.prev_left_arm_time = now
            else:
                self.prev_left_arm_rot = None
        else:
            self.prev_left_arm_rot = None

        # --- Right arm (user's right = UF850) ---
        r_shoulder, r_s_vis = lm_xyz(POSE_RIGHT_SHOULDER)
        r_elbow, r_e_vis = lm_xyz(POSE_RIGHT_ELBOW)
        r_wrist, r_w_vis = lm_xyz(POSE_RIGHT_WRIST)

        if all(v > POSE_MIN_VISIBILITY for v in [r_s_vis, r_e_vis, r_w_vis]):
            rot = self._compute_arm_rotation(r_shoulder, r_elbow, r_wrist)
            if rot is not None:
                q = rot.as_quat()  # [x, y, z, w]
                result['right_quat'] = q.tolist()
                if self.prev_right_arm_rot is not None:
                    dt = now - self.prev_right_arm_time
                    if 0 < dt < 0.5:
                        delta = rot * self.prev_right_arm_rot.inv()
                        raw_ang = delta.as_rotvec() / dt
                        result['right_ang_vel'] = \
                            self.filter_right_arm_angular(
                                raw_ang.tolist(), now)
                self.prev_right_arm_rot = rot
                self.prev_right_arm_time = now
            else:
                self.prev_right_arm_rot = None
        else:
            self.prev_right_arm_rot = None

        return result

    def _draw_pose_skeleton(self, frame, pose_landmarks):
        """Draw arm skeleton from body pose on visualization."""
        if not pose_landmarks:
            return
        h, w = frame.shape[:2]
        lms = pose_landmarks[0]

        # Draw arm connections: shoulder-elbow-wrist for each arm
        arm_pairs = [
            # Left arm (green = xArm5)
            (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, self.COLOR_LEFT),
            (POSE_LEFT_ELBOW, POSE_LEFT_WRIST, self.COLOR_LEFT),
            # Right arm (orange = UF850)
            (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, self.COLOR_RIGHT),
            (POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST, self.COLOR_RIGHT),
        ]
        for start_idx, end_idx, color in arm_pairs:
            s = lms[start_idx]
            e = lms[end_idx]
            if s.visibility < POSE_MIN_VISIBILITY or \
               e.visibility < POSE_MIN_VISIBILITY:
                continue
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), color, 3, cv2.LINE_AA)

        # Draw shoulder line (torso reference)
        ls = lms[POSE_LEFT_SHOULDER]
        rs = lms[POSE_RIGHT_SHOULDER]
        if ls.visibility > POSE_MIN_VISIBILITY and \
           rs.visibility > POSE_MIN_VISIBILITY:
            lsx, lsy = int(ls.x * w), int(ls.y * h)
            rsx, rsy = int(rs.x * w), int(rs.y * h)
            cv2.line(frame, (lsx, lsy), (rsx, rsy),
                     (100, 100, 100), 2, cv2.LINE_AA)

        # Draw joint dots
        joint_indices = [
            (POSE_LEFT_SHOULDER, self.COLOR_LEFT),
            (POSE_LEFT_ELBOW, self.COLOR_LEFT),
            (POSE_LEFT_WRIST, self.COLOR_LEFT),
            (POSE_RIGHT_SHOULDER, self.COLOR_RIGHT),
            (POSE_RIGHT_ELBOW, self.COLOR_RIGHT),
            (POSE_RIGHT_WRIST, self.COLOR_RIGHT),
        ]
        for idx, color in joint_indices:
            lm = lms[idx]
            if lm.visibility < POSE_MIN_VISIBILITY:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

    def _assign_identities(self, detections):
        """Assign Left/Right identity using spatial tracking.

        On first detection with 2 hands: use x-position (leftmost = Left).
        After that: match each detection to the nearest tracked position.
        Falls back to MediaPipe handedness only when spatial match fails.

        Args:
            detections: list of (landmarks, mp_label, score) tuples

        Returns:
            dict mapping 'Left'/'Right' to (landmarks, score)
        """
        if not detections:
            return {}

        assigned = {}

        if len(detections) == 1:
            lms, mp_label, score = detections[0]
            wrist = lms[WRIST]
            pos = (wrist.x, wrist.y)

            if not self.identity_initialized:
                # No prior tracking — use position in frame as primary cue:
                # In mirrored view, left side of frame = user's left hand
                # Fall back to corrected MediaPipe label only if ambiguous
                if pos[0] < 0.4:
                    label = 'Left'
                elif pos[0] > 0.6:
                    label = 'Right'
                else:
                    # Ambiguous center region — use MediaPipe label
                    # When flip_h=True, MP sees mirrored anatomy so we invert
                    if self.flip_h:
                        label = 'Right' if mp_label == 'Left' else 'Left'
                    else:
                        label = mp_label
                assigned[label] = (lms, score)
                if label == 'Left':
                    self.tracked_left_pos = pos
                else:
                    self.tracked_right_pos = pos
                return assigned

            # Match to nearest tracked position
            dist_left = (np.sqrt((pos[0] - self.tracked_left_pos[0]) ** 2 +
                                  (pos[1] - self.tracked_left_pos[1]) ** 2)
                         if self.tracked_left_pos else float('inf'))
            dist_right = (np.sqrt((pos[0] - self.tracked_right_pos[0]) ** 2 +
                                   (pos[1] - self.tracked_right_pos[1]) ** 2)
                          if self.tracked_right_pos else float('inf'))

            if dist_left <= dist_right and dist_left < self.max_match_dist:
                assigned['Left'] = (lms, score)
                self.tracked_left_pos = pos
            elif dist_right < self.max_match_dist:
                assigned['Right'] = (lms, score)
                self.tracked_right_pos = pos
            else:
                # Too far from any known position — use frame position
                if pos[0] < 0.4:
                    label = 'Left'
                elif pos[0] > 0.6:
                    label = 'Right'
                else:
                    if self.flip_h:
                        label = 'Right' if mp_label == 'Left' else 'Left'
                    else:
                        label = mp_label
                assigned[label] = (lms, score)
                if label == 'Left':
                    self.tracked_left_pos = pos
                else:
                    self.tracked_right_pos = pos

            return assigned

        # --- Two hands detected ---
        positions = []
        for lms, mp_label, score in detections:
            wrist = lms[WRIST]
            positions.append((wrist.x, wrist.y))

        if not self.identity_initialized:
            # First time with 2 hands: leftmost in frame = Left hand
            if positions[0][0] < positions[1][0]:
                left_idx, right_idx = 0, 1
            else:
                left_idx, right_idx = 1, 0

            assigned['Left'] = (detections[left_idx][0], detections[left_idx][2])
            assigned['Right'] = (detections[right_idx][0], detections[right_idx][2])
            self.tracked_left_pos = positions[left_idx]
            self.tracked_right_pos = positions[right_idx]
            self.identity_initialized = True
            return assigned

        # Spatial matching: compute 2x2 distance matrix and pick best assignment
        if self.tracked_left_pos and self.tracked_right_pos:
            d00 = np.sqrt((positions[0][0] - self.tracked_left_pos[0]) ** 2 +
                          (positions[0][1] - self.tracked_left_pos[1]) ** 2)
            d01 = np.sqrt((positions[0][0] - self.tracked_right_pos[0]) ** 2 +
                          (positions[0][1] - self.tracked_right_pos[1]) ** 2)
            d10 = np.sqrt((positions[1][0] - self.tracked_left_pos[0]) ** 2 +
                          (positions[1][1] - self.tracked_left_pos[1]) ** 2)
            d11 = np.sqrt((positions[1][0] - self.tracked_right_pos[0]) ** 2 +
                          (positions[1][1] - self.tracked_right_pos[1]) ** 2)

            # Best assignment: minimize total distance
            cost_straight = d00 + d11  # 0->Left, 1->Right
            cost_cross = d01 + d10     # 0->Right, 1->Left

            if cost_straight <= cost_cross:
                left_idx, right_idx = 0, 1
            else:
                left_idx, right_idx = 1, 0
        else:
            # Fallback: leftmost = Left
            if positions[0][0] < positions[1][0]:
                left_idx, right_idx = 0, 1
            else:
                left_idx, right_idx = 1, 0

        assigned['Left'] = (detections[left_idx][0], detections[left_idx][2])
        assigned['Right'] = (detections[right_idx][0], detections[right_idx][2])
        self.tracked_left_pos = positions[left_idx]
        self.tracked_right_pos = positions[right_idx]
        self.identity_initialized = True

        return assigned

    def _make_wrist_pose(self, wrist_xyz, frame_id='camera_frame'):
        """Create a PoseStamped from wrist position."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(wrist_xyz[0])
        msg.pose.position.y = float(wrist_xyz[1])
        msg.pose.position.z = float(wrist_xyz[2])
        msg.pose.orientation.w = 1.0
        return msg

    def _draw_center_reference(self, frame):
        """Draw center crosshair, control radius circle, and displacement."""
        h, w = frame.shape[:2]
        # control_radius in normalized coords — use avg of w/h to get pixel radius
        ctrl_r_px = int(0.25 * (w + h) / 2)

        for label, center_pos, active, color in [
            ('xArm5', self.left_center_pos, self.left_active, self.COLOR_LEFT),
            ('UF850', self.right_center_pos, self.right_active, self.COLOR_RIGHT),
        ]:
            if center_pos is None:
                continue

            cx = int(center_pos[0] * w)
            cy = int(center_pos[1] * h)

            # Draw control radius circle (max-speed boundary)
            cv2.circle(frame, (cx, cy), ctrl_r_px, color, 1, cv2.LINE_AA)
            cv2.putText(frame, 'MAX', (cx + ctrl_r_px + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Draw center crosshair
            cross_size = 20
            cv2.line(frame, (cx - cross_size, cy), (cx - 5, cy), color, 2)
            cv2.line(frame, (cx + 5, cy), (cx + cross_size, cy), color, 2)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy - 5), color, 2)
            cv2.line(frame, (cx, cy + 5), (cx, cy + cross_size), color, 2)
            cv2.circle(frame, (cx, cy), 6, color, 2)
            cv2.circle(frame, (cx, cy), 2, color, -1)

            cv2.putText(frame, f'{label} CENTER', (cx - 40, cy - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw displacement arrow and speed indicator
            if active:
                cached = (self.cached_left if label == 'xArm5'
                          else self.cached_right)
                if cached is not None:
                    wrist = cached[0]['wrist']
                    wx = int(wrist[0] * w)
                    wy = int(wrist[1] * h)
                    # Arrow from center to hand
                    cv2.arrowedLine(frame, (cx, cy), (wx, wy),
                                   color, 2, tipLength=0.15)
                    # Speed fraction (distance / control_radius, clamped)
                    dx = (wrist[0] - center_pos[0])
                    dy = (wrist[1] - center_pos[1])
                    dist_norm = min(1.0, (dx**2 + dy**2)**0.5 / 0.25)
                    speed_pct = int(dist_norm**2 * 100)  # quadratic
                    cv2.putText(frame, f'{speed_pct}%',
                                (wx + 8, wy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_info_panel(self, frame, left_data, right_data):
        """Draw status panel at top of frame."""
        h, w = frame.shape[:2]
        panel_h = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        title = 'HAND+BODY TELEOP' if self.enable_pose else 'HAND TELEOP'
        cv2.putText(frame, title, (10, 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_TEXT, 2)
        if self.enable_pose:
            pose_color = (0, 255, 0) if self.cached_pose_landmarks else (80, 80, 80)
            cv2.putText(frame, 'POSE', (260, 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_color, 1)
        if self.enable_depth_model:
            depth_color = (0, 255, 0) if (self.depth_estimator and
                                           self.depth_estimator.ready) else (80, 80, 80)
            cv2.putText(frame, 'DEPTH', (320, 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, depth_color, 1)

        bs = self.bridge_status
        xa = bs.get('xarm5', {})
        uf = bs.get('uf850', {})
        grip_closed = bs.get('gripper_closed', False)

        # --- LEFT: xArm5 ---
        left_color = self.COLOR_LEFT if left_data else (80, 80, 80)
        xarm_active = xa.get('active', False)
        status_l = 'ACTIVE' if xarm_active else ('DETECTED' if left_data else 'NOT FOUND')
        thickness_l = 2 if xarm_active else 1
        cv2.putText(frame, f'L: xArm5  {status_l}', (10, 48),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, thickness_l)

        if left_data:
            gesture_l = ('FIST' if left_data.get('fist') else
                         'PINCH' if left_data.get('pinch') else 'OPEN')
            mode_l = xa.get('mode', 'position').upper()
            cv2.putText(frame, f'  {gesture_l}  |  {mode_l}', (10, 70),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, left_color, 1)

        # --- RIGHT: UF850 ---
        right_color = self.COLOR_RIGHT if right_data else (80, 80, 80)
        uf_active = uf.get('active', False)
        status_r = 'ACTIVE' if uf_active else ('DETECTED' if right_data else 'NOT FOUND')
        thickness_r = 2 if uf_active else 1
        cv2.putText(frame, f'R: UF850  {status_r}', (w // 2, 48),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, thickness_r)

        if right_data:
            gesture_r = ('FIST' if right_data.get('fist') else
                         'PINCH' if right_data.get('pinch') else 'OPEN')
            mode_r = uf.get('mode', 'position').upper()
            # Mode color: cyan for orientation, default for position
            mode_color = (0, 255, 255) if mode_r == 'ORIENTATION' else right_color
            cv2.putText(frame, f'  {gesture_r}  |  {mode_r}', (w // 2, 70),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_color, 1)

        # --- Gripper status (bottom of panel) ---
        grip_color = (0, 80, 255) if grip_closed else (0, 200, 100)
        grip_str = 'GRIPPER: CLOSED' if grip_closed else 'GRIPPER: OPEN'
        cv2.putText(frame, grip_str, (w // 2 - 10, 95),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, grip_color, 2)

        # --- Pinch hint ---
        cv2.putText(frame,
                     'L-Pinch=Mode  |  R-Pinch=Gripper',
                     (10, 95),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130, 130, 130), 1)

    def _draw_hand_landmarks(self, frame, landmarks, hand_label, pinch,
                              fist, pinch_dist):
        """Draw hand landmarks with custom styling."""
        h, w = frame.shape[:2]
        color = self.COLOR_LEFT if hand_label == 'Left' else self.COLOR_RIGHT
        arm_label = 'xArm5' if hand_label == 'Left' else 'UF850'

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            sx, sy = int(start.x * w), int(start.y * h)
            ex, ey = int(end.x * w), int(end.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            radius = 6 if i in [WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP,
                                  RING_TIP, PINKY_TIP] else 3
            lm_color = color
            if pinch and i in [THUMB_TIP, PINKY_TIP]:
                lm_color = self.COLOR_PINCH
                radius = 10
            if fist:
                lm_color = self.COLOR_FIST
            cv2.circle(frame, (cx, cy), radius, lm_color, -1)
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)

        # Draw pinch line (thumb to pinky)
        thumb = landmarks[THUMB_TIP]
        pinky = landmarks[PINKY_TIP]
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        px, py = int(pinky.x * w), int(pinky.y * h)
        pinch_color = self.COLOR_PINCH if pinch else (150, 150, 150)
        cv2.line(frame, (tx, ty), (px, py), pinch_color, 2 if pinch else 1)

        mid_x, mid_y = (tx + px) // 2, (ty + py) // 2
        cv2.putText(frame, f'{pinch_dist:.3f}', (mid_x + 5, mid_y - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, pinch_color, 1)

        # Draw wrist label and crosshair
        wrist = landmarks[WRIST]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, f'{hand_label} ({arm_label})',
                     (wx - 40, wy + 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cross_size = 15
        cv2.line(frame, (wx - cross_size, wy), (wx + cross_size, wy), color, 2)
        cv2.line(frame, (wx, wy - cross_size), (wx, wy + cross_size), color, 2)

    def _draw_workspace_guide(self, frame):
        """Draw shared workspace boundary (both arms overlap)."""
        h, w = frame.shape[:2]
        margin = 60
        cv2.rectangle(frame, (margin, 100), (w - margin, h - 20),
                       (80, 80, 80), 1)
        cv2.putText(frame, 'SHARED WORKSPACE (xArm5 + UF850)',
                     (margin + 10, h - 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    def _process_frame(self):
        """Main processing loop -- called at 30Hz by ROS timer."""
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn(
                'Failed to read camera frame', throttle_duration_sec=5.0)
            return

        if self.flip_h:
            frame = cv2.flip(frame, 1)

        # Convert BGR->RGB and wrap in MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hands (VIDEO mode)
        self._frame_ts_ms += 33
        result = self.landmarker.detect_for_video(
            mp_image, self._frame_ts_ms)

        # FPS tracking
        now = time.monotonic()
        self._fps_count += 1
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_time = now

        # --- Run depth estimation on GPU (every N frames) ---
        if self.depth_estimator is not None:
            self.depth_estimator.update(rgb)

        # --- Collect raw detections ---
        raw_detections = []
        if result.hand_landmarks and result.handedness:
            for hand_lms, handedness_list in zip(
                    result.hand_landmarks, result.handedness):
                mp_label = handedness_list[0].category_name
                score = handedness_list[0].score
                raw_detections.append((hand_lms, mp_label, score))

        # --- Assign identities via spatial tracking ---
        assigned = self._assign_identities(raw_detections)

        left_data = None
        right_data = None
        detected_labels = set(assigned.keys())

        for label, (landmarks, score) in assigned.items():
            wrist = landmarks[WRIST]

            # Z (depth) estimation:
            # Primary: Depth Anything V2 model (real monocular depth)
            # Fallback: hand apparent scale (wrist-to-MCP distance)
            if self.depth_estimator is not None:
                depth_val = self.depth_estimator.get_depth_at(
                    wrist.x, wrist.y)
                if depth_val is not None:
                    z_value = depth_val
                else:
                    z_value = self._calc_hand_scale(landmarks)
            else:
                z_value = self._calc_hand_scale(landmarks)

            # 1-Euro filter for smooth, low-latency wrist tracking
            # Filter X, Y from wrist; Z slot carries depth (filtered too)
            filt = self.filter_left if label == 'Left' else self.filter_right
            wrist_xyz = filt([wrist.x, wrist.y, z_value], now)

            # Gesture detection
            fist = self._calc_fist(landmarks)
            pinch, pinch_dist = self._calc_pinch(landmarks)
            # Suppress pinch when fist is active — closing fist brings
            # thumb near index finger, which falsely triggers pinch
            if fist:
                pinch = False

            # Track center reference, scale baseline, and active state
            if label == 'Left':
                if fist:
                    if self.left_center_pos is None:
                        self.left_center_pos = (wrist_xyz[0], wrist_xyz[1])
                        self.left_baseline_scale = wrist_xyz[2]
                    self.left_active = True
                    # Replace Z with delta from baseline (positive = closer)
                    if self.left_baseline_scale is not None:
                        wrist_xyz[2] = wrist_xyz[2] - self.left_baseline_scale
                else:
                    self.left_active = False
                    self.left_center_pos = None
                    self.left_baseline_scale = None
                    wrist_xyz[2] = 0.0
            else:
                if fist:
                    if self.right_center_pos is None:
                        self.right_center_pos = (wrist_xyz[0], wrist_xyz[1])
                        self.right_baseline_scale = wrist_xyz[2]
                    self.right_active = True
                    if self.right_baseline_scale is not None:
                        wrist_xyz[2] = wrist_xyz[2] - self.right_baseline_scale
                else:
                    self.right_active = False
                    self.right_center_pos = None
                    self.right_baseline_scale = None
                    wrist_xyz[2] = 0.0

            hand_info = {
                'label': label,
                'confidence': float(score),
                'wrist': wrist_xyz,
                'pinch': pinch,
                'pinch_dist': pinch_dist,
                'fist': fist,
                'landmarks': [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z}
                    for lm in landmarks
                ],
            }

            if label == 'Left':
                left_data = hand_info
                self.cached_left = (hand_info, landmarks)
                self.left_miss_count = 0
            else:
                right_data = hand_info
                self.cached_right = (hand_info, landmarks)
                self.right_miss_count = 0

        # Detection persistence: use cached data for briefly lost hands
        if 'Left' not in detected_labels:
            self.left_miss_count += 1
            if (self.cached_left is not None
                    and self.left_miss_count <= self.persistence_frames):
                left_data = self.cached_left[0]
            else:
                self.cached_left = None

        if 'Right' not in detected_labels:
            self.right_miss_count += 1
            if (self.cached_right is not None
                    and self.right_miss_count <= self.persistence_frames):
                right_data = self.cached_right[0]
            else:
                self.cached_right = None

        # Reset spatial tracking if both hands lost for too long
        if self.cached_left is None and self.cached_right is None:
            self.identity_initialized = False
            self.tracked_left_pos = None
            self.tracked_right_pos = None

        # --- Compute arm angular velocities from body pose ---
        # Primary: use body pose (shoulder/elbow/wrist chain) for arm orientation
        # Fallback: use hand rotation (wrist landmarks) if pose unavailable
        pose_ang = None
        if self.enable_pose and self.pose_landmarker is not None:
            pose_result = self.pose_landmarker.detect_for_video(
                mp_image, self._frame_ts_ms)
            if pose_result.pose_landmarks:
                self.cached_pose_landmarks = pose_result.pose_landmarks
                pose_ang = self._process_pose(
                    pose_result.pose_landmarks, now)
            else:
                self.cached_pose_landmarks = None

        # Publish left arm angular velocity
        left_ang = (pose_ang['left_ang_vel']
                    if pose_ang else [0.0, 0.0, 0.0])
        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = self.get_clock().now().to_msg()
        ang_msg.header.frame_id = 'camera_frame'
        ang_msg.vector.x = float(left_ang[0])
        ang_msg.vector.y = float(left_ang[1])
        ang_msg.vector.z = float(left_ang[2])
        self.pub_left_angular_vel.publish(ang_msg)

        # Publish right arm angular velocity
        if pose_ang:
            # Body pose available — use arm chain orientation
            right_ang = pose_ang['right_ang_vel']
            self.prev_rot_right = None  # Reset hand fallback
        elif right_data is not None and self.cached_right is not None:
            # Fallback: hand rotation tracking
            _, right_lms = self.cached_right
            rot = self._calc_hand_orientation(right_lms)
            right_ang = [0.0, 0.0, 0.0]
            if rot is not None:
                if self.prev_rot_right is not None:
                    dt = now - self.prev_rot_time_right
                    if 0 < dt < 0.5:
                        r_delta = rot * self.prev_rot_right.inv()
                        raw_ang = r_delta.as_rotvec() / dt
                        right_ang = self.filter_right_angular(
                            raw_ang.tolist(), now)
                self.prev_rot_right = rot
                self.prev_rot_time_right = now
            else:
                self.prev_rot_right = None
        else:
            right_ang = [0.0, 0.0, 0.0]
            self.prev_rot_right = None

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = self.get_clock().now().to_msg()
        ang_msg.header.frame_id = 'camera_frame'
        ang_msg.vector.x = float(right_ang[0])
        ang_msg.vector.y = float(right_ang[1])
        ang_msg.vector.z = float(right_ang[2])
        self.pub_right_angular_vel.publish(ang_msg)

        # Publish absolute arm orientation quaternions (for pose matching)
        stamp = self.get_clock().now().to_msg()
        for side, quat, pub in [
            ('left', pose_ang.get('left_quat') if pose_ang else None,
             self.pub_left_arm_orient),
            ('right', pose_ang.get('right_quat') if pose_ang else None,
             self.pub_right_arm_orient),
        ]:
            if quat is not None:
                orient_msg = PoseStamped()
                orient_msg.header.stamp = stamp
                orient_msg.header.frame_id = 'camera_frame'
                orient_msg.pose.orientation.x = float(quat[0])
                orient_msg.pose.orientation.y = float(quat[1])
                orient_msg.pose.orientation.z = float(quat[2])
                orient_msg.pose.orientation.w = float(quat[3])
                pub.publish(orient_msg)

        # Publish per-hand data
        for label, data in [('Left', left_data), ('Right', right_data)]:
            if data is None:
                continue
            pose_msg = self._make_wrist_pose(data['wrist'])
            pinch_msg = Bool(data=data['pinch'])
            fist_msg = Bool(data=data['fist'])
            if label == 'Left':
                self.pub_left_wrist.publish(pose_msg)
                self.pub_left_pinch.publish(pinch_msg)
                self.pub_left_fist.publish(fist_msg)
            else:
                self.pub_right_wrist.publish(pose_msg)
                self.pub_right_pinch.publish(pinch_msg)
                self.pub_right_fist.publish(fist_msg)

        # Draw visualization
        if self.show_viz:
            for label, data_cache in [('Left', self.cached_left),
                                       ('Right', self.cached_right)]:
                if data_cache is None:
                    continue
                info, lms = data_cache
                miss = (self.left_miss_count if label == 'Left'
                        else self.right_miss_count)
                if miss <= self.persistence_frames:
                    self._draw_hand_landmarks(
                        frame, lms, label, info['pinch'],
                        info['fist'], info['pinch_dist'])

        # Publish combined landmarks JSON
        all_landmarks = {}
        if left_data:
            all_landmarks['Left'] = left_data
        if right_data:
            all_landmarks['Right'] = right_data
        if all_landmarks:
            lm_msg = String()
            lm_msg.data = json.dumps(all_landmarks, default=str)
            self.pub_landmarks.publish(lm_msg)

        # Publish tracking status
        status_msg = String()
        status = {
            'left_tracked': left_data is not None,
            'right_tracked': right_data is not None,
            'hands_detected': len(detected_labels),
        }
        status_msg.data = json.dumps(status)
        self.pub_status.publish(status_msg)

        # Visualization
        if self.show_viz:
            # Draw body pose skeleton (arms) behind hand landmarks
            if self.cached_pose_landmarks:
                self._draw_pose_skeleton(frame, self.cached_pose_landmarks)
            self._draw_workspace_guide(frame)
            self._draw_center_reference(frame)
            self._draw_info_panel(frame, left_data, right_data)

            fps_color = (0, 255, 0) if self._fps_display >= 25 else (0, 200, 255)
            cv2.putText(frame, f'FPS: {self._fps_display:.0f}',
                         (self.cam_w - 280, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            cv2.putText(frame, 'Q=quit', (self.cam_w - 80, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow('Hand Teleop Tracker', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested -- shutting down')
                self.running = False
                rclpy.shutdown()

    def destroy_node(self):
        """Cleanup on shutdown."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
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
