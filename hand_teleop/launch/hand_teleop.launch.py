#!/usr/bin/env python3
"""Launch hand tracking teleoperation visualizer."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('hand_teleop')
    params_file = os.path.join(pkg_dir, 'config', 'hand_teleop_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_id', default_value='0',
            description='Camera device index'),

        DeclareLaunchArgument(
            'show_viz', default_value='true',
            description='Show OpenCV visualization window'),

        Node(
            package='hand_teleop',
            executable='hand_tracker',
            name='hand_tracker',
            output='screen',
            parameters=[params_file, {
                'camera_id': LaunchConfiguration('camera_id'),
                'show_visualization': LaunchConfiguration('show_viz'),
            }],
        ),
    ])
