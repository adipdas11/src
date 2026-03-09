#!/usr/bin/env python3
"""Launch hand tracker + robot bridge for hand teleoperation."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('hand_teleop')
    tracker_params = os.path.join(pkg_dir, 'config', 'hand_teleop_params.yaml')
    bridge_params = os.path.join(pkg_dir, 'config', 'hand_bridge_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_id', default_value='0',
            description='Camera device index'),

        DeclareLaunchArgument(
            'show_viz', default_value='true',
            description='Show OpenCV hand tracking visualization'),

        DeclareLaunchArgument(
            'linear_scale', default_value='0.4',
            description='Displacement-to-velocity gain'),

        DeclareLaunchArgument(
            'max_vel', default_value='0.15',
            description='Max linear velocity (m/s)'),

        # Hand tracker node
        Node(
            package='hand_teleop',
            executable='hand_tracker',
            name='hand_tracker',
            output='screen',
            parameters=[tracker_params, {
                'camera_id': LaunchConfiguration('camera_id'),
                'show_visualization': LaunchConfiguration('show_viz'),
            }],
        ),

        # Hand-to-robot bridge node
        Node(
            package='hand_teleop',
            executable='hand_robot_bridge',
            name='hand_robot_bridge',
            output='screen',
            parameters=[bridge_params, {
                'linear_scale': LaunchConfiguration('linear_scale'),
                'max_linear_vel': LaunchConfiguration('max_vel'),
            }],
        ),
    ])
