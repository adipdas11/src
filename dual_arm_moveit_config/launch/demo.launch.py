import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    # 1. SETUP CONFIG
    moveit_config_pkg = "dual_arm_moveit_config"
    
    # 1. Define the path to your RViz config
    rviz_config_file = os.path.join(
        get_package_share_directory("dual_arm_moveit_config"),
        "rviz",
        "dual_arm.rviz"
    )
    
    moveit_config = (
        MoveItConfigsBuilder("dual_arm_world", package_name=moveit_config_pkg)
        .robot_description(file_path="config/dual_arm_world.urdf")
        .robot_description_semantic(file_path="config/dual_arm_world.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # 2. DEFINE NODES

    # B. Move Group
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": False},
            {"planning_scene_monitor_options": {
                "name": "planning_scene_monitor",
                "robot_description": "robot_description",
                "default_robot_padding": 0.02,
                "default_robot_scale": 1.0,
            }}
        ],
    )
    
    # C. Robot State Publisher (Still needed for TF)
    run_rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    
    # D. Rviz & Static TF (Same as before)
    run_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],  
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )
    
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "world_world"],
    )

    return LaunchDescription([
        static_tf,
        run_rsp_node,
        run_move_group_node,
        run_rviz_node,


    ])