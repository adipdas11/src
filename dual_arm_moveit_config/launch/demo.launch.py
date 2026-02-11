import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, 
                            ExecuteProcess, TimerAction, OpaqueFunction, 
                            LogInfo, Shutdown)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def launch_setup(context, *args, **kwargs):
    hw_type = LaunchConfiguration('hardware_type').perform(context)
    
    print(f"\n{'='*50}\n🚀 STARTING DUAL-ARM SYSTEM IN [{hw_type.upper()}] MODE\n{'='*50}\n")

    moveit_config_pkg = "dual_arm_moveit_config"
    pkg_share = get_package_share_directory(moveit_config_pkg)
    rviz_config_file = os.path.join(pkg_share, "rviz", "dual_arm.rviz")
    
    # 1. Load MoveIt Config
    moveit_config = (
        MoveItConfigsBuilder("dual_arm_world", package_name=moveit_config_pkg)
        .robot_description(file_path=os.path.join(pkg_share, "config", "dual_arm_world.urdf"))
        .robot_description_semantic(file_path=os.path.join(pkg_share, "config", "dual_arm_world.srdf"))
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # 2. Core Nodes
    static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "world_world"]
    )
    
    run_rsp_node = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    # 3. Hardware Nodes
    hardware_actions = []
    if hw_type == 'real':
        # FT Sensor
        ft_sensor_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('robotiq_ft_sensor_hardware'),
                'launch', 'ft_sensor_standalone.launch.py'))
        )
        # Real Hardware Driver (with on_exit shutdown)
        real_hw_process = ExecuteProcess(
            cmd=['ros2', 'run', 'dual_arm_moveit_config', 'real_hardware.py'],
            output='screen',
            on_exit=Shutdown()
        )
        hardware_actions = [
            LogInfo(msg="🔌 Launching REAL hardware drivers and FT sensor..."),
            real_hw_process,
            ft_sensor_launch
        ]
    else:
        # Fake Hardware
        fake_hw_process = ExecuteProcess(
            cmd=['ros2', 'run', 'dual_arm_moveit_config', 'fake_hardware.py'],
            output='screen'
        )
        hardware_actions = [
            LogInfo(msg="💻 Launching FAKE hardware simulation..."),
            fake_hw_process
        ]

    # 4. Tools & Perception (UPDATED)
    tool_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('tool_controller'),
            'launch', 'tool_launch.py'))
    )

    # Hand-Eye Calibration Publisher
    handeye_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('easy_handeye2'),
            'launch', 'publish.launch.py')),
        launch_arguments={
            'name': 'realsense_handeye',
            'calibration_file': '/home/adip/workspaces/disassembly_ws/src/disassembly_skills/config/realsense_handeye.calib'
        }.items()
    )

    # 5. MoveGroup & RViz
    run_move_group_node = Node(
        package="moveit_ros_move_group", executable="move_group",
        parameters=[moveit_config.to_dict(), {"use_sim_time": False}],
    )

    run_rviz_node = Node(
        package="rviz2", executable="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[moveit_config.to_dict()],
    )

    # 6. Execution Sequence
    return [
        LogInfo(msg="🛰️  Step 1: Loading Robot State and Static TFs..."),
        static_tf,
        run_rsp_node,

        TimerAction(period=4.0, actions=[
            LogInfo(msg="🦾 Step 2: Initializing Hardware..."),
            *hardware_actions
        ]),

        TimerAction(period=8.0, actions=[
            LogInfo(msg="🔧 Step 3: Launching Tool Controller & Hand-Eye Calibration..."),
            tool_controller,
            handeye_publisher
        ]),

        TimerAction(period=12.0, actions=[
            LogInfo(msg="🧠 Step 4: Starting MoveGroup Brain..."),
            run_move_group_node
        ]),

        TimerAction(period=16.0, actions=[
            LogInfo(msg="📊 Step 5: Opening RViz Visualization..."),
            run_rviz_node,
            LogInfo(msg="✅ SYSTEM READY. Happy Disassembling!")
        ]),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'hardware_type',
            default_value='fake',
            description='Select hardware type: "real" or "fake"'),
        OpaqueFunction(function=launch_setup)
    ])