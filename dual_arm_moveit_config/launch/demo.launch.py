import os
import yaml
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
    ft_sensor_ftdi_id = LaunchConfiguration('ft_sensor_ftdi_id')
    ft_sensor_max_retries = LaunchConfiguration('ft_sensor_max_retries')
    ft_sensor_frame_id = LaunchConfiguration('ft_sensor_frame_id')
    
    print(f"\n{'='*50}\n🚀 STARTING DUAL-ARM SYSTEM IN [{hw_type.upper()}] MODE\n{'='*50}\n")

    moveit_config_pkg = "dual_arm_moveit_config"
    pkg_share = get_package_share_directory(moveit_config_pkg)
    rviz_config_file = os.path.join(pkg_share, "rviz", "dual_arm.rviz")
    sensors_3d_config = os.path.join(pkg_share, "config", "sensors_3d.yaml")
    sensors_3d_params = []
    if os.path.exists(sensors_3d_config):
        with open(sensors_3d_config, 'r') as file:
            sensors_3d_params.append(yaml.safe_load(file))
    
    # 1. Load MoveIt Config 
    moveit_config = (
        MoveItConfigsBuilder("dual_arm_world", package_name=moveit_config_pkg)
        .robot_description(
            file_path=os.path.join(pkg_share, "config", "dual_arm_world.urdf.xacro"),
            mappings={"hw_type": hw_type}
        )
        .robot_description_semantic(file_path=os.path.join(pkg_share, "config", "dual_arm_world.srdf"))
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
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

    # 3. Hardware Nodes & Logic
    ros2_controllers_path = os.path.join(pkg_share, "config", "ros2_controllers.yaml")

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, ros2_controllers_path],
        output="screen",
    )

    ft_sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('robotiq_ft_sensor_hardware'),
            'launch', 'ft_sensor_standalone.launch.py')),
        launch_arguments={
            'ftdi_id': ft_sensor_ftdi_id,
            'max_retries': ft_sensor_max_retries,
            'frame_id': ft_sensor_frame_id,
        }.items()
    )
    
    real_hw_process = ExecuteProcess(
        cmd=['ros2', 'run', 'dual_arm_moveit_config', 'real_hardware.py'],
        output='screen',
        on_exit=Shutdown()
    )

    # 4. Tools & Perception
    tool_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('tool_controller'),
            'launch', 'tool_launch.py'))
    )

    handeye_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('easy_handeye2'),
            'launch', 'publish.launch.py')),
        launch_arguments={
            'name': 'realsense_handeye',
            'calibration_file': '/home/adip/workspace/disassembly_ws/src/disassembly_skills/config/realsense_handeye.calib'
        }.items()
    )

    # 5. MoveGroup, RViz & Custom Nodes
    run_move_group_node = Node(
        package="moveit_ros_move_group", executable="move_group",
        parameters=[moveit_config.to_dict(), *sensors_3d_params, {"use_sim_time": False, "octomap_resolution": 0.05}],
    )

    run_rviz_node = Node(
        package="rviz2", executable="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[moveit_config.to_dict(), {"octomap_resolution": 0.05}],
    )
    
    state_manager = ExecuteProcess(cmd=['ros2', 'run', 'dual_arm_moveit_config', 'state_manager.py'], output='screen', on_exit=Shutdown())
    hold_state = ExecuteProcess(cmd=['ros2', 'run', 'dual_arm_moveit_config', 'object_hold_state.py'], output='screen', on_exit=Shutdown())
    
    # ==========================================
    # DUAL-ARM SERVO CONFIGURATION
    # ==========================================
    
    # Load xArm5 Servo Config
    xarm_servo_yaml = os.path.join(pkg_share, 'config', 'xarm_servo.yaml')
    with open(xarm_servo_yaml, 'r') as file:
        xarm_servo_params = yaml.safe_load(file)

    xarm_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="xarm_servo_node",
        parameters=[
            {"moveit_servo": xarm_servo_params},
            {"use_sim_time": False, "octomap_resolution": 0.05},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            *sensors_3d_params,
        ],
        output="screen",
    )

    # Load UF850 Servo Config
    uf_servo_yaml = os.path.join(pkg_share, 'config', 'uf_servo.yaml')
    with open(uf_servo_yaml, 'r') as file:
        uf_servo_params = yaml.safe_load(file)

    uf_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="uf_servo_node",
        parameters=[
            {"moveit_servo": uf_servo_params}, 
            {"use_sim_time": False, "octomap_resolution": 0.05},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            *sensors_3d_params,
        ],
        output="screen",
    )
    
    teleoperate = ExecuteProcess(cmd=['ros2', 'run', 'joy', 'joy_node'], output='screen', on_exit=Shutdown())
    
    teleop_bridge = ExecuteProcess(
        cmd=['ros2', 'run', 'dual_arm_moveit_config', 'teleop_bridge.py'], 
        output='screen', 
        on_exit=Shutdown()
    )

    # ==========================================
    # ROS 2 Controller Spawners
    # ==========================================
    jsb_spawner = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"])
    xarm_spawner = Node(package="controller_manager", executable="spawner", arguments=["xarm_controller", "--controller-manager", "/controller_manager"])
    uf_spawner = Node(package="controller_manager", executable="spawner", arguments=["uf_controller", "--controller-manager", "/controller_manager"])
    rg6_spawner = Node(package="controller_manager", executable="spawner", arguments=["rg6_controller", "--controller-manager", "/controller_manager"])
    slider_spawner = Node(package="controller_manager", executable="spawner", arguments=["slider_controller", "--controller-manager", "/controller_manager"])

    # ==========================================
    # 6. STRICT EXECUTION SEQUENCE (THE FIX)
    # ==========================================
    launch_sequence = [
        LogInfo(msg="🛰️  Step 1: Loading Robot State and Static TFs..."),
        static_tf,
        run_rsp_node,
    ]

    # Hardware-specific execution timing
    if hw_type == 'real':
        launch_sequence.append(
            TimerAction(period=2.0, actions=[
                LogInfo(msg="🔌 Step 2A: Booting Python Hardware API... (Waiting for Real Pose)"),
                real_hw_process,
                ft_sensor_launch
            ])
        )
        launch_sequence.append(
            TimerAction(period=8.0, actions=[
                LogInfo(msg="⚙️ Step 2B: Starting ROS 2 Control Node (Topics are now populated)..."),
                ros2_control_node
            ])
        )
    elif hw_type == 'fake':
        launch_sequence.append(
            TimerAction(period=2.0, actions=[
                LogInfo(msg="💻 Step 2: Launching Controller Manager (Mock Fake Hardware)..."),
                ros2_control_node
            ])
        )
    elif hw_type == 'isaac':
        # topic_based_ros2_control bridges ROS 2 control <-> Isaac Sim via topics:
        #   /isaac_joint_states  <- Isaac Sim publishes joint states
        #   /isaac_joint_commands -> Isaac Sim receives joint commands
        # ros2_control_node still runs on this machine; Isaac Sim is NOT the controller manager.
        launch_sequence.append(
            TimerAction(period=2.0, actions=[
                LogInfo(msg="🌌 Step 2: Starting ROS 2 Control Node (Isaac Sim bridge on /isaac_joint_states + /isaac_joint_commands)..."),
                ros2_control_node
            ])
        )

    # Remaining sequence starts safely AFTER Step 2B
    launch_sequence.extend([
        TimerAction(period=12.0, actions=[
            LogInfo(msg="📡 Step 3: Spawning Broadcaster & Controllers..."),
            jsb_spawner, xarm_spawner, uf_spawner, rg6_spawner, slider_spawner
        ]),

        TimerAction(period=15.0, actions=[
            LogInfo(msg="🧠 Step 4: Starting MoveGroup, Tools, & Perception..."),
            run_move_group_node, tool_controller, handeye_publisher
        ]),

        # TimerAction(period=18.0, actions=[
        #     LogInfo(msg="🕹️  Step 5: Starting Teleop & MoveIt Servo..."),
        #     teleoperate, teleop_bridge, xarm_servo_node, uf_servo_node
        # ]),
        
        TimerAction(period=18.0, actions=[
            LogInfo(msg="🕹️  Step 5: Starting Teleop & MoveIt Servo..."),
            xarm_servo_node, uf_servo_node
        ]),

        TimerAction(period=21.0, actions=[
            LogInfo(msg="✅ Step 6: Starting Managers & RViz..."),
            state_manager, hold_state, run_rviz_node,
            LogInfo(msg="🚀 SYSTEM READY. NO JUMPS ALLOWED.")
        ])
    ])

    return launch_sequence

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'hardware_type',
            default_value='fake',
            description='Select hardware type: "real", "fake", or "isaac"'),
        DeclareLaunchArgument(
            'ft_sensor_ftdi_id',
            default_value='',
            description='Optional FTDI device id for the Robotiq FT sensor'),
        DeclareLaunchArgument(
            'ft_sensor_max_retries',
            default_value='100',
            description='Number of stream retries before reinitializing the Robotiq FT sensor'),
        DeclareLaunchArgument(
            'ft_sensor_frame_id',
            default_value='robotiq_ft_frame_id',
            description='Frame id for published Robotiq FT wrench messages'),
        OpaqueFunction(function=launch_setup)
    ])
