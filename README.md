# Disassembly Workspace

## Overview
This repository contains the source code for the Disassembly Automation project, integrated with ROS 2. It combines robotic manipulation, computer vision, and specialized tooling to perform autonomous disassembly tasks.

## Repository Structure

### Core Packages
- **`disassembly_skills`**: Implements high-level skills and logic for disassembly operations.
- **`vision_agent`**: Autonomous agents for visual perception and decision making.
- **`tool_controller`**: Controllers and interfaces for the custom disassembly tool.
- **`tool_camera_pkg`**: Drivers and processing for the tool-mounted camera.
- **`scene_description`**: URDFs, meshes, and configuration for the workcell environment.

### Robot Descriptions
- **`robots`**: Contains robot descriptions (URDF/Xacro) for:
  - `uf850` (Universal Robots / Factory Automation arm)
  - `xarm5` (UFACTORY xArm 5)
  - `so-arm`
  - `screwdriver_description`
  - `rg6_description` (OnRobot RG6 Gripper)

### Machine Learning & Vision
- **`vision_training`**: Training scripts, datasets, and model checkpoints for vision tasks.
  - *Note: Large data directories (`realsense_data`, `record_data`) are git-ignored.*

### Drivers & Hardware Integration
- **`rq_fts_ros2_driver`**: Driver for Robotiq Force Torque Sensors.
- **`bio_ik`**: BioIK inverse kinematics solver for ROS 2.
- **`camera_calibaration`**: Hand-eye calibration tools (Easy HandEye, Aruco ROS).
- **`moveit_2`**: Integrated MoveIt 2 framework for motion planning.

## Getting Started

### Prerequisites
- **ROS 2** (Humble Hawksbill or Rolling Ridley recommended)
- **Colcon** build tool

### Installation

1. **Clone the repository:**
   ```bash
   # If you haven't already
   git clone <repository_url> src
   ```

2. **Install Dependencies:**
   Navigate to the workspace root (one level up from `src`) and run:
   ```bash
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build the Workspace:**
   ```bash
   colcon build --symlink-install
   ```

4. **Source the Workspace:**
   ```bash
   source install/setup.bash
   ```

## Usage

### Running the Vision Agent
```bash
ros2 launch vision_agent vision_agent.launch.py
```

### Launching the Cell
```bash
ros2 launch scene_description view_scene.launch.py
```

*(Add specific launch commands here as the project evolves)*

## License
[Insert License Information Here]
