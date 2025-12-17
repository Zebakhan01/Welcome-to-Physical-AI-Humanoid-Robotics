---
sidebar_position: 3
---

# Setup Scripts

## Introduction to Setup Scripts

Setup scripts automate the configuration and installation of development environments for Physical AI and humanoid robotics projects. These scripts handle system dependencies, environment variables, development tools, and initial project setup to ensure consistent and reproducible development environments across different systems and teams.

## System Prerequisites Script

### Ubuntu/Debian Setup Script

```bash
#!/bin/bash
# setup_ubuntu.sh - Ubuntu/Debian system setup for robotics development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Ubuntu/Debian Robotics Development Setup${NC}"

# Function to print status
print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential development tools
print_status "Installing essential development tools..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    iotop \
    sysstat \
    gnupg \
    lsb-release \
    software-properties-common \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel

# Install ROS Noetic (or appropriate ROS version)
print_status "Adding ROS repository..."
if [ ! -f /etc/apt/sources.list.d/ros-latest.list ]; then
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt update
fi

print_status "Installing ROS Noetic..."
sudo apt install -y ros-noetic-desktop-full

# Initialize rosdep
print_status "Initializing rosdep..."
sudo rosdep init || echo "rosdep already initialized"
rosdep update

# Install ROS dependencies
print_status "Installing ROS dependencies..."
sudo apt install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential

# Setup ROS environment
print_status "Setting up ROS environment..."
if ! grep -q "source /opt/ros/noetic/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
fi

# Install additional robotics libraries
print_status "Installing additional robotics libraries..."
sudo apt install -y \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    pcl-tools \
    libgflags-dev \
    libgoogle-glog-dev \
    libceres-dev \
    libyaml-cpp-dev \
    libjsoncpp-dev \
    libboost-all-dev \
    libyaml-dev \
    libxml2-dev \
    libxslt1-dev \
    libbz2-dev \
    libffi-dev \
    libssl-dev

# Install simulation tools
print_status "Installing simulation tools..."
sudo apt install -y \
    gazebo11 \
    libgazebo11-dev \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers

# Install Python packages
print_status "Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    jupyter \
    ipython \
    cython \
    scikit-learn \
    opencv-python \
    transforms3d \
    pyquaternion \
    sympy \
    control \
    python-rtmidi \
    pygame

# Install development tools
print_status "Installing development tools..."
sudo apt install -y \
    terminator \
    meld \
    geany \
    code || echo "VS Code not available, installing alternatives"

# Install Docker (optional)
read -p "Install Docker? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo systemctl enable docker
    rm get-docker.sh
fi

# Create workspace directory
print_status "Creating workspace..."
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws
catkin_make

# Setup environment variables
print_status "Setting up environment variables..."
if ! grep -q "source ~/robotics_ws/devel/setup.bash" ~/.bashrc; then
    echo "source ~/robotics_ws/devel/setup.bash" >> ~/.bashrc
fi

# Install additional tools
print_status "Installing additional tools..."
sudo apt install -y \
    ros-noetic-moveit \
    ros-noetic-navigation \
    ros-noetic-interactive-markers \
    ros-noetic-tf2-tools \
    ros-noetic-rviz \
    ros-noetic-xacro

# Install GPU tools (if NVIDIA GPU detected)
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing GPU tools..."
    sudo apt install -y \
        nvidia-driver-470 \
        nvidia-utils-470 \
        cuda-toolkit-11-4
fi

print_status "Ubuntu/Debian setup complete!"
echo -e "${GREEN}Setup complete! Please run:${NC}"
echo "  source ~/.bashrc"
echo "  cd ~/robotics_ws && source devel/setup.bash"
echo "  roscore # to test ROS installation"
```

### Python Environment Setup Script

```bash
#!/usr/bin/env python3
# setup_python_env.py - Python environment setup for robotics

import os
import sys
import subprocess
import venv
import shutil
from pathlib import Path

def run_command(cmd, description="Running command"):
    """Run a command and handle errors"""
    print(f"[INFO] {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                               capture_output=True, text=True)
        print("Success!")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_robotics_venv():
    """Create and configure Python virtual environment for robotics"""
    venv_path = Path.home() / "robotics_env"

    print("[INFO] Creating Python virtual environment...")

    # Create virtual environment
    venv.create(venv_path, with_pip=True)

    # Get the correct Python executable
    python_exe = venv_path / "bin" / "python"
    pip_exe = venv_path / "bin" / "pip"

    # Upgrade pip
    run_command(f"{pip_exe} install --upgrade pip",
                "Upgrading pip")

    # Install core robotics packages
    robotics_packages = [
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "opencv-python>=4.5.0",
        "transforms3d>=0.3.1",
        "pyquaternion>=0.9.9",
        "sympy>=1.7.0",
        "control>=0.8.3",
        "pygame>=2.0.0",
        "pyserial>=3.5",
        "keyboard>=0.13.5",
        "inputs>=0.5.0"
    ]

    print("[INFO] Installing core robotics packages...")
    for package in robotics_packages:
        run_command(f"{pip_exe} install {package}",
                   f"Installing {package}")

    # Install ROS Python interfaces
    print("[INFO] Installing ROS Python interfaces...")
    run_command(f"{pip_exe} install roslibpy rospy_message_converter",
                "Installing ROS Python packages")

    # Install machine learning packages
    ml_packages = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "tensorflow>=2.5.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.2.0",
        "h5py>=3.1.0"
    ]

    print("[INFO] Installing machine learning packages...")
    for package in ml_packages:
        run_command(f"{pip_exe} install {package}",
                   f"Installing ML package: {package}")

    # Install development tools
    dev_packages = [
        "pytest>=6.2.0",
        "pytest-cov>=2.11.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.812",
        "jupyter>=1.0.0",
        "ipython>=7.20.0",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "pdoc3>=0.10.0"
    ]

    print("[INFO] Installing development tools...")
    for package in dev_packages:
        run_command(f"{pip_exe} install {package}",
                   f"Installing dev tool: {package}")

    # Create requirements file
    print("[INFO] Creating requirements file...")
    requirements_path = venv_path / "requirements.txt"
    result = subprocess.run([str(pip_exe), "freeze"],
                           capture_output=True, text=True)
    if result.returncode == 0:
        with open(requirements_path, 'w') as f:
            f.write(result.stdout)

    print(f"[INFO] Virtual environment created at: {venv_path}")
    print("[INFO] To activate: source ~/robotics_env/bin/activate")

def setup_jupyter_notebooks():
    """Setup Jupyter for robotics development"""
    venv_path = Path.home() / "robotics_env"
    pip_exe = venv_path / "bin" / "pip"

    print("[INFO] Setting up Jupyter for robotics...")

    # Install Jupyter extensions
    jupyter_extensions = [
        "jupyter_contrib_nbextensions",
        "jupyter_nbextensions_configurator",
        "ipywidgets",
        "plotly",
        "dash"
    ]

    for ext in jupyter_extensions:
        run_command(f"{pip_exe} install {ext}",
                   f"Installing Jupyter extension: {ext}")

    # Install Jupyter kernel
    run_command(f"{pip_exe} install ipykernel",
               "Installing IPython kernel")

    # Install robotics-specific Jupyter extensions
    run_command(f"{pip_exe} install --upgrade nbconvert",
               "Installing nbconvert")

    print("[INFO] Jupyter setup complete!")

def main():
    """Main setup function"""
    print("Python Environment Setup for Robotics Development")
    print("=" * 50)

    # Create virtual environment
    create_robotics_venv()

    # Setup Jupyter
    setup_jupyter_notebooks()

    print("=" * 50)
    print("Python environment setup complete!")
    print("To activate: source ~/robotics_env/bin/activate")
    print("To start Jupyter: jupyter notebook")

if __name__ == "__main__":
    main()
```

## ROS Workspace Setup Script

### Workspace Creation Script

```bash
#!/bin/bash
# setup_ros_workspace.sh - Create and configure ROS workspace

set -e

# Configuration
WORKSPACE_NAME="robotics_ws"
ROS_DISTRO="noetic"  # Change as needed

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if ROS is installed
if ! command -v rosversion &> /dev/null; then
    print_error "ROS is not installed. Please install ROS first."
    exit 1
fi

# Check ROS version
ROS_VERSION=$(rosversion -d)
if [ "$ROS_VERSION" != "$ROS_DISTRO" ]; then
    print_warning "Expected ROS $ROS_DISTRO, but found $ROS_VERSION"
fi

# Create workspace
print_status "Creating ROS workspace: $WORKSPACE_NAME"
HOME_DIR="$HOME/$WORKSPACE_NAME"

if [ -d "$HOME_DIR" ]; then
    print_error "Workspace already exists: $HOME_DIR"
    read -p "Remove existing workspace? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$HOME_DIR"
        mkdir -p "$HOME_DIR/src"
    else
        print_error "Exiting without creating workspace"
        exit 1
    fi
else
    mkdir -p "$HOME_DIR/src"
fi

cd "$HOME_DIR"

# Initialize workspace
print_status "Initializing workspace..."
catkin_make

# Source workspace in bashrc if not already done
if ! grep -q "source $HOME_DIR/devel/setup.bash" ~/.bashrc; then
    echo "source $HOME_DIR/devel/setup.bash" >> ~/.bashrc
    print_status "Added workspace to ~/.bashrc"
fi

# Create standard package structure
print_status "Creating standard package structure..."

# Create common robotics packages
mkdir -p src/{common_msgs,hardware_drivers,sensors,control,planning,navigation,visualization,utilities}

# Create a basic robot package template
cat > src/robot_bringup/package.xml << EOF
<?xml version="1.0"?>
<package format="2">
  <name>robot_bringup</name>
  <version>0.0.1</version>
  <description>Robot bringup and configuration package</description>
  <maintainer email="developer@robotics.com">Developer</maintainer>
  <license>MIT</license>

  <buildtool_depend>catkin</buildtool_depend>
  <build_depend>roscpp</build_depend>
  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_export_depend>roscpp</build_export_depend>
  <build_export_depend>rospy</build_export_depend>
  <build_export_depend>std_msgs</build_export_depend>
  <exec_depend>roscpp</exec_depend>
  <exec_depend>rospy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
</package>
EOF

# Create CMakeLists.txt for robot_bringup
cat > src/robot_bringup/CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.0.2)
project(robot_bringup)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories(
  \${catkin_INCLUDE_DIRS}
)
EOF

# Create launch directory structure
mkdir -p src/robot_bringup/launch
mkdir -p src/robot_bringup/config
mkdir -p src/robot_bringup/urdf

# Create a basic launch file
cat > src/robot_bringup/launch/robot.launch << EOF
<launch>
  <!-- Robot description -->
  <param name="robot_description" command="\$(find xacro)/xacro \$(find robot_bringup)/urdf/robot.xacro" />

  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />

  <!-- Joint state publisher -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" />

  <!-- Robot bringup node -->
  <node name="robot_bringup_node" pkg="robot_bringup" type="robot_bringup_node" output="screen" />
</launch>
EOF

# Create URDF directory structure
mkdir -p src/robot_bringup/urdf
mkdir -p src/robot_bringup/meshes

# Create a basic URDF template
cat > src/robot_bringup/urdf/robot.xacro << EOF
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot">
  <!-- Include other xacro files -->
  <xacro:include filename="\$(find robot_bringup)/urdf/materials.xacro" />
  <xacro:include filename="\$(find robot_bringup)/urdf/transmissions.xacro" />

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Example joint and link -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
  </link>
</robot>
EOF

# Create materials file
cat > src/robot_bringup/urdf/materials.xacro << EOF
<?xml version="1.0"?>
<robot>
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="red">
    <color rgba="0.8 0 0 1"/>
  </material>
</robot>
EOF

print_success "ROS workspace created successfully!"
print_status "Workspace location: $HOME_DIR"
print_status "To build: cd $HOME_DIR && catkin_make"
print_status "To source: source $HOME_DIR/devel/setup.bash"

# Build the workspace
print_status "Building workspace..."
cd "$HOME_DIR"
catkin_make

print_success "Workspace build complete!"
```

## Simulation Environment Setup

### Gazebo Setup Script

```bash
#!/bin/bash
# setup_gazebo.sh - Setup Gazebo simulation environment

set -e

# Configuration
GAZEBO_VERSION="11"  # Change as needed
ROS_DISTRO="noetic"  # Change as needed

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system compatibility
print_status "Checking system compatibility..."

# Check for Ubuntu/Debian
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        print_error "This script is designed for Ubuntu/Debian systems"
        exit 1
    fi
else
    print_error "Unable to determine OS"
    exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Gazebo
print_status "Installing Gazebo $GAZEBO_VERSION..."
sudo apt install -y \
    gazebo$GAZEBO_VERSION \
    libgazebo$GAZEBO_VERSION-dev

# Install Gazebo ROS plugins
print_status "Installing Gazebo ROS plugins..."
sudo apt install -y \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-gazebo-ros-control \
    ros-$ROS_DISTRO-ros-control \
    ros-$ROS_DISTRO-ros-controllers

# Install additional Gazebo tools
print_status "Installing Gazebo tools..."
sudo apt install -y \
    gazebo$GAZEBO_VERSION-plugins \
    gazebo$GAZEBO_VERSION-utils

# Setup Gazebo environment
print_status "Setting up Gazebo environment..."

# Add to bashrc if not already present
if ! grep -q "GAZEBO_MODEL_PATH" ~/.bashrc; then
    echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models:/usr/share/gazebo-'"$GAZEBO_VERSION"'/models' >> ~/.bashrc
fi

if ! grep -q "GAZEBO_RESOURCE_PATH" ~/.bashrc; then
    echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo:/usr/share/gazebo-'"$GAZEBO_VERSION" >> ~/.bashrc
fi

if ! grep -q "GAZEBO_PLUGIN_PATH" ~/.bashrc; then
    echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-'"$GAZEBO_VERSION"'/plugins' >> ~/.bashrc
fi

# Create Gazebo configuration directory
mkdir -p ~/.gazebo/models ~/.gazebo/worlds

# Install common Gazebo models
print_status "Installing common Gazebo models..."
sudo apt install -y \
    gazebo$GAZEBO_VERSION-common \
    gazebo$GAZEBO_VERSION-data

# Setup ROS Gazebo integration
print_status "Setting up ROS Gazebo integration..."

# Create a Gazebo robot package template
ROBOT_WS="$HOME/robotics_ws"
if [ -d "$ROBOT_WS" ]; then
    cd "$ROBOT_WS/src"

    # Create Gazebo robot package
    catkin_create_pkg robot_gazebo std_msgs rospy roscpp \
        gazebo_ros_control \
        controller_manager \
        joint_state_controller \
        robot_state_publisher \
        xacro

    cd robot_gazebo

    # Create launch directory
    mkdir -p launch worlds config

    # Create Gazebo launch file
    cat > launch/robot_gazebo.launch << EOF
<launch>
  <!-- these are the arguments you can pass this launch file, for example paused:=true -->
  <arg name="paused" default="false"/>
  <arg name="use_sim_time" default="true"/>
  <arg name="gui" default="true"/>
  <arg name="headless" default="false"/>
  <arg name="debug" default="false"/>

  <!-- We resume the logic in empty_world.launch, changing only the name of the world to be launched -->
  <include file="\$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="\$(find robot_gazebo)/worlds/empty.world"/>
    <arg name="debug" value="\$(arg debug)" />
    <arg name="gui" value="\$(arg gui)" />
    <arg name="paused" value="\$(arg paused)"/>
    <arg name="use_sim_time" value="\$(arg use_sim_time)"/>
    <arg name="headless" value="\$(arg headless)"/>
  </include>

  <!-- Load the URDF into the ROS Parameter Server -->
  <param name="robot_description" command="\$(find xacro)/xacro \$(find robot_gazebo)/urdf/robot.xacro" />

  <!-- Run a python script to the send a service call to gazebo_ros to spawn a URDF robot -->
  <node name="urdf_spawner" pkg="gazebo_ros" type="spawn_model" respawn="false" output="screen"
    args="-param robot_description -urdf -model robot -x 0 -y 0 -z 1"/>

  <!-- ros_control robot hardware interface -->
  <include file="\$(find robot_gazebo)/launch/controller_utils.launch"/>

  <!-- Load joint controller configurations from YAML file to parameter server -->
  <rosparam file="\$(find robot_gazebo)/config/controllers.yaml" command="load"/>

  <!-- Load the controllers -->
  <node name="controller_spawner" pkg="controller_manager" type="spawner" respawn="false"
    output="screen" args="joint_state_controller
                         position_controllers
                         velocity_controllers
                         effort_controllers"/>
</launch>
EOF

    # Create controller launch file
    cat > launch/controller_utils.launch << EOF
<launch>
  <!-- Load joint controller configurations from YAML file to parameter server -->
  <rosparam file="\$(find robot_gazebo)/config/controllers.yaml" command="load"/>

  <!-- Load the controllers -->
  <node name="controller_spawner" pkg="controller_manager" type="spawner" respawn="false"
    output="screen" args="joint_state_controller
                         position_controllers
                         velocity_controllers
                         effort_controllers"/>
</launch>
EOF

    # Create controllers configuration
    mkdir -p config
    cat > config/controllers.yaml << EOF
# Publish all joint states
joint_state_controller:
  type: joint_state_controller/JointStateController
  publish_rate: 50

# Position controllers
position_controllers:
  type: position_controllers/JointGroupPositionController
  joints:
    - joint1
    - joint2
    - joint3

# Velocity controllers
velocity_controllers:
  type: velocity_controllers/JointGroupVelocityController
  joints:
    - joint1
    - joint2
    - joint3

# Effort controllers
effort_controllers:
  type: effort_controllers/JointGroupEffortController
  joints:
    - joint1
    - joint2
    - joint3
EOF

    print_success "Gazebo robot package created!"
else
    print_status "Robot workspace not found, skipping Gazebo package creation"
fi

# Test Gazebo installation
print_status "Testing Gazebo installation..."
if command -v gazebo &> /dev/null; then
    print_success "Gazebo installation verified!"
    print_status "To test: gazebo --verbose"
else
    print_error "Gazebo installation failed!"
    exit 1
fi

print_success "Gazebo setup complete!"
echo -e "${GREEN}Next steps:${NC}"
echo "  source ~/.bashrc"
echo "  gazebo  # Test Gazebo"
echo "  roslaunch robot_gazebo robot_gazebo.launch  # Test ROS integration"
```

## Development Environment Configuration

### VS Code Configuration Script

```bash
#!/bin/bash
# setup_vscode.sh - Setup VS Code for robotics development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if VS Code is installed
if ! command -v code &> /dev/null; then
    print_status "VS Code not found, installing..."
    sudo snap install --classic code
fi

# Install VS Code extensions for robotics development
print_status "Installing VS Code extensions..."

EXTENSIONS=(
    "ms-python.python"                    # Python support
    "ms-python.vscode-pylance"            # Python IntelliSense
    "ms-toolsai.jupyter"                  # Jupyter notebooks
    "ms-iot.vscode-ros"                   # ROS support
    "ms-iot.vscode-ros-debug"             # ROS debugging
    "ms-vscode.cpptools"                  # C++ support
    "twxs.cmake"                          # CMake support
    "ms-vscode.cmake-tools"               # CMake tools
    "matepek.vscode-catch2-test-adapter"  # C++ testing
    "ms-azuretools.vscode-docker"         # Docker support
    "ms-kubernetes-tools.vscode-kubernetes-tools"  # Kubernetes
    "shardulm94.trailing-spaces"          # Show trailing spaces
    "ms-vscode.makefile-tools"            # Makefile support
    "formulahendry.auto-rename-tag"       # HTML/XML auto rename
    "christian-kohler.path-intellisense"  # Path autocompletion
    "ms-python.flake8"                    # Python linting
    "ms-python.black-formatter"           # Python formatting
    "ms-python.mypy-type-checker"         # MyPy integration
)

for extension in "${EXTENSIONS[@]}"; do
    print_status "Installing $extension..."
    code --install-extension "$extension" || echo "Failed to install $extension"
done

# Create VS Code settings directory
SETTINGS_DIR="$HOME/.config/Code/User"
mkdir -p "$SETTINGS_DIR"

# Create settings.json for robotics development
cat > "$SETTINGS_DIR/settings.json" << EOF
{
    "python.defaultInterpreterPath": "~/robotics_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true,
    "terminal.integrated.shell.linux": "/bin/bash",
    "terminal.integrated.env.linux": {
        "BASH_ENV": "/home/\${env:USER}/.bashrc"
    },
    "ros.distro": "noetic",
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug"
    ],
    "C_Cpp.clang_format_style": "{ BasedOnStyle: Google, ColumnLimit: 120 }",
    "workbench.colorTheme": "Default Dark+",
    "explorer.confirmDelete": false,
    "explorer.confirmDragAndDrop": false,
    "extensions.ignoreRecommendations": false,
    "python.terminal.activateEnvironment": true
}
EOF

# Create recommended extensions file
cat > "$SETTINGS_DIR/recommended_extensions.json" << EOF
{
    "recommendations": [
        "ms-python.python",
        "ms-iot.vscode-ros",
        "ms-vscode.cpptools",
        "twxs.cmake",
        "ms-azuretools.vscode-docker"
    ]
}
EOF

# Create workspace settings template
mkdir -p "$HOME/robotics_ws/.vscode"
cat > "$HOME/robotics_ws/.vscode/settings.json" << EOF
{
    "python.defaultInterpreterPath": "~/robotics_env/bin/python",
    "python.analysis.extraPaths": [
        "~/robotics_ws/devel/lib/python3/dist-packages",
        "/opt/ros/noetic/lib/python3/dist-packages"
    ],
    "cmake.sourceDirectory": "\${workspaceFolder}/src",
    "cmake.buildDirectory": "\${workspaceFolder}/build",
    "terminal.integrated.cwd": "\${workspaceFolder}",
    "terminal.integrated.env.linux": {
        "ROS_PACKAGE_PATH": "\${workspaceFolder}/src:\${env:ROS_PACKAGE_PATH}",
        "CMAKE_PREFIX_PATH": "\${workspaceFolder}/devel:\${env:CMAKE_PREFIX_PATH}",
        "PKG_CONFIG_PATH": "\${workspaceFolder}/devel/lib/pkgconfig:\${env:PKG_CONFIG_PATH}",
        "LD_LIBRARY_PATH": "\${workspaceFolder}/devel/lib:\${env:LD_LIBRARY_PATH}",
        "PYTHONPATH": "\${workspaceFolder}/devel/lib/python3/dist-packages:\${env:PYTHONPATH}",
        "PATH": "\${workspaceFolder}/devel/bin:\${env:PATH}"
    }
}
EOF

# Create launch configuration for debugging
cat > "$HOME/robotics_ws/.vscode/launch.json" << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: ROS Node",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/src/package_name/nodes/node_name.py",
            "console": "integratedTerminal",
            "python": "~/robotics_env/bin/python",
            "env": {
                "ROS_PACKAGE_PATH": "\${workspaceFolder}/src:\${env:ROS_PACKAGE_PATH}",
                "PYTHONPATH": "\${workspaceFolder}/devel/lib/python3/dist-packages:\${env:PYTHONPATH}"
            }
        },
        {
            "name": "C++: ROS Node",
            "type": "cppdbg",
            "request": "launch",
            "program": "\${workspaceFolder}/devel/lib/package_name/node_name",
            "args": [],
            "stopAtEntry": false,
            "cwd": "\${workspaceFolder}",
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "CMake: build"
        }
    ]
}
EOF

print_success "VS Code setup complete!"
print_status "Extensions installed and settings configured."
print_status "Open VS Code and open your robotics workspace for full functionality."
```

## Week Summary

This section provided comprehensive setup scripts for Physical AI and humanoid robotics development environments. The scripts cover system prerequisites, Python environment setup, ROS workspace creation, simulation environment configuration, and development tool integration. These automated setup procedures ensure consistent and reproducible development environments across different systems and teams, reducing setup time and configuration errors.