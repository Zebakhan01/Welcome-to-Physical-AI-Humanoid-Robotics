---
sidebar_position: 4
---

# Tools and Resources

## Development Environment Setup

### Required Software Stack

#### Operating System and Development Environment

**Primary Development Environment:**
- **Ubuntu 20.04 LTS or 22.04 LTS**: Recommended for full ROS 2 compatibility
- **WSL2 (Windows Subsystem for Linux)**: For Windows users requiring native Linux experience
- **Docker**: For containerized development and deployment
- **Git**: Version control system for code management

**Installation Commands:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
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

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
rm get-docker.sh
```

#### ROS 2 Installation

**ROS 2 Humble Hawksbill (Recommended):**
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
sudo apt install -y ros-humble-rosbridge-suite
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Simulation Environments

#### Isaac Sim Installation

**NVIDIA Isaac Sim Requirements:**
- NVIDIA GPU with CUDA support (RTX series recommended)
- NVIDIA GPU drivers (latest recommended)
- CUDA Toolkit 11.8 or later
- Isaac Sim from NVIDIA Developer Portal

**Installation Steps:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt -y install cuda-toolkit-11-8

# Download and install Isaac Sim from NVIDIA Developer Portal
# Follow the official Isaac Sim installation guide
```

#### Gazebo Installation

**Gazebo Garden (Latest Stable):**
```bash
# Add Gazebo repository
sudo curl -sSL https://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gz-garden
```

### Development Tools

#### IDE and Editors

**Visual Studio Code with Robotics Extensions:**
```bash
# Install VS Code
wget -qO - https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo install -o root -g root -m 644 microsoft.gpg /usr/share/keyrings/microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'

sudo apt update
sudo apt install -y code

# Install essential extensions
code --install-extension ms-python.python
code --install-extension ms-iot.vscode-ros
code --install-extension ms-iot.vscode-ros-debug
code --install-extension ms-vscode.cpptools
code --install-extension twxs.cmake
code --install-extension ms-vscode.cmake-tools
code --install-extension ms-azuretools.vscode-docker
code --install-extension formulahendry.auto-rename-tag
code --install-extension ms-python.flake8
code --install-extension ms-python.black-formatter
```

#### Version Control and Collaboration

**Git Configuration for Robotics:**
```bash
# Configure Git for robotics development
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set up Git aliases for common operations
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.lg "log --oneline --graph --decorate --all"

# Configure Git for large files (important for robotics data)
git lfs install
```

### Python Environment Management

#### Virtual Environment Setup

```bash
# Create and configure Python virtual environment
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Install core robotics packages
pip install --upgrade pip
pip install setuptools wheel

# Install essential robotics libraries
pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    pandas>=1.3.0 \
    opencv-python>=4.5.0 \
    transforms3d>=0.4.0 \
    pyquaternion>=0.9.9 \
    sympy>=1.9.0 \
    control>=0.9.0 \
    pygame>=2.1.0 \
    pyserial>=3.5 \
    roslibpy>=1.3.0 \
    rospy-message-converter>=0.5.0

# Install AI/ML packages
pip install \
    torch>=1.12.0 \
    torchvision>=0.13.0 \
    tensorflow>=2.9.0 \
    scikit-learn>=1.1.0 \
    pillow>=9.0.0 \
    h5py>=3.7.0 \
    tensorboard>=2.9.0

# Install development tools
pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=22.0.0 \
    flake8>=5.0.0 \
    mypy>=0.971 \
    jupyter>=1.0.0 \
    sphinx>=5.0.0 \
    sphinx-rtd-theme>=1.0.0
```

#### Conda Environment (Alternative)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Create robotics environment
conda create -n robotics python=3.10
conda activate robotics

# Install packages via conda/mamba
conda install -c conda-forge \
    numpy scipy matplotlib pandas \
    opencv transforms3d pyquaternion \
    jupyter notebook spyder

# Install additional packages via pip
pip install \
    torch torchvision tensorflow \
    roslibpy rospy-message-converter \
    control scikit-learn
```

## Hardware Resources

### Recommended Hardware Specifications

#### Development Workstation

**Minimum Requirements:**
- CPU: Intel i7 or AMD Ryzen 7 (8+ cores)
- RAM: 32GB DDR4
- GPU: NVIDIA RTX 3070 or equivalent
- Storage: 1TB NVMe SSD
- Network: Gigabit Ethernet

**Recommended Configuration:**
- CPU: Intel i9 or AMD Ryzen 9 (16+ cores)
- RAM: 64GB DDR4-3200
- GPU: NVIDIA RTX 4080/4090 or RTX A5000/A6000
- Storage: 2TB+ NVMe SSD + 4TB HDD for datasets
- Network: 10GbE for high-bandwidth applications

#### Robot Hardware Platforms

**Simulation-Only Development:**
- Any modern PC meeting minimum requirements
- Focus on GPU for simulation acceleration
- Multiple monitors recommended for development

**Physical Robot Integration:**
- Real-time capable computer (Intel NUC, Jetson AGX, etc.)
- Real-time operating system (PREEMPT_RT Linux recommended)
- High-bandwidth network connection to robot
- Emergency stop interface capability

### Sensor Integration Tools

#### Camera Systems

**Recommended Camera Specifications:**
- RGB-D cameras (Intel RealSense, Intel Realsense D435/D455)
- High-resolution (640x480 or higher)
- Good low-light performance
- Accurate depth sensing (10cm - 10m range)

**Camera Calibration Tools:**
```bash
# Install camera calibration packages
sudo apt install ros-humble-camera-calibration
sudo apt install ros-humble-image-geometry
sudo apt install ros-humble-vision-opencv
```

#### LIDAR Systems

**Compatible LIDAR Sensors:**
- 2D LIDAR: Hokuyo UTM-30LX, Sick TIM551, Velodyne Puck
- 3D LIDAR: Velodyne VLP-16, Ouster OS1, Livox Horizon/Mid-40
- ROS drivers available for most common models

#### IMU and Inertial Sensors

**Recommended IMU Specifications:**
- 9-axis (gyro + accelerometer + magnetometer)
- High update rate (200Hz+)
- Low noise and drift characteristics
- ROS integration support

## Software Development Resources

### Package Managers and Dependencies

#### APT Package Management for Robotics

```bash
# Install robotics-specific packages
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros-control \
    ros-humble-ros-control \
    ros-humble-ros-controllers \
    ros-humble-joint-state-controller \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-teleop-twist-keyboard \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-moveit \
    ros-humble-moveit-ros \
    ros-humble-moveit-ros-planners \
    ros-humble-moveit-ros-visualization \
    ros-humble-rosbridge-server \
    ros-humble-web-video-server \
    ros-humble-interactive-markers \
    ros-humble-tf2-tools \
    ros-humble-rviz \
    ros-humble-xacro \
    ros-humble-urdf-tutorial \
    ros-humble-urdfdom-py \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-depthimage-to-laserscan \
    ros-humble-laser-filters \
    ros-humble-robot-self-filter \
    ros-humble-costmap-2d \
    ros-humble-amcl \
    ros-humble-map-server
```

#### Pip Package Management

```bash
# Create requirements file for robotics projects
cat > robotics_requirements.txt << EOF
# Core robotics libraries
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0

# Computer vision
opencv-python>=4.5.0
Pillow>=9.0.0
transforms3d>=0.4.0
pyquaternion>=0.9.9

# AI and Machine Learning
torch>=1.12.0
torchvision>=0.13.0
tensorflow>=2.9.0
scikit-learn>=1.1.0
tensorboard>=2.9.0

# ROS integration
roslibpy>=1.3.0
rospy-message-converter>=0.5.0

# Robotics specific
control>=0.9.0
pybullet>=3.2.5
gym>=0.21.0
stable-baselines3>=1.7.0

# Development tools
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.971
jupyter>=1.0.0
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0

# Utilities
pyserial>=3.5
keyboard>=0.13.5
inputs>=0.5.0
psutil>=5.9.0
pyyaml>=6.0
requests>=2.28.0
EOF

pip install -r robotics_requirements.txt
```

### Containerization with Docker

#### Docker for Robotics Development

```dockerfile
# Dockerfile for robotics development environment
FROM osrf/ros:humble-desktop

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV COLCON_WS=/opt/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    gnupg \
    lsb-release \
    software-properties-common \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    && rm -rf /var/lib/apt/lists/*

# Install additional robotics tools
RUN apt-get update && apt-get install -y \
    gazebo \
    libgazebo-dev \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros-control \
    ros-humble-ros-control \
    ros-humble-ros-controllers \
    ros-humble-navigation2 \
    ros-humble-moveit \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS workspace
RUN mkdir -p ${COLCON_WS}/src
WORKDIR ${COLCON_WS}

# Install Python packages
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Install additional Python packages for robotics
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    transforms3d \
    pyquaternion \
    torch \
    torchvision \
    tensorflow \
    roslibpy \
    rospy-message-converter \
    control \
    pygame

# Setup entrypoint
RUN echo '#!/bin/bash\n\
source "/opt/ros/${ROS_DISTRO}/setup.bash"\n\
source "${COLCON_WS}/install/setup.bash" 2>/dev/null || true\n\
exec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]

# requirements.txt for Docker
RUN echo "numpy>=1.21.0" > requirements.txt && \
    echo "scipy>=1.7.0" >> requirements.txt && \
    echo "matplotlib>=3.4.0" >> requirements.txt && \
    echo "opencv-python>=4.5.0" >> requirements.txt && \
    echo "transforms3d>=0.4.0" >> requirements.txt && \
    echo "pyquaternion>=0.9.9" >> requirements.txt && \
    echo "torch>=1.12.0" >> requirements.txt && \
    echo "torchvision>=0.13.0" >> requirements.txt && \
    echo "tensorflow>=2.9.0" >> requirements.txt && \
    echo "roslibpy>=1.3.0" >> requirements.txt && \
    echo "rospy-message-converter>=0.5.0" >> requirements.txt && \
    echo "control>=0.9.0" >> requirements.txt && \
    echo "pygame>=2.1.0" >> requirements.txt && \
    echo "pyserial>=3.5" >> requirements.txt && \
    echo "pytest>=7.0.0" >> requirements.txt && \
    echo "black>=22.0.0" >> requirements.txt && \
    echo "flake8>=5.0.0" >> requirements.txt
```

#### Docker Compose for Multi-Container Robotics Systems

```yaml
# docker-compose.yml for robotics development
version: '3.8'

services:
  ros-master:
    image: osrf/ros:humble-desktop
    container_name: ros-master
    command: roscore
    networks:
      - robotics-net
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1

  gazebo:
    image: osrf/gazebo:gzserver-harmonic
    container_name: gazebo-server
    depends_on:
      - ros-master
    networks:
      - robotics-net
    environment:
      - ROS_DOMAIN_ID=0
      - ROS_MASTER_URI=http://ros-master:11311
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.x11-unix:rw
      - ./worlds:/root/.gazebo/worlds:rw
    devices:
      - /dev/dri:/dev/dri
    ports:
      - "11345:11345/udp"
    command: gzserver --verbose /root/.gazebo/worlds/empty.sdf

  robot-controller:
    build: .
    container_name: robot-controller
    depends_on:
      - ros-master
    networks:
      - robotics-net
    environment:
      - ROS_DOMAIN_ID=0
      - ROS_MASTER_URI=http://ros-master:11311
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./src:/opt/workspace/src:rw
      - ./config:/opt/workspace/config:rw
    devices:
      - /dev:/dev
    privileged: true
    command: >
      bash -c "
        source /opt/ros/humble/setup.bash &&
        cd /opt/workspace &&
        colcon build &&
        source install/setup.bash &&
        ros2 run robot_control robot_node
      "

  visualization:
    image: osrf/ros:humble-desktop
    container_name: rviz-visualization
    depends_on:
      - ros-master
      - robot-controller
    networks:
      - robotics-net
    environment:
      - ROS_DOMAIN_ID=0
      - ROS_MASTER_URI=http://ros-master:11311
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    devices:
      - /dev/dri:/dev/dri
    command: >
      bash -c "
        source /opt/ros/humble/setup.bash &&
        rviz2
      "

networks:
  robotics-net:
    driver: bridge
```

## Cloud and Remote Development

### Remote Development Setup

#### SSH Configuration for Robotics Development

```bash
# SSH configuration for robotics development
cat >> ~/.ssh/config << EOF
Host robotics-dev
    HostName your-robotics-server.com
    User robotics
    Port 22
    IdentityFile ~/.ssh/robotics_key
    ForwardAgent yes
    ForwardX11 yes
    ForwardX11Trusted yes
    LocalForward 11311 localhost:11311
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host simulation-server
    HostName simulation-server.company.com
    User simuser
    Port 22
    IdentityFile ~/.ssh/simulation_key
    ForwardAgent yes
    Compression yes
    ServerAliveInterval 60
    RequestTTY yes
EOF

chmod 600 ~/.ssh/config
```

#### VS Code Remote Development

```json
// .vscode/settings.json for remote robotics development
{
    "python.defaultInterpreterPath": "/home/robotics/robotics_env/bin/python3",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/node_modules": true,
        "**/*.code-search": true,
        "**/build": true,
        "**/devel": true,
        "**/.ros": true
    },
    "terminal.integrated.env.linux": {
        "ROS_DISTRO": "humble",
        "ROS_DOMAIN_ID": "0",
        "ROS_MASTER_URI": "http://localhost:11311"
    },
    "remote.SSH.showLoginTerminal": true,
    "remote.SSH.useLocalServer": false
}
```

### Cloud Robotics Platforms

#### AWS RoboMaker Setup

```bash
# Install AWS CLI and configure for RoboMaker
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure

# Install RoboMaker dependencies
pip3 install boto3
sudo apt install ros-humble-aws-robomaker-small-heap-monitoring
```

#### Google Cloud Robotics

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install Cloud Robotics dependencies
gcloud components install cloud-robotics
pip3 install google-cloud-robotics
```

## Simulation-Specific Tools

### Isaac Sim Tools and Extensions

#### Isaac Sim Extensions for Robotics

```python
# Example: Isaac Sim extension for robotics development
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.kit.commands

class RoboticsExtension:
    """
    Isaac Sim extension for robotics development workflows
    """

    def __init__(self):
        self.world = None
        self.assets_root = get_assets_root_path()
        self.robots = {}
        self.scenes = {}

    def initialize_world(self):
        """Initialize Isaac Sim world for robotics"""
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        print("Isaac Sim world initialized for robotics")

    def load_robot(self, robot_name, urdf_path, position=(0, 0, 0)):
        """Load a robot into the simulation"""
        if self.world is None:
            self.initialize_world()

        # Add robot to stage
        robot_path = f"/World/{robot_name}"
        add_reference_to_stage(
            usd_path=urdf_path,
            prim_path=robot_path
        )

        # Create robot object
        robot = Robot(
            prim_path=robot_path,
            name=robot_name,
            position=position
        )

        # Add to world
        self.world.scene.add(robot)
        self.robots[robot_name] = robot

        print(f"Robot {robot_name} loaded at {position}")
        return robot

    def setup_sensors(self, robot_name):
        """Setup sensors for the robot"""
        robot = self.robots.get(robot_name)
        if not robot:
            print(f"Robot {robot_name} not found")
            return

        # Add camera
        camera_path = f"{robot.prim_path}/camera"
        camera = Camera(
            prim_path=camera_path,
            position=(0.3, 0, 0.2),
            frequency=30
        )
        self.world.scene.add_sensor(camera)

        # Add IMU
        imu_path = f"{robot.prim_path}/imu"
        imu = IMUSensor(
            prim_path=imu_path,
            position=(0, 0, 0.5)
        )
        self.world.scene.add_sensor(imu)

        print(f"Sensors added to robot {robot_name}")

    def run_simulation(self, steps=1000):
        """Run the simulation for specified steps"""
        if self.world is None:
            print("World not initialized")
            return

        self.world.reset()

        for i in range(steps):
            self.world.step(render=True)

            # Process sensor data and control commands
            if i % 10 == 0:  # Print every 10 steps
                print(f"Simulation step: {i}/{steps}")

        print("Simulation completed")

    def get_sensor_data(self, robot_name, sensor_type="camera"):
        """Get sensor data from robot"""
        robot = self.robots.get(robot_name)
        if not robot:
            return None

        # Implementation depends on specific sensor access methods
        # This is a simplified example
        if sensor_type == "camera":
            # Get camera data
            pass
        elif sensor_type == "lidar":
            # Get LIDAR data
            pass
        elif sensor_type == "imu":
            # Get IMU data
            pass

        return None

    def add_obstacles(self, obstacle_configs):
        """Add obstacles to the environment"""
        for i, config in enumerate(obstacle_configs):
            obstacle_path = f"/World/Obstacle_{i}"

            # Add obstacle based on configuration
            if config['type'] == 'box':
                add_reference_to_stage(
                    usd_path=f"{self.assets_root}/Isaac/Props/Blocks/block_instanceable.usd",
                    prim_path=obstacle_path
                )

                # Set position and scale
                from pxr import Gf
                obstacle_prim = self.world.stage.GetPrimAtPath(obstacle_path)
                if obstacle_prim:
                    import omni.usd
                    omni.usd.get_context().get_stage_update_events().set_translate_op(
                        obstacle_path,
                        Gf.Vec3d(config['position'])
                    )

    def setup_navigation_goal(self, goal_position):
        """Setup navigation goal in the environment"""
        # Add goal marker to environment
        goal_path = "/World/Goal"
        goal_marker = add_reference_to_stage(
            usd_path=f"{self.assets_root}/Isaac/Props/KIT/wood_cube_0_5meter.usd",
            prim_path=goal_path
        )

        # Position goal
        from pxr import Gf
        import omni.usd
        omni.usd.get_context().get_stage_update_events().set_translate_op(
            goal_path,
            Gf.Vec3d(goal_position[0], goal_position[1], goal_position[2])
        )

        print(f"Navigation goal set at {goal_position}")

# Example usage
def main():
    extension = RoboticsExtension()

    # Initialize simulation
    extension.initialize_world()

    # Load robot
    robot = extension.load_robot(
        robot_name="franka_robot",
        urdf_path="/path/to/robot/urdf",
        position=(0, 0, 0.5)
    )

    # Setup sensors
    extension.setup_sensors("franka_robot")

    # Add obstacles
    obstacles = [
        {'type': 'box', 'position': (2, 0, 0.5)},
        {'type': 'box', 'position': (0, 2, 0.5)},
        {'type': 'box', 'position': (-2, 0, 0.5)}
    ]
    extension.add_obstacles(obstacles)

    # Set navigation goal
    extension.setup_navigation_goal((3, 3, 0.5))

    # Run simulation
    extension.run_simulation(steps=5000)

if __name__ == "__main__":
    main()
```

### Unity Robotics Setup

#### Unity Robotics Package Integration

```csharp
// Unity Robotics Package setup script
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityRoboticsSetup : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIpAddress = "127.0.0.1";
    public int rosPort = 10000;
    public bool autoConnect = true;

    [Header("Robot Configuration")]
    public GameObject robotPrefab;
    public Transform spawnPoint;

    private ROSConnection ros;
    private bool isConnected = false;

    void Start()
    {
        if (autoConnect)
        {
            ConnectToROS();
        }

        // Setup robot in Unity scene
        SetupRobot();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIpAddress, rosPort);

        // Register connection callbacks
        ros.onConnected += OnConnected;
        ros.onDisconnected += OnDisconnected;

        Debug.Log($"Connecting to ROS at {rosIpAddress}:{rosPort}");
    }

    void SetupRobot()
    {
        if (robotPrefab != null && spawnPoint != null)
        {
            Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
            Debug.Log("Robot instantiated in Unity scene");
        }
    }

    void OnConnected()
    {
        isConnected = true;
        Debug.Log("Connected to ROS successfully");

        // Start publishing Unity data to ROS
        StartCoroutine(PublishUnityData());
    }

    void OnDisconnected()
    {
        isConnected = false;
        Debug.Log("Disconnected from ROS");
    }

    System.Collections.IEnumerator PublishUnityData()
    {
        // Publish Unity scene data to ROS
        while (isConnected)
        {
            // Publish robot transforms
            PublishRobotTransforms();

            // Publish sensor data if available
            PublishSensorData();

            yield return new WaitForSeconds(0.033f); // ~30 FPS
        }
    }

    void PublishRobotTransforms()
    {
        // Get robot transforms and publish to ROS TF
        // Implementation depends on robot structure
    }

    void PublishSensorData()
    {
        // Publish Unity-generated sensor data to ROS topics
        // Example: Camera images, IMU data, etc.
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

## Performance Monitoring and Optimization Tools

### System Monitoring for Robotics

#### Real-time Performance Monitoring

```python
# performance_monitor.py
import psutil
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    process_count: int
    robot_cpu_usage: float

class PerformanceMonitor:
    """Performance monitoring for robotics applications"""

    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        self.metrics_history = deque(maxlen=max_samples)
        self.is_monitoring = False
        self.monitoring_thread = None

        # GPU monitoring (if available)
        try:
            import GPUtil
            self.gputil_available = True
        except ImportError:
            self.gputil_available = False

    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            print("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop running in background thread"""
        while self.is_monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            time.sleep(0.1)  # 10 Hz monitoring

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        timestamp = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU memory (if available)
        gpu_memory = 0.0
        if self.gputil_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = gpus[0].memoryUtil * 100  # Percentage
            except:
                gpu_memory = 0.0

        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_io_data = {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        else:
            disk_io_data = {}

        # Process count
        process_count = len(psutil.pids())

        # Robot-specific CPU usage (placeholder)
        robot_cpu_usage = self._get_robot_process_cpu()

        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory=gpu_memory,
            network_io=network_io,
            disk_io=disk_io_data,
            process_count=process_count,
            robot_cpu_usage=robot_cpu_usage
        )

    def _get_robot_process_cpu(self) -> float:
        """Get CPU usage of robot processes"""
        robot_processes = ['ros', 'gazebo', 'isaac', 'robot']
        total_robot_cpu = 0.0

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                proc_name = proc.info['name'].lower()
                if any(robot_proc in proc_name for robot_proc in robot_processes):
                    total_robot_cpu += proc.info['cpu_percent'] or 0.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total_robot_cpu

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get the most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_average_metrics(self, window_size=100) -> PerformanceMetrics:
        """Get average metrics over specified window"""
        if len(self.metrics_history) < window_size:
            window = list(self.metrics_history)
        else:
            window = list(self.metrics_history)[-window_size:]

        if not window:
            return None

        # Calculate averages
        avg_timestamp = sum(m.timestamp for m in window) / len(window)
        avg_cpu = sum(m.cpu_percent for m in window) / len(window)
        avg_memory = sum(m.memory_percent for m in window) / len(window)
        avg_gpu = sum(m.gpu_memory for m in window) / len(window)
        avg_robot_cpu = sum(m.robot_cpu_usage for m in window) / len(window)

        # For network and disk, we might want rates instead of totals
        return PerformanceMetrics(
            timestamp=avg_timestamp,
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            gpu_memory=avg_gpu,
            network_io={'avg_rate': 'calculate_from_deltas'},  # Placeholder
            disk_io={'avg_rate': 'calculate_from_deltas'},    # Placeholder
            process_count=window[-1].process_count,  # Last value
            robot_cpu_usage=avg_robot_cpu
        )

    def generate_performance_report(self) -> str:
        """Generate a performance report"""
        if not self.metrics_history:
            return "No performance data available"

        # Calculate statistics
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        gpu_values = [m.gpu_memory for m in self.metrics_history]
        robot_cpu_values = [m.robot_cpu_usage for m in self.metrics_history]

        report = []
        report.append("# Performance Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Sample count: {len(self.metrics_history)}")
        report.append("")

        # CPU Performance
        report.append("## CPU Performance")
        report.append(f"- Average CPU Usage: {sum(cpu_values)/len(cpu_values):.2f}%")
        report.append(f"- Peak CPU Usage: {max(cpu_values):.2f}%")
        report.append(f"- Min CPU Usage: {min(cpu_values):.2f}%")
        report.append(f"- Robot CPU Usage: {sum(robot_cpu_values)/len(robot_cpu_values):.2f}%")
        report.append("")

        # Memory Performance
        report.append("## Memory Performance")
        report.append(f"- Average Memory Usage: {sum(memory_values)/len(memory_values):.2f}%")
        report.append(f"- Peak Memory Usage: {max(memory_values):.2f}%")
        report.append(f"- Min Memory Usage: {min(memory_values):.2f}%")
        report.append("")

        # GPU Performance (if monitored)
        if any(gpu_values):
            report.append("## GPU Performance")
            valid_gpu_values = [v for v in gpu_values if v > 0]
            if valid_gpu_values:
                report.append(f"- Average GPU Memory: {sum(valid_gpu_values)/len(valid_gpu_values):.2f}%")
                report.append(f"- Peak GPU Memory: {max(valid_gpu_values):.2f}%")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        avg_cpu = sum(cpu_values)/len(cpu_values)
        avg_memory = sum(memory_values)/len(memory_values)

        if avg_cpu > 80:
            report.append("- High CPU usage detected - consider optimization")
        if avg_memory > 85:
            report.append("- High memory usage detected - check for leaks")
        if any(v > 90 for v in gpu_values):
            report.append("- High GPU memory usage detected - optimize graphics")

        return "\n".join(report)

    def plot_performance(self, save_path=None):
        """Plot performance metrics over time"""
        if not self.metrics_history:
            print("No data to plot")
            return

        timestamps = [m.timestamp for m in self.metrics_history]
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        gpu_values = [m.gpu_memory for m in self.metrics_history if m.gpu_memory > 0]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # CPU Usage
        axes[0, 0].plot(timestamps, cpu_values)
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True)

        # Memory Usage
        axes[0, 1].plot(timestamps, memory_values)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True)

        # GPU Usage (if available)
        if gpu_values:
            gpu_timestamps = [m.timestamp for m in self.metrics_history if m.gpu_memory > 0]
            axes[1, 0].plot(gpu_timestamps, gpu_values)
            axes[1, 0].set_title('GPU Memory Usage Over Time')
            axes[1, 0].set_ylabel('GPU Memory %')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].set_title('GPU Memory (Not Available)')
            axes[1, 0].text(0.5, 0.5, 'GPU Monitoring\nNot Available',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)

        # Robot CPU Usage
        robot_cpu_values = [m.robot_cpu_usage for m in self.metrics_history]
        axes[1, 1].plot(timestamps, robot_cpu_values)
        axes[1, 1].set_title('Robot Process CPU Usage')
        axes[1, 1].set_ylabel('Robot CPU %')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Performance plot saved to {save_path}")

        plt.show()

# Example usage
def example_performance_monitoring():
    """Example of using the performance monitor"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()

    # Simulate some robot operation
    import time
    time.sleep(10)  # Let it run for 10 seconds

    # Generate report
    report = monitor.generate_performance_report()
    print(report)

    # Plot performance
    monitor.plot_performance('performance_chart.png')

    # Stop monitoring
    monitor.stop_monitoring()

if __name__ == "__main__":
    example_performance_monitoring()
```

## Debugging and Profiling Tools

### Robotics Debugging Tools

```python
# debug_tools.py
import traceback
import sys
import logging
from functools import wraps
import time
import cProfile
import pstats
from io import StringIO

class RoboticsDebugger:
    """
    Comprehensive debugging tools for robotics applications
    """

    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('RoboticsDebugger')
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(ch)

        # Performance tracking
        self.performance_data = {}

    def debug_wrapper(self, func):
        """
        Decorator for debugging function calls
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__

            self.logger.debug(f"Entering {func_name}")
            self.logger.debug(f"Arguments: args={args}, kwargs={kwargs}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.logger.debug(f"{func_name} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                self.logger.error(f"Error in {func_name}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            finally:
                self.logger.debug(f"Exiting {func_name}")

        return wrapper

    def profile_function(self, func):
        """
        Decorator for profiling function performance
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                pr.disable()

            # Store profile data
            s = StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions

            profile_data = s.getvalue()
            self.performance_data[func.__name__] = {
                'profile': profile_data,
                'call_count': ps.total_calls,
                'total_time': ps.total_tt
            }

            self.logger.info(f"Profiled {func.__name__}: {ps.total_tt:.4f}s total, {ps.total_calls} calls")
            return result

        return wrapper

    def log_sensor_data(self, sensor_name, data):
        """
        Log sensor data with timestamp and validation
        """
        timestamp = time.time()

        # Validate data
        if data is None:
            self.logger.warning(f"Null data received from {sensor_name}")
            return

        # Check for NaN or infinite values
        import numpy as np
        if isinstance(data, (list, np.ndarray)):
            if np.any(np.isnan(data)):
                self.logger.warning(f"NaN values detected in {sensor_name} data")
            if np.any(np.isinf(data)):
                self.logger.warning(f"Infinite values detected in {sensor_name} data")

        # Log data
        self.logger.debug(f"Sensor {sensor_name}: {data} at {timestamp}")

    def validate_robot_state(self, robot_state):
        """
        Validate robot state for common issues
        """
        issues = []

        # Check joint limits
        if 'joint_positions' in robot_state:
            joint_positions = robot_state['joint_positions']
            joint_limits = robot_state.get('joint_limits', {})

            for joint_name, position in joint_positions.items():
                if joint_name in joint_limits:
                    limits = joint_limits[joint_name]
                    if position < limits['min'] or position > limits['max']:
                        issues.append(f"Joint {joint_name} out of limits: {position} (limits: {limits})")

        # Check for valid values
        for key, value in robot_state.items():
            if isinstance(value, float):
                if np.isnan(value):
                    issues.append(f"NaN value in {key}: {value}")
                elif np.isinf(value):
                    issues.append(f"Infinite value in {key}: {value}")

        # Check velocity limits
        if 'joint_velocities' in robot_state:
            max_velocity = robot_state.get('max_velocity', 10.0)  # Default max
            for joint_name, velocity in robot_state['joint_velocities'].items():
                if abs(velocity) > max_velocity:
                    issues.append(f"Velocity limit exceeded for {joint_name}: {velocity} > {max_velocity}")

        if issues:
            for issue in issues:
                self.logger.warning(f"Robot state issue: {issue}")
        else:
            self.logger.debug("Robot state validation passed")

        return issues

    def performance_monitor(self, operation_name, expected_duration=None):
        """
        Context manager for performance monitoring
        """
        class PerformanceMonitor:
            def __enter__(monitor_self):
                self.logger.debug(f"Starting operation: {operation_name}")
                self.start_time = time.time()
                return self

            def __exit__(monitor_self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.logger.debug(f"Completed operation: {operation_name} in {duration:.4f}s")

                if expected_duration and duration > expected_duration:
                    self.logger.warning(f"Operation {operation_name} took longer than expected: {duration:.4f}s > {expected_duration:.4f}s")

                if exc_type:
                    self.logger.error(f"Exception in {operation_name}: {exc_type.__name__}: {exc_val}")

        return PerformanceMonitor()

    def create_debug_visualization(self, data, visualization_type='plot'):
        """
        Create debug visualizations for robotics data
        """
        import matplotlib.pyplot as plt

        if visualization_type == 'plot':
            plt.figure(figsize=(12, 8))

            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, (list, tuple)):
                        plt.plot(values, label=key)
            else:
                plt.plot(data)

            plt.legend()
            plt.title('Debug Visualization')
            plt.grid(True)
            plt.show()

        elif visualization_type == 'heatmap':
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, annot=True, cmap='viridis')
            plt.title('Debug Heatmap')
            plt.show()

    def get_performance_summary(self):
        """
        Get summary of performance data
        """
        if not self.performance_data:
            return "No performance data collected"

        summary = []
        summary.append("## Performance Summary")
        summary.append(f"Functions profiled: {len(self.performance_data)}")

        for func_name, data in self.performance_data.items():
            summary.append(f"- {func_name}: {data['total_time']:.4f}s, {data['call_count']} calls")

        return "\n".join(summary)

# Example usage of debugging tools
def example_debugging_usage():
    debugger = RoboticsDebugger()

    @debugger.debug_wrapper
    @debugger.profile_function
    def example_robot_control(joint_angles, target_positions):
        """Example robot control function"""
        # Simulate some computation
        import time
        time.sleep(0.01)  # Simulate processing time

        # Simulate control calculation
        errors = []
        for joint, target in target_positions.items():
            if joint in joint_angles:
                error = target - joint_angles[joint]
                errors.append(error)

        return sum(errors) / len(errors) if errors else 0.0

    # Use performance monitor
    with debugger.performance_monitor("robot_control_cycle", expected_duration=0.1):
        result = example_robot_control(
            {'joint1': 0.5, 'joint2': 1.0},
            {'joint1': 0.6, 'joint2': 1.1}
        )
        print(f"Control result: {result}")

    # Log sensor data
    debugger.log_sensor_data("imu_sensor", [0.1, 0.2, 9.8, 0.0, 0.0, 0.0])

    # Validate robot state
    robot_state = {
        'joint_positions': {'joint1': 1.5, 'joint2': 2.0},
        'joint_velocities': {'joint1': 0.1, 'joint2': 0.2},
        'joint_limits': {'joint1': {'min': -1.0, 'max': 1.0}, 'joint2': {'min': -2.0, 'max': 2.0}},
        'max_velocity': 1.0
    }
    issues = debugger.validate_robot_state(robot_state)

    # Print performance summary
    print(debugger.get_performance_summary())

if __name__ == "__main__":
    example_debugging_usage()
```

## Resource Management

### Efficient Resource Utilization

```python
# resource_manager.py
import gc
import psutil
import threading
import time
from collections import defaultdict
from enum import Enum

class ResourceType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    DISK = "disk"

class ResourceManager:
    """
    Resource management for robotics applications
    """

    def __init__(self):
        self.resource_limits = {
            ResourceType.MEMORY: 0.8,  # 80% of available memory
            ResourceType.CPU: 0.9,     # 90% of available CPU
            ResourceType.GPU: 0.85,    # 85% of available GPU
        }

        self.resource_usage = defaultdict(float)
        self.resource_callbacks = defaultdict(list)
        self.is_monitoring = False
        self.monitoring_thread = None

    def register_resource_callback(self, resource_type, callback):
        """
        Register callback for resource limit notifications
        """
        self.resource_callbacks[resource_type].append(callback)

    def start_resource_monitoring(self):
        """
        Start background resource monitoring
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_resources)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            print("Resource monitoring started")

    def stop_resource_monitoring(self):
        """
        Stop resource monitoring
        """
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Resource monitoring stopped")

    def _monitor_resources(self):
        """
        Background thread for resource monitoring
        """
        while self.is_monitoring:
            self._check_resource_usage()
            time.sleep(1.0)  # Check every second

    def _check_resource_usage(self):
        """
        Check current resource usage against limits
        """
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent / 100.0
        self.resource_usage[ResourceType.MEMORY] = memory_percent

        if memory_percent > self.resource_limits[ResourceType.MEMORY]:
            self._trigger_resource_callback(ResourceType.MEMORY, memory_percent)

        # Check CPU usage
        cpu_percent = psutil.cpu_percent() / 100.0
        self.resource_usage[ResourceType.CPU] = cpu_percent

        if cpu_percent > self.resource_limits[ResourceType.CPU]:
            self._trigger_resource_callback(ResourceType.CPU, cpu_percent)

        # Check GPU usage (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].memoryUtil
                self.resource_usage[ResourceType.GPU] = gpu_percent

                if gpu_percent > self.resource_limits[ResourceType.GPU]:
                    self._trigger_resource_callback(ResourceType.GPU, gpu_percent)
        except ImportError:
            pass  # GPU monitoring not available

    def _trigger_resource_callback(self, resource_type, usage):
        """
        Trigger registered callbacks for resource limit exceeded
        """
        for callback in self.resource_callbacks[resource_type]:
            try:
                callback(resource_type, usage)
            except Exception as e:
                print(f"Error in resource callback: {e}")

    def get_resource_usage(self, resource_type):
        """
        Get current resource usage
        """
        return self.resource_usage.get(resource_type, 0.0)

    def optimize_memory_usage(self):
        """
        Optimize memory usage by cleaning up unused objects
        """
        # Force garbage collection
        collected = gc.collect()
        print(f"Garbage collection: {collected} objects collected")

        # Clear matplotlib figures if they exist
        try:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.close('all')
        except ImportError:
            pass

        # Clear cached data
        self._clear_cached_data()

    def _clear_cached_data(self):
        """
        Clear cached data that may be consuming memory
        """
        # This would include clearing image caches, sensor data caches, etc.
        # Implementation depends on specific application needs
        pass

    def set_resource_limit(self, resource_type, limit):
        """
        Set resource usage limit
        """
        self.resource_limits[resource_type] = limit
        print(f"Set {resource_type.value} limit to {limit:.2%}")

    def get_system_resources(self):
        """
        Get comprehensive system resource information
        """
        system_info = {
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent_used': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }

        # Add GPU info if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                system_info['gpu'] = [{
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100
                } for gpu in gpus]
        except ImportError:
            system_info['gpu'] = None

        return system_info

    def adaptive_resource_allocation(self, priority_tasks):
        """
        Adaptively allocate resources based on task priorities
        """
        system_resources = self.get_system_resources()

        allocation_strategy = {}
        total_cpu = system_resources['cpu']['count']
        total_memory = system_resources['memory']['available']

        for task in priority_tasks:
            # Allocate resources based on task priority and requirements
            priority = task.get('priority', 'medium')
            cpu_req = task.get('cpu_requirement', 1.0)  # CPU cores needed
            mem_req = task.get('memory_requirement', 100)  # MB needed

            # Calculate allocation based on priority
            if priority == 'high':
                cpu_alloc = min(cpu_req, total_cpu * 0.4)  # Up to 40% of CPU
                mem_alloc = min(mem_req, total_memory * 0.4)  # Up to 40% of memory
            elif priority == 'medium':
                cpu_alloc = min(cpu_req, total_cpu * 0.3)  # Up to 30% of CPU
                mem_alloc = min(mem_req, total_memory * 0.3)  # Up to 30% of memory
            else:  # low
                cpu_alloc = min(cpu_req, total_cpu * 0.2)  # Up to 20% of CPU
                mem_alloc = min(mem_req, total_memory * 0.2)  # Up to 20% of memory

            allocation_strategy[task['name']] = {
                'cpu_cores': cpu_alloc,
                'memory_mb': mem_alloc / (1024*1024)  # Convert to MB
            }

        return allocation_strategy

class MemoryOptimizer:
    """
    Memory optimization utilities for robotics applications
    """

    def __init__(self):
        self.large_objects = {}
        self.object_references = defaultdict(list)

    def register_large_object(self, obj_id, obj, reference_point="unknown"):
        """
        Register large objects for memory optimization tracking
        """
        import sys
        size = sys.getsizeof(obj)
        self.large_objects[obj_id] = {
            'object': obj,
            'size': size,
            'reference_point': reference_point,
            'timestamp': time.time()
        }
        print(f"Registered large object {obj_id}: {size} bytes")

    def get_memory_usage_report(self):
        """
        Get memory usage report for registered objects
        """
        report = []
        report.append("## Memory Usage Report")

        total_size = 0
        for obj_id, obj_info in self.large_objects.items():
            size_mb = obj_info['size'] / (1024 * 1024)
            age = time.time() - obj_info['timestamp']
            report.append(f"- {obj_id}: {size_mb:.2f} MB, age: {age:.1f}s, ref: {obj_info['reference_point']}")
            total_size += obj_info['size']

        total_mb = total_size / (1024 * 1024)
        report.append(f"\nTotal tracked memory: {total_mb:.2f} MB")

        return "\n".join(report)

    def optimize_memory_pool(self, size_threshold=10*1024*1024):  # 10MB threshold
        """
        Optimize memory by removing old large objects
        """
        current_time = time.time()
        removed_count = 0

        for obj_id, obj_info in list(self.large_objects.items()):
            age = current_time - obj_info['timestamp']

            # Remove objects older than 30 seconds and larger than threshold
            if age > 30 and obj_info['size'] > size_threshold:
                del self.large_objects[obj_id]
                removed_count += 1
                print(f"Removed old large object: {obj_id}")

        if removed_count > 0:
            gc.collect()  # Force garbage collection
            print(f"Memory optimization: removed {removed_count} objects")

    def profile_memory_usage(self, func):
        """
        Decorator to profile memory usage of functions
        """
        from memory_profiler import profile
        return profile(func)

# Example usage
def example_resource_management():
    # Initialize resource manager
    resource_mgr = ResourceManager()

    # Set custom limits
    resource_mgr.set_resource_limit(ResourceType.MEMORY, 0.75)  # 75% memory limit
    resource_mgr.set_resource_limit(ResourceType.CPU, 0.85)    # 85% CPU limit

    # Register resource callback
    def memory_warning_callback(resource_type, usage):
        print(f"WARNING: {resource_type.value} usage is high: {usage:.2%}")

    resource_mgr.register_resource_callback(ResourceType.MEMORY, memory_warning_callback)

    # Start monitoring
    resource_mgr.start_resource_monitoring()

    # Example priority tasks
    priority_tasks = [
        {'name': 'control_system', 'priority': 'high', 'cpu_requirement': 2.0, 'memory_requirement': 500*1024*1024},
        {'name': 'perception', 'priority': 'high', 'cpu_requirement': 4.0, 'memory_requirement': 1000*1024*1024},
        {'name': 'navigation', 'priority': 'medium', 'cpu_requirement': 2.0, 'memory_requirement': 300*1024*1024},
        {'name': 'logging', 'priority': 'low', 'cpu_requirement': 0.5, 'memory_requirement': 100*1024*1024}
    ]

    allocation = resource_mgr.adaptive_resource_allocation(priority_tasks)
    for task, alloc in allocation.items():
        print(f"{task}: {alloc['cpu_cores']:.1f} cores, {alloc['memory_mb']:.1f} MB")

    # Memory optimization example
    mem_optimizer = MemoryOptimizer()

    # Simulate creating large objects
    large_array = np.random.rand(1000, 1000)  # ~8MB array
    mem_optimizer.register_large_object('test_array', large_array, 'simulation')

    print(mem_optimizer.get_memory_usage_report())

    # Stop monitoring after some time
    time.sleep(5)
    resource_mgr.stop_resource_monitoring()

if __name__ == "__main__":
    example_resource_management()
```

## Week Summary

This section provided comprehensive information about tools and resources for Physical AI and Humanoid Robotics development. We covered development environment setup, simulation optimization, sensor integration, debugging tools, performance monitoring, and resource management. The tools and techniques described enable efficient development, testing, and deployment of complex robotics systems while maintaining optimal performance and reliability.