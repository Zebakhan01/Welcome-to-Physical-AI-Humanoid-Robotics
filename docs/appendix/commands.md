---
sidebar_position: 2
---

# Command Reference

## Introduction to Commands

This section provides a comprehensive reference for commands used in Physical AI and humanoid robotics development. The commands cover development environments, simulation tools, deployment procedures, and system administration tasks. This reference serves as a quick lookup for commonly used commands and procedures.

## Development Environment Commands

### ROS/ROS2 Commands

#### Basic ROS Commands

```bash
# Initialize ROS environment
source /opt/ros/noetic/setup.bash  # For ROS1
source /opt/ros/foxy/setup.bash    # For ROS2

# Create a new ROS package
catkin_create_pkg package_name std_msgs rospy roscpp
colcon build  # For ROS2

# Run ROS core
roscore  # For ROS1
ros2 daemon start  # For ROS2

# List ROS nodes, topics, and services
rosnode list
rostopic list
rosservice list

# Echo messages on a topic
rostopic echo /topic_name
ros2 topic echo /topic_name std_msgs/msg/String

# Call a service
rosservice call /service_name "request_data"
ros2 service call /service_name std_srvs/srv/Empty

# Run a ROS node
rosrun package_name node_name
ros2 run package_name node_name

# Launch a ROS system
roslaunch package_name launch_file.launch
ros2 launch package_name launch_file.py
```

#### Advanced ROS Commands

```bash
# ROS bag recording and playback
rosbag record -a  # Record all topics
rosbag play recorded_bag.bag

# ROS parameter server
rosparam list
rosparam get param_name
rosparam set param_name value

# ROS launch arguments
roslaunch package_name launch_file.launch arg_name:=value

# ROS node management
rosnode info node_name
rosnode kill node_name

# ROS topic monitoring
rostopic info /topic_name
rostopic hz /topic_name  # Check topic frequency
rostopic bw /topic_name  # Check topic bandwidth
```

### Git Commands for Robotics Development

```bash
# Basic Git operations
git clone repository_url
git status
git add .
git commit -m "Commit message"
git push origin branch_name
git pull origin branch_name

# Branch management
git branch  # List branches
git branch new_branch  # Create new branch
git checkout branch_name  # Switch to branch
git checkout -b new_branch  # Create and switch

# Advanced Git operations
git log --oneline  # Concise commit history
git diff  # Show changes
git stash  # Temporarily save changes
git merge branch_name  # Merge branch
git rebase branch_name  # Rebase onto branch

# Git for collaborative development
git fetch origin  # Fetch remote changes
git merge origin/main  # Merge remote main
git pull --rebase origin main  # Pull with rebase

# Tagging for releases
git tag -a v1.0 -m "Version 1.0"
git push origin --tags
```

## Simulation Environment Commands

### Gazebo Commands

```bash
# Start Gazebo
gazebo  # Start empty world
gazebo world_file.world  # Start with specific world

# Gazebo command line options
gazebo --verbose  # Verbose output
gazebo --play=paused_world.world  # Start paused
gazebo --record  # Record simulation

# Gazebo tools
gz topic -l  # List topics
gz topic -i /gazebo/default/physics/contacts  # Inspect topic
gz model -l  # List models in simulation
gz service -l  # List services

# Gazebo plugins and configuration
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/path/to/resources
```

### Isaac Sim Commands

```bash
# Start Isaac Sim
./isaac-sim.sh  # Or the appropriate startup script

# Isaac Sim environment variables
export ISAACSIM_PYTHON_EXE=/path/to/python
export ISAACSIM_HEADLESS=1  # For headless operation

# Isaac Sim extensions management
python -m omni.tools.updater  # Extension updater
isaac-sim --ext-folder /path/to/extensions  # Load extensions

# Isaac Sim configuration
isaac-sim --config config_file.json  # Load configuration
isaac-sim --enable-extensions  # Enable specific extensions
```

## Build System Commands

### CMake Commands

```bash
# Basic CMake operations
cmake ..  # Configure project
cmake --build .  # Build project
cmake --build . --parallel  # Build in parallel

# CMake with specific generators
cmake -G "Unix Makefiles" ..
cmake -G "Ninja" ..  # Faster builds

# CMake with specific configurations
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..

# CMake installation
cmake --install . --prefix /install/path
cmake --build . --target install

# CMake package management
cmake -DCMAKE_PREFIX_PATH=/path/to/packages ..
find_package(PackageName REQUIRED)
```

### Colcon Commands (ROS2)

```bash
# Basic colcon build
colcon build
colcon build --packages-select package_name  # Build specific package
colcon build --symlink-install  # Symlink for development

# Colcon with specific options
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon build --executor sequential  # Sequential builds

# Colcon test and install
colcon test
colcon test-result --all  # Show test results
colcon build --install-symlinks  # Install with symlinks

# Colcon clean operations
rm -rf build/ install/ log/  # Clean build artifacts
colcon build --packages-up-to package_name  # Build package and dependencies
```

## Docker Commands for Robotics

### Basic Docker Operations

```bash
# Docker image management
docker build -t image_name .
docker images  # List images
docker rmi image_id  # Remove image
docker pull image_name  # Pull from registry

# Docker container management
docker run -it image_name  # Run container interactively
docker run -d --name container_name image_name  # Run detached
docker ps  # List running containers
docker stop container_name  # Stop container
docker rm container_name  # Remove container

# Docker networking
docker run -p host_port:container_port image_name
docker network ls  # List networks
docker network create network_name

# Docker volumes
docker run -v host_path:container_path image_name
docker volume ls  # List volumes
docker volume create volume_name
```

### Docker for ROS Development

```bash
# Run ROS in Docker
docker run -it --rm \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --privileged \
  ros:noetic \
  bash

# Build ROS Docker image
docker build -t ros_robotics:latest -f Dockerfile .

# Docker Compose for robotics systems
docker-compose up
docker-compose down
docker-compose logs service_name
```

## Python Development Commands

### Virtual Environment Management

```bash
# Create virtual environment
python -m venv venv_name
virtualenv venv_name

# Activate virtual environment
source venv_name/bin/activate  # Linux/Mac
venv_name\Scripts\activate  # Windows

# Deactivate virtual environment
deactivate

# Install packages
pip install package_name
pip install -r requirements.txt
pip freeze > requirements.txt  # Export current packages

# Package management
pip list
pip show package_name
pip uninstall package_name
```

### Python Development Tools

```bash
# Code formatting
black .  # Format all Python files
black --line-length 88 file.py  # Format with line length

# Code linting
flake8 .  # Lint all files
pylint module_name  # Analyze module

# Type checking
mypy .  # Type check all files
mypy --strict file.py  # Strict type checking

# Testing
pytest  # Run tests
pytest -v  # Verbose output
pytest --cov=module_name  # Coverage analysis
unittest discover  # Discover and run tests

# Documentation generation
sphinx-build -b html source_dir build_dir
pdoc module_name > docs.html
```

## System Administration Commands

### Process Management

```bash
# Process monitoring
ps aux  # List all processes
top  # Real-time process monitoring
htop  # Interactive process viewer
ps aux | grep process_name  # Find specific process

# Process control
kill process_id  # Terminate process
kill -9 process_id  # Force terminate
pkill process_name  # Kill by name
killall process_name  # Kill all processes by name

# System monitoring
df -h  # Disk space usage
du -sh directory  # Directory size
free -h  # Memory usage
iotop  # I/O monitoring
vmstat  # Virtual memory statistics
```

### Network Configuration

```bash
# Network interface information
ifconfig  # Interface configuration
ip addr show  # IP addresses
route -n  # Routing table
netstat -tuln  # Active connections

# Network testing
ping hostname  # Test connectivity
traceroute hostname  # Trace route
telnet hostname port  # Test port connectivity
nc -zv hostname port  # Netcat port check

# Network configuration
ifconfig eth0 192.168.1.100  # Set IP address
route add default gw 192.168.1.1  # Set default gateway
echo "nameserver 8.8.8.8" >> /etc/resolv.conf  # DNS configuration
```

## Hardware Interface Commands

### Serial Communication

```bash
# List serial ports
ls /dev/tty*  # List all serial devices
dmesg | grep tty  # Check for new devices

# Serial communication
screen /dev/ttyUSB0 115200  # Connect with screen
picocom -b 115200 /dev/ttyUSB0  # Connect with picocom
minicom -D /dev/ttyUSB0  # Connect with minicom

# Set serial port permissions
sudo chmod 666 /dev/ttyUSB0
sudo usermod -a -G dialout $USER  # Add user to dialout group
```

### USB Device Management

```bash
# USB device information
lsusb  # List USB devices
lsusb -v  # Verbose USB information
dmesg | grep USB  # Check USB events

# USB device permissions
sudo chmod 666 /dev/bus/usb/vendor_id/product_id
# Or add udev rules in /etc/udev/rules.d/
```

## Performance Monitoring Commands

### Real-time Performance

```bash
# CPU and memory monitoring
htop  # Interactive system monitor
glances  # Comprehensive system monitor
iotop  # I/O monitoring
nethogs  # Network bandwidth per process

# Process-specific monitoring
pidstat -p process_id 1  # Monitor specific process
strace -p process_id  # System call tracing
lsof -p process_id  # Files opened by process

# Performance analysis
perf record -g ./program  # Performance profiling
perf report  # View performance report
valgrind --tool=memcheck ./program  # Memory checking
```

## Troubleshooting Commands

### System Diagnostics

```bash
# System information
uname -a  # System information
lscpu  # CPU information
lsmem  # Memory information
lspci  # PCI devices
lsblk  # Block devices

# Hardware diagnostics
sudo dmidecode  # Hardware information
sensors  # Temperature sensors
smartctl -a /dev/sda  # Hard drive health

# Log analysis
journalctl -u service_name  # Service logs
dmesg  # Kernel messages
tail -f /var/log/syslog  # Follow system log
grep "error" /var/log/syslog  # Search for errors
```

## Common Robotics Development Workflows

### Complete Development Cycle

```bash
# 1. Setup development environment
git clone repository
cd repository
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. Run tests
make test
# or
pytest tests/

# 4. Launch the system
source ../devel/setup.bash  # For ROS
roslaunch package_name system.launch

# 5. Monitor and debug
rostopic echo /debug_topic
rosrun rqt_plot rqt_plot
# Use appropriate debugging tools
```

### Simulation Development Workflow

```bash
# 1. Start simulation environment
gazebo --verbose worlds/empty.world &
sleep 5  # Wait for Gazebo to start

# 2. Launch robot in simulation
roslaunch robot_gazebo robot_world.launch

# 3. Run robot controllers
roslaunch robot_control controllers.launch

# 4. Monitor system
rosrun rqt_gui rqt_gui
rosrun rviz rviz

# 5. Test functionality
rostopic pub /cmd_vel geometry_msgs/Twist "linear: [0.5, 0.0, 0.0]"
```

## Safety and Security Commands

### Security Checks

```bash
# Check file permissions
ls -la file_or_directory
chmod 755 executable_file
chown user:group file

# Security audit
sudo apt-get update && sudo apt-get upgrade
sudo ufw enable  # Enable firewall
sudo ufw allow port_number
ssh-keygen -t rsa -b 4096  # Generate SSH key
```

## Week Summary

This command reference provides a comprehensive collection of commands commonly used in Physical AI and humanoid robotics development. From basic system commands to specialized ROS and simulation tools, this reference serves as a quick lookup for development workflows, troubleshooting, and system administration tasks. Regular use of these commands will streamline the development process and improve productivity.