---
sidebar_position: 4
---

# Implementation Guide

## Introduction to Implementation

This implementation guide provides step-by-step instructions for developing the Physical AI & Humanoid Robotics capstone project. The guide follows a systematic approach that integrates all concepts learned throughout the curriculum, emphasizing safety, reliability, and professional software engineering practices.

## Phase 1: Project Setup and Environment Configuration

### 1.1 Development Environment Setup

#### System Requirements Verification

```bash
# Verify system meets requirements
echo "=== System Requirements Verification ==="

# Check GPU availability (NVIDIA recommended)
nvidia-smi || echo "Warning: NVIDIA GPU not detected"

# Check available memory
free -h

# Check disk space
df -h $HOME

# Verify ROS installation
if command -v ros2 &> /dev/null; then
    echo "ROS 2 installed: $(ros2 --version)"
else
    echo "Error: ROS 2 not found"
    exit 1
fi

# Check Isaac Sim availability
if [ -d "/opt/isaac-sim" ]; then
    echo "Isaac Sim found at /opt/isaac-sim"
else
    echo "Warning: Isaac Sim not found"
fi

# Check Python environment
python3 --version
pip3 --version
```

#### Workspace Creation

```bash
# Create project workspace
mkdir -p ~/physical_ai_capstone/src
cd ~/physical_ai_capstone

# Initialize workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash

echo "Workspace created at ~/physical_ai_capstone"
```

#### Virtual Environment Setup

```bash
# Create Python virtual environment for the project
cd ~/physical_ai_capstone

python3 -m venv capstone_env
source capstone_env/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install setuptools wheel

# Install robotics-specific packages
pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    opencv-python>=4.5.0 \
    transforms3d>=0.4.0 \
    pyquaternion>=0.9.9 \
    control>=0.9.0 \
    pygame>=2.1.0

# Install AI/ML packages
pip install \
    torch>=1.12.0 \
    torchvision>=0.13.0 \
    tensorflow>=2.9.0 \
    scikit-learn>=1.1.0 \
    stable-baselines3>=1.7.0

# Install ROS interfaces
pip install \
    roslibpy>=1.3.0 \
    rospy-message-converter>=0.5.0

# Install development tools
pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=22.0.0 \
    flake8>=5.0.0 \
    mypy>=0.971

echo "Virtual environment created and packages installed"
```

### 1.2 Project Structure Setup

```bash
# Create comprehensive project structure
cd ~/physical_ai_capstone/src

# Create main package structure
mkdir -p physical_ai_system/{config,launch,scripts,src,include,msg,srv,action,test,docs}

# Create subdirectories for different system components
mkdir -p physical_ai_system/{perception,cognition,control,interaction,safety,utils}

# Create config directory structure
mkdir -p physical_ai_system/config/{sensors,controllers,planning,navigation,manipulation}

# Create launch directory structure
mkdir -p physical_ai_system/launch/{simulation,real_robot,perception,cognition,control}

# Create test structure
mkdir -p physical_ai_system/test/{unit,integration,system}

echo "Project structure created successfully"
```

## Phase 2: Core System Architecture Implementation

### 2.1 System Architecture Definition

```python
# physical_ai_system/src/system_architecture.py
"""
System architecture for Physical AI & Humanoid Robotics capstone project
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image, JointState, Imu, LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import threading
import time
from typing import Dict, List, Optional, Any

class PhysicalAISystem(Node):
    """
    Main system node for Physical AI & Humanoid Robotics
    """

    def __init__(self):
        super().__init__('physical_ai_system')

        # System state
        self.system_state = 'startup'  # startup, ready, executing, error, shutdown
        self.safety_status = 'safe'    # safe, warning, emergency
        self.emergency_stop = False

        # Initialize components
        self._initialize_components()

        # Setup communication
        self._setup_communication()

        # Start system monitoring
        self._start_monitoring()

        self.get_logger().info('Physical AI System initialized')

    def _initialize_components(self):
        """Initialize all system components"""
        self.get_logger().info('Initializing system components...')

        # Initialize perception system
        self.perception_system = PerceptionSystem(self)

        # Initialize cognition system
        self.cognition_system = CognitionSystem(self)

        # Initialize control system
        self.control_system = ControlSystem(self)

        # Initialize interaction system
        self.interaction_system = InteractionSystem(self)

        # Initialize safety system
        self.safety_system = SafetySystem(self)

        self.get_logger().info('All components initialized')

    def _setup_communication(self):
        """Setup ROS communication infrastructure"""
        # QoS profiles for different data types
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 1)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', cmd_qos)

        # Subscribers
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, sensor_qos
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('Communication infrastructure setup complete')

    def _start_monitoring(self):
        """Start system monitoring and health checks"""
        # System status monitoring
        self.status_timer = self.create_timer(1.0, self.publish_system_status)

        # Health check timer
        self.health_timer = self.create_timer(5.0, self.health_check)

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop commands"""
        if msg.data:
            self.emergency_stop = True
            self.safety_system.trigger_emergency_stop()
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        else:
            self.emergency_stop = False
            self.safety_system.clear_emergency_stop()
            self.get_logger().info('Emergency stop cleared')

    def joint_state_callback(self, msg: JointState):
        """Handle joint state updates"""
        if not self.emergency_stop:
            self.control_system.update_joint_states(msg)

    def odom_callback(self, msg: Odometry):
        """Handle odometry updates"""
        if not self.emergency_stop:
            self.control_system.update_odometry(msg)

    def publish_system_status(self):
        """Publish system status updates"""
        status_msg = String()
        status_msg.data = f"state:{self.system_state},safety:{self.safety_status},emergency:{self.emergency_stop}"
        self.status_pub.publish(status_msg)

    def health_check(self):
        """Perform system health checks"""
        # Check all subsystems
        subsystem_health = {
            'perception': self.perception_system.is_healthy(),
            'cognition': self.cognition_system.is_healthy(),
            'control': self.control_system.is_healthy(),
            'interaction': self.interaction_system.is_healthy(),
            'safety': self.safety_system.is_healthy()
        }

        # Determine overall system health
        all_healthy = all(subsystem_health.values())

        if all_healthy:
            self.system_state = 'ready'
            self.safety_status = 'safe'
        else:
            self.system_state = 'warning'
            self.safety_status = 'warning'

            unhealthy_systems = [name for name, healthy in subsystem_health.items() if not healthy]
            self.get_logger().warn(f'Unhealthy subsystems: {unhealthy_systems}')

    def run_system(self):
        """Main system execution loop"""
        self.get_logger().info('Starting system execution...')
        self.system_state = 'executing'

        try:
            while rclpy.ok() and not self.emergency_stop:
                # Update perception system
                perception_data = self.perception_system.process_sensors()

                # Update cognition system
                cognitive_state = self.cognition_system.process_perceptions(perception_data)

                # Generate control commands
                control_commands = self.control_system.generate_commands(cognitive_state)

                # Execute commands
                self.control_system.execute_commands(control_commands)

                # Update interaction system
                self.interaction_system.update()

                # Update safety system
                self.safety_system.update()

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except Exception as e:
            self.get_logger().error(f'System execution error: {e}')
            self.system_state = 'error'
            self.safety_system.trigger_emergency_stop()

        finally:
            self.system_state = 'shutdown'
            self.get_logger().info('System shutdown complete')


class PerceptionSystem:
    """
    Perception system component
    """

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.sensors = {}
        self.perception_data = {}
        self.is_initialized = False

        self._initialize_sensors()
        self.is_initialized = True

    def _initialize_sensors(self):
        """Initialize sensor interfaces"""
        self.parent_node.get_logger().info('Initializing perception system...')

        # Camera sensors
        self.camera_sub = self.parent_node.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        # LIDAR sensors
        self.lidar_sub = self.parent_node.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )

        # IMU sensors
        self.imu_sub = self.parent_node.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Initialize perception algorithms
        self.object_detector = ObjectDetectionModule()
        self.localization_system = LocalizationModule()
        self.mapping_system = MappingModule()

    def camera_callback(self, msg: Image):
        """Process camera data"""
        # Convert ROS image to OpenCV format
        image = self.ros_image_to_cv2(msg)

        # Run object detection
        objects = self.object_detector.detect(image)

        # Store in perception data
        self.perception_data['objects'] = objects
        self.perception_data['camera_image'] = image

    def lidar_callback(self, msg: LaserScan):
        """Process LIDAR data"""
        # Process LIDAR scan
        obstacles = self.process_lidar_scan(msg)

        # Store in perception data
        self.perception_data['obstacles'] = obstacles
        self.perception_data['lidar_data'] = msg

    def imu_callback(self, msg: Imu):
        """Process IMU data"""
        # Extract orientation and acceleration
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

        # Store in perception data
        self.perception_data['orientation'] = orientation
        self.perception_data['acceleration'] = acceleration
        self.perception_data['angular_velocity'] = angular_velocity

    def process_sensors(self):
        """Process all sensor data and return integrated perception"""
        if not self.is_initialized:
            return {}

        # Update sensor data
        # (This would be called from the main loop)

        # Integrate sensor data
        integrated_perception = {
            'environment_map': self.mapping_system.get_map(),
            'object_locations': self.extract_object_locations(),
            'robot_pose': self.localization_system.get_pose(),
            'obstacle_distances': self.perception_data.get('obstacles', []),
            'camera_data': self.perception_data.get('camera_image', None)
        }

        return integrated_perception

    def extract_object_locations(self):
        """Extract object locations from perception data"""
        objects = self.perception_data.get('objects', [])
        locations = []

        for obj in objects:
            if 'bbox' in obj and 'confidence' in obj:
                # Calculate 3D position from 2D detection and depth
                bbox = obj['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # This would use depth information to get 3D position
                # For now, return 2D position with confidence
                locations.append({
                    'class': obj.get('class', 'unknown'),
                    'confidence': obj['confidence'],
                    'position_2d': [center_x, center_y]
                })

        return locations

    def is_healthy(self):
        """Check if perception system is healthy"""
        # Check if sensors are receiving data
        return len(self.perception_data) > 0


class CognitionSystem:
    """
    Cognition system component
    """

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.behavior_tree = None
        self.task_planner = None
        self.memory_system = None
        self.is_initialized = False

        self._initialize_cognition()
        self.is_initialized = True

    def _initialize_cognition(self):
        """Initialize cognition system components"""
        self.parent_node.get_logger().info('Initializing cognition system...')

        # Initialize behavior tree
        self.behavior_tree = BehaviorTree()

        # Initialize task planner
        self.task_planner = TaskPlanner()

        # Initialize memory system
        self.memory_system = MemorySystem()

    def process_perceptions(self, perception_data):
        """Process perception data and generate cognitive state"""
        if not self.is_initialized:
            return {}

        # Update memory with new perceptions
        self.memory_system.update(perception_data)

        # Plan tasks based on perceptions and goals
        current_goals = self.memory_system.get_goals()
        planned_tasks = self.task_planner.plan_tasks(
            perception_data, current_goals
        )

        # Execute behavior tree
        cognitive_state = self.behavior_tree.execute(
            perception_data, planned_tasks, self.memory_system
        )

        return cognitive_state

    def is_healthy(self):
        """Check if cognition system is healthy"""
        return self.is_initialized and self.behavior_tree is not None


class ControlSystem:
    """
    Control system component
    """

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.joint_states = {}
        self.odometry = None
        self.controllers = {}
        self.motion_planner = None
        self.is_initialized = False

        self._initialize_control()
        self.is_initialized = True

    def _initialize_control(self):
        """Initialize control system components"""
        self.parent_node.get_logger().info('Initializing control system...')

        # Initialize controllers for different joints/actuators
        self.controllers['navigation'] = NavigationController()
        self.controllers['manipulation'] = ManipulationController()
        self.controllers['balance'] = BalanceController()

        # Initialize motion planner
        self.motion_planner = MotionPlanner()

        # Create publishers for control commands
        self.joint_cmd_pub = self.parent_node.create_publisher(
            JointState, '/joint_commands', 10
        )
        self.cmd_vel_pub = self.parent_node.create_publisher(
            Twist, '/cmd_vel', 10
        )

    def update_joint_states(self, joint_state_msg: JointState):
        """Update joint state information"""
        for i, name in enumerate(joint_state_msg.name):
            if i < len(joint_state_msg.position):
                self.joint_states[name] = {
                    'position': joint_state_msg.position[i],
                    'velocity': joint_state_msg.velocity[i] if i < len(joint_state_msg.velocity) else 0.0,
                    'effort': joint_state_msg.effort[i] if i < len(joint_state_msg.effort) else 0.0
                }

    def update_odometry(self, odom_msg: Odometry):
        """Update odometry information"""
        self.odometry = {
            'position': [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ],
            'orientation': [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w
            ],
            'linear_velocity': [
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.linear.z
            ],
            'angular_velocity': [
                odom_msg.twist.twist.angular.x,
                odom_msg.twist.twist.angular.y,
                odom_msg.twist.twist.angular.z
            ]
        }

    def generate_commands(self, cognitive_state):
        """Generate control commands from cognitive state"""
        if not self.is_initialized:
            return {}

        commands = {}

        # Generate navigation commands if needed
        if 'navigation_goal' in cognitive_state:
            nav_cmd = self.controllers['navigation'].generate_command(
                cognitive_state['navigation_goal'],
                self.odometry
            )
            commands['navigation'] = nav_cmd

        # Generate manipulation commands if needed
        if 'manipulation_goal' in cognitive_state:
            manip_cmd = self.controllers['manipulation'].generate_command(
                cognitive_state['manipulation_goal'],
                self.joint_states
            )
            commands['manipulation'] = manip_cmd

        # Generate balance commands
        balance_cmd = self.controllers['balance'].generate_command(
            self.joint_states,
            self.odometry
        )
        commands['balance'] = balance_cmd

        return commands

    def execute_commands(self, commands):
        """Execute control commands"""
        if 'navigation' in commands:
            cmd_vel = commands['navigation']
            self.cmd_vel_pub.publish(cmd_vel)

        if 'manipulation' in commands:
            joint_cmd = commands['manipulation']
            self.joint_cmd_pub.publish(joint_cmd)

    def is_healthy(self):
        """Check if control system is healthy"""
        return self.is_initialized and len(self.joint_states) > 0


class InteractionSystem:
    """
    Human-robot interaction system component
    """

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.speech_recognizer = None
        self.speech_synthesizer = None
        self.gesture_detector = None
        self.is_initialized = False

        self._initialize_interaction()
        self.is_initialized = True

    def _initialize_interaction(self):
        """Initialize interaction system components"""
        self.parent_node.get_logger().info('Initializing interaction system...')

        # Initialize speech recognition
        self.speech_recognizer = SpeechRecognizer()

        # Initialize speech synthesis
        self.speech_synthesizer = SpeechSynthesizer()

        # Initialize gesture detection
        self.gesture_detector = GestureDetector()

        # Setup interaction interfaces
        self.speech_sub = self.parent_node.create_subscription(
            String, '/speech_input', self.speech_callback, 10
        )
        self.interaction_pub = self.parent_node.create_publisher(
            String, '/interaction_output', 10
        )

    def speech_callback(self, msg: String):
        """Handle speech input"""
        # Process speech command
        interpreted_command = self.speech_recognizer.interpret(msg.data)

        # Update cognitive system with command
        # This would typically go through the cognition system
        pass

    def update(self):
        """Update interaction system"""
        # Process any interaction updates
        pass

    def is_healthy(self):
        """Check if interaction system is healthy"""
        return self.is_initialized


class SafetySystem:
    """
    Safety system component
    """

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.safety_monitors = []
        self.emergency_stop_active = False
        self.safety_limits = {}
        self.is_initialized = False

        self._initialize_safety()
        self.is_initialized = True

    def _initialize_safety(self):
        """Initialize safety system components"""
        self.parent_node.get_logger().info('Initializing safety system...')

        # Initialize safety monitors
        self.safety_monitors.append(CollisionAvoidanceMonitor())
        self.safety_monitors.append(VelocityLimitMonitor())
        self.safety_monitors.append(ForceLimitMonitor())

        # Set safety limits
        self.safety_limits = {
            'max_velocity': 1.0,  # m/s
            'max_angular_velocity': 1.0,  # rad/s
            'max_joint_velocity': 2.0,  # rad/s
            'max_force': 100.0,  # N
            'min_distance': 0.5  # m
        }

    def update(self):
        """Update safety system"""
        if self.emergency_stop_active:
            return

        # Check all safety monitors
        for monitor in self.safety_monitors:
            if not monitor.check_safety():
                self.trigger_emergency_stop()
                self.parent_node.get_logger().error(
                    f'Safety violation detected by {monitor.__class__.__name__}'
                )
                break

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True

        # Publish emergency stop command
        emergency_msg = Bool()
        emergency_msg.data = True
        # self.parent_node.emergency_stop_pub.publish(emergency_msg)  # Would publish to the main node's publisher

        self.parent_node.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def clear_emergency_stop(self):
        """Clear emergency stop"""
        self.emergency_stop_active = False
        self.parent_node.get_logger().info('Emergency stop cleared')

    def is_healthy(self):
        """Check if safety system is healthy"""
        return self.is_initialized and not self.emergency_stop_active


# Supporting modules for the system
class ObjectDetectionModule:
    """Object detection module for perception system"""

    def __init__(self):
        # Initialize object detection model
        # This would load a pre-trained model like YOLO, SSD, etc.
        pass

    def detect(self, image):
        """Detect objects in image"""
        # This would run the object detection model
        # For now, return mock detections
        return [
            {'class': 'person', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
            {'class': 'chair', 'bbox': [300, 200, 400, 300], 'confidence': 0.85}
        ]


class LocalizationModule:
    """Localization module for perception system"""

    def __init__(self):
        # Initialize localization algorithms
        pass

    def get_pose(self):
        """Get robot pose in map"""
        # This would return the current robot pose
        return {'x': 0.0, 'y': 0.0, 'theta': 0.0}


class MappingModule:
    """Mapping module for perception system"""

    def __init__(self):
        # Initialize mapping algorithms
        pass

    def get_map(self):
        """Get current environment map"""
        # This would return the current map
        return {'resolution': 0.05, 'data': []}


class BehaviorTree:
    """Behavior tree for cognition system"""

    def __init__(self):
        # Initialize behavior tree structure
        pass

    def execute(self, perception_data, tasks, memory):
        """Execute behavior tree"""
        # This would execute the behavior tree logic
        return {'action': 'move_forward', 'parameters': {}}


class TaskPlanner:
    """Task planner for cognition system"""

    def __init__(self):
        # Initialize task planning algorithms
        pass

    def plan_tasks(self, perception_data, goals):
        """Plan tasks to achieve goals"""
        # This would generate a task plan
        return [{'task': 'navigate', 'goal': [1.0, 1.0, 0.0]}]


class MemorySystem:
    """Memory system for cognition system"""

    def __init__(self):
        self.memory = {}

    def update(self, perception_data):
        """Update memory with new perceptions"""
        self.memory['last_perception'] = perception_data
        self.memory['timestamp'] = time.time()

    def get_goals(self):
        """Get current goals"""
        return self.memory.get('goals', [])


class NavigationController:
    """Navigation controller for control system"""

    def __init__(self):
        # Initialize navigation controller
        pass

    def generate_command(self, goal, odometry):
        """Generate navigation command"""
        cmd = Twist()
        # Calculate appropriate velocities based on goal and current pose
        cmd.linear.x = 0.5  # Example velocity
        cmd.angular.z = 0.1  # Example angular velocity
        return cmd


class ManipulationController:
    """Manipulation controller for control system"""

    def __init__(self):
        # Initialize manipulation controller
        pass

    def generate_command(self, goal, joint_states):
        """Generate manipulation command"""
        cmd = JointState()
        # Calculate appropriate joint commands based on goal
        cmd.name = list(joint_states.keys())
        cmd.position = [0.0] * len(cmd.name)  # Example positions
        return cmd


class BalanceController:
    """Balance controller for control system"""

    def __init__(self):
        # Initialize balance controller
        pass

    def generate_command(self, joint_states, odometry):
        """Generate balance command"""
        # Calculate balance corrections based on current state
        cmd = JointState()
        cmd.name = list(joint_states.keys())
        cmd.position = [0.0] * len(cmd.name)  # Example balance corrections
        return cmd


class MotionPlanner:
    """Motion planner for control system"""

    def __init__(self):
        # Initialize motion planning algorithms
        pass


class SpeechRecognizer:
    """Speech recognition module for interaction system"""

    def __init__(self):
        # Initialize speech recognition
        pass

    def interpret(self, speech_text):
        """Interpret speech text into commands"""
        # This would parse natural language into robot commands
        return {'command': 'interpret', 'text': speech_text}


class SpeechSynthesizer:
    """Speech synthesis module for interaction system"""

    def __init__(self):
        # Initialize speech synthesis
        pass


class GestureDetector:
    """Gesture detection module for interaction system"""

    def __init__(self):
        # Initialize gesture detection
        pass


class CollisionAvoidanceMonitor:
    """Collision avoidance monitor for safety system"""

    def __init__(self):
        # Initialize collision detection algorithms
        pass

    def check_safety(self):
        """Check for collision safety"""
        # This would check for potential collisions
        return True  # Assume safe for now


class VelocityLimitMonitor:
    """Velocity limit monitor for safety system"""

    def __init__(self):
        # Initialize velocity monitoring
        pass

    def check_safety(self):
        """Check velocity limits"""
        # This would check if velocities are within limits
        return True  # Assume within limits for now


class ForceLimitMonitor:
    """Force limit monitor for safety system"""

    def __init__(self):
        # Initialize force monitoring
        pass

    def check_safety(self):
        """Check force limits"""
        # This would check if forces are within limits
        return True  # Assume within limits for now


def main(args=None):
    """Main function to run the Physical AI system"""
    rclpy.init(args=args)

    try:
        # Create system node
        system = PhysicalAISystem()

        # Spin in a separate thread
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(system)

        # Start system execution in background
        system_thread = threading.Thread(target=system.run_system)
        system_thread.daemon = True
        system_thread.start()

        # Run executor
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            system.system_state = 'shutdown'
            executor.shutdown()
            system.destroy_node()

    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Phase 3: Advanced Component Implementation

### 3.1 Perception System Implementation

```python
# physical_ai_system/src/perception_system.py
"""
Advanced perception system implementation
"""

import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy

class AdvancedPerceptionSystem:
    """
    Advanced perception system with AI integration
    """

    def __init__(self, node):
        self.node = node
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize AI models
        self.object_detector = self._initialize_object_detector()
        self.segmentation_model = self._initialize_segmentation_model()
        self.depth_estimator = self._initialize_depth_estimator()
        self.pose_estimator = self._initialize_pose_estimator()

        # Data buffers
        self.image_buffer = []
        self.point_cloud_buffer = []
        self.max_buffer_size = 5

        # Feature extraction
        self.feature_extractor = FeatureExtractor()

        # Object tracking
        self.object_tracker = ObjectTracker()

        # Scene understanding
        self.scene_analyzer = SceneAnalyzer()

    def _initialize_object_detector(self):
        """
        Initialize object detection model
        """
        try:
            # Load pre-trained model (e.g., YOLOv5, Detectron2, etc.)
            import torch.hub
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(self.device)
            model.eval()
            self.node.get_logger().info('Object detector initialized')
            return model
        except Exception as e:
            self.node.get_logger().error(f'Failed to initialize object detector: {e}')
            return None

    def _initialize_segmentation_model(self):
        """
        Initialize semantic segmentation model
        """
        try:
            # Load segmentation model (e.g., DeepLab, UNet, etc.)
            import torchvision.models.segmentation as segmentation_models
            model = segmentation_models.deeplabv3_resnet101(pretrained=True)
            model.to(self.device)
            model.eval()
            self.node.get_logger().info('Segmentation model initialized')
            return model
        except Exception as e:
            self.node.get_logger().error(f'Failed to initialize segmentation model: {e}')
            return None

    def _initialize_depth_estimator(self):
        """
        Initialize depth estimation model
        """
        try:
            # Load depth estimation model (e.g., MiDaS, etc.)
            import torch.hub
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
            model.to(self.device)
            model.eval()
            self.node.get_logger().info('Depth estimator initialized')
            return model
        except Exception as e:
            self.node.get_logger().error(f'Failed to initialize depth estimator: {e}')
            return None

    def _initialize_pose_estimator(self):
        """
        Initialize human pose estimation model
        """
        try:
            # Load pose estimation model (e.g., OpenPose, etc.)
            # For this example, we'll use a simple placeholder
            self.node.get_logger().info('Pose estimator initialized')
            return True  # Placeholder
        except Exception as e:
            self.node.get_logger().error(f'Failed to initialize pose estimator: {e}')
            return None

    def process_camera_image(self, image_msg):
        """
        Process camera image and extract perception information
        """
        try:
            # Convert ROS image to OpenCV format
            image = self._ros_image_to_cv2(image_msg)

            # Store in buffer
            self._add_to_buffer(self.image_buffer, image)

            # Run object detection
            detections = self._run_object_detection(image)

            # Run semantic segmentation
            segmentation = self._run_segmentation(image)

            # Run depth estimation
            depth_map = self._run_depth_estimation(image)

            # Extract features
            features = self.feature_extractor.extract_features(image)

            # Track objects
            tracked_objects = self.object_tracker.update(detections)

            # Analyze scene
            scene_info = self.scene_analyzer.analyze(
                image, detections, segmentation, depth_map
            )

            # Create perception result
            perception_result = {
                'timestamp': image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9,
                'image': image,
                'detections': detections,
                'segmentation': segmentation,
                'depth_map': depth_map,
                'features': features,
                'tracked_objects': tracked_objects,
                'scene_analysis': scene_info,
                'frame_id': image_msg.header.frame_id
            }

            return perception_result

        except Exception as e:
            self.node.get_logger().error(f'Error processing camera image: {e}')
            return None

    def _ros_image_to_cv2(self, image_msg):
        """
        Convert ROS Image message to OpenCV image
        """
        import ros_numpy
        # Convert using ros_numpy or similar
        image = ros_numpy.numpify(image_msg)
        return image

    def _run_object_detection(self, image):
        """
        Run object detection on image
        """
        if self.object_detector is None:
            return []

        # Preprocess image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.object_detector(img_rgb)

        # Extract detections
        detections = []
        for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
            x1, y1, x2, y2, conf, cls = detection
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.object_detector.names[int(cls)],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            })

        return detections

    def _run_segmentation(self, image):
        """
        Run semantic segmentation on image
        """
        if self.segmentation_model is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Preprocess image
        img_tensor = self._preprocess_for_segmentation(image)

        with torch.no_grad():
            output = self.segmentation_model(img_tensor)
            predicted = torch.argmax(output['out'], dim=1)
            segmentation_map = predicted.squeeze().cpu().numpy()

        return segmentation_map

    def _run_depth_estimation(self, image):
        """
        Run depth estimation on image
        """
        if self.depth_estimator is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Preprocess image
        img_tensor = self._preprocess_for_depth(image)

        with torch.no_grad():
            prediction = self.depth_estimator(img_tensor)
            depth_map = prediction.squeeze().cpu().numpy()

        return depth_map

    def _preprocess_for_segmentation(self, image):
        """
        Preprocess image for segmentation model
        """
        # Resize and normalize image
        img_resized = cv2.resize(image, (512, 512))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _preprocess_for_depth(self, image):
        """
        Preprocess image for depth estimation model
        """
        # Resize and normalize image
        img_resized = cv2.resize(image, (384, 384))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _add_to_buffer(self, buffer, item):
        """
        Add item to circular buffer
        """
        buffer.append(item)
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)

    def process_point_cloud(self, point_cloud_msg):
        """
        Process point cloud data
        """
        try:
            # Extract point cloud data
            points = list(pc2.read_points(
                point_cloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            ))

            # Store in buffer
            self._add_to_buffer(self.point_cloud_buffer, points)

            # Process point cloud
            processed_data = {
                'points': points,
                'num_points': len(points),
                'bounds': self._calculate_bounds(points),
                'clusters': self._cluster_points(points)
            }

            return processed_data

        except Exception as e:
            self.node.get_logger().error(f'Error processing point cloud: {e}')
            return None

    def _calculate_bounds(self, points):
        """
        Calculate bounding box for point cloud
        """
        if not points:
            return None

        pts_array = np.array(points)
        min_bounds = np.min(pts_array, axis=0)
        max_bounds = np.max(pts_array, axis=0)

        return {
            'min': min_bounds.tolist(),
            'max': max_bounds.tolist(),
            'center': ((min_bounds + max_bounds) / 2).tolist()
        }

    def _cluster_points(self, points, eps=0.1, min_samples=10):
        """
        Cluster points using DBSCAN
        """
        from sklearn.cluster import DBSCAN

        if len(points) < min_samples:
            return []

        pts_array = np.array(points)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_array)
        labels = clustering.labels_

        clusters = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue

            cluster_points = pts_array[labels == label]
            clusters.append({
                'label': int(label),
                'points': cluster_points.tolist(),
                'center': np.mean(cluster_points, axis=0).tolist(),
                'size': len(cluster_points)
            })

        return clusters

    def fuse_sensor_data(self, camera_data, lidar_data, imu_data):
        """
        Fuse data from multiple sensors
        """
        fused_data = {
            'timestamp': max(
                camera_data.get('timestamp', 0),
                lidar_data.get('timestamp', 0),
                imu_data.get('timestamp', 0)
            ),
            'environment_map': self._create_environment_map(camera_data, lidar_data),
            'object_locations': self._locate_objects_3d(
                camera_data['detections'],
                depth_map=camera_data['depth_map'],
                lidar_data=lidar_data
            ),
            'robot_pose': self._estimate_robot_pose(imu_data),
            'traversability': self._analyze_traversability(
                camera_data['segmentation'],
                lidar_data['clusters']
            )
        }

        return fused_data

    def _create_environment_map(self, camera_data, lidar_data):
        """
        Create environment map from camera and lidar data
        """
        # This would integrate camera semantic segmentation with lidar point clouds
        # to create a 3D semantic map
        return {
            'semantic_map': camera_data.get('segmentation'),
            'obstacle_map': self._create_obstacle_map(lidar_data),
            'traversable_areas': []
        }

    def _create_obstacle_map(self, lidar_data):
        """
        Create obstacle map from lidar data
        """
        if 'clusters' in lidar_data:
            obstacles = []
            for cluster in lidar_data['clusters']:
                obstacles.append({
                    'center': cluster['center'],
                    'size': cluster['size'],
                    'type': 'unknown'
                })
            return obstacles

        return []

    def _locate_objects_3d(self, detections, depth_map=None, lidar_data=None):
        """
        Locate objects in 3D space
        """
        objects_3d = []

        for detection in detections:
            bbox = detection['bbox']
            center_2d = detection['center']

            # Estimate 3D position using depth
            if depth_map is not None:
                center_x, center_y = int(center_2d[0]), int(center_2d[1])
                if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                    depth = depth_map[center_y, center_x]

                    # Convert 2D pixel coordinates to 3D world coordinates
                    # This requires camera intrinsic parameters
                    # For now, return a simplified estimate
                    position_3d = [
                        (center_x - 320) * depth * 0.001,  # Approximate conversion
                        (center_y - 240) * depth * 0.001,  # Approximate conversion
                        depth
                    ]

                    objects_3d.append({
                        'class': detection['class_name'],
                        'confidence': detection['confidence'],
                        'position_2d': center_2d,
                        'position_3d': position_3d,
                        'bbox': bbox
                    })

        return objects_3d

    def _estimate_robot_pose(self, imu_data):
        """
        Estimate robot pose from IMU data
        """
        # This would integrate IMU data over time to estimate pose
        # For now, return a simplified estimate
        return {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # Quaternion (w, x, y, z)
            'velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }

    def _analyze_traversability(self, segmentation_map, lidar_clusters):
        """
        Analyze environment traversability
        """
        traversability_map = np.ones_like(segmentation_map, dtype=np.float32)

        # Mark areas as non-traversable based on segmentation
        # This is a simplified example - in practice, this would be more sophisticated
        for cluster in lidar_clusters:
            if cluster['size'] > 100:  # Large clusters likely obstacles
                # Mark area as non-traversable
                # This would require mapping 3D clusters to 2D image coordinates
                pass

        return traversability_map


class FeatureExtractor:
    """
    Feature extraction module for perception system
    """

    def __init__(self):
        # Initialize feature extraction models
        pass

    def extract_features(self, image):
        """
        Extract features from image
        """
        # Extract various features
        features = {
            'edges': self._extract_edges(image),
            'corners': self._extract_corners(image),
            'descriptors': self._extract_descriptors(image),
            'texture': self._extract_texture_features(image)
        }

        return features

    def _extract_edges(self, image):
        """
        Extract edges from image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def _extract_corners(self, image):
        """
        Extract corners from image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        return corners

    def _extract_descriptors(self, image):
        """
        Extract feature descriptors from image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return {'keypoints': keypoints, 'descriptors': descriptors}

    def _extract_texture_features(self, image):
        """
        Extract texture features from image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Local Binary Pattern (LBP) features
        lbp = self._calculate_lbp(gray)

        # Calculate Haralick texture features
        haralick = self._calculate_haralick(gray)

        return {
            'lbp': lbp,
            'haralick': haralick
        }

    def _calculate_lbp(self, image):
        """
        Calculate Local Binary Pattern
        """
        # Simplified LBP calculation
        lbp_image = np.zeros_like(image)

        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                binary_pattern = 0

                # Compare with 8 neighbors
                for k, (di, dj) in enumerate([(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]):
                    neighbor = image[i+di, j+dj]
                    if neighbor >= center:
                        binary_pattern |= (1 << k)

                lbp_image[i, j] = binary_pattern

        return lbp_image

    def _calculate_haralick(self, image):
        """
        Calculate Haralick texture features
        """
        # This would calculate GLCM-based texture features
        # For brevity, returning a placeholder
        return {'contrast': 0.0, 'correlation': 0.0, 'energy': 0.0, 'homogeneity': 0.0}


class ObjectTracker:
    """
    Object tracking module for perception system
    """

    def __init__(self):
        # Initialize tracker
        self.trackers = {}  # {track_id: tracker_object}
        self.next_track_id = 0
        self.max_displacement = 50  # pixels
        self.iou_threshold = 0.3

    def update(self, detections):
        """
        Update object trackers with new detections
        """
        if not detections:
            return []

        # Create new trackers for detections without matches
        unassigned_detections = list(range(len(detections)))
        tracked_objects = []

        # Update existing trackers
        for track_id, tracker in list(self.trackers.items()):
            # Update tracker with prediction
            predicted_bbox = tracker.predict()

            # Find best matching detection
            best_match_idx = self._find_best_match(predicted_bbox, detections, unassigned_detections)

            if best_match_idx is not None:
                # Update tracker with matched detection
                detection = detections[best_match_idx]
                tracker.update(detection['bbox'])

                # Add to tracked objects
                tracked_objects.append({
                    'id': track_id,
                    'bbox': detection['bbox'],
                    'class': detection['class_name'],
                    'confidence': detection['confidence'],
                    'center': detection['center']
                })

                # Remove from unassigned
                unassigned_detections.remove(best_match_idx)
            else:
                # No match found, decrease confidence
                tracker.decrease_confidence()

                # Remove tracker if confidence too low
                if tracker.confidence < 0.1:
                    del self.trackers[track_id]

        # Create new trackers for unassigned detections
        for det_idx in unassigned_detections:
            detection = detections[det_idx]
            new_track_id = self._create_new_tracker(detection)

            tracked_objects.append({
                'id': new_track_id,
                'bbox': detection['bbox'],
                'class': detection['class_name'],
                'confidence': detection['confidence'],
                'center': detection['center']
            })

        return tracked_objects

    def _find_best_match(self, predicted_bbox, detections, unassigned_indices):
        """
        Find best matching detection for predicted bounding box
        """
        best_iou = 0
        best_idx = None

        for idx in unassigned_indices:
            detection = detections[idx]
            iou = self._calculate_iou(predicted_bbox, detection['bbox'])

            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_idx = idx

        return best_idx

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x1_int < x2_int and y1_int < y2_int:
            intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area

            return intersection_area / union_area if union_area > 0 else 0
        else:
            return 0

    def _create_new_tracker(self, detection):
        """
        Create new tracker for detection
        """
        track_id = self.next_track_id
        self.next_track_id += 1

        # Create and store tracker
        tracker = SimpleTracker(detection['bbox'])
        self.trackers[track_id] = tracker

        return track_id


class SimpleTracker:
    """
    Simple object tracker implementation
    """

    def __init__(self, initial_bbox):
        self.bbox = initial_bbox
        self.confidence = 1.0
        self.age = 0

    def predict(self):
        """
        Predict next position (simple implementation)
        """
        # In a real implementation, this would use motion models
        return self.bbox

    def update(self, bbox):
        """
        Update tracker with new bounding box
        """
        self.bbox = bbox
        self.confidence = min(1.0, self.confidence + 0.1)  # Increase confidence
        self.age += 1

    def decrease_confidence(self):
        """
        Decrease confidence when no match found
        """
        self.confidence = max(0.0, self.confidence - 0.2)


class SceneAnalyzer:
    """
    Scene analysis module for perception system
    """

    def __init__(self):
        pass

    def analyze(self, image, detections, segmentation, depth_map):
        """
        Analyze scene for high-level understanding
        """
        scene_info = {
            'layout': self._analyze_layout(segmentation),
            'objects_relationships': self._analyze_object_relationships(detections),
            'activity': self._infer_activity(detections, segmentation),
            'context': self._analyze_context(image, detections, segmentation)
        }

        return scene_info

    def _analyze_layout(self, segmentation):
        """
        Analyze scene layout from segmentation
        """
        # Identify different regions in the scene
        unique_labels = np.unique(segmentation)
        regions = {}

        for label in unique_labels:
            mask = segmentation == label
            region_area = np.sum(mask)

            if region_area > 100:  # Only consider significant regions
                y_coords, x_coords = np.where(mask)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)

                regions[label] = {
                    'area': int(region_area),
                    'center': [float(center_x), float(center_y)],
                    'bbox': [
                        int(np.min(x_coords)), int(np.min(y_coords)),
                        int(np.max(x_coords)), int(np.max(y_coords))
                    ]
                }

        return regions

    def _analyze_object_relationships(self, detections):
        """
        Analyze relationships between detected objects
        """
        relationships = []

        for i, obj1 in enumerate(detections):
            for j, obj2 in enumerate(detections):
                if i != j:
                    # Calculate spatial relationship
                    center1 = np.array(obj1['center'])
                    center2 = np.array(obj2['center'])
                    distance = np.linalg.norm(center1 - center2)

                    # Determine relationship based on distance and classes
                    if distance < 100:  # pixels
                        relationship = {
                            'object1': obj1['class_name'],
                            'object2': obj2['class_name'],
                            'relationship': 'near',
                            'distance': float(distance)
                        }
                        relationships.append(relationship)

        return relationships

    def _infer_activity(self, detections, segmentation):
        """
        Infer activities from detected objects and scene
        """
        # Simple activity inference based on object co-occurrence
        person_count = sum(1 for det in detections if det['class_name'] == 'person')
        chair_count = sum(1 for det in detections if det['class_name'] == 'chair')
        table_count = sum(1 for det in detections if det['class_name'] == 'table')

        if person_count > 0 and chair_count > 0 and table_count > 0:
            return 'working_or_meeting'
        elif person_count > 0 and chair_count > 0:
            return 'sitting'
        elif person_count > 0:
            return 'standing'
        else:
            return 'empty'

    def _analyze_context(self, image, detections, segmentation):
        """
        Analyze scene context
        """
        # Analyze context based on dominant objects and scene layout
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Determine scene context based on object composition
        if 'bed' in class_counts or 'bedroom' in class_counts:
            context = 'bedroom'
        elif 'kitchen' in class_counts or 'refrigerator' in class_counts:
            context = 'kitchen'
        elif 'desk' in class_counts or 'office' in class_counts:
            context = 'office'
        elif 'person' in class_counts and 'chair' in class_counts:
            context = 'living_space'
        else:
            context = 'unknown'

        return {
            'type': context,
            'dominant_objects': sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'complexity': len(detections)
        }
```

## Phase 4: Integration and Testing

### 4.1 System Integration Testing

```python
# integration_test.py
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
import rclpy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
import time

class TestPhysicalAISystemIntegration(unittest.TestCase):
    """
    Integration tests for the Physical AI system
    """

    def setUp(self):
        """
        Set up test environment
        """
        rclpy.init()

        # Create mock node for testing
        self.node = Mock()
        self.node.get_logger = Mock()
        self.node.get_logger.return_value.info = Mock()
        self.node.get_logger.return_value.warn = Mock()
        self.node.get_logger.return_value.error = Mock()

        # Create system instance
        self.system = PhysicalAISystem(self.node)

    def tearDown(self):
        """
        Clean up test environment
        """
        rclpy.shutdown()

    def test_perception_control_integration(self):
        """
        Test integration between perception and control systems
        """
        # Create mock perception data
        mock_perception_data = {
            'environment_map': {'resolution': 0.05, 'data': [0] * 100},
            'object_locations': [
                {'class': 'person', 'position_2d': [320, 240], 'confidence': 0.9}
            ],
            'robot_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'obstacle_distances': [1.0, 1.5, 2.0, 2.5, 3.0] * 72  # 360 degrees at 5-degree intervals
        }

        # Mock cognition system to return navigation goal
        self.system.cognition_system = Mock()
        self.system.cognition_system.process_perceptions = Mock()
        self.system.cognition_system.process_perceptions.return_value = {
            'navigation_goal': [1.0, 1.0, 0.0],
            'task': 'follow_person'
        }

        # Mock control system
        self.system.control_system = Mock()
        self.system.control_system.generate_commands = Mock()
        self.system.control_system.generate_commands.return_value = {
            'navigation': Twist(),
            'balance': JointState()
        }
        self.system.control_system.execute_commands = Mock()

        # Process perception data through cognition and control
        cognitive_state = self.system.cognition_system.process_perceptions(mock_perception_data)
        control_commands = self.system.control_system.generate_commands(cognitive_state)
        self.system.control_system.execute_commands(control_commands)

        # Verify interactions occurred
        self.system.cognition_system.process_perceptions.assert_called_once()
        self.system.control_system.generate_commands.assert_called_once()
        self.system.control_system.execute_commands.assert_called_once()

    def test_safety_integration(self):
        """
        Test integration between safety system and other components
        """
        # Mock emergency stop condition
        self.system.emergency_stop = True
        self.system.safety_system = Mock()
        self.system.safety_system.trigger_emergency_stop = Mock()

        # Try to run system - should not execute due to emergency stop
        self.system.run_system()

        # Verify safety system was called
        self.system.safety_system.trigger_emergency_stop.assert_called()

    def test_multi_sensor_fusion(self):
        """
        Test fusion of multiple sensor inputs
        """
        # Create perception system
        perception = AdvancedPerceptionSystem(self.node)

        # Create mock sensor data
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_lidar_data = [1.0, 1.5, 2.0] * 120  # 360 points
        mock_imu_data = {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'angular_velocity': [0.0, 0.0, 0.0],
            'linear_acceleration': [0.0, 0.0, -9.81]
        }

        # Mock the image processing function
        perception._ros_image_to_cv2 = Mock(return_value=mock_image)

        # Process sensor data
        camera_result = {
            'timestamp': time.time(),
            'image': mock_image,
            'detections': [{'class_name': 'person', 'confidence': 0.9}],
            'segmentation': np.zeros((480, 640)),
            'depth_map': np.ones((480, 640)) * 2.0
        }

        lidar_result = {
            'timestamp': time.time(),
            'clusters': [{'center': [1.0, 0.0, 0.0], 'size': 10, 'type': 'obstacle'}]
        }

        # Fuse sensor data
        fused_data = perception.fuse_sensor_data(
            camera_result, lidar_result, mock_imu_data
        )

        # Verify fusion occurred
        self.assertIn('environment_map', fused_data)
        self.assertIn('object_locations', fused_data)
        self.assertIn('robot_pose', fused_data)

    def test_real_time_performance(self):
        """
        Test real-time performance constraints
        """
        # Create perception system
        perception = AdvancedPerceptionSystem(self.node)

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Measure processing time
        start_time = time.time()
        perception_result = perception.process_camera_image(test_image)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within 100ms for real-time performance
        self.assertLess(processing_time, 0.1,
                       f"Processing took {processing_time:.3f}s, exceeds 100ms limit")

    def test_error_handling(self):
        """
        Test error handling in system components
        """
        # Test perception system with invalid image
        perception = AdvancedPerceptionSystem(self.node)
        invalid_result = perception.process_camera_image(None)

        # Should return None for invalid input
        self.assertIsNone(invalid_result)

        # Test with empty image
        empty_image = np.array([])
        empty_result = perception.process_camera_image(empty_image)

        # Should handle gracefully
        self.assertIsNone(empty_result)

    def test_data_flow_consistency(self):
        """
        Test consistency of data flow between components
        """
        # Create mock data pipeline
        mock_perception = {
            'objects': [{'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}],
            'camera_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }

        # Mock cognition system
        cognitive_state = {
            'intent': 'follow_person',
            'target_object': mock_perception['objects'][0],
            'navigation_goal': [2.0, 2.0, 0.0]
        }

        # Mock control system
        control_commands = {
            'navigation': {'linear': [0.5, 0.0, 0.0], 'angular': [0.0, 0.0, 0.1]},
            'manipulation': {'joint_positions': [0.0] * 7}
        }

        # Verify data flows correctly
        self.assertEqual(cognitive_state['target_object']['class'], 'person')
        self.assertEqual(control_commands['navigation']['linear'][0], 0.5)
        self.assertEqual(len(control_commands['manipulation']['joint_positions']), 7)


class PerformanceTestSuite:
    """
    Performance testing suite for Physical AI system
    """

    def __init__(self):
        self.test_results = []
        self.metrics = {
            'processing_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'throughput': []
        }

    def run_performance_tests(self):
        """
        Run comprehensive performance tests
        """
        import psutil
        import time

        print("Running Performance Tests...")

        # Test 1: Perception processing speed
        self.test_perception_speed()

        # Test 2: Control loop frequency
        self.test_control_frequency()

        # Test 3: Memory usage over time
        self.test_memory_usage()

        # Test 4: System throughput
        self.test_throughput()

        # Generate performance report
        self.generate_performance_report()

    def test_perception_speed(self):
        """
        Test perception processing speed
        """
        print("Testing perception processing speed...")

        # Create test images of different sizes
        test_sizes = [(320, 240), (640, 480), (1280, 720)]

        for width, height in test_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # Measure processing time
            start_time = time.perf_counter()
            for _ in range(10):  # Average over 10 runs
                result = self.process_single_image(test_image)
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / 10
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')

            print(f"  {width}x{height}: {avg_time:.4f}s ({fps:.1f} FPS)")

            self.metrics['processing_time'].append({
                'size': f"{width}x{height}",
                'time': avg_time,
                'fps': fps
            })

    def test_control_frequency(self):
        """
        Test control loop frequency
        """
        print("Testing control loop frequency...")

        # Simulate control loop execution
        start_time = time.time()
        loop_count = 0
        target_duration = 10.0  # seconds

        while time.time() - start_time < target_duration:
            self.simulate_control_loop()
            loop_count += 1

        actual_duration = time.time() - start_time
        frequency = loop_count / actual_duration

        print(f"  Achieved {frequency:.1f} Hz control frequency")

        self.metrics['throughput'].append({
            'type': 'control_frequency',
            'frequency': frequency,
            'duration': actual_duration
        })

    def test_memory_usage(self):
        """
        Test memory usage over time
        """
        print("Testing memory usage...")

        # Monitor memory over time
        memory_readings = []
        start_time = time.time()

        for _ in range(100):  # Take 100 readings
            memory_percent = psutil.virtual_memory().percent
            memory_readings.append(memory_percent)
            time.sleep(0.1)

        avg_memory = sum(memory_readings) / len(memory_readings)
        peak_memory = max(memory_readings)

        print(f"  Average memory usage: {avg_memory:.1f}%")
        print(f"  Peak memory usage: {peak_memory:.1f}%")

        self.metrics['memory_usage'].append({
            'average': avg_memory,
            'peak': peak_memory,
            'readings': memory_readings
        })

    def test_throughput(self):
        """
        Test system throughput
        """
        print("Testing system throughput...")

        # Test message processing throughput
        import queue
        import threading

        msg_queue = queue.Queue()
        processed_count = 0
        start_time = time.time()
        duration = 5.0  # seconds

        def message_processor():
            nonlocal processed_count
            while time.time() - start_time < duration:
                try:
                    msg = msg_queue.get(timeout=0.01)
                    # Simulate message processing
                    time.sleep(0.001)  # Simulate processing time
                    processed_count += 1
                    msg_queue.task_done()
                except queue.Empty:
                    continue

        # Start processor thread
        processor_thread = threading.Thread(target=message_processor)
        processor_thread.start()

        # Generate messages
        msg_count = 0
        while time.time() - start_time < duration:
            try:
                msg_queue.put(f"message_{msg_count}")
                msg_count += 1
                time.sleep(0.0001)  # Small delay to prevent overwhelming
            except:
                break

        # Wait for processing to complete
        msg_queue.join()
        processor_thread.join()

        throughput = processed_count / duration

        print(f"  Processed {processed_count} messages in {duration}s")
        print(f"  Throughput: {throughput:.1f} messages/second")

        self.metrics['throughput'].append({
            'type': 'message_throughput',
            'processed_count': processed_count,
            'duration': duration,
            'throughput': throughput
        })

    def process_single_image(self, image):
        """
        Process single image for performance testing
        """
        # Simulate image processing
        # In real implementation, this would call perception system
        time.sleep(0.01)  # Simulate processing time
        return {'objects': [], 'features': []}

    def simulate_control_loop(self):
        """
        Simulate control loop execution
        """
        # Simulate control computation
        time.sleep(0.005)  # Simulate control computation time

    def generate_performance_report(self):
        """
        Generate performance test report
        """
        report = []
        report.append("# Performance Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Processing time results
        report.append("## Processing Time Results")
        for metric in self.metrics['processing_time']:
            report.append(f"- {metric['size']}: {metric['fps']:.1f} FPS ({metric['time']:.4f}s)")
        report.append("")

        # Memory usage results
        if self.metrics['memory_usage']:
            mem_metric = self.metrics['memory_usage'][0]
            report.append("## Memory Usage Results")
            report.append(f"- Average: {mem_metric['average']:.1f}%")
            report.append(f"- Peak: {mem_metric['peak']:.1f}%")
            report.append("")

        # Throughput results
        report.append("## Throughput Results")
        for metric in self.metrics['throughput']:
            if metric['type'] == 'control_frequency':
                report.append(f"- Control Frequency: {metric['frequency']:.1f} Hz")
            elif metric['type'] == 'message_throughput':
                report.append(f"- Message Throughput: {metric['throughput']:.1f} msg/sec")
        report.append("")

        # Overall assessment
        report.append("## Performance Assessment")

        # Check if performance meets requirements
        fps_ok = any(m['fps'] >= 30 for m in self.metrics['processing_time'])
        control_freq_ok = any(m['frequency'] >= 50 for m in self.metrics['throughput']
                             if m['type'] == 'control_frequency')
        mem_ok = any(m['average'] <= 80 for m in self.metrics['memory_usage'])

        report.append(f"- Real-time processing: {'PASS' if fps_ok else 'FAIL'}")
        report.append(f"- Control frequency: {'PASS' if control_freq_ok else 'FAIL'}")
        report.append(f"- Memory usage: {'PASS' if mem_ok else 'FAIL'}")

        performance_report = "\n".join(report)
        print(performance_report)

        # Save report to file
        with open('performance_report.txt', 'w') as f:
            f.write(performance_report)

        print("Performance report saved to performance_report.txt")


def run_integration_tests():
    """
    Run all integration tests
    """
    print("Running Integration Tests...")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhysicalAISystemIntegration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run performance tests
    perf_tester = PerformanceTestSuite()
    perf_tester.run_performance_tests()

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
```

## Phase 5: Safety and Validation

### 5.1 Safety System Implementation

```python
# safety_validation.py
import threading
import time
from enum import Enum
import numpy as np

class SafetyLevel(Enum):
    SAFE = 1
    WARNING = 2
    EMERGENCY = 3
    CRITICAL = 4

class SafetyValidator:
    """
    Comprehensive safety validation system for Physical AI
    """

    def __init__(self):
        self.safety_level = SafetyLevel.SAFE
        self.emergency_stop_active = False
        self.safety_monitors = []
        self.safety_limits = {}
        self.violation_log = []
        self.is_active = True

        self._initialize_safety_monitors()
        self._set_safety_limits()

    def _initialize_safety_monitors(self):
        """
        Initialize all safety monitoring systems
        """
        self.safety_monitors = [
            CollisionAvoidanceMonitor(self),
            VelocityLimitMonitor(self),
            ForceLimitMonitor(self),
            JointLimitMonitor(self),
            StabilityMonitor(self),
            EnvironmentalSafetyMonitor(self)
        ]

    def _set_safety_limits(self):
        """
        Set safety limits for the system
        """
        self.safety_limits = {
            'velocity': {
                'linear_max': 1.0,      # m/s
                'angular_max': 1.0,     # rad/s
                'joint_max': 5.0        # rad/s
            },
            'force': {
                'max_contact_force': 100.0,  # N
                'max_gripper_force': 50.0    # N
            },
            'position': {
                'workspace_bounds': [-5, 5, -5, 5, 0, 3],  # [xmin, xmax, ymin, ymax, zmin, zmax]
                'joint_limits': {
                    'hip_yaw': [-1.57, 1.57],
                    'hip_roll': [-0.785, 0.785],
                    'hip_pitch': [-1.57, 1.57],
                    'knee': [0.0, 2.356],
                    'ankle_pitch': [-0.785, 0.785],
                    'ankle_roll': [-0.349, 0.349]
                }
            },
            'stability': {
                'max_tilt_angle': 15.0,  # degrees
                'min_support_polygon': 0.1  # m^2
            },
            'environment': {
                'min_obstacle_distance': 0.5,  # m
                'max_slope_angle': 30.0      # degrees
            }
        }

    def validate_system_state(self, system_state):
        """
        Validate current system state against safety requirements
        """
        if not self.is_active:
            return True

        violations = []

        # Check each safety monitor
        for monitor in self.safety_monitors:
            try:
                monitor_violations = monitor.check_safety(system_state)
                violations.extend(monitor_violations)
            except Exception as e:
                violations.append({
                    'monitor': monitor.__class__.__name__,
                    'violation': f'Monitor error: {str(e)}',
                    'severity': 'critical'
                })

        # Log violations
        if violations:
            self._log_violations(violations)

            # Determine overall safety level
            critical_violations = [v for v in violations if v['severity'] == 'critical']
            warning_violations = [v for v in violations if v['severity'] == 'warning']

            if critical_violations:
                self.safety_level = SafetyLevel.CRITICAL
                self.trigger_emergency_stop()
            elif warning_violations:
                self.safety_level = SafetyLevel.WARNING
            else:
                self.safety_level = SafetyLevel.SAFE

        return len([v for v in violations if v['severity'] == 'critical']) == 0

    def trigger_emergency_stop(self):
        """
        Trigger emergency stop procedure
        """
        self.emergency_stop_active = True
        print("EMERGENCY STOP TRIGGERED!")

        # Log the emergency stop
        self.violation_log.append({
            'timestamp': time.time(),
            'type': 'EMERGENCY_STOP',
            'message': 'Emergency stop triggered due to safety violations',
            'violations': self._get_current_violations()
        })

    def clear_emergency_stop(self):
        """
        Clear emergency stop condition
        """
        self.emergency_stop_active = False
        self.safety_level = SafetyLevel.SAFE
        print("Emergency stop cleared")

    def _log_violations(self, violations):
        """
        Log safety violations
        """
        for violation in violations:
            self.violation_log.append({
                'timestamp': time.time(),
                'monitor': violation.get('monitor', 'Unknown'),
                'violation': violation.get('violation', 'Unknown violation'),
                'severity': violation.get('severity', 'warning'),
                'details': violation.get('details', {})
            })

    def _get_current_violations(self):
        """
        Get recent violations
        """
        # Return violations from last 10 seconds
        current_time = time.time()
        recent_violations = [
            v for v in self.violation_log
            if current_time - v['timestamp'] < 10
        ]
        return recent_violations

    def get_safety_status(self):
        """
        Get current safety status
        """
        return {
            'level': self.safety_level.name,
            'emergency_stop': self.emergency_stop_active,
            'violations_count': len(self.violation_log),
            'last_violation': self.violation_log[-1] if self.violation_log else None
        }


class CollisionAvoidanceMonitor:
    """
    Monitor for collision avoidance safety
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator
        self.collision_threshold = 0.3  # meters
        self.human_safety_distance = 1.0  # meters

    def check_safety(self, system_state):
        """
        Check for collision safety violations
        """
        violations = []

        # Check obstacle distances
        if 'sensor_data' in system_state and 'lidar' in system_state['sensor_data']:
            lidar_data = system_state['sensor_data']['lidar']
            min_distance = min(lidar_data['ranges']) if lidar_data.get('ranges') else float('inf')

            if min_distance < self.collision_threshold:
                violations.append({
                    'monitor': 'CollisionAvoidance',
                    'violation': f'Obstacle too close: {min_distance:.2f}m',
                    'severity': 'critical',
                    'details': {
                        'min_distance': min_distance,
                        'threshold': self.collision_threshold
                    }
                })

        # Check for human safety
        if 'perception_data' in system_state:
            objects = system_state['perception_data'].get('objects', [])
            humans_nearby = [obj for obj in objects if obj.get('class') == 'person']

            for human in humans_nearby:
                distance = self._calculate_distance_to_robot(human)
                if distance < self.human_safety_distance:
                    violations.append({
                        'monitor': 'CollisionAvoidance',
                        'violation': f'Human too close: {distance:.2f}m',
                        'severity': 'critical',
                        'details': {
                            'human_distance': distance,
                            'safety_distance': self.human_safety_distance
                        }
                    })

        return violations

    def _calculate_distance_to_robot(self, object_info):
        """
        Calculate distance from object to robot
        """
        # This would use robot position and object position
        # For now, return a placeholder
        return 0.5  # meters


class VelocityLimitMonitor:
    """
    Monitor for velocity limit safety
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator

    def check_safety(self, system_state):
        """
        Check for velocity limit violations
        """
        violations = []

        # Check linear velocity
        if 'control_state' in system_state and 'velocity' in system_state['control_state']:
            linear_vel = system_state['control_state']['velocity'].get('linear', [0, 0, 0])
            linear_speed = np.linalg.norm(linear_vel)

            max_linear = self.validator.safety_limits['velocity']['linear_max']
            if linear_speed > max_linear:
                violations.append({
                    'monitor': 'VelocityLimit',
                    'violation': f'Linear velocity too high: {linear_speed:.2f}m/s',
                    'severity': 'critical',
                    'details': {
                        'actual': linear_speed,
                        'limit': max_linear
                    }
                })

        # Check angular velocity
        if 'control_state' in system_state and 'velocity' in system_state['control_state']:
            angular_vel = system_state['control_state']['velocity'].get('angular', [0, 0, 0])
            angular_speed = np.linalg.norm(angular_vel)

            max_angular = self.validator.safety_limits['velocity']['angular_max']
            if angular_speed > max_angular:
                violations.append({
                    'monitor': 'VelocityLimit',
                    'violation': f'Angular velocity too high: {angular_speed:.2f}rad/s',
                    'severity': 'critical',
                    'details': {
                        'actual': angular_speed,
                        'limit': max_angular
                    }
                })

        # Check joint velocities
        if 'joint_states' in system_state:
            for joint_name, joint_state in system_state['joint_states'].items():
                if 'velocity' in joint_state:
                    joint_vel = abs(joint_state['velocity'])
                    max_joint = self.validator.safety_limits['velocity']['joint_max']

                    if joint_vel > max_joint:
                        violations.append({
                            'monitor': 'VelocityLimit',
                            'violation': f'Joint {joint_name} velocity too high: {joint_vel:.2f}rad/s',
                            'severity': 'critical',
                            'details': {
                                'joint': joint_name,
                                'actual': joint_vel,
                                'limit': max_joint
                            }
                        })

        return violations


class ForceLimitMonitor:
    """
    Monitor for force limit safety
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator

    def check_safety(self, system_state):
        """
        Check for force limit violations
        """
        violations = []

        # Check contact forces
        if 'sensor_data' in system_state and 'force_torque' in system_state['sensor_data']:
            ft_sensors = system_state['sensor_data']['force_torque']

            for sensor_name, force_data in ft_sensors.items():
                if 'force' in force_data:
                    force_magnitude = np.linalg.norm(force_data['force'])
                    max_force = self.validator.safety_limits['force']['max_contact_force']

                    if force_magnitude > max_force:
                        violations.append({
                            'monitor': 'ForceLimit',
                            'violation': f'Force sensor {sensor_name} force too high: {force_magnitude:.2f}N',
                            'severity': 'critical',
                            'details': {
                                'sensor': sensor_name,
                                'actual': force_magnitude,
                                'limit': max_force
                            }
                        })

        # Check gripper forces
        if 'gripper_state' in system_state:
            gripper_force = system_state['gripper_state'].get('force', 0)
            max_gripper_force = self.validator.safety_limits['force']['max_gripper_force']

            if gripper_force > max_gripper_force:
                violations.append({
                    'monitor': 'ForceLimit',
                    'violation': f'Gripper force too high: {gripper_force:.2f}N',
                    'severity': 'warning',
                    'details': {
                        'actual': gripper_force,
                        'limit': max_gripper_force
                    }
                })

        return violations


class JointLimitMonitor:
    """
    Monitor for joint position limits
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator

    def check_safety(self, system_state):
        """
        Check for joint limit violations
        """
        violations = []

        if 'joint_states' in system_state:
            joint_limits = self.validator.safety_limits['position']['joint_limits']

            for joint_name, joint_state in system_state['joint_states'].items():
                if joint_name in joint_limits and 'position' in joint_state:
                    position = joint_state['position']
                    limits = joint_limits[joint_name]

                    if position < limits[0] or position > limits[1]:
                        violations.append({
                            'monitor': 'JointLimit',
                            'violation': f'Joint {joint_name} out of limits: {position:.3f}',
                            'severity': 'critical',
                            'details': {
                                'joint': joint_name,
                                'actual': position,
                                'min_limit': limits[0],
                                'max_limit': limits[1]
                            }
                        })

        return violations


class StabilityMonitor:
    """
    Monitor for robot stability
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator

    def check_safety(self, system_state):
        """
        Check for stability violations
        """
        violations = []

        # Check tilt angle
        if 'imu_data' in system_state:
            orientation = system_state['imu_data'].get('orientation', [0, 0, 0, 1])
            tilt_angle = self._calculate_tilt_angle(orientation)
            max_tilt = self.validator.safety_limits['stability']['max_tilt_angle']

            if tilt_angle > max_tilt:
                violations.append({
                    'monitor': 'Stability',
                    'violation': f'Robot tilt angle too high: {tilt_angle:.2f}°',
                    'severity': 'critical',
                    'details': {
                        'actual': tilt_angle,
                        'limit': max_tilt
                    }
                })

        # Check center of mass
        if 'robot_state' in system_state:
            com = system_state['robot_state'].get('center_of_mass', [0, 0, 0])
            support_polygon = system_state['robot_state'].get('support_polygon', [])

            if not self._is_com_stable(com, support_polygon):
                violations.append({
                    'monitor': 'Stability',
                    'violation': 'Center of mass outside support polygon',
                    'severity': 'critical',
                    'details': {
                        'com': com,
                        'support_polygon': support_polygon
                    }
                })

        return violations

    def _calculate_tilt_angle(self, orientation):
        """
        Calculate tilt angle from orientation quaternion
        """
        # Convert quaternion to Euler angles
        w, x, y, z = orientation

        # Calculate roll and pitch angles
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Return maximum tilt angle
        return max(abs(np.degrees(roll)), abs(np.degrees(pitch)))

    def _is_com_stable(self, com, support_polygon):
        """
        Check if center of mass is within support polygon
        """
        if not support_polygon:
            return False

        # Project 3D CoM to 2D ground plane
        com_2d = [com[0], com[1]]

        # Check if point is inside polygon using ray casting
        x, y = com_2d
        n = len(support_polygon)
        inside = False

        p1x, p1y = support_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = support_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


class EnvironmentalSafetyMonitor:
    """
    Monitor for environmental safety
    """

    def __init__(self, safety_validator):
        self.validator = safety_validator

    def check_safety(self, system_state):
        """
        Check for environmental safety violations
        """
        violations = []

        # Check workspace bounds
        if 'robot_state' in system_state:
            position = system_state['robot_state'].get('position', [0, 0, 0])
            bounds = self.validator.safety_limits['position']['workspace_bounds']

            if (position[0] < bounds[0] or position[0] > bounds[1] or
                position[1] < bounds[2] or position[1] > bounds[3] or
                position[2] < bounds[4] or position[2] > bounds[5]):

                violations.append({
                    'monitor': 'Environmental',
                    'violation': f'Robot outside workspace bounds: {position}',
                    'severity': 'warning',
                    'details': {
                        'position': position,
                        'bounds': bounds
                    }
                })

        # Check for hazardous conditions
        if 'environment_data' in system_state:
            temperature = system_state['environment_data'].get('temperature', 25)
            humidity = system_state['environment_data'].get('humidity', 50)

            if temperature > 60:  # Too hot
                violations.append({
                    'monitor': 'Environmental',
                    'violation': f'Hazardous temperature: {temperature}°C',
                    'severity': 'warning',
                    'details': {
                        'temperature': temperature
                    }
                })

            if humidity > 90:  # Too humid
                violations.append({
                    'monitor': 'Environmental',
                    'violation': f'Hazardous humidity: {humidity}%',
                    'severity': 'warning',
                    'details': {
                        'humidity': humidity
                    }
                })

        return violations


class SafetyTestSuite:
    """
    Test suite for safety validation
    """

    def __init__(self):
        self.validator = SafetyValidator()

    def run_safety_tests(self):
        """
        Run comprehensive safety tests
        """
        print("Running Safety Validation Tests...")

        test_results = {
            'collision_avoidance': self.test_collision_avoidance(),
            'velocity_limits': self.test_velocity_limits(),
            'force_limits': self.test_force_limits(),
            'joint_limits': self.test_joint_limits(),
            'stability': self.test_stability(),
            'emergency_stop': self.test_emergency_stop()
        }

        self._generate_safety_report(test_results)
        return test_results

    def test_collision_avoidance(self):
        """
        Test collision avoidance safety
        """
        print("Testing collision avoidance...")

        # Test 1: Safe distance
        safe_state = {
            'sensor_data': {
                'lidar': {
                    'ranges': [2.0] * 360  # All distances are safe
                }
            },
            'perception_data': {
                'objects': []
            }
        }

        result_safe = self.validator.validate_system_state(safe_state)

        # Test 2: Unsafe distance
        unsafe_state = {
            'sensor_data': {
                'lidar': {
                    'ranges': [0.1] + [2.0] * 359  # One close obstacle
                }
            },
            'perception_data': {
                'objects': []
            }
        }

        result_unsafe = self.validator.validate_system_state(unsafe_state)

        return {
            'safe_case': result_safe,
            'unsafe_case': result_unsafe,
            'passed': result_safe and not result_unsafe
        }

    def test_velocity_limits(self):
        """
        Test velocity limit safety
        """
        print("Testing velocity limits...")

        # Test 1: Safe velocities
        safe_state = {
            'control_state': {
                'velocity': {
                    'linear': [0.5, 0, 0],  # Below limit
                    'angular': [0.5, 0, 0]  # Below limit
                }
            },
            'joint_states': {
                'joint1': {'velocity': 2.0},  # Below limit
                'joint2': {'velocity': 3.0}   # Below limit
            }
        }

        result_safe = self.validator.validate_system_state(safe_state)

        # Test 2: Unsafe velocities
        unsafe_state = {
            'control_state': {
                'velocity': {
                    'linear': [2.0, 0, 0],  # Above limit
                    'angular': [0, 0, 0]
                }
            },
            'joint_states': {
                'joint1': {'velocity': 6.0},  # Above limit
                'joint2': {'velocity': 3.0}   # Below limit
            }
        }

        result_unsafe = self.validator.validate_system_state(unsafe_state)

        return {
            'safe_case': result_safe,
            'unsafe_case': result_unsafe,
            'passed': result_safe and not result_unsafe
        }

    def test_force_limits(self):
        """
        Test force limit safety
        """
        print("Testing force limits...")

        # Test 1: Safe forces
        safe_state = {
            'sensor_data': {
                'force_torque': {
                    'sensor1': {'force': [10, 5, 2]},  # Below limit
                    'sensor2': {'force': [8, 3, 1]}    # Below limit
                }
            },
            'gripper_state': {
                'force': 20  # Below limit
            }
        }

        result_safe = self.validator.validate_system_state(safe_state)

        # Test 2: Unsafe forces
        unsafe_state = {
            'sensor_data': {
                'force_torque': {
                    'sensor1': {'force': [150, 50, 20]},  # Above limit
                    'sensor2': {'force': [8, 3, 1]}       # Below limit
                }
            },
            'gripper_state': {
                'force': 60  # Above limit
            }
        }

        result_unsafe = self.validator.validate_system_state(unsafe_state)

        return {
            'safe_case': result_safe,
            'unsafe_case': result_unsafe,
            'passed': result_safe and not result_unsafe
        }

    def test_joint_limits(self):
        """
        Test joint limit safety
        """
        print("Testing joint limits...")

        # Test 1: Safe joint positions
        safe_state = {
            'joint_states': {
                'hip_yaw': {'position': 0.5},    # Within limits
                'hip_roll': {'position': 0.3},   # Within limits
                'hip_pitch': {'position': 1.0}   # Within limits
            }
        }

        result_safe = self.validator.validate_system_state(safe_state)

        # Test 2: Unsafe joint positions
        unsafe_state = {
            'joint_states': {
                'hip_yaw': {'position': 2.0},    # Outside limits
                'hip_roll': {'position': 0.3},   # Within limits
                'hip_pitch': {'position': 1.0}   # Within limits
            }
        }

        result_unsafe = self.validator.validate_system_state(unsafe_state)

        return {
            'safe_case': result_safe,
            'unsafe_case': result_unsafe,
            'passed': result_safe and not result_unsafe
        }

    def test_stability(self):
        """
        Test stability safety
        """
        print("Testing stability...")

        # Test 1: Stable state
        stable_state = {
            'imu_data': {
                'orientation': [0, 0, 0, 1]  # No tilt
            },
            'robot_state': {
                'center_of_mass': [0, 0, 0.8],  # Stable position
                'support_polygon': [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]  # Square support
            }
        }

        result_stable = self.validator.validate_system_state(stable_state)

        # Test 2: Unstable state
        unstable_state = {
            'imu_data': {
                'orientation': [0.3, 0.3, 0, 0.9]  # Significant tilt
            },
            'robot_state': {
                'center_of_mass': [0.5, 0.5, 0.8],  # Outside support polygon
                'support_polygon': [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]  # Small support
            }
        }

        result_unstable = self.validator.validate_system_state(unstable_state)

        return {
            'stable_case': result_stable,
            'unstable_case': result_unstable,
            'passed': result_stable and not result_unstable
        }

    def test_emergency_stop(self):
        """
        Test emergency stop functionality
        """
        print("Testing emergency stop...")

        # Reset emergency stop
        self.validator.clear_emergency_stop()

        # Test normal operation
        normal_state = {
            'control_state': {'velocity': {'linear': [0.5, 0, 0]}}
        }
        result_normal = self.validator.validate_system_state(normal_state)
        normal_emergency = self.validator.emergency_stop_active

        # Create unsafe state to trigger emergency stop
        unsafe_state = {
            'control_state': {'velocity': {'linear': [5.0, 0, 0]}},  # Way above limit
            'joint_states': {'joint1': {'velocity': 10.0}}  # Way above limit
        }
        result_unsafe = self.validator.validate_system_state(unsafe_state)
        unsafe_emergency = self.validator.emergency_stop_active

        # Clear emergency stop and test recovery
        self.validator.clear_emergency_stop()
        result_recovered = self.validator.validate_system_state(normal_state)
        recovered_emergency = self.validator.emergency_stop_active

        return {
            'normal_case': result_normal,
            'emergency_triggered': unsafe_emergency,
            'recovered_case': not recovered_emergency,
            'passed': result_normal and unsafe_emergency and not recovered_emergency
        }

    def _generate_safety_report(self, test_results):
        """
        Generate safety validation report
        """
        report = []
        report.append("# Safety Validation Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        overall_passed = all(result['passed'] for result in test_results.values())

        report.append(f"## Overall Result: {'PASS' if overall_passed else 'FAIL'}")
        report.append("")

        for test_name, result in test_results.items():
            status = "PASS" if result['passed'] else "FAIL"
            report.append(f"### {test_name.replace('_', ' ').title()}: {status}")
            report.append(f"- Safe case: {'PASS' if result['safe_case'] else 'FAIL'}")
            report.append(f"- Unsafe case: {'PASS' if not result['unsafe_case'] else 'FAIL'}")
            report.append("")

        # Add safety status
        safety_status = self.validator.get_safety_status()
        report.append("## Current Safety Status")
        report.append(f"- Level: {safety_status['level']}")
        report.append(f"- Emergency Stop: {'ACTIVE' if safety_status['emergency_stop'] else 'INACTIVE'}")
        report.append(f"- Violations Logged: {safety_status['violations_count']}")

        safety_report = "\n".join(report)
        print(safety_report)

        # Save report to file
        with open('safety_validation_report.txt', 'w') as f:
            f.write(safety_report)

        print("Safety validation report saved to safety_validation_report.txt")


def run_safety_validation():
    """
    Run safety validation tests
    """
    safety_tester = SafetyTestSuite()
    results = safety_tester.run_safety_tests()

    overall_passed = all(result['passed'] for result in results.values())
    print(f"\nSafety validation: {'PASSED' if overall_passed else 'FAILED'}")

    return overall_passed


if __name__ == "__main__":
    success = run_safety_validation()
    exit(0 if success else 1)
```

## Week Summary

This section covered advanced simulation techniques for Physical AI and humanoid robotics, including optimization strategies, domain randomization, sensor simulation, physics modeling, and safety validation. The content emphasized performance optimization, realistic simulation, and comprehensive safety systems to ensure reliable operation of AI-powered robotic systems. These techniques are essential for creating effective simulation environments that can accelerate development while maintaining safety and accuracy requirements.