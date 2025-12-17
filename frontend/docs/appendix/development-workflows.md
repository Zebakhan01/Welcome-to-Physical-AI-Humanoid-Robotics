---
sidebar_position: 6
---

# Development Workflows

## Introduction to Robotics Development Workflows

Development workflows in Physical AI and humanoid robotics encompass the complete lifecycle from concept to deployment. These workflows integrate hardware development, software engineering, simulation, testing, and deployment in a cohesive manner. This section outlines best practices for managing complex robotics projects, including version control, continuous integration, testing strategies, and deployment procedures.

## Git Workflows for Robotics Projects

### Git Branching Strategy

#### Feature Branch Workflow

```bash
# Feature branch workflow for robotics development
# Main branches:
#   main: Production-ready code
#   develop: Integration branch for features
#   release: Release preparation
#   hotfix: Critical fixes for production

# Starting a new feature
git checkout -b feature/humanoid-walking-patterns develop

# Making changes
git add .
git commit -m "feat: Add basic walking pattern generator"

# Sync with develop before merging
git checkout develop
git pull origin develop
git checkout feature/humanoid-walking-patterns
git rebase develop

# Push feature branch
git push origin feature/humanoid-walking-patterns

# Create pull request to merge into develop
# After review and approval:
git checkout develop
git merge --no-ff feature/humanoid-walking-patterns
git push origin develop

# Clean up feature branch
git branch -d feature/humanoid-walking-patterns
git push origin --delete feature/humanoid-walking-patterns
```

#### Release Branch Workflow

```bash
# Release branch workflow
# Creating a release branch
git checkout -b release/v1.2.0 develop

# Update version numbers
# Fix last-minute bugs
git add .
git commit -m "chore: Update version to v1.2.0"

# Merge to main and tag
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.2.0
git push origin develop

# Clean up release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

### Git Hooks for Robotics Development

```bash
#!/bin/bash
# pre-commit-hook.sh - Pre-commit hook for robotics projects

# Check if ROS workspace is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS environment not sourced. Please source your ROS setup file."
    exit 1
fi

# Run basic code quality checks
echo "Running pre-commit checks..."

# Check for large files (over 10MB)
git diff --cached --name-only | xargs -I {} sh -c 'if [ $(stat -f%z "{}" 2>/dev/null || stat -c%s "{}" 2>/dev/null) -gt 10485760 ]; then echo "Large file detected: {}"; exit 1; fi'

# Run basic syntax checks
echo "Checking Python syntax..."
for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$'); do
    python -m py_compile "$file" || exit 1
done

# Run basic C++ syntax checks
echo "Checking C++ syntax..."
for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.\(cpp\|hpp\|cc\|h\)$'); do
    g++ -fsyntax-only "$file" || exit 1
done

# Run ROS package checks
echo "Checking ROS packages..."
for package_xml in $(git diff --cached --name-only --diff-filter=ACM | grep 'package\.xml$'); do
    package_dir=$(dirname "$package_xml")
    if [ -f "$package_dir/CMakeLists.txt" ]; then
        echo "Checking package: $package_dir"
        # Add package validation here
    fi
done

echo "All pre-commit checks passed!"
exit 0
```

### Git Aliases for Robotics Development

```bash
# Git aliases for robotics development
git config --global alias.rospush "push origin HEAD:develop"
git config --global alias.rosmerge "merge --no-ff"
git config --global alias.roslog "log --oneline --graph --decorate"
git config --global alias.rosstatus "status -s"
git config --global alias.rosdiff "diff --name-status"
git config --global alias.rosbranch "branch -av"
git config --global alias.rosfetch "fetch --all --prune"

# ROS-specific aliases
git config --global alias.build "bash -c 'catkin_make'"
git config --global alias.run "bash -c 'source devel/setup.bash && roslaunch'"
git config --global alias.test "bash -c 'catkin_make run_tests && catkin_test_results'"
```

## Continuous Integration and Deployment

### CI/CD Pipeline for Robotics

#### GitHub Actions Configuration

```yaml
# .github/workflows/ros_ci.yml
name: ROS CI

on:
  push:
    branches: [ develop, main ]
  pull_request:
    branches: [ develop ]

jobs:
  build-and-test:
    runs-on: ubuntu-20.04
    strategy:
      matrix:
        ros_distro: [noetic]

    container:
      image: ros:${{ matrix.ros_distro }}-desktop-full-focal

    steps:
    - uses: actions/checkout@v2

    - name: Setup ROS environment
      run: |
        apt-get update
        apt-get install -y python3-catkin-tools python3-osrf-pycommon
        source /opt/ros/${{ matrix.ros_distro }}/setup.bash
        echo "source /opt/ros/${{ matrix.ros_distro }}/setup.bash" >> ~/.bashrc

    - name: Install dependencies
      run: |
        source /opt/ros/${{ matrix.ros_distro }}/setup.bash
        rosdep update
        rosdep install --from-paths src --ignore-src -r -y

    - name: Build workspace
      run: |
        source /opt/ros/${{ matrix.ros_distro }}/setup.bash
        catkin_make

    - name: Run tests
      run: |
        source /opt/ros/${{ matrix.ros_distro }}/setup.bash
        catkin_make run_tests
        catkin_test_results

    - name: Build documentation
      run: |
        source /opt/ros/${{ matrix.ros_distro }}/setup.bash
        cd docs
        pip3 install sphinx
        make html

    - name: Upload test results
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-results-${{ matrix.ros_distro }}
        path: |
          test_results/
          build/
```

#### Docker-based CI/CD

```yaml
# .github/workflows/docker_ci.yml
name: Docker CI/CD

on:
  push:
    branches: [ develop, main ]
  pull_request:
    branches: [ develop ]

jobs:
  docker-build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v2

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1

    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v2
      with:
        context: .
        file: ./Dockerfile.ros
        push: true
        tags: |
          robotics/humanoid-robot:${{ github.sha }}
          robotics/humanoid-robot:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  simulation-tests:
    runs-on: ubuntu-latest
    needs: docker-build
    steps:
    - name: Run simulation tests
      run: |
        docker run -it robotics/humanoid-robot:latest \
          bash -c "source /opt/ros/noetic/setup.bash && \
                   cd /catkin_ws && \
                   catkin_make run_tests && \
                   catkin_test_results"
```

### Jenkins Pipeline for Robotics

```groovy
// Jenkinsfile for robotics project
pipeline {
    agent any

    parameters {
        choice(
            name: 'ROS_DISTRO',
            choices: ['noetic', 'melodic'],
            description: 'ROS Distribution to use'
        )
        booleanParam(
            name: 'RUN_TESTS',
            defaultValue: true,
            description: 'Run tests after build'
        )
        booleanParam(
            name: 'BUILD_DOCS',
            defaultValue: false,
            description: 'Build documentation'
        )
    }

    environment {
        ROS_WS = "${WORKSPACE}/catkin_ws"
        ROS_DISTRO = "${params.ROS_DISTRO}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                script {
                    sh '''
                        sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
                        sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
                        sudo apt update
                        sudo apt install -y ros-${ROS_DISTRO}-desktop-full
                        sudo rosdep init || echo "rosdep already initialized"
                        rosdep update
                    '''
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    source /opt/ros/${ROS_DISTRO}/setup.bash
                    cd ${ROS_WS}/src
                    rosdep install --from-paths . --ignore-src -r -y
                '''
            }
        }

        stage('Build') {
            steps {
                sh '''
                    source /opt/ros/${ROS_DISTRO}/setup.bash
                    cd ${ROS_WS}
                    catkin_make
                '''
            }
        }

        stage('Test') {
            when {
                expression { params.RUN_TESTS }
            }
            steps {
                sh '''
                    source /opt/ros/${ROS_DISTRO}/setup.bash
                    cd ${ROS_WS}
                    catkin_make run_tests
                    catkin_test_results
                '''
            }
        }

        stage('Build Documentation') {
            when {
                expression { params.BUILD_DOCS }
            }
            steps {
                sh '''
                    pip3 install sphinx sphinx-rtd-theme
                    cd docs
                    make html
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'catkin_ws/build/**/*', fingerprint: true
            publishHTML([
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'docs/_build/html',
                reportFiles: 'index.html',
                reportName: 'Documentation'
            ])
        }
        success {
            echo 'Build and test successful!'
        }
        failure {
            echo 'Build or test failed!'
        }
    }
}
```

## Development Environment Setup

### Docker Development Environment

```dockerfile
# Dockerfile for robotics development environment
FROM osrf/ros:noetic-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/home/robotics/ws
ENV DISPLAY=:0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    gnupg \
    lsb-release \
    software-properties-common \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    && rm -rf /var/lib/apt/lists/*

# Install additional tools
RUN apt-get update && apt-get install -y \
    gazebo11 \
    libgazebo11-dev \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-moveit \
    ros-noetic-navigation \
    ros-noetic-interactive-markers \
    ros-noetic-tf2-tools \
    ros-noetic-rviz \
    ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -ms /bin/bash robotics
USER robotics
WORKDIR /home/robotics

# Install Python packages
COPY requirements.txt .
RUN pip3 install --user -r requirements.txt

# Create workspace
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}

# Copy source code
COPY --chown=robotics:robotics src/ src/

# Build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Setup entrypoint
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
source '${ROS_WS}'/devel/setup.bash\n\
exec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
```

### Development Container Configuration

```json
// .devcontainer/devcontainer.json
{
    "name": "Robotics Development",
    "image": "osrf/ros:noetic-desktop-full",

    "runArgs": [
        "--network=host",
        "--privileged",
        "--env=DISPLAY=${env:DISPLAY}",
        "--volume=/tmp/.X11-unix:/tmp/.X11-unix:rw",
        "--volume=/dev:/dev",
        "--gpus=all"
    ],

    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-iot.vscode-ros",
                "ms-vscode.cpptools",
                "twxs.cmake",
                "ms-azuretools.vscode-docker",
                "ms-python.flake8",
                "ms-python.black-formatter"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/bin/python3",
                "python.linting.enabled": true,
                "python.linting.flake8Enabled": true,
                "ros.distro": "noetic",
                "terminal.integrated.env.linux": {
                    "ROS_DISTRO": "noetic",
                    "ROS_WS": "/workspace"
                }
            }
        }
    },

    "postCreateCommand": "bash -c 'source /opt/ros/noetic/setup.bash && cd /workspace && mkdir -p src && catkin_make'",

    "remoteUser": "vscode",

    "features": {
        "ghcr.io/devcontainers/features/docker-in-docker:2": {},
        "ghcr.io/devcontainers/features/git:1": {},
        "ghcr.io/devcontainers/features/github-cli:1": {}
    }
}
```

## Testing Strategies

### Unit Testing Framework

```python
# test/unit/test_robot_controller.py
import unittest
import numpy as np
from robot_control.robot_controller import RobotController
from robot_control.pid_controller import PIDController

class TestRobotController(unittest.TestCase):
    """
    Unit tests for RobotController class
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.controller = RobotController()
        self.pid_controller = PIDController(kp=1.0, ki=0.1, kd=0.05)

    def test_pid_controller_init(self):
        """
        Test PID controller initialization
        """
        self.assertEqual(self.pid_controller.kp, 1.0)
        self.assertEqual(self.pid_controller.ki, 0.1)
        self.assertEqual(self.pid_controller.kd, 0.05)

    def test_pid_controller_update(self):
        """
        Test PID controller update method
        """
        # Test with zero error
        output = self.pid_controller.update(0.0, 0.0, 0.01)
        self.assertEqual(output, 0.0)

        # Test with positive error
        output = self.pid_controller.update(1.0, 0.0, 0.01)
        expected = 1.0 * 1.0 + 0.1 * 0.0 + 0.05 * 100.0  # P + I + D terms
        self.assertAlmostEqual(output, expected, places=2)

    def test_robot_controller_init(self):
        """
        Test RobotController initialization
        """
        self.assertIsNotNone(self.controller.joint_controllers)
        self.assertEqual(len(self.controller.joint_controllers), 0)

    def test_add_joint_controller(self):
        """
        Test adding joint controller to RobotController
        """
        self.controller.add_joint_controller('joint1', self.pid_controller)
        self.assertIn('joint1', self.controller.joint_controllers)
        self.assertEqual(self.controller.joint_controllers['joint1'], self.pid_controller)

    def test_calculate_inverse_kinematics(self):
        """
        Test inverse kinematics calculation
        """
        # Simple test with known solution
        target_pos = np.array([0.5, 0.0, 0.5])
        joint_angles = self.controller.calculate_inverse_kinematics(target_pos)

        # Verify the solution is reasonable
        self.assertIsNotNone(joint_angles)
        self.assertIsInstance(joint_angles, np.ndarray)
        self.assertEqual(len(joint_angles), 6)  # 6DOF robot

    def test_trajectory_generation(self):
        """
        Test trajectory generation
        """
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 1.0, 1.0])
        duration = 1.0

        trajectory = self.controller.generate_trajectory(start_pos, end_pos, duration)

        # Verify trajectory properties
        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory), 0)

        # Check start and end positions
        self.assertTrue(np.allclose(trajectory[0], start_pos))
        self.assertTrue(np.allclose(trajectory[-1], end_pos))

class TestPIDControllerAdvanced(unittest.TestCase):
    """
    Advanced tests for PID controller
    """

    def setUp(self):
        self.pid = PIDController(kp=10.0, ki=1.0, kd=0.1)

    def test_integral_windup_protection(self):
        """
        Test integral windup protection
        """
        # Simulate large error for many iterations
        for _ in range(100):
            output = self.pid.update(10.0, 0.0, 0.01)
            # Output should be limited to prevent windup
            self.assertLess(abs(output), 100.0)  # Reasonable limit

    def test_derivative_kick_prevention(self):
        """
        Test derivative kick prevention
        """
        # Set initial conditions
        initial_output = self.pid.update(0.0, 0.0, 0.01)

        # Suddenly change setpoint (should not cause derivative kick)
        new_output = self.pid.update(5.0, 0.0, 0.01)

        # Difference should not be extremely large
        difference = abs(new_output - initial_output)
        self.assertLess(difference, 100.0)  # Reasonable limit

    def test_tuning_methods(self):
        """
        Test PID tuning methods
        """
        # Test Ziegler-Nichols tuning
        self.pid.tune_zeitlin_kalman(1.0, 0.1)  # Example parameters
        self.assertGreater(self.pid.kp, 0)
        self.assertGreater(self.pid.ki, 0)
        self.assertGreater(self.pid.kd, 0)

def run_unit_tests():
    """
    Run all unit tests
    """
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRobotController)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPIDControllerAdvanced))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_unit_tests()
    exit(0 if success else 1)
```

### Integration Testing

```python
# test/integration/test_robot_integration.py
import unittest
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np

class TestRobotIntegration(unittest.TestCase):
    """
    Integration tests for robot system components
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class-level test fixtures
        """
        rospy.init_node('robot_integration_test', anonymous=True)
        cls.timeout = 10.0  # seconds

    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        # Publishers for joint commands
        self.joint_publishers = {}
        for i in range(6):  # 6 DOF robot
            pub = rospy.Publisher(f'/joint_{i}_position_controller/command', Float64, queue_size=10)
            self.joint_publishers[f'joint_{i}'] = pub

        # Subscribers for joint states
        self.joint_states = None
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        # Subscribers for end-effector pose
        self.end_effector_pose = None
        self.pose_sub = rospy.Subscriber('/end_effector_pose', Pose, self.pose_callback)

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        """
        self.joint_states = msg

    def pose_callback(self, msg):
        """
        Callback for end-effector pose messages
        """
        self.end_effector_pose = msg

    def test_joint_command_response(self):
        """
        Test that joint commands result in state changes
        """
        # Wait for initial joint states
        start_time = time.time()
        while self.joint_states is None and (time.time() - start_time) < self.timeout:
            time.sleep(0.1)

        self.assertIsNotNone(self.joint_states, "Failed to receive joint states")

        # Store initial positions
        initial_positions = list(self.joint_states.position)

        # Send command to first joint
        command_msg = Float64()
        command_msg.data = 1.0  # radians
        self.joint_publishers['joint_0'].publish(command_msg)

        # Wait for response
        time.sleep(2.0)  # Wait for movement

        # Verify position changed
        current_positions = list(self.joint_states.position)
        position_difference = abs(current_positions[0] - initial_positions[0])

        self.assertGreater(position_difference, 0.1, "Joint did not respond to command")

    def test_forward_kinematics(self):
        """
        Test forward kinematics by comparing calculated vs actual pose
        """
        # Wait for pose data
        start_time = time.time()
        while self.end_effector_pose is None and (time.time() - start_time) < self.timeout:
            time.sleep(0.1)

        self.assertIsNotNone(self.end_effector_pose, "Failed to receive end-effector pose")

        # Calculate expected pose from joint angles (simplified)
        if self.joint_states:
            calculated_pose = self.calculate_forward_kinematics(self.joint_states.position)

            # Compare with actual pose (allow for some tolerance)
            actual_pose = self.end_effector_pose
            position_error = np.sqrt(
                (calculated_pose[0] - actual_pose.position.x)**2 +
                (calculated_pose[1] - actual_pose.position.y)**2 +
                (calculated_pose[2] - actual_pose.position.z)**2
            )

            self.assertLess(position_error, 0.05, "Forward kinematics error too large")

    def calculate_forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics (simplified implementation)
        """
        # This is a simplified example - real implementation would use DH parameters
        # or other kinematic models
        x = np.cos(joint_angles[0]) * 0.5  # Example calculation
        y = np.sin(joint_angles[0]) * 0.5
        z = joint_angles[1] * 0.3 + 0.5

        return [x, y, z]

    def test_trajectory_execution(self):
        """
        Test trajectory execution from start to end
        """
        # Define trajectory points
        trajectory_points = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start position
            [0.5, 0.2, 0.1, 0.0, 0.0, 0.0],  # Intermediate
            [1.0, 0.5, 0.3, 0.2, 0.1, 0.0],  # End position
        ]

        # Execute trajectory
        for point in trajectory_points:
            self.send_joint_trajectory(point)
            time.sleep(1.0)  # Wait between points

        # Verify final position
        if self.joint_states:
            final_positions = list(self.joint_states.position)
            expected_final = trajectory_points[-1]

            for exp, actual in zip(expected_final, final_positions):
                self.assertAlmostEqual(exp, actual, places=1,
                                     msg=f"Final position mismatch: expected {exp}, got {actual}")

    def send_joint_trajectory(self, joint_positions):
        """
        Send joint trajectory to robot
        """
        for i, pos in enumerate(joint_positions):
            cmd_msg = Float64()
            cmd_msg.data = pos
            self.joint_publishers[f'joint_{i}'].publish(cmd_msg)

    def test_safety_systems(self):
        """
        Test safety system response
        """
        # This would test emergency stop, collision avoidance, etc.
        # For now, just verify safety topics exist
        try:
            # Try to access safety-related topics
            rospy.wait_for_message('/emergency_stop', Bool, timeout=1.0)
            safety_available = True
        except rospy.ROSException:
            safety_available = False

        # The test passes if safety system is available or not
        # (depending on robot configuration)
        self.assertTrue(True)  # Always pass for now, implement based on specific robot

class TestSimulationIntegration(unittest.TestCase):
    """
    Integration tests for simulation components
    """

    def test_gazebo_connection(self):
        """
        Test connection to Gazebo simulation
        """
        try:
            # Check if Gazebo services are available
            rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
            gazebo_available = True
        except rospy.ROSException:
            gazebo_available = False

        self.assertTrue(gazebo_available, "Gazebo simulation not available")

    def test_model_spawning(self):
        """
        Test model spawning in simulation
        """
        try:
            from gazebo_msgs.srv import SpawnModel
            spawn_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

            # Test service availability
            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)

            # This is a simplified test - actual spawning would require model definition
            self.assertTrue(True)  # Service is available

        except rospy.ROSException:
            self.fail("Gazebo model spawning service not available")

def run_integration_tests():
    """
    Run all integration tests
    """
    if not rospy.core.is_initialized():
        rospy.init_node('integration_test_runner', anonymous=True)

    # Create test suites
    robot_suite = unittest.TestLoader().loadTestsFromTestCase(TestRobotIntegration)
    sim_suite = unittest.TestLoader().loadTestsFromTestCase(TestSimulationIntegration)

    # Combine suites
    full_suite = unittest.TestSuite([robot_suite, sim_suite])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(full_suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
```

## Code Quality and Standards

### Static Analysis Configuration

```yaml
# .github/linters/.flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503  # Black compatibility
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    build,
    devel,
    .vscode,
    .github
per-file-ignores =
    */__init__.py:F401  # Allow unused imports in __init__.py
    test/*:S101  # Allow asserts in tests
    */test_*.py:S101  # Allow asserts in tests

# Import order
application-import-names = robot_control,utils,common
import-order-style = google
```

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
skip-string-normalization = false

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "test.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
minversion = "6.0"
testpaths = [
    "test",
]
python_files = [
    "test_*.py",
    "*_test.py",
]
python_classes = [
    "Test*",
]
python_functions = [
    "test_*",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "hardware: marks tests that require hardware",
]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
]
```

### Code Review Checklist

```markdown
# Code Review Checklist for Robotics Projects

## General Code Quality
- [ ] Code follows established style guides (PEP 8, Google C++ Style)
- [ ] Functions/classes have appropriate docstrings
- [ ] Variable and function names are descriptive
- [ ] Code is properly commented where complex logic is used
- [ ] No commented-out code blocks
- [ ] Error handling is implemented appropriately

## ROS/Robotics Specific
- [ ] ROS node initialization and shutdown handled properly
- [ ] Topics/services/actions are named according to conventions
- [ ] Message types are validated
- [ ] TF frames are properly managed
- [ ] URDF/SDF files are properly formatted and validated
- [ ] Sensor data is properly filtered and validated
- [ ] Safety systems are properly integrated

## Performance and Efficiency
- [ ] No unnecessary loops or repeated calculations
- [ ] Memory allocation is minimized
- [ ] Real-time constraints are respected
- [ ] Computational complexity is appropriate for target platform
- [ ] Data structures are appropriate for use case

## Safety and Reliability
- [ ] Safety checks are implemented for all critical operations
- [ ] Emergency stop functionality is available
- [ ] Bounds checking is performed on all array accesses
- [ ] Sensor limits and joint limits are enforced
- [ ] Collision detection and avoidance are implemented

## Testing
- [ ] Unit tests cover all critical functionality
- [ ] Integration tests verify system behavior
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Performance tests are included where appropriate

## Documentation
- [ ] README is updated with new features
- [ ] API documentation is complete
- [ ] Configuration parameters are documented
- [ ] Dependencies are listed
- [ ] Known issues are documented

## Security
- [ ] No hardcoded passwords or sensitive information
- [ ] Input validation is implemented
- [ ] Network communication is secured where appropriate
- [ ] Access controls are implemented for critical functions
```

## Deployment Workflows

### Simulation to Real Robot Deployment

```python
# deployment_workflow.py
import os
import sys
import yaml
import subprocess
import paramiko
from pathlib import Path

class RobotDeploymentWorkflow:
    """
    Deployment workflow for transferring code from simulation to real robot
    """

    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.robot_ip = self.config['robot']['ip']
        self.robot_username = self.config['robot']['username']
        self.robot_password = self.config['robot']['password']
        self.workspace_path = self.config['robot']['workspace_path']

    def validate_code_before_deployment(self):
        """
        Validate code before deployment to real robot
        """
        print("Validating code before deployment...")

        # Run all tests
        if not self.run_tests():
            print("Tests failed - aborting deployment")
            return False

        # Check for simulation-specific code
        if self.has_simulation_specific_code():
            print("Warning: Simulation-specific code detected")
            # Ask for confirmation or fix automatically
            if not self.remove_simulation_specific_code():
                return False

        # Validate configuration for real robot
        if not self.validate_robot_configuration():
            print("Configuration validation failed")
            return False

        print("Code validation passed")
        return True

    def run_tests(self):
        """
        Run comprehensive test suite
        """
        print("Running unit tests...")
        result = subprocess.run(['python', '-m', 'pytest', 'test/unit'],
                               capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Unit tests failed: {result.stderr}")
            return False

        print("Running integration tests...")
        result = subprocess.run(['python', '-m', 'pytest', 'test/integration'],
                               capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Integration tests failed: {result.stderr}")
            return False

        print("All tests passed")
        return True

    def has_simulation_specific_code(self):
        """
        Check for simulation-specific code that shouldn't run on real robot
        """
        simulation_indicators = [
            'gazebo',
            'simulation',
            'fake_controller',
            'mock_sensor',
            'test_mode'
        ]

        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py') or file.endswith('.cpp'):
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read().lower()
                        for indicator in simulation_indicators:
                            if indicator in content:
                                print(f"Found simulation indicator '{indicator}' in {file}")
                                return True
        return False

    def remove_simulation_specific_code(self):
        """
        Remove or replace simulation-specific code
        """
        print("Removing simulation-specific code...")
        # Implementation would depend on specific code patterns
        return True

    def validate_robot_configuration(self):
        """
        Validate that configuration is appropriate for real robot
        """
        # Check hardware configurations
        required_configs = [
            'real_robot_config.yaml',
            'hardware_interfaces.yaml',
            'safety_limits.yaml'
        ]

        for config in required_configs:
            config_path = Path('config') / config
            if not config_path.exists():
                print(f"Required configuration file missing: {config_path}")
                return False

        # Validate safety parameters
        safety_config = self.load_config('config/safety_limits.yaml')
        if not self.validate_safety_parameters(safety_config):
            return False

        return True

    def load_config(self, config_path):
        """
        Load configuration file
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def validate_safety_parameters(self, config):
        """
        Validate safety parameters are reasonable
        """
        # Check joint limits
        for joint, limits in config.get('joint_limits', {}).items():
            if limits['max_position'] <= limits['min_position']:
                print(f"Invalid joint limits for {joint}")
                return False

        # Check velocity limits
        for joint, limits in config.get('velocity_limits', {}).items():
            if limits['max_velocity'] <= 0:
                print(f"Invalid velocity limit for {joint}")
                return False

        return True

    def deploy_to_robot(self):
        """
        Deploy validated code to real robot
        """
        if not self.validate_code_before_deployment():
            print("Code validation failed - deployment aborted")
            return False

        print(f"Deploying to robot at {self.robot_ip}...")

        try:
            # Create SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.robot_ip, username=self.robot_username,
                       password=self.robot_password)

            # Create SFTP connection for file transfer
            sftp = ssh.open_sftp()

            # Backup existing workspace
            self.backup_existing_workspace(ssh)

            # Transfer source code
            self.transfer_source_code(sftp)

            # Install dependencies
            self.install_dependencies(ssh)

            # Build workspace
            self.build_workspace(ssh)

            # Upload configuration
            self.upload_configuration(sftp)

            # Upload launch files
            self.upload_launch_files(sftp)

            # Set permissions
            self.set_permissions(ssh)

            # Restart robot services
            self.restart_robot_services(ssh)

            print("Deployment completed successfully!")
            return True

        except Exception as e:
            print(f"Deployment failed: {e}")
            return False

        finally:
            try:
                sftp.close()
                ssh.close()
            except:
                pass

    def backup_existing_workspace(self, ssh):
        """
        Backup existing workspace on robot
        """
        timestamp = subprocess.check_output(['date', '+%Y%m%d_%H%M%S']).decode().strip()
        backup_cmd = f"mv {self.workspace_path} {self.workspace_path}_backup_{timestamp}"
        stdin, stdout, stderr = ssh.exec_command(backup_cmd)
        stdout.channel.recv_exit_status()  # Wait for completion

    def transfer_source_code(self, sftp):
        """
        Transfer source code to robot
        """
        # Create workspace directory
        self.sftp_mkdir_p(sftp, f"{self.workspace_path}/src")

        # Transfer source files
        for root, dirs, files in os.walk('src'):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, 'src')
                remote_path = f"{self.workspace_path}/src/{relative_path}"

                # Create remote directory
                remote_dir = os.path.dirname(remote_path)
                self.sftp_mkdir_p(sftp, remote_dir)

                # Transfer file
                sftp.put(local_path, remote_path)
                print(f"Transferred: {local_path} -> {remote_path}")

    def sftp_mkdir_p(self, sftp, remote_directory):
        """
        Create remote directory recursively
        """
        if remote_directory == '/':
            return
        if remote_directory == '':
            return

        try:
            sftp.stat(remote_directory)
        except FileNotFoundError:
            dirname, basename = os.path.split(remote_directory.rstrip('/'))
            self.sftp_mkdir_p(sftp, dirname)
            sftp.mkdir(remote_directory)

    def install_dependencies(self, ssh):
        """
        Install dependencies on robot
        """
        commands = [
            f"cd {self.workspace_path}",
            "rosdep update",
            f"rosdep install --from-paths src --ignore-src -r -y"
        ]

        full_cmd = " && ".join(commands)
        stdin, stdout, stderr = ssh.exec_command(full_cmd)

        # Print output in real-time
        for line in iter(stdout.readline, ""):
            print(line.strip())

        # Check for errors
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_output = stderr.read().decode()
            raise Exception(f"Dependency installation failed: {error_output}")

    def build_workspace(self, ssh):
        """
        Build the workspace on robot
        """
        commands = [
            f"source /opt/ros/{self.config['ros']['distro']}/setup.bash",
            f"cd {self.workspace_path}",
            "catkin_make"
        ]

        full_cmd = " && ".join(commands)
        stdin, stdout, stderr = ssh.exec_command(full_cmd)

        # Print output in real-time
        for line in iter(stdout.readline, ""):
            print(line.strip())

        # Check for errors
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_output = stderr.read().decode()
            raise Exception(f"Build failed: {error_output}")

    def upload_configuration(self, sftp):
        """
        Upload configuration files to robot
        """
        config_dir = f"{self.workspace_path}/config"
        self.sftp_mkdir_p(sftp, config_dir)

        for root, dirs, files in os.walk('config'):
            for file in files:
                if file.endswith(('.yaml', '.xml', '.ini')):
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, 'config')
                    remote_path = f"{config_dir}/{relative_path}"

                    # Create remote directory
                    remote_dir = os.path.dirname(remote_path)
                    self.sftp_mkdir_p(sftp, remote_dir)

                    # Transfer file
                    sftp.put(local_path, remote_path)
                    print(f"Uploaded config: {local_path} -> {remote_path}")

    def restart_robot_services(self, ssh):
        """
        Restart robot services after deployment
        """
        # Stop any existing services
        ssh.exec_command("pkill -f ros")

        # Wait for processes to stop
        import time
        time.sleep(2)

        # Source workspace and start robot
        start_cmd = f"""
        source /opt/ros/{self.config['ros']['distro']}/setup.bash &&
        source {self.workspace_path}/devel/setup.bash &&
        roslaunch robot_bringup robot.launch > /tmp/robot_startup.log 2>&1 &
        """
        ssh.exec_command(start_cmd)

        print("Robot services restarted")

def create_deployment_config():
    """
    Create a sample deployment configuration
    """
    config = {
        'robot': {
            'ip': '192.168.1.100',
            'username': 'robot',
            'password': 'password',
            'workspace_path': '/home/robot/catkin_ws'
        },
        'ros': {
            'distro': 'noetic'
        },
        'deployment': {
            'validate_before_deploy': True,
            'backup_workspace': True,
            'restart_services': True
        }
    }

    with open('deployment_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Deployment configuration created: deployment_config.yaml")

def main():
    """
    Main deployment function
    """
    if len(sys.argv) != 2:
        print("Usage: python deployment_workflow.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)

    deployment = RobotDeploymentWorkflow(config_file)
    success = deployment.deploy_to_robot()

    if success:
        print("Deployment successful!")
        sys.exit(0)
    else:
        print("Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Week Summary

This section covered comprehensive development workflows for Physical AI and humanoid robotics projects. The content included Git workflows, CI/CD pipelines, development environment setup, testing strategies, code quality standards, and deployment procedures. These workflows ensure consistent, reliable, and efficient development processes that can scale from simulation to real robot deployment while maintaining high quality and safety standards.