---
sidebar_position: 3
---

# ROS 2 Packages

## Package Management in ROS 2

ROS 2 packages are the fundamental building blocks of ROS 2 applications. Unlike ROS 1, ROS 2 uses the colcon build system and follows a more standardized approach to package management and dependencies.

## Package Structure

### Standard Package Layout

```
my_robot_package/
├── CMakeLists.txt          # Build configuration
├── package.xml             # Package metadata and dependencies
├── src/                    # Source code
│   ├── main.cpp
│   └── my_robot_node.cpp
├── include/                # Header files
│   └── my_robot_package/
├── launch/                 # Launch files
│   └── robot.launch.py
├── config/                 # Configuration files
│   └── robot_params.yaml
├── msg/                    # Custom message definitions
├── srv/                    # Custom service definitions
├── action/                 # Custom action definitions
└── test/                   # Test files
```

### package.xml

The package.xml file defines metadata and dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.1.0</version>
  <description>My robot package for ROS 2</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Build System: Colcon

### Colcon vs Catkin

- **Colcon**: Modern, modular build system
- **Parallel Builds**: Faster compilation
- **Multiple Build Systems**: Support for CMake, Ament, etc.
- **Clean Architecture**: Better separation of concerns

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Create executable
add_executable(robot_node
  src/robot_node.cpp
)

# Link libraries
ament_target_dependencies(robot_node
  rclcpp
  std_msgs
  sensor_msgs
)

# Install targets
install(TARGETS
  robot_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Package configuration
ament_package()
```

## Creating Custom Messages

### Message Definition (.msg)

Custom messages are defined in .msg files:

```
# In msg/RobotStatus.msg
string robot_name
int32 battery_level
float64[] joint_positions
bool is_moving
geometry_msgs/Pose current_pose
```

### Service Definition (.srv)

```
# In srv/MoveRobot.srv
geometry_msgs/Pose target_pose
---
bool success
string message
```

### Action Definition (.action)

```
# In action/Navigation.action
# Goal definition
geometry_msgs/PoseStamped target_pose
float32 tolerance

# Result definition
bool success
float32 distance_traveled

# Feedback definition
float32 distance_remaining
geometry_msgs/Pose current_pose
```

## Package Dependencies

### Build Dependencies

Dependencies needed during compilation:
- `buildtool_depend`: Build system requirements
- `build_depend`: Compilation dependencies
- `build_export_depend`: Dependencies for packages using this package

### Execution Dependencies

Dependencies needed at runtime:
- `exec_depend`: Runtime dependencies
- `test_depend`: Testing dependencies

## Advanced Package Features

### Interface Definition Packages

For packages that only define messages, services, and actions:

```xml
<package format="3">
  <name>my_robot_interfaces</name>
  <!-- ... other metadata ... -->
  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Composition Packages

For node composition:

```cmake
# In CMakeLists.txt
find_package(rclcpp_components REQUIRED)

add_library(my_robot_components SHARED)
target_sources(my_robot_components PRIVATE
  src/motion_controller.cpp
  src/perception_node.cpp
)
rclcpp_components_register_nodes(my_robot_components "my_robot_package::MotionController")
rclcpp_components_register_nodes(my_robot_components "my_robot_package::PerceptionNode")

install(TARGETS my_robot_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
```

## Package Management Tools

### rosdep

Manage system dependencies:

```bash
# Install dependencies for current workspace
rosdep install --from-paths src --ignore-src -r -y

# Check dependencies
rosdep check --from-paths src --ignore-src
```

### vcs (vcstool)

Manage multiple repositories:

```bash
# Import repositories from .repos file
vcs import src < repositories.repos

# Update all repositories
vcs pull src
```

## Best Practices for Package Development

### Package Naming

- Use lowercase with underscores
- Be descriptive but concise
- Follow ROS naming conventions
- Avoid generic names

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version numbers appropriately
- Tag releases in version control
- Maintain changelogs

### Documentation

- Include README.md with package overview
- Document public APIs
- Provide usage examples
- Include configuration examples

### Testing

- Unit tests for individual components
- Integration tests for subsystems
- System tests for complete functionality
- Performance tests for critical paths

## Package Distribution

### Release Process

1. **Version Bump**: Update version in package.xml
2. **Changelog**: Update changelog with changes
3. **Tag**: Create Git tag for release
4. **Bloom**: Use bloom for Debian package generation
5. **Build Farm**: Submit to ROS build farm

### Repository Structure

```
my_robot_repo/
├── ros2/
│   ├── my_robot_core/
│   ├── my_robot_navigation/
│   ├── my_robot_manipulation/
│   └── my_robot_examples/
├── .github/
├── .gitlab/
└── README.md
```

## Common Package Patterns

### Driver Packages

- Interface with hardware devices
- Publish sensor data
- Subscribe to control commands
- Handle device-specific protocols

### Algorithm Packages

- Implement specific algorithms
- Provide reusable components
- Follow design patterns
- Include parameter configuration

### Application Packages

- Coordinate multiple components
- Implement complete behaviors
- Handle user interaction
- Manage system state

## Week Summary

This section covered the comprehensive approach to package management in ROS 2, including the colcon build system, dependency management, custom message definitions, and best practices for package development. Understanding these concepts is crucial for developing maintainable and reusable ROS 2 applications.