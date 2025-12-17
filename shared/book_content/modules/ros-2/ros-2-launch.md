---
sidebar_position: 4
---

# ROS 2 Launch

## Introduction to ROS 2 Launch System

The ROS 2 launch system provides a powerful and flexible way to start multiple nodes and configure their parameters. Unlike ROS 1's XML-based launch files, ROS 2 uses Python for launch descriptions, offering greater flexibility and programmability.

## Launch File Structure

### Basic Launch File

```python
# launch/basic_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            output='screen'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            output='screen',
            prefix='xterm -e'  # Run in separate terminal
        )
    ])
```

### Launch Arguments

```python
# launch/launch_with_args.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot1',
        description='Robot namespace'
    )

    # Use launch configuration
    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        namespace_arg,
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=namespace,
            parameters=[
                {'robot_namespace': namespace},
                'config/robot_config.yaml'
            ]
        )
    ])
```

## Advanced Launch Features

### Conditional Launch

```python
from launch import LaunchDescription, LaunchCondition
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Declare configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_viz = LaunchConfiguration('enable_viz')

    return LaunchDescription([
        # Launch simulation if use_sim_time is true
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(['path/to/sim.launch.py']),
            condition=IfCondition(use_sim_time)
        ),
        # Launch visualization if enabled
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            condition=IfCondition(enable_viz)
        )
    ])
```

### Launch Substitutions

```python
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable(
            name='RCUTILS_LOGGING_SEVERITY_THRESHOLD',
            value='INFO'
        ),
        Node(
            package='my_robot_package',
            executable='robot_node',
            name='robot_node',
            parameters=[
                PathJoinSubstitution([
                    'config',
                    [LaunchConfiguration('robot_model'), '.yaml']
                ])
            ]
        )
    ])
```

## Launch Actions

### Node Actions

```python
from launch_ros.actions import Node

# Basic node
basic_node = Node(
    package='package_name',
    executable='executable_name',
    name='node_name',
    namespace='namespace',
    parameters=[
        {'param1': 'value1'},
        'path/to/params.yaml'
    ],
    remappings=[
        ('original_topic', 'new_topic'),
        ('original_service', 'new_service')
    ],
    arguments=['--arg1', 'value1'],
    output='screen'
)

# Lifecycle node
lifecycle_node = Node(
    package='lifecycle',
    executable='lifecycle_talker',
    name='lc_talker',
    parameters=[{'use_intra_process_comms': True}]
)
```

### Group Actions

```python
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        # Group with namespace
        GroupAction(
            actions=[
                PushRosNamespace('robot1'),
                Node(
                    package='navigation',
                    executable='amcl',
                    name='amcl'
                ),
                Node(
                    package='navigation',
                    executable='move_base',
                    name='move_base'
                )
            ]
        )
    ])
```

## Parameter Management

### YAML Parameter Files

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    use_sim_time: false
    robot_radius: 0.3
    max_velocity: 1.0
    acceleration_limits:
      linear: 2.0
      angular: 3.14

robot_controller:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
```

### Parameter Loading in Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    config_file = LaunchConfiguration('config_file')

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value='config/default.yaml',
        description='Configuration file path'
    )

    return LaunchDescription([
        declare_config_file,
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            parameters=[
                config_file,  # Load from launch argument
                {'param_override': 'value'},  # Override specific parameters
            ]
        )
    ])
```

## Complex Launch Scenarios

### Multi-Robot Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    num_robots = LaunchConfiguration('num_robots')
    declare_num_robots = DeclareLaunchArgument(
        'num_robots',
        default_value='2',
        description='Number of robots to launch'
    )

    launch_description = LaunchDescription([declare_num_robots])

    # Dynamically create robot nodes
    for i in range(int(num_robots.perform(None))):
        robot_name = f'robot_{i}'
        launch_description.add_action(
            Node(
                package='my_robot_package',
                executable='robot_node',
                name=f'{robot_name}_node',
                namespace=robot_name,
                parameters=[
                    {'robot_name': robot_name},
                    f'config/{robot_name}.yaml'
                ]
            )
        )

    return launch_description
```

### Launch File Inclusion

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Include other launch files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    'path',
                    'to',
                    'other_launch_file.py'
                ])
            )
        ),
        # Additional nodes
        Node(
            package='my_robot_package',
            executable='monitor_node',
            name='system_monitor'
        )
    ])
```

## Launch Configuration Patterns

### Environment-Specific Launch

```python
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, LogInfo
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable(
            name='ROS_DOMAIN_ID',
            value='1'
        ),
        SetEnvironmentVariable(
            name='RCUTILS_LOGGING_USE_STDOUT',
            value='1'
        ),
        LogInfo(
            msg=['ROS_DOMAIN_ID is: ', EnvironmentVariable(name='ROS_DOMAIN_ID')]
        ),
        Node(
            package='my_robot_package',
            executable='robot_node',
            name='robot_node'
        )
    ])
```

### Conditional Parameter Loading

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Environment configuration
    sim_time = LaunchConfiguration('use_sim_time')
    debug_mode = LaunchConfiguration('debug')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('debug', default_value='false'),

        Node(
            package='navigation',
            executable='nav2_bringup',
            name='navigator',
            parameters=[
                'config/navigation.yaml',
                {'use_sim_time': sim_time}
            ],
            condition=IfCondition(LaunchConfiguration('navigation_enabled'))
        )
    ])
```

## Launch Testing

### Launch Test Integration

```python
# test/launch_test.py
import launch
import launch_ros.actions
import launch_testing.actions
import pytest

@pytest.mark.launch_test
def generate_test_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='my_robot_package',
            executable='test_node',
            name='test_node'
        ),
        launch_testing.actions.ReadyToTest()
    ])

def test_node_launch(test_runner):
    # Test that node launches successfully
    assert test_runner.get_node_by_name('test_node') is not None
```

## Best Practices

### Organizational Best Practices

- **Modular Launch Files**: Break complex systems into smaller launch files
- **Parameter Separation**: Keep parameters in separate YAML files
- **Naming Conventions**: Use consistent naming for launch files
- **Documentation**: Comment complex launch logic

### Performance Considerations

- **Lazy Loading**: Use conditional launches to avoid unnecessary nodes
- **Resource Management**: Monitor resource usage of launched systems
- **Error Handling**: Implement proper error handling in launch files
- **Debugging**: Use appropriate output settings for debugging

### Security Considerations

- **Namespace Isolation**: Use namespaces to isolate robot systems
- **Parameter Validation**: Validate launch parameters
- **Access Control**: Consider security implications of launched nodes
- **Environment Variables**: Secure environment variable settings

## Common Launch Patterns

### Bringup Launch Files

Standard pattern for robot bringup:

```python
# launch/robot_bringup.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Common launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time},
                'config/robot_description.yaml'
            ]
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'use_sim_time': use_sim_time}]
        )
    ])
```

## Week Summary

This section covered the comprehensive ROS 2 launch system, including basic launch files, advanced features like conditional launching, parameter management, and best practices. The Python-based launch system provides powerful capabilities for managing complex robotic systems with proper configuration and organization.