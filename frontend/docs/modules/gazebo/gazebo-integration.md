---
sidebar_position: 5
---

# Gazebo Integration

## Introduction to Gazebo Integration

Gazebo integration involves connecting simulation environments with real-world systems, development frameworks, and other tools. The most common integration is with ROS (Robot Operating System), which allows seamless communication between simulated and real robots. This section covers various integration techniques and best practices for incorporating Gazebo into robotics development workflows.

## ROS-Gazebo Integration

### Gazebo ROS Packages

The `gazebo_ros` package provides the core integration between Gazebo and ROS:

```xml
<!-- In package.xml -->
<depend>gazebo_ros</depend>
<depend>gazebo_plugins</depend>
<depend>gazebo_msgs</depend>
```

### Launch File Integration

```xml
<!-- launch/gazebo_simulation.launch -->
<launch>
  <!-- Set Gazebo-specific arguments -->
  <arg name="world" default="worlds/empty.world"/>
  <arg name="paused" default="false"/>
  <arg name="use_sim_time" default="true"/>
  <arg name="gui" default="true"/>
  <arg name="headless" default="false"/>
  <arg name="debug" default="false"/>

  <!-- Launch Gazebo with custom world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(arg world)"/>
    <arg name="paused" value="$(arg paused)"/>
    <arg name="use_sim_time" value="$(arg use_sim_time)"/>
    <arg name="gui" value="$(arg gui)"/>
    <arg name="headless" value="$(arg headless)"/>
    <arg name="debug" value="$(arg debug)"/>
  </include>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-file $(find my_robot_description)/urdf/my_robot.urdf
              -urdf -model my_robot -x 0 -y 0 -z 0.1"
        respawn="false" output="screen"/>
</launch>
```

## Robot Description Integration

### URDF with Gazebo Elements

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.5</mu1>
    <mu2>0.5</mu2>
  </gazebo>

  <!-- Joints -->
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
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
      <leftJoint>left_wheel_joint</leftJoint>
      <rightJoint>right_wheel_joint</rightJoint>
      <wheelSeparation>0.3</wheelSeparation>
      <wheelDiameter>0.2</wheelDiameter>
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <robotBaseFrame>base_link</robotBaseFrame>
      <publishWheelTF>false</publishWheelTF>
      <publishOdomTF>true</publishOdomTF>
      <odometrySource>world</odometrySource>
      <updateRate>30</updateRate>
    </plugin>
  </gazebo>
</robot>
```

## Common Gazebo Plugins

### Differential Drive Plugin

```xml
<gazebo>
  <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
    <leftJoint>left_wheel_joint</leftJoint>
    <rightJoint>right_wheel_joint</rightJoint>
    <wheelSeparation>0.3</wheelSeparation>
    <wheelDiameter>0.2</wheelDiameter>
    <commandTopic>cmd_vel</commandTopic>
    <odometryTopic>odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <robotBaseFrame>base_link</robotBaseFrame>
    <publishWheelTF>false</publishWheelTF>
    <publishOdomTF>true</publishOdomTF>
    <odometrySource>world</odometrySource>
    <updateRate>30</updateRate>
    <legacyMode>false</legacyMode>
  </plugin>
</gazebo>
```

### Joint State Publisher Plugin

```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <jointName>joint1, joint2, joint3</jointName>
    <updateRate>30</updateRate>
    <alwaysOn>true</alwaysOn>
  </plugin>
</gazebo>
```

### IMU Sensor Plugin

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topicName>imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <updateRateHZ>100.0</updateRateHZ>
      <gaussianNoise>0.01</gaussianNoise>
      <xyzOffset>0 0 0</xyzOffset>
      <rpyOffset>0 0 0</rpyOffset>
      <frameName>imu_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Plugin

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>my_camera</cameraName>
      <imageTopicName>image_raw</imageTopicName>
      <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      <frameName>camera_link</frameName>
      <hackBaseline>0.07</hackBaseline>
      <distortionK1>0.0</distortionK1>
      <distortionK2>0.0</distortionK2>
      <distortionK3>0.0</distortionK3>
      <distortionT1>0.0</distortionT1>
      <distortionT2>0.0</distortionT2>
    </plugin>
  </sensor>
</gazebo>
```

## Advanced Integration Techniques

### Custom Controller Integration

```cpp
// custom_controller.cpp
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>

namespace my_robot_controller
{
  class CustomController : public controller_interface::Controller<hardware_interface::VelocityJointInterface>
  {
  public:
    bool init(hardware_interface::VelocityJointInterface* hw, ros::NodeHandle& n)
    {
      // Get joint name from parameter server
      std::string joint_name;
      if (!n.getParam("joint", joint_name)) {
        ROS_ERROR("No joint given in namespace: %s", n.getNamespace().c_str());
        return false;
      }

      // Get joint handle
      try {
        joint_ = hw->getHandle(joint_name);
      } catch (const hardware_interface::HardwareInterfaceException& e) {
        ROS_ERROR_STREAM("Exception thrown: " << e.what());
        return false;
      }

      // Initialize controller parameters
      n.getParam("kp", kp_);
      n.getParam("max_velocity", max_velocity_);

      return true;
    }

    void update(const ros::Time& time, const ros::Duration& period)
    {
      // Simple PD controller example
      double error = target_position_ - joint_.getPosition();
      double vel_cmd = kp_ * error;

      // Apply velocity limits
      if (vel_cmd > max_velocity_) vel_cmd = max_velocity_;
      if (vel_cmd < -max_velocity_) vel_cmd = -max_velocity_;

      joint_.setCommand(vel_cmd);
    }

  private:
    hardware_interface::JointHandle joint_;
    double kp_;
    double max_velocity_;
    double target_position_;
  };

  PLUGINLIB_EXPORT_CLASS(my_robot_controller::CustomController, controller_interface::ControllerBase)
}
```

### ROS 2 Integration

For ROS 2, the integration uses different packages and approaches:

```python
# launch/gazebo_simulation_ros2.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('my_robot_gazebo'),
                    'worlds',
                    'my_world.world'
                ])
            }.items()
        ),

        # Spawn robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'my_robot',
                '-x', '0', '-y', '0', '-z', '0.1'
            ],
            output='screen'
        )
    ])
```

## Simulation vs. Real Robot Transitions

### Hardware Abstraction

```cpp
// hardware_interface.h
class RobotHardwareInterface
{
public:
    virtual void read() = 0;
    virtual void write() = 0;
    virtual bool init() = 0;
};

// simulation_hardware_interface.h
class SimulationHardwareInterface : public RobotHardwareInterface
{
public:
    bool init() override {
        // Initialize Gazebo interfaces
        return true;
    }

    void read() override {
        // Read from Gazebo simulation
    }

    void write() override {
        // Write to Gazebo simulation
    }
};

// real_hardware_interface.h
class RealHardwareInterface : public RobotHardwareInterface
{
public:
    bool init() override {
        // Initialize real hardware interfaces
        return true;
    }

    void read() override {
        // Read from real sensors
    }

    void write() override {
        // Write to real actuators
    }
};
```

### Parameter-Based Switching

```yaml
# config/robot_config.yaml
simulation:
  use_sim_time: true
  joint_state_topic: "joint_states"
  cmd_vel_topic: "cmd_vel"

real_robot:
  use_sim_time: false
  joint_state_topic: "joint_states"
  cmd_vel_topic: "cmd_vel"
```

```cpp
// controller.cpp
void Controller::init(ros::NodeHandle& nh) {
    bool use_simulation;
    nh.getParam("use_simulation", use_simulation);

    if (use_simulation) {
        joint_state_sub_ = nh.subscribe("joint_states", 1, &Controller::jointStateCallback, this);
        cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    } else {
        // Real robot interfaces
        joint_state_sub_ = nh.subscribe("real_joint_states", 1, &Controller::jointStateCallback, this);
        cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("real_cmd_vel", 1);
    }
}
```

## Performance Optimization

### Simulation Optimization

1. **Update Rates**: Balance accuracy with performance
2. **Physics Settings**: Adjust solver parameters appropriately
3. **Sensor Configuration**: Optimize sensor update rates
4. **Model Complexity**: Use appropriate level of detail

### Communication Optimization

```cpp
// Efficient message publishing
class OptimizedPublisher
{
public:
    OptimizedPublisher(ros::NodeHandle& nh) :
        pub_(nh.advertise<SensorMsgs>("topic", 1, true)) // latch for static transforms
    {
        // Use appropriate queue sizes
        last_publish_time_ = ros::Time::now();
    }

    void publishIfNeeded(const SensorMsgs& msg) {
        ros::Time current_time = ros::Time::now();

        // Rate limiting
        if ((current_time - last_publish_time_).toSec() > 1.0/update_rate_) {
            pub_.publish(msg);
            last_publish_time_ = current_time;
        }
    }

private:
    ros::Publisher pub_;
    ros::Time last_publish_time_;
    double update_rate_ = 30.0; // Hz
};
```

## Debugging Integration Issues

### Common Problems and Solutions

1. **TF Issues**: Ensure proper frame relationships
2. **Timing**: Synchronize simulation and ROS time
3. **Topics**: Verify topic names and message types
4. **Permissions**: Check file permissions for plugins

### Debugging Tools

```bash
# Check TF tree
rosrun tf tf_echo base_link laser_link

# Monitor topics
rostopic echo /joint_states

# Check Gazebo topics
gz topic -l

# Debug with RQT
rqt
```

## Best Practices

### Architecture

- Use hardware abstraction layers
- Separate simulation and real code cleanly
- Implement proper error handling
- Follow ROS conventions

### Configuration

- Use parameter servers for configuration
- Provide default configurations
- Document configuration parameters
- Use launch files for complex setups

### Testing

- Test simulation and real systems separately
- Validate sensor data consistency
- Check timing and synchronization
- Verify controller performance

## Week Summary

This section covered the integration of Gazebo with ROS and other systems, including plugin usage, hardware abstraction, and best practices for simulation-development workflows. Proper integration enables seamless transition between simulation and real robot development.