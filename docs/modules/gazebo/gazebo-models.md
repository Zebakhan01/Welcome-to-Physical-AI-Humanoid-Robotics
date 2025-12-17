---
sidebar_position: 2
---

# Gazebo Models

## Introduction to Gazebo Models

Gazebo models are the fundamental building blocks of simulation environments. They define the physical properties, visual appearance, and behavior of objects in the simulation. Understanding how to create and configure models is essential for effective robotics simulation.

## SDF (Simulation Description Format)

### SDF Structure

SDF is an XML-based format that describes objects in Gazebo. The basic structure includes:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <!-- Model properties and components -->
  </model>
</sdf>
```

### Model Definition

```xml
<sdf version="1.7">
  <model name="simple_robot">
    <pose>0 0 0.5 0 0 0</pose>

    <!-- Links define the physical parts of the robot -->
    <link name="chassis">
      <pose>0 0 0 0 0 0</pose>

      <!-- Inertial properties -->
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <!-- Visual properties -->
      <visual name="chassis_visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>

      <!-- Collision properties -->
      <collision name="chassis_collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joints connect links -->
    <joint name="wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="wheel">
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.001</iyy>
          <iyz>0</iyz>
          <izz>0.002</izz>
        </inertia>
      </inertial>

      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>

      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

## Robot Model Components

### Links

Links represent rigid bodies in the model:

```xml
<link name="sensor_link">
  <!-- Inertial properties are crucial for physics simulation -->
  <inertial>
    <mass>0.05</mass>
    <inertia>
      <ixx>0.0001</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.0001</iyy>
      <iyz>0</iyz>
      <izz>0.0001</izz>
    </inertia>
  </inertial>

  <!-- Visual representation -->
  <visual name="sensor_visual">
    <geometry>
      <box>
        <size>0.02 0.02 0.02</size>
      </box>
    </geometry>
    <material>
      <ambient>0.5 0.5 1 1</ambient>
      <diffuse>0.5 0.5 1 1</diffuse>
    </material>
  </visual>

  <!-- Collision geometry -->
  <collision name="sensor_collision">
    <geometry>
      <box>
        <size>0.02 0.02 0.02</size>
      </box>
    </geometry>
  </collision>
</link>
```

### Joints

Joints connect links and define their relative motion:

```xml
<!-- Revolute joint (rotational) -->
<joint name="revolute_joint" type="revolute">
  <parent>link1</parent>
  <child>link2</child>
  <axis>
    <xyz>0 0 1</xyz>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
      <effort>10</effort>
      <velocity>1</velocity>
    </limit>
  </axis>
</joint>

<!-- Prismatic joint (linear) -->
<joint name="prismatic_joint" type="prismatic">
  <parent>base_link</parent>
  <child>slider</child>
  <axis>
    <xyz>1 0 0</xyz>
    <limit>
      <lower>-0.1</lower>
      <upper>0.1</upper>
      <effort>50</effort>
      <velocity>0.5</velocity>
    </limit>
  </axis>
</joint>

<!-- Fixed joint (no motion) -->
<joint name="fixed_joint" type="fixed">
  <parent>base_link</parent>
  <child>sensor_mount</child>
</joint>
```

## Sensor Models

### Camera Sensor

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LIDAR Sensor

```xml
<sensor name="lidar" type="ray">
  <pose>0.1 0 0.2 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensor

```xml
<sensor name="imu" type="imu">
  <pose>0 0 0.1 0 0 0</pose>
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Model Configuration Files

### Model.config

Every model should have a configuration file:

```xml
<?xml version="1.0"?>
<model>
  <name>my_robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>

  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>

  <description>
    A simple robot model for Gazebo simulation.
  </description>
</model>
```

## URDF to SDF Conversion

### Using xacro for Complex Models

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complex_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="prefix *joint_pose">
    <link name="${prefix}_wheel">
      <inertial>
        <mass value="0.5" />
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02" />
      </inertial>

      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1" />
        </material>
      </visual>

      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </collision>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="joint_pose" />
      <axis xyz="0 1 0" />
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1" />
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
    </collision>

    <inertial>
      <mass value="10" />
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1" />
    </inertial>
  </link>

  <!-- Wheels -->
  <xacro:wheel prefix="front_left">
    <origin xyz="0.2 0.2 0" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="front_left_wheel" />
  </xacro:wheel>

  <xacro:wheel prefix="front_right">
    <origin xyz="0.2 -0.2 0" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="front_right_wheel" />
  </xacro:wheel>
</robot>
```

## Model Best Practices

### Performance Optimization

1. **Simplify Collision Geometry**: Use simple shapes for collision detection
2. **Level of Detail**: Use detailed meshes only for visualization
3. **Appropriate Inertia Values**: Ensure realistic physical behavior
4. **Sensor Optimization**: Balance accuracy with performance

### File Organization

```
my_robot/
├── model.sdf
├── model.config
├── meshes/
│   ├── chassis.dae
│   ├── wheel.dae
│   └── sensor.dae
├── materials/
│   └── textures/
└── plugins/
    └── custom_controller.so
```

## Advanced Model Features

### Transmission Elements

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

```xml
<gazebo reference="chassis">
  <material>Gazebo/Blue</material>
  <mu1>0.5</mu1>
  <mu2>0.5</mu2>
</gazebo>

<gazebo reference="wheel_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

## Model Validation

### Checking Model Integrity

Before using a model in simulation:

1. **SDF Validation**: Use `gz sdf -k model.sdf` to check syntax
2. **Physical Properties**: Verify mass, inertia, and joint limits
3. **Collision Detection**: Ensure collision geometry is properly defined
4. **Visual Quality**: Check that visual elements appear correctly

## Week Summary

This section covered the fundamentals of creating and configuring Gazebo models, including SDF structure, link and joint definitions, sensor integration, and best practices for model development. Proper model creation is essential for effective robotics simulation and development.