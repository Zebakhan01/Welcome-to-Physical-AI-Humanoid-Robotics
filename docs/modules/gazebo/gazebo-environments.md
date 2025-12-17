---
sidebar_position: 4
---

# Gazebo Environments

## Introduction to Gazebo Environments

Gazebo environments define the complete simulation world where robots operate. This includes the physical world properties, terrain, objects, lighting, and other environmental factors that affect robot behavior. Creating realistic and useful environments is crucial for effective robotics simulation and development.

## World File Structure

### Basic World Definition

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Scene properties -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.0 -1.0</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Models in the world -->
    <model name="my_robot">
      <!-- Robot model definition -->
    </model>
  </world>
</sdf>
```

## Physics Configuration

### Physics Engine Options

```xml
<!-- ODE (Open Dynamics Engine) -->
<physics name="ode_physics" type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>

<!-- Bullet Physics -->
<physics name="bullet_physics" type="bullet">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <bullet>
    <solver>
      <type>dantzig</type>
      <iters>50</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
    </constraints>
  </bullet>
</physics>
```

## Terrain and Ground Models

### Flat Ground Plane

```xml
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
        <contact>
          <ode>
            <kp>1e+16</kp>
            <kd>1e+13</kd>
          </ode>
        </contact>
      </surface>
    </collision>
    <visual name="visual">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <material>
        <script>
          <name>Gazebo/Grey</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

### Complex Terrain

```xml
<model name="complex_terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="terrain_collision">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="terrain_visual">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
      <material>
        <script>
          <name>Gazebo/Dirt</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

## Environmental Objects

### Static Objects

```xml
<!-- Table -->
<model name="table">
  <pose>2 0 0 0 0 0</pose>
  <static>true</static>
  <link name="table_top">
    <inertial>
      <mass>10</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.5 0.8 0.02</size>
        </box>
      </geometry>
      <material>
        <script>
          <name>Gazebo/Wood</name>
        </script>
      </material>
    </visual>
    <collision name="collision">
      <geometry>
        <box>
          <size>1.5 0.8 0.02</size>
        </box>
      </geometry>
    </collision>
  </link>
  <link name="leg1">
    <pose>-0.6 -0.35 0 0 0 0</pose>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.05 0.05 0.7</size>
        </box>
      </geometry>
      <material>
        <script>
          <name>Gazebo/Wood</name>
        </script>
      </material>
    </visual>
    <collision name="collision">
      <geometry>
        <box>
          <size>0.05 0.05 0.7</size>
        </box>
      </geometry>
    </collision>
  </link>
</model>
```

### Dynamic Objects

```xml
<!-- Moving object -->
<model name="moving_object">
  <pose>0 0 1 0 0 0</pose>
  <link name="object_link">
    <inertial>
      <mass>1</mass>
      <inertia>
        <ixx>0.1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.1</iyy>
        <iyz>0</iyz>
        <izz>0.1</izz>
      </inertia>
    </inertial>
    <visual name="visual">
      <geometry>
        <sphere>
          <radius>0.1</radius>
        </sphere>
      </geometry>
      <material>
        <script>
          <name>Gazebo/Red</name>
        </script>
      </material>
    </visual>
    <collision name="collision">
      <geometry>
        <sphere>
          <radius>0.1</radius>
        </sphere>
      </geometry>
    </collision>
  </link>
</model>
```

## Lighting and Atmosphere

### Directional Light (Sun)

```xml
<light name="sun" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.3 0.0 -1.0</direction>
  <cast_shadows>true</cast_shadows>
</light>
```

### Point Light

```xml
<light name="room_light" type="point">
  <pose>0 0 3 0 0 0</pose>
  <diffuse>0.9 0.9 0.9 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>5</range>
    <constant>0.2</constant>
    <linear>0.5</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>
```

### Atmospheric Effects

```xml
<scene>
  <ambient>0.3 0.3 0.3 1</ambient>
  <background>0.6 0.7 0.8 1</background>
  <shadows>true</shadows>
  <fog>
    <type>linear</type>
    <color>0.8 0.8 0.8 1</color>
    <density>0.01</density>
    <start>10</start>
    <end>50</end>
  </fog>
</scene>
```

## Complex World Examples

### Indoor Environment

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_world">
    <!-- Physics -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Scene -->
    <scene>
      <ambient>0.3 0.3 0.3 1</ambient>
      <background>0.8 0.8 0.8 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <light name="ceiling_light" type="point">
      <pose>0 0 2.5 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.1</constant>
        <linear>0.2</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>

    <!-- Ground -->
    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/White</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="wall_north">
      <static>true</static>
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/White</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <static>true</static>
      <!-- Table definition -->
    </model>

    <!-- Robot -->
    <model name="my_robot">
      <pose>0 0 0.1 0 0 0</pose>
      <!-- Robot model -->
    </model>
  </world>
</sdf>
```

### Outdoor Environment

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="outdoor_world">
    <!-- Physics -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Scene with fog -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.8 1.0 1</background>
      <shadows>true</shadows>
      <fog>
        <type>exp</type>
        <color>0.9 0.9 0.9 1</color>
        <density>0.002</density>
      </fog>
    </scene>

    <!-- Sun -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 -0.3 -1.0</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Terrain features -->
    <model name="hill">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>model://terrain/meshes/hill.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://terrain/meshes/hill.dae</uri>
            </mesh>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Dirt</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Trees and obstacles -->
    <model name="tree_1">
      <pose>5 5 0 0 0 0</pose>
      <static>true</static>
      <!-- Tree model -->
    </model>

    <model name="rock_1">
      <pose>-3 -2 0 0 0 0</pose>
      <static>true</static>
      <!-- Rock model -->
    </model>
  </world>
</sdf>
```

## World Spawn and Management

### Spawning Models at Runtime

```cpp
#include <gazebo_msgs/SpawnModel.h>
#include <ros/ros.h>

bool spawnModel(const std::string& model_name,
                const geometry_msgs::Pose& pose) {
  ros::NodeHandle nh;
  ros::ServiceClient spawn_client =
      nh.serviceClient<gazebo_msgs::SpawnModel>("/gazebo/spawn_sdf_model");

  gazebo_msgs::SpawnModel spawn_srv;
  spawn_srv.request.model_name = model_name;
  spawn_srv.request.initial_pose = pose;

  // Load model from file or string
  std::ifstream model_file("path/to/model.sdf");
  std::string model_xml;
  std::string line;
  while (std::getline(model_file, line)) {
    model_xml += line + "\n";
  }

  spawn_srv.request.model_xml = model_xml;
  spawn_srv.request.robot_namespace = "";

  if (spawn_client.call(spawn_srv)) {
    if (spawn_srv.response.success) {
      ROS_INFO("Model spawned successfully");
      return true;
    } else {
      ROS_ERROR("Spawn service failed: %s",
                spawn_srv.response.status_message.c_str());
      return false;
    }
  } else {
    ROS_ERROR("Failed to call spawn service");
    return false;
  }
}
```

## Environment Optimization

### Performance Considerations

1. **Model Complexity**: Balance visual quality with performance
2. **Collision Geometry**: Use simple shapes for collision detection
3. **Update Rates**: Optimize physics and rendering rates
4. **Lighting**: Use appropriate lighting complexity
5. **LOD (Level of Detail)**: Implement distance-based detail reduction

### Memory Management

- Use static models when possible
- Optimize mesh complexity
- Manage texture memory efficiently
- Consider streaming for large environments

## Custom Environment Development

### Creating Custom Models

```bash
# Model directory structure
my_environment/
├── models/
│   ├── custom_table/
│   │   ├── model.sdf
│   │   ├── model.config
│   │   └── meshes/
│   │       └── table.dae
│   └── custom_obstacle/
│       ├── model.sdf
│       ├── model.config
│       └── meshes/
├── worlds/
│   └── custom_world.world
└── materials/
    └── textures/
```

### Environment Configuration

```xml
<!-- In world file -->
<world name="custom_environment">
  <!-- Include custom models -->
  <include>
    <uri>model://custom_table</uri>
    <pose>2 0 0 0 0 0</pose>
  </include>

  <include>
    <uri>model://custom_obstacle</uri>
    <pose>-1 1 0 0 0 1.57</pose>
  </include>

  <!-- Custom plugins for environment -->
  <plugin name="weather_plugin" filename="libweather_plugin.so">
    <wind_speed>2.0</wind_speed>
    <rain_intensity>0.0</rain_intensity>
  </plugin>
</world>
```

## Best Practices

### Organization

- Use consistent naming conventions
- Organize models in logical hierarchies
- Document complex environments
- Version control environment files

### Testing

- Test environments with various robots
- Validate physics interactions
- Check performance under load
- Verify sensor functionality

### Reusability

- Create modular, reusable components
- Use parameterization for flexibility
- Provide multiple configuration options
- Include example usage scenarios

## Week Summary

This section covered the creation and configuration of Gazebo environments, including world files, physics settings, lighting, and complex scene construction. Understanding how to create realistic and efficient simulation environments is crucial for effective robotics development and testing.