---
sidebar_position: 3
---

# Gazebo Plugins

## Introduction to Gazebo Plugins

Gazebo plugins are shared libraries that extend the functionality of the Gazebo simulation environment. They allow developers to customize simulation behavior, integrate with external systems like ROS, and add specialized simulation features. Understanding plugin development is crucial for creating sophisticated simulation environments.

## Plugin Types

### World Plugins

World plugins operate on the entire simulation world:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomWorldPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      this->world = _world;

      // Read plugin parameters from SDF
      if (_sdf->HasElement("update_rate"))
        this->updateRate = _sdf->Get<double>("update_rate");
      else
        this->updateRate = 1000; // default to 1kHz

      // Connect to pre-update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CustomWorldPlugin::OnUpdate, this));

      gzmsg << "Custom world plugin loaded with update rate: "
            << this->updateRate << " Hz\n";
    }

    public: void OnUpdate()
    {
      // Custom world update logic here
      common::Time simTime = this->world->SimTime();

      // Example: Apply custom forces or modify world properties
      // based on simulation time or other conditions
    }

    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;
    private: double updateRate;
  };

  GZ_REGISTER_WORLD_PLUGIN(CustomWorldPlugin)
}
```

### Model Plugins

Model plugins attach to specific models in the simulation:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomModelPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Get links
      this->link = _model->GetLink("link_name");
      if (!this->link)
      {
        gzerr << "Link not found!\n";
        return;
      }

      // Parse SDF parameters
      if (_sdf->HasElement("force_magnitude"))
        this->forceMagnitude = _sdf->Get<double>("force_magnitude");
      else
        this->forceMagnitude = 1.0;

      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CustomModelPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Apply forces or torques to the model
      math::Vector3 force(0, 0, this->forceMagnitude);
      this->link->AddForce(force);
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: physics::LinkPtr link;
    private: event::ConnectionPtr updateConnection;
    private: double forceMagnitude;
  };

  GZ_REGISTER_MODEL_PLUGIN(CustomModelPlugin)
}
```

### Sensor Plugins

Sensor plugins process data from Gazebo sensors:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomCameraPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Cast the sensor to a camera sensor
      this->cameraSensor =
          std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);

      if (!this->cameraSensor)
      {
        gzerr << "CustomCameraPlugin not attached to a camera sensor\n";
        return;
      }

      // Connect to camera data update
      this->newImageConnection = this->cameraSensor->Camera()->ConnectNewImageFrame(
          std::bind(&CustomCameraPlugin::OnNewFrame, this,
                   std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3, std::placeholders::_4,
                   std::placeholders::_5));
    }

    public: void OnNewFrame(const unsigned char * _image,
                           unsigned int _width, unsigned int _height,
                           unsigned int _depth, const std::string & _format)
    {
      // Process the image data
      // Example: Apply custom image processing or send to external system
      unsigned int pixelCount = _width * _height * _depth;

      // Perform custom image processing
      // ...

      // Example: Calculate average intensity
      double sum = 0;
      for (unsigned int i = 0; i < pixelCount; ++i)
      {
        sum += _image[i];
      }
      double average = sum / pixelCount;

      gzdbg << "Average pixel intensity: " << average << std::endl;
    }

    private: sensors::CameraSensorPtr cameraSensor;
    private: event::ConnectionPtr newImageConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomCameraPlugin)
}
```

## CMakeLists.txt for Plugin Development

```cmake
cmake_minimum_required(VERSION 3.5)
project(gazebo_plugins)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Gazebo
find_package(gazebo REQUIRED)

# Include Gazebo headers
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Create plugin library
add_library(custom_world_plugin SHARED custom_world_plugin.cc)
target_link_libraries(custom_world_plugin ${GAZEBO_LIBRARIES})

add_library(custom_model_plugin SHARED custom_model_plugin.cc)
target_link_libraries(custom_model_plugin ${GAZEBO_LIBRARIES})

add_library(custom_camera_plugin SHARED custom_camera_plugin.cc)
target_link_libraries(custom_camera_plugin ${GAZEBO_LIBRARIES})

# Install plugins
install(TARGETS custom_world_plugin custom_model_plugin custom_camera_plugin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
```

## ROS Integration Plugins

### ROS Publisher Plugin

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <boost/bind.hpp>

namespace gazebo
{
  class ROSPublisherPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;

      // Initialize ROS if not already initialized
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_client",
                 ros::init_options::NoSigintHandler);
      }

      // Create ROS node handle
      this->rosNode.reset(new ros::NodeHandle("gazebo_ros"));

      // Create publisher
      this->pub = this->rosNode->advertise<std_msgs::Float64>(
          "model_pose_z", 1);

      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          boost::bind(&ROSPublisherPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Get model position
      math::Pose pose = this->model->GetWorldPose();

      // Publish Z position
      std_msgs::Float64 msg;
      msg.data = pose.pos.z;
      this->pub.publish(msg);
    }

    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
    private: boost::shared_ptr<ros::NodeHandle> rosNode;
    private: ros::Publisher pub;
  };

  GZ_REGISTER_MODEL_PLUGIN(ROSPublisherPlugin)
}
```

### ROS Subscriber Plugin

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <boost/bind.hpp>

namespace gazebo
{
  class ROSSubscriberPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->link = _model->GetLink();

      // Initialize ROS
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_ros",
                 ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("gazebo_ros"));

      // Create subscriber
      this->sub = this->rosNode->subscribe<geometry_msgs::Twist>(
          "cmd_vel", 1, &ROSSubscriberPlugin::OnCmdVel, this);

      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          boost::bind(&ROSSubscriberPlugin::OnUpdate, this));
    }

    public: void OnCmdVel(const geometry_msgs::Twist::ConstPtr &_msg)
    {
      this->cmdVel = *_msg;
    }

    public: void OnUpdate()
    {
      // Apply velocity commands to the model
      math::Vector3 vel(this->cmdVel.linear.x,
                       this->cmdVel.linear.y,
                       this->cmdVel.linear.z);

      this->link->SetLinearVel(vel);

      math::Vector3 angVel(this->cmdVel.angular.x,
                          this->cmdVel.angular.y,
                          this->cmdVel.angular.z);

      this->link->SetAngularVel(angVel);
    }

    private: physics::ModelPtr model;
    private: physics::LinkPtr link;
    private: event::ConnectionPtr updateConnection;
    private: boost::shared_ptr<ros::NodeHandle> rosNode;
    private: ros::Subscriber sub;
    private: geometry_msgs::Twist cmdVel;
  };

  GZ_REGISTER_MODEL_PLUGIN(ROSSubscriberPlugin)
}
```

## Advanced Plugin Features

### Custom Physics Plugin

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomPhysicsPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      this->world = _world;

      // Set custom physics parameters
      physics::PhysicsEnginePtr physics = this->world->Physics();
      physics->SetGravity(math::Vector3(0, 0, -9.8));

      // Connect to pre-step event for custom physics
      this->preUpdateConnection = event::Events::ConnectPreRender(
          std::bind(&CustomPhysicsPlugin::OnPreRender, this));
    }

    public: void OnPreRender()
    {
      // Custom physics calculations before rendering
      // This is called before each physics update
    }

    private: physics::WorldPtr world;
    private: event::ConnectionPtr preUpdateConnection;
  };

  GZ_REGISTER_WORLD_PLUGIN(CustomPhysicsPlugin)
}
```

### Joint Control Plugin

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class JointControlPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;

      // Get joints
      this->joint = _model->GetJoint("joint_name");
      if (!this->joint)
      {
        gzerr << "Joint not found!\n";
        return;
      }

      // Set initial position
      this->joint->SetPosition(0, 0.0);

      // Connect to update
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&JointControlPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Simple PD controller
      double targetPos = 1.0; // Desired position
      double currentPos = this->joint->GetAngle(0).Radian();
      double error = targetPos - currentPos;

      double kp = 10.0; // Proportional gain
      double kd = 1.0;  // Derivative gain

      double vel = this->joint->GetVelocity(0);
      double force = kp * error - kd * vel;

      this->joint->SetForce(0, force);
    }

    private: physics::ModelPtr model;
    private: physics::JointPtr joint;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(JointControlPlugin)
}
```

## Plugin Configuration in SDF

### Adding Plugin to Model

```xml
<model name="robot_with_plugin">
  <link name="base_link">
    <pose>0 0 0.1 0 0 0</pose>
    <!-- Link definition here -->
  </link>

  <plugin name="custom_model_plugin" filename="libcustom_model_plugin.so">
    <update_rate>100</update_rate>
    <force_magnitude>5.0</force_magnitude>
  </plugin>
</model>
```

### World Plugin in World File

```xml
<sdf version="1.7">
  <world name="default">
    <!-- World definition here -->

    <plugin name="custom_world_plugin" filename="libcustom_world_plugin.so">
      <update_rate>50</update_rate>
    </plugin>
  </world>
</sdf>
```

## Plugin Best Practices

### Error Handling

```cpp
public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  this->model = _model;

  // Always check for null pointers
  if (!_model)
  {
    gzerr << "Model pointer is null!\n";
    return;
  }

  // Check for required SDF elements
  if (!_sdf->HasElement("required_param"))
  {
    gzerr << "Missing required parameter 'required_param'\n";
    return;
  }

  // Validate parameter values
  double param = _sdf->Get<double>("param_name");
  if (param <= 0)
  {
    gzerr << "Parameter must be positive, got: " << param << "\n";
    return;
  }

  // Continue with initialization
}
```

### Memory Management

- Use smart pointers when possible
- Be careful with callbacks and object lifetimes
- Avoid memory leaks in long-running simulations
- Clean up resources in destructors

### Performance Considerations

- Minimize calculations in update loops
- Cache frequently accessed values
- Use appropriate update rates
- Consider threading for heavy computations

## Debugging Plugins

### Logging

```cpp
// Use Gazebo's logging system
gzmsg << "Informative message" << std::endl;
gzdbg << "Debug message" << std::endl;
gzwarn << "Warning message" << std::endl;
gzerr << "Error message" << std::endl;
```

### Common Issues

1. **Plugin Not Loading**: Check file permissions and paths
2. **Symbol Resolution**: Ensure proper linking and library paths
3. **Update Rate**: Balance between accuracy and performance
4. **Threading Issues**: Be careful with ROS integration

## Week Summary

This section covered the development of Gazebo plugins, including different plugin types, integration with ROS, and best practices for plugin development. Plugins are essential for extending Gazebo's functionality and creating sophisticated simulation environments for robotics development.