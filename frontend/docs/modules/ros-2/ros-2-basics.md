---
sidebar_position: 2
---

# ROS 2 Basics

## Introduction to ROS 2

ROS 2 is the next generation of the Robot Operating System, designed to address the limitations of ROS 1 and provide enhanced features for modern robotics applications. Key improvements include better security, real-time support, and improved architecture for production systems.

## Architecture Differences

### DDS Integration

ROS 2 uses Data Distribution Service (DDS) as its middleware:
- **Decentralized Architecture**: No central master node required
- **Language Independence**: Better support for multiple programming languages
- **Quality of Service (QoS)**: Configurable communication policies
- **Security**: Built-in security features for production environments

### Client Library (rclcpp/rclpy)

- **Language-Agnostic Core**: Common interface across languages
- **Improved Lifecycle**: Better node lifecycle management
- **Modern C++**: C++14/17 features and best practices

## Core Concepts

### Nodes and Communication

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher() : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

### Quality of Service (QoS)

QoS policies allow fine-tuning communication behavior:

```cpp
// Reliability policies
rclcpp::QoS qos_profile(10);
qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
// or RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT

// Durability policies
qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
// or RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL

// History policies
qos_profile.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
qos_profile.depth(7);
```

## Advanced Features

### Lifecycle Nodes

Lifecycle nodes provide better control over node state:

```cpp
#include "rclcpp_lifecycle/lifecycle_node.hpp"

class LifecycleNodeDemo : public rclcpp_lifecycle::LifecycleNode
{
public:
    LifecycleNodeDemo() : rclcpp_lifecycle::LifecycleNode("lifecycle_node")
    {
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Configuring");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &)
    {
        RCLCPP_INFO(get_logger(), "Activating");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};
```

### Parameters

Dynamic parameter handling with callbacks:

```cpp
class ParameterNode : public rclcpp::Node
{
public:
    ParameterNode() : rclcpp::Node("parameter_node")
    {
        // Declare parameters with default values
        this->declare_parameter("my_parameter", 42);

        // Register parameter callback
        this->set_on_parameters_set_callback(
            [this](const std::vector<rclcpp::Parameter> & parameters) {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;
                for (const auto & parameter : parameters) {
                    if (parameter.get_name() == "my_parameter") {
                        RCLCPP_INFO(get_logger(), "Parameter updated: %ld",
                                   parameter.as_int());
                    }
                }
                return result;
            });
    }
};
```

## Security Features

ROS 2 includes built-in security capabilities:
- **Authentication**: Identity verification
- **Authorization**: Access control policies
- **Encryption**: Data protection in transit
- **Secure Communication**: End-to-end security

## Real-time Considerations

ROS 2 supports real-time applications:
- **SCHED_FIFO**: Real-time scheduling policies
- **Memory Management**: Predictable allocation patterns
- **Deterministic Communication**: QoS for timing guarantees
- **Process Isolation**: Better resource management

## Migration from ROS 1

Key differences to consider:
- **Build System**: CMake instead of rosbuild/catkin
- **Communication**: DDS instead of roscpp/rosjava
- **Tools**: New command-line tools (ros2 command)
- **Concepts**: Lifecycle nodes, QoS policies

## Best Practices

### Node Design

- Use composition over inheritance
- Implement proper error handling
- Follow naming conventions
- Use appropriate QoS settings

### Package Structure

```
my_package/
├── CMakeLists.txt
├── package.xml
├── include/
│   └── my_package/
├── src/
├── launch/
├── config/
└── test/
```

## Week Summary

This section covered the fundamental differences between ROS 1 and ROS 2, focusing on the DDS middleware, quality of service policies, and lifecycle nodes. These features make ROS 2 more suitable for production robotics applications with enhanced security, real-time capabilities, and improved architecture.