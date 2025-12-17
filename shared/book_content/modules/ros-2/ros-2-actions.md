---
sidebar_position: 5
---

# ROS 2 Actions

## Introduction to ROS 2 Actions

Actions in ROS 2 provide a communication pattern for long-running tasks that require feedback and the ability to cancel. Unlike services (which are synchronous) or topics (which are asynchronous), actions combine the best of both: they are asynchronous like topics but provide feedback like services, with the added capability of cancellation.

## Action Concepts

### When to Use Actions

Actions are appropriate for tasks that:
- Take a significant amount of time to complete
- Require feedback during execution
- May need to be cancelled before completion
- Have intermediate results to report
- Need status monitoring

### Action vs Service vs Topic

| Communication Type | Synchronous | Feedback | Cancellation | Use Case |
|-------------------|-------------|----------|--------------|----------|
| Topic | No | Continuous | No | Sensor data, status |
| Service | Yes | One-time | No | Simple requests |
| Action | No | Continuous | Yes | Long-running tasks |

## Action Structure

### Action Definition

Actions are defined in `.action` files with three parts:

```
# Goal definition (input to the action server)
geometry_msgs/PoseStamped target_pose
float32 tolerance

# Result definition (output from the action server)
---
bool success
float32 distance_traveled

# Feedback definition (continuous feedback during execution)
---
float32 distance_remaining
geometry_msgs/Pose current_pose
```

### Generated Messages

From an action definition, ROS 2 generates three message types:
- `ActionName_Goal`: Contains goal parameters
- `ActionName_Result`: Contains final result
- `ActionName_Feedback`: Contains feedback during execution

## Action Server Implementation

### Basic Action Server

```cpp
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

class NavigateToPoseActionServer : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandle = rclcpp_action::ServerGoalHandle<NavigateToPose>;

    NavigateToPoseActionServer()
    : Node("navigate_to_pose_action_server")
    {
        using namespace std::placeholders;

        // Create action server
        this->action_server_ = rclcpp_action::create_server<NavigateToPose>(
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "navigate_to_pose",
            std::bind(&NavigateToPoseActionServer::handle_goal, this, _1, _2),
            std::bind(&NavigateToPoseActionServer::handle_cancel, this, _1),
            std::bind(&NavigateToPoseActionServer::handle_accepted, this, _1)
        );
    }

private:
    rclcpp_action::Server<NavigateToPose>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const NavigateToPose::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request with pose x: %f, y: %f",
                   goal->pose.pose.position.x, goal->pose.pose.position.y);
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandle> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
    {
        using namespace std::placeholders;
        // This needs to return quickly to avoid blocking the executor
        std::thread{std::bind(&NavigateToPoseActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandle> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Executing goal");

        // Get goal
        auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<NavigateToPose::Feedback>();
        auto result = std::make_shared<NavigateToPose::Result>();

        // Simulate navigation execution
        auto start_time = std::chrono::high_resolution_clock::now();
        rclcpp::Rate rate(1);

        for (int i = 0; i < 10; ++i) {
            // Check if there was a cancel request
            if (goal_handle->is_canceling()) {
                result->distance_traveled = 0.0;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }

            // Publish feedback
            feedback->distance_remaining = 10.0 - i;
            feedback->current_pose.pose.position.x = goal->pose.pose.position.x * (i / 10.0);
            feedback->current_pose.pose.position.y = goal->pose.pose.position.y * (i / 10.0);
            goal_handle->publish_feedback(feedback);

            rate.sleep();
        }

        // Check if goal was cancelled
        if (rclcpp::ok()) {
            result->success = true;
            result->distance_traveled = 5.0; // Example value
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
    }
};
```

## Action Client Implementation

### Basic Action Client

```cpp
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

class NavigateToPoseActionClient : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;

    NavigateToPoseActionClient()
    : Node("navigate_to_pose_action_client")
    {
        this->client_ptr_ = rclcpp_action::create_client<NavigateToPose>(
            this->get_node_base_interface(),
            this->get_node_graph_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "navigate_to_pose");

        this->timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&NavigateToPoseActionClient::send_goal, this));
    }

private:
    rclcpp_action::Client<NavigateToPose>::SharedPtr client_ptr_;
    rclcpp::TimerBase::SharedPtr timer_;
    GoalHandle::SharedPtr goal_handle_;

    void send_goal()
    {
        using namespace std::placeholders;

        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(1))) {
            RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
            return;
        }

        // Create goal
        auto goal_msg = NavigateToPose::Goal();
        goal_msg.pose.header.frame_id = "map";
        goal_msg.pose.pose.position.x = 1.0;
        goal_msg.pose.pose.position.y = 2.0;
        goal_msg.pose.pose.orientation.w = 1.0;

        // Set options
        auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&NavigateToPoseActionClient::goal_response_callback, this, _1);
        send_goal_options.feedback_callback =
            std::bind(&NavigateToPoseActionClient::feedback_callback, this, _1, _2);
        send_goal_options.result_callback =
            std::bind(&NavigateToPoseActionClient::result_callback, this, _1);

        // Send goal
        RCLCPP_INFO(this->get_logger(), "Sending goal");
        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }

    void goal_response_callback(std::shared_ptr<GoalHandle> goal_handle)
    {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
            goal_handle_ = goal_handle;
        }
    }

    void feedback_callback(
        std::shared_ptr<GoalHandle> goal_handle,
        const std::shared_ptr<const NavigateToPose::Feedback> feedback)
    {
        (void)goal_handle;
        RCLCPP_INFO(this->get_logger(),
                   "Current distance remaining: %f, Current pose: (%f, %f)",
                   feedback->distance_remaining,
                   feedback->current_pose.pose.position.x,
                   feedback->current_pose.pose.position.y);
    }

    void result_callback(const GoalHandle::WrappedResult & result)
    {
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Goal succeeded!");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_INFO(this->get_logger(), "Goal was canceled");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
                return;
        }

        rclcpp::shutdown();
    }
};
```

## Advanced Action Features

### Goal Preemption

Action servers can handle multiple goals and implement preemption:

```cpp
class PreemptableActionServer : public rclcpp::Node
{
public:
    using ExampleAction = example_interfaces::action::Fibonacci;
    using GoalHandle = rclcpp_action::ServerGoalHandle<ExampleAction>;

    PreemptableActionServer()
    : Node("preemptable_action_server")
    {
        this->action_server_ = rclcpp_action::create_server<ExampleAction>(
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "fibonacci",
            std::bind(&PreemptableActionServer::handle_goal, this, _1, _2),
            std::bind(&PreemptableActionServer::handle_cancel, this, _1),
            std::bind(&PreemptableActionServer::handle_accepted, this, _1)
        );
    }

private:
    rclcpp_action::Server<ExampleAction>::SharedPtr action_server_;
    std::shared_ptr<GoalHandle> current_goal_handle_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const ExampleAction::Goal> goal)
    {
        (void)uuid;
        RCLCPP_INFO(this->get_logger(), "Received Fibonacci goal request with order %d",
                   goal->order);

        // Accept new goal only if no current goal or current goal is canceling
        if (!current_goal_handle_ || current_goal_handle_->is_canceling()) {
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
        return rclcpp_action::GoalResponse::REJECT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
    {
        // If there's a current goal, cancel it (preemption)
        if (current_goal_handle_ && !current_goal_handle_->is_canceling()) {
            current_goal_handle_->abort(std::make_shared<ExampleAction::Result>());
        }

        current_goal_handle_ = goal_handle;
        std::thread{std::bind(&PreemptableActionServer::execute, this, _1), goal_handle}.detach();
    }
};
```

### Goal Tracking and Management

```cpp
class TrackedActionServer : public rclcpp::Node
{
public:
    using ExampleAction = example_interfaces::action::Fibonacci;
    using GoalHandle = rclcpp_action::ServerGoalHandle<ExampleAction>;

    TrackedActionServer()
    : Node("tracked_action_server")
    {
        this->action_server_ = rclcpp_action::create_server<ExampleAction>(
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "fibonacci",
            std::bind(&TrackedActionServer::handle_goal, this, _1, _2),
            std::bind(&TrackedActionServer::handle_cancel, this, _1),
            std::bind(&TrackedActionServer::handle_accepted, this, _1)
        );
    }

private:
    rclcpp_action::Server<ExampleAction>::SharedPtr action_server_;
    std::map<rclcpp_action::GoalUUID, std::shared_ptr<GoalHandle>> active_goals_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const ExampleAction::Goal> goal)
    {
        (void)goal;
        auto inserted = active_goals_.insert(std::make_pair(uuid, nullptr));
        if (!inserted.second) {
            RCLCPP_ERROR(this->get_logger(), "A goal with the same UUID already exists");
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandle> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
    {
        using namespace std::placeholders;
        // Update the map entry with the actual goal handle
        auto uuid = goal_handle->get_goal_id();
        auto it = active_goals_.find(uuid);
        if (it != active_goals_.end()) {
            it->second = goal_handle;
        }

        std::thread{std::bind(&TrackedActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandle> goal_handle)
    {
        // Execution logic here
        // When done, remove from active goals
        auto uuid = goal_handle->get_goal_id();
        active_goals_.erase(uuid);
    }
};
```

## Action Best Practices

### Error Handling

```cpp
void execute(const std::shared_ptr<GoalHandle> goal_handle)
{
    try {
        // Check if goal is still valid
        if (!goal_handle->is_active()) {
            return;
        }

        // Your execution logic
        // ...

        // Check for cancellation during execution
        if (goal_handle->is_canceling()) {
            auto result = std::make_shared<NavigateToPose::Result>();
            result->success = false;
            goal_handle->canceled(result);
            return;
        }

        // Success case
        auto result = std::make_shared<NavigateToPose::Result>();
        result->success = true;
        goal_handle->succeed(result);

    } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in action execution: %s", e.what());
        auto result = std::make_shared<NavigateToPose::Result>();
        result->success = false;
        if (goal_handle->is_active()) {
            goal_handle->abort(result);
        }
    }
}
```

### Resource Management

```cpp
class ResourceManagedActionServer : public rclcpp::Node
{
public:
    ResourceManagedActionServer()
    : Node("resource_managed_action_server")
    {
        // Initialize resources
        resource_manager_ = std::make_shared<ResourceManager>();

        this->action_server_ = rclcpp_action::create_server<NavigateToPose>(
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "navigate_to_pose",
            [this](const auto & uuid, const auto & goal) {
                return this->handle_goal(uuid, goal);
            },
            [this](const auto & goal_handle) {
                return this->handle_cancel(goal_handle);
            },
            [this](const auto & goal_handle) {
                this->handle_accepted(goal_handle);
            }
        );
    }

private:
    std::shared_ptr<ResourceManager> resource_manager_;
    rclcpp_action::Server<NavigateToPose>::SharedPtr action_server_;

    void execute(const std::shared_ptr<GoalHandle> goal_handle)
    {
        // Acquire resources
        auto resources = resource_manager_->acquire_resources();
        if (!resources) {
            auto result = std::make_shared<NavigateToPose::Result>();
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // Execute with resources
        // ...

        // Release resources
        resource_manager_->release_resources(resources);
    }
};
```

## Common Action Use Cases

### Navigation Actions

- MoveBase actions for path following
- Exploration actions for autonomous mapping
- Patrol actions for route following
- Docking actions for charging

### Manipulation Actions

- Pick and place actions
- Grasp actions with force control
- Assembly actions with precision requirements
- Tool use actions

### System Management Actions

- Calibration actions
- Homing actions for robot initialization
- Recovery actions for error handling
- Diagnostic actions for system health

## Performance Considerations

### Threading and Concurrency

- Use separate threads for execution to avoid blocking the executor
- Consider thread safety when sharing data
- Use appropriate synchronization mechanisms
- Monitor resource usage during concurrent execution

### Feedback Frequency

- Balance feedback frequency with performance
- Avoid excessive feedback that impacts execution
- Consider network bandwidth for remote systems
- Use appropriate feedback intervals for the task

## Week Summary

This section covered ROS 2 actions in depth, including their purpose, implementation, and best practices. Actions provide a powerful communication pattern for long-running tasks that require feedback and cancellation capabilities, making them essential for many robotics applications such as navigation, manipulation, and system management tasks.