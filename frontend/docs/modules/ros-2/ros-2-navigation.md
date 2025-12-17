---
sidebar_position: 6
---

# ROS 2 Navigation

## Introduction to ROS 2 Navigation

ROS 2 Navigation (Nav2) is the official navigation stack for ROS 2, providing a comprehensive set of tools and algorithms for robot navigation in various environments. The Navigation 2 stack builds upon the lessons learned from ROS 1's navigation stack while addressing its limitations and incorporating modern robotics capabilities.

## Navigation Stack Architecture

### Core Components

The Nav2 stack consists of several key components working together:

**Navigation Server**: Main orchestration node that coordinates all navigation activities
**Planners**: Global and local path planning algorithms
**Controllers**: Local trajectory generation and execution
**Recovery Behaviors**: Strategies for handling navigation failures
**Behavior Trees**: Task planning and execution framework

### System Overview

```
[Navigation Server]
    ├── [Global Planner] → Path planning
    ├── [Local Planner] → Trajectory execution
    ├── [Controller] → Robot control
    ├── [Recovery] → Failure handling
    └── [Behavior Tree] → Task orchestration
```

## Navigation Server

### Basic Navigation Server

```cpp
#include "nav2_behavior_tree/bt_navigator.hpp"
#include "nav2_lifecycle_node.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

class NavigationServer : public nav2_lifecycle_node::LifecycleNode
{
public:
    explicit NavigationServer(const std::string & name)
    : nav2_lifecycle_node::LifecycleNode(name)
    {
    }

    nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state)
    {
        // Configure navigation components
        bt_navigator_ = std::make_unique<nav2_behavior_tree::BtNavigator>(
            "navigate_to_pose", shared_from_this());

        return nav2_util::CallbackReturn::SUCCESS;
    }

    nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state)
    {
        // Activate navigation components
        bt_navigator_->activate();
        return nav2_util::CallbackReturn::SUCCESS;
    }

    nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state)
    {
        // Deactivate navigation components
        bt_navigator_->deactivate();
        return nav2_util::CallbackReturn::SUCCESS;
    }

private:
    std::unique_ptr<nav2_behavior_tree::BtNavigator> bt_navigator_;
};
```

## Global Path Planning

### Global Planner Interface

```cpp
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

class CustomGlobalPlanner : public nav2_core::GlobalPlanner
{
public:
    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros,
        const std::shared_ptr<nav2_costmap_2d::Costmap2D> & costmap,
        const std::string & global_frame,
        const std::string & robot_base_frame,
        const tf2::BufferCore::SharedPtr & tf) override
    {
        parent_ = parent;
        name_ = name;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap;
        global_frame_ = global_frame;
        robot_base_frame_ = robot_base_frame;
        tf_ = tf;

        // Additional configuration
        RCLCPP_INFO(rclcpp::get_logger(name_), "Configured global planner");
    }

    void cleanup() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Cleaning up global planner");
    }

    void activate() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Activating global planner");
    }

    void deactivate() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Deactivating global planner");
    }

    nav2_util::Nav2Status createPlan(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal,
        std::vector<geometry_msgs::msg::PoseStamped> & path) override
    {
        // Implementation of path planning algorithm
        // Example: A* or Dijkstra's algorithm
        RCLCPP_INFO(rclcpp::get_logger(name_), "Creating global plan");

        // Check if goal is valid
        if (!isGoalValid(start, goal)) {
            return nav2_util::Nav2Status::FAILURE;
        }

        // Plan path using algorithm
        if (!planPath(start, goal, path)) {
            return nav2_util::Nav2Status::FAILURE;
        }

        return nav2_util::Nav2Status::SUCCEEDED;
    }

private:
    bool isGoalValid(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal)
    {
        // Check if goal is in valid location
        auto costmap = costmap_->getCostmap();
        unsigned int mx, my;

        if (!costmap->worldToMap(goal.pose.position.x, goal.pose.position.y, mx, my)) {
            RCLCPP_WARN(rclcpp::get_logger(name_), "Goal is off the costmap");
            return false;
        }

        unsigned char cost = costmap->getCost(mx, my);
        if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
            RCLCPP_WARN(rclcpp::get_logger(name_), "Goal is in an obstacle");
            return false;
        }

        return true;
    }

    bool planPath(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal,
        std::vector<geometry_msgs::msg::PoseStamped> & path)
    {
        // Implementation of actual path planning algorithm
        // This is a simplified example
        path.clear();

        // Add start pose
        path.push_back(start);

        // Add intermediate poses (in a real implementation, this would be the planned path)
        geometry_msgs::msg::PoseStamped intermediate = goal;
        intermediate.pose.position.x = (start.pose.position.x + goal.pose.position.x) / 2.0;
        intermediate.pose.position.y = (start.pose.position.y + goal.pose.position.y) / 2.0;
        path.push_back(intermediate);

        // Add goal pose
        path.push_back(goal);

        return true;
    }

    // Member variables
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
    std::string name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap_;
    std::string global_frame_, robot_base_frame_;
    tf2::BufferCore::SharedPtr tf_;
};
```

## Local Path Planning and Control

### Local Planner Interface

```cpp
#include "nav2_core/local_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"

class CustomLocalPlanner : public nav2_core::LocalPlanner
{
public:
    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros,
        const std::shared_ptr<nav2_costmap_2d::Costmap2D> & costmap,
        const std::string & global_frame,
        const std::string & robot_base_frame,
        const tf2::BufferCore::SharedPtr & tf) override
    {
        parent_ = parent;
        name_ = name;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap;
        global_frame_ = global_frame;
        robot_base_frame_ = robot_base_frame;
        tf_ = tf;

        // Initialize controller parameters
        max_vel_x_ = this->get_parameter("max_vel_x").as_double();
        min_vel_x_ = this->get_parameter("min_vel_x").as_double();
        max_vel_theta_ = this->get_parameter("max_vel_theta").as_double();
    }

    void cleanup() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Cleaning up local planner");
    }

    void activate() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Activating local planner");
    }

    void deactivate() override
    {
        RCLCPP_INFO(rclcpp::get_logger(name_), "Deactivating local planner");
    }

    geometry_msgs::msg::Twist computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override
    {
        geometry_msgs::msg::Twist cmd_vel;

        // Get next pose in global plan
        if (global_plan_.empty()) {
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
            return cmd_vel;
        }

        // Simple proportional controller example
        auto goal = global_plan_.front();
        double dx = goal.pose.position.x - pose.pose.position.x;
        double dy = goal.pose.position.y - pose.pose.position.y;

        double distance = sqrt(dx * dx + dy * dy);
        double angle_to_goal = atan2(dy, dx);

        // Adjust velocity based on distance to goal
        cmd_vel.linear.x = std::min(max_vel_x_, distance * 0.5);
        cmd_vel.angular.z = angle_to_goal * 0.5;

        // Apply safety checks
        if (isCollisionImminent(pose, cmd_vel)) {
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
        }

        return cmd_vel;
    }

    bool isGoalReached(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override
    {
        if (global_plan_.empty()) {
            return true;
        }

        auto goal = global_plan_.back();
        double dx = goal.pose.position.x - pose.pose.position.x;
        double dy = goal.pose.position.y - pose.pose.position.y;
        double distance = sqrt(dx * dx + dy * dy);

        return distance < 0.2 && velocity.linear.x < 0.05;
    }

    void setPlan(const std::vector<geometry_msgs::msg::PoseStamped> & path) override
    {
        global_plan_ = path;
        RCLCPP_INFO(rclcpp::get_logger(name_), "Set new local plan with %zu poses", path.size());
    }

private:
    bool isCollisionImminent(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & cmd_vel)
    {
        // Check costmap for potential collisions in the direction of movement
        auto costmap = costmap_->getCostmap();

        // Calculate position in front of robot based on velocity
        double check_distance = 0.5; // meters ahead to check
        double check_x = pose.pose.position.x + check_distance * cos(0);
        double check_y = pose.pose.position.y + check_distance * sin(0);

        unsigned int mx, my;
        if (costmap->worldToMap(check_x, check_y, mx, my)) {
            unsigned char cost = costmap->getCost(mx, my);
            return cost >= nav2_costmap_2d::LETHAL_OBSTACLE;
        }

        return false; // Can't determine, assume safe
    }

    // Member variables
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
    std::string name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap_;
    std::string global_frame_, robot_base_frame_;
    tf2::BufferCore::SharedPtr tf_;

    std::vector<geometry_msgs::msg::PoseStamped> global_plan_;

    // Parameters
    double max_vel_x_, min_vel_x_, max_vel_theta_;
};
```

## Costmap Configuration

### Costmap Setup

```yaml
# config/costmap.yaml
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  static_map: false
  rolling_window: true
  width: 3
  height: 3
  resolution: 0.05
  origin_x: 0.0
  origin_y: 0.0

  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}

  obstacle_layer:
    enabled: true
    observation_sources: scan
    scan:
      topic: /scan
      max_obstacle_height: 2.0
      clearing: true
      marking: true
      data_type: LaserScan
      obstacle_range: 2.5
      raytrace_range: 3.0

  inflation_layer:
    enabled: true
    cost_scaling_factor: 3.0
    inflation_radius: 0.55

global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 1.0
  static_map: true

  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
```

## Behavior Trees for Navigation

### Navigation Behavior Tree

```xml
<!-- launch/navigation_tree.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RateController hz="1.0">
                <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
            </RateController>
            <FollowPath path="{path}" controller_id="FollowPath"/>
        </PipelineSequence>
    </BehaviorTree>

    <BehaviorTree ID="RecoveryNode">
        <ReactiveSequence>
            <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
            <Spin spin_dist="1.57"/>
            <Wait wait_duration="5"/>
        </ReactiveSequence>
    </BehaviorTree>
</root>
```

## Navigation Parameters

### Parameter Configuration

```yaml
# config/navigation_params.yaml
amcl:
  ros__parameters:
    use_sim_time: false
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_duration: 0.5
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "nav2_bt_navigator/navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node