---
sidebar_position: 4
---

# Configuration Templates

## Introduction to Configuration Templates

Configuration templates provide standardized, reusable configurations for various components in Physical AI and humanoid robotics systems. These templates ensure consistency across deployments, reduce configuration errors, and accelerate development by providing tested starting points for common system setups. This section covers configuration files for ROS packages, simulation environments, control systems, and deployment scenarios.

## ROS Package Configuration Templates

### Package.xml Template

```xml
<?xml version="1.0"?>
<package format="2">
  <name>robot_control</name>
  <version>0.1.0</version>
  <description>Robot control package for humanoid robot</description>

  <maintainer email="developer@robotics.org">Robotics Developer</maintainer>
  <license>MIT</license>

  <url type="website">http://wiki.ros.org/robot_control</url>
  <author email="developer@robotics.org">Robotics Developer</author>

  <buildtool_depend>catkin</buildtool_depend>

  <build_depend>roscpp</build_depend>
  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>geometry_msgs</build_depend>
  <build_depend>nav_msgs</build_depend>
  <build_depend>tf2</build_depend>
  <build_depend>tf2_ros</build_depend>
  <build_depend>message_generation</build_depend>
  <build_depend>control_msgs</build_depend>
  <build_depend>trajectory_msgs</build_depend>
  <build_depend>actionlib_msgs</build_depend>
  <build_depend>actionlib</build_depend>

  <build_export_depend>roscpp</build_export_depend>
  <build_export_depend>rospy</build_export_depend>
  <build_export_depend>std_msgs</build_export_depend>
  <build_export_depend>sensor_msgs</build_export_depend>
  <build_export_depend>geometry_msgs</build_export_depend>
  <build_export_depend>nav_msgs</build_export_depend>
  <build_export_depend>tf2</build_export_depend>
  <build_export_depend>tf2_ros</build_export_depend>
  <build_export_depend>control_msgs</build_export_depend>
  <build_export_depend>trajectory_msgs</build_export_depend>
  <build_export_depend>actionlib_msgs</build_export_depend>
  <build_export_depend>actionlib</build_export_depend>

  <exec_depend>roscpp</exec_depend>
  <exec_depend>rospy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>tf2</exec_depend>
  <exec_depend>tf2_ros</exec_depend>
  <exec_depend>message_runtime</exec_depend>
  <exec_depend>control_msgs</exec_depend>
  <exec_depend>trajectory_msgs</exec_depend>
  <exec_depend>actionlib_msgs</exec_depend>
  <exec_depend>actionlib</exec_depend>

  <export>
    <!-- Other tools can request additional information be placed here -->
  </export>
</package>
```

### CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(robot_control)

## Compile as C++14, supported in ROS Noetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  message_generation
  control_msgs
  trajectory_msgs
  actionlib_msgs
  actionlib
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_control
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/robot_control.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/robot_control_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_robot_control.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
```

## Launch File Templates

### Robot Launch Template

```xml
<launch>
  <!-- Robot description -->
  <param name="robot_description" command="$(find xacro)/xacro '$(find robot_description)/urdf/robot.xacro'" />

  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />

  <!-- Joint state publisher -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
    <param name="use_gui" value="false" />
    <param name="rate" value="50" />
  </node>

  <!-- Robot bringup node -->
  <node name="robot_bringup" pkg="robot_control" type="robot_bringup.py" output="screen">
    <param name="loop_hz" value="50" />
    <param name="robot_name" value="humanoid_robot" />
    <remap from="joint_states" to="/joint_states" />
    <remap from="cmd_vel" to="/cmd_vel" />
  </node>

  <!-- Control manager -->
  <node name="controller_manager" pkg="controller_manager" type="spawner" respawn="false"
        output="screen" ns="/" args="
        joint_state_controller
        position_controllers
        velocity_controllers
        effort_controllers
        " />

  <!-- Visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find robot_description)/config/robot.rviz" />

  <!-- Gazebo simulation (if needed) -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find robot_gazebo)/worlds/empty.world" />
    <arg name="paused" value="false" />
    <arg name="use_sim_time" value="true" />
    <arg name="gui" value="true" />
    <arg name="recording" value="false" />
    <arg name="debug" value="false" />
  </include>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" respawn="false" output="screen"
        args="-param robot_description -urdf -model robot -x 0 -y 0 -z 1" />

</launch>
```

### Control Configuration Template

```yaml
# controllers.yaml - Robot controller configuration
# Joint State Controller
joint_state_controller:
  type: joint_state_controller/JointStateController
  publish_rate: 50

# Position Controllers
position_controllers:
  type: position_controllers/JointGroupPositionController
  joints:
    - left_hip_yaw
    - left_hip_roll
    - left_hip_pitch
    - left_knee
    - left_ankle_pitch
    - left_ankle_roll
    - right_hip_yaw
    - right_hip_roll
    - right_hip_pitch
    - right_knee
    - right_ankle_pitch
    - right_ankle_roll
    - left_shoulder_pitch
    - left_shoulder_roll
    - left_shoulder_yaw
    - left_elbow
    - left_wrist_yaw
    - left_wrist_pitch
    - right_shoulder_pitch
    - right_shoulder_roll
    - right_shoulder_yaw
    - right_elbow
    - right_wrist_yaw
    - right_wrist_pitch

# Velocity Controllers
velocity_controllers:
  type: velocity_controllers/JointGroupVelocityController
  joints:
    - left_hip_yaw
    - left_hip_roll
    - left_hip_pitch
    - left_knee
    - left_ankle_pitch
    - left_ankle_roll
    - right_hip_yaw
    - right_hip_roll
    - right_hip_pitch
    - right_knee
    - right_ankle_pitch
    - right_ankle_roll

# Effort Controllers
effort_controllers:
  type: effort_controllers/JointGroupEffortController
  joints:
    - left_hip_yaw
    - left_hip_roll
    - left_hip_pitch
    - left_knee
    - left_ankle_pitch
    - left_ankle_roll
    - right_hip_yaw
    - right_hip_roll
    - right_hip_pitch
    - right_knee
    - right_ankle_pitch
    - right_ankle_roll

# Balance Controller
balance_controller:
  type: robot_control/BalanceController
  kp: [100.0, 100.0, 100.0]  # Position gains
  kd: [10.0, 10.0, 10.0]     # Damping gains
  max_force: 1000.0          # Maximum force limit
  publish_rate: 100          # Update rate

# Trajectory Controller
trajectory_controller:
  type: position_controllers/JointTrajectoryController
  joints:
    - left_hip_yaw
    - left_hip_roll
    - left_hip_pitch
    - left_knee
    - left_ankle_pitch
    - left_ankle_roll
    - right_hip_yaw
    - right_hip_roll
    - right_hip_pitch
    - right_knee
    - right_ankle_pitch
    - right_ankle_roll
  gains:
    left_hip_yaw: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    left_hip_roll: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    left_hip_pitch: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    left_knee: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    left_ankle_pitch: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1}
    left_ankle_roll: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1}
    right_hip_yaw: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    right_hip_roll: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    right_hip_pitch: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    right_knee: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1}
    right_ankle_pitch: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1}
    right_ankle_roll: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1}
```

## URDF/Xacro Templates

### Robot URDF Template

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Include other xacro files -->
  <xacro:include filename="$(find robot_description)/urdf/materials.xacro" />
  <xacro:include filename="$(find robot_description)/urdf/transmissions.xacro" />
  <xacro:include filename="$(find robot_description)/urdf/gazebo.xacro" />

  <!-- Robot constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_mass" value="70.0" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${robot_mass}"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 ${base_height/2 + 0.1}" rpy="0 0 0"/>
  </joint>

  <link name="torso_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="fixed">
    <parent link="torso_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="head_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="base_to_left_hip" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0 0.1 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="200" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <joint name="base_to_right_hip" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="200" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

</robot>
```

### Materials Xacro Template

```xml
<?xml version="1.0"?>
<robot>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>

  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>

  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>

  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>

  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>

  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>

  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <material name="yellow">
    <color rgba="1.0 1.0 0.0 1.0"/>
  </material>
</robot>
```

## Gazebo Configuration Templates

### Gazebo Plugin Configuration

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
      <legacyModeNS>true</legacyModeNS>
    </plugin>
  </gazebo>

  <!-- Gazebo camera plugin -->
  <gazebo reference="head_link">
    <sensor type="camera" name="head_camera">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>0.0</updateRate>
        <cameraName>head_camera</cameraName>
        <imageTopicName>image_raw</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>head_camera_optical_frame</frameName>
        <hackBaseline>0.07</hackBaseline>
        <distortionK1>0.0</distortionK1>
        <distortionK2>0.0</distortionK2>
        <distortionK3>0.0</distortionK3>
        <distortionT1>0.0</distortionT1>
        <distortionT2>0.0</distortionT2>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Gazebo IMU plugin -->
  <gazebo reference="torso_link">
    <sensor type="imu" name="torso_imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <topicName>imu/data</topicName>
        <bodyName>torso_link</bodyName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.01</gaussianNoise>
        <xyzOffset>0 0 0</xyzOffset>
        <rpyOffset>0 0 0</rpyOffset>
        <frameName>torso_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Gazebo joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <robotNamespace>/</robotNamespace>
      <jointName>
        left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
        right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll,
        left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_yaw, left_wrist_pitch,
        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_yaw, right_wrist_pitch
      </jointName>
      <updateRate>30.0</updateRate>
    </plugin>
  </gazebo>

</robot>
```

## Control System Configuration Templates

### PID Controller Configuration

```yaml
# pid_controllers.yaml
# PID controller configurations for robot joints

# Hip controllers
left_hip_yaw_controller:
  type: position_controllers/JointPositionController
  joint: left_hip_yaw
  pid: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1.0}

left_hip_roll_controller:
  type: position_controllers/JointPositionController
  joint: left_hip_roll
  pid: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1.0}

left_hip_pitch_controller:
  type: position_controllers/JointPositionController
  joint: left_hip_pitch
  pid: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1.0}

# Knee controller
left_knee_controller:
  type: position_controllers/JointPositionController
  joint: left_knee
  pid: {p: 1000.0, i: 0.01, d: 10.0, i_clamp: 1.0}

# Ankle controllers
left_ankle_pitch_controller:
  type: position_controllers/JointPositionController
  joint: left_ankle_pitch
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

left_ankle_roll_controller:
  type: position_controllers/JointPositionController
  joint: left_ankle_roll
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

# Shoulder controllers
left_shoulder_pitch_controller:
  type: position_controllers/JointPositionController
  joint: left_shoulder_pitch
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

left_shoulder_roll_controller:
  type: position_controllers/JointPositionController
  joint: left_shoulder_roll
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

left_shoulder_yaw_controller:
  type: position_controllers/JointPositionController
  joint: left_shoulder_yaw
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

# Elbow controller
left_elbow_controller:
  type: position_controllers/JointPositionController
  joint: left_elbow
  pid: {p: 500.0, i: 0.01, d: 5.0, i_clamp: 1.0}

# Wrist controllers
left_wrist_yaw_controller:
  type: position_controllers/JointPositionController
  joint: left_wrist_yaw
  pid: {p: 200.0, i: 0.01, d: 2.0, i_clamp: 1.0}

left_wrist_pitch_controller:
  type: position_controllers/JointPositionController
  joint: left_wrist_pitch
  pid: {p: 200.0, i: 0.01, d: 2.0, i_clamp: 1.0}

# Balance controller configuration
balance_controller:
  type: robot_control/BalanceController
  kp: [100.0, 100.0, 100.0]  # Position gains
  kd: [10.0, 10.0, 10.0]     # Damping gains
  max_force: 1000.0          # Maximum force limit
  publish_rate: 100          # Update rate
  com_reference: [0.0, 0.0, 0.8]  # Center of mass reference
```

## Simulation Configuration Templates

### Gazebo World Configuration

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">

    <!-- Physics engine -->
    <physics name="default_physics" type="ode">
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

    <!-- Scene properties -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 -0.4 -0.8</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Robot model -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <!-- Additional objects -->
    <include>
      <uri>model://table</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://box</uri>
      <pose>3 1 0.5 0 0 0</pose>
    </include>

  </world>
</sdf>
```

## ROS Node Configuration Templates

### Parameter Server Configuration

```yaml
# robot_params.yaml
# Robot parameter configuration

robot_description:
  # Robot physical properties
  mass: 70.0
  height: 1.7
  width: 0.5
  depth: 0.3

  # Center of mass
  com_offset: [0.0, 0.0, 0.8]

  # Joint limits
  joint_limits:
    hip_yaw: [-1.57, 1.57]
    hip_roll: [-0.785, 0.785]
    hip_pitch: [-1.57, 1.57]
    knee: [0.0, 2.356]
    ankle_pitch: [-0.785, 0.785]
    ankle_roll: [-0.349, 0.349]

  # Actuator specifications
  actuators:
    max_torque: 100.0
    max_velocity: 5.0
    gear_ratio: 100.0

control:
  # Control system parameters
  loop_rate: 100
  dt: 0.01

  # Balance control
  balance:
    kp: [100.0, 100.0, 100.0]
    kd: [10.0, 10.0, 10.0]
    max_force: 1000.0
    com_reference: [0.0, 0.0, 0.8]

  # Walking parameters
  walking:
    step_length: 0.3
    step_height: 0.1
    step_duration: 1.0
    swing_height: 0.05

sensors:
  # Sensor configurations
  camera:
    resolution: [640, 480]
    fov: 60.0
    frame_rate: 30

  imu:
    update_rate: 100
    noise_density: 0.01
    bias_correlation_time: 1000.0

  lidar:
    range_min: 0.1
    range_max: 10.0
    resolution: 0.01
    update_rate: 10

navigation:
  # Navigation parameters
  planner:
    planner_type: "dijkstra"
    resolution: 0.05
    origin_x: 0.0
    origin_y: 0.0

  controller:
    max_vel_x: 0.5
    max_vel_theta: 1.0
    min_vel_x: 0.1
    min_vel_theta: 0.2

  recovery:
    rotate_recovery: true
    clear_costmap: true

perception:
  # Perception system parameters
  object_detection:
    confidence_threshold: 0.7
    max_objects: 10
    detection_range: 5.0

  segmentation:
    min_cluster_size: 100
    max_cluster_size: 25000
    tolerance: 0.02

  tracking:
    max_distance: 0.5
    min_points: 10

safety:
  # Safety system parameters
  emergency_stop:
    enabled: true
    timeout: 1.0
    velocity_threshold: 0.5

  collision_detection:
    enabled: true
    safety_distance: 0.3
    detection_range: 2.0

  force_limits:
    max_force: 100.0
    max_torque: 50.0
```

## Docker Configuration Templates

### Dockerfile Template

```dockerfile
# Dockerfile for robotics development
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV ROS_WS=/home/robotics/ws

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-catkin-tools \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    tmux \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-desktop-full
RUN apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential

# Initialize rosdep
RUN rosdep init || echo "rosdep already initialized"
RUN rosdep update

# Create non-root user
RUN useradd -ms /bin/bash robotics
USER robotics
WORKDIR /home/robotics

# Setup ROS environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Create workspace
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}

# Install Python packages
COPY requirements.txt .
RUN pip3 install --user -r requirements.txt

# Build workspace
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

# Source workspace
RUN echo "source ${ROS_WS}/devel/setup.bash" >> ~/.bashrc

# Setup entrypoint
RUN echo '#!/bin/bash\n\
source /opt/ros/'${ROS_DISTRO}'/setup.bash\n\
source '${ROS_WS}'/devel/setup.bash\n\
exec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
```

### Docker Compose Template

```yaml
version: '3.8'

services:
  ros-master:
    image: osrf/ros:noetic-desktop-full
    container_name: ros_master
    command: roscore
    networks:
      - robotics_net
    ports:
      - "11311:11311"

  robot-control:
    build: .
    container_name: robot_control
    depends_on:
      - ros-master
    environment:
      - ROS_HOSTNAME=robot-control
      - ROS_MASTER_URI=http://ros-master:11311
    volumes:
      - ./src:/home/robotics/ws/src
      - ./config:/home/robotics/config
    networks:
      - robotics_net
    devices:
      - /dev:/dev
    privileged: true

  simulation:
    image: osrf/gazebo:gzserver11
    container_name: gazebo_simulation
    depends_on:
      - ros-master
    environment:
      - ROS_HOSTNAME=simulation
      - ROS_MASTER_URI=http://ros-master:11311
    volumes:
      - ./worlds:/root/.gazebo/worlds
    networks:
      - robotics_net
    ports:
      - "8080:8080"
    command: gazebo --verbose worlds/empty.world

  visualization:
    image: osrf/ros:noetic-desktop-full
    container_name: rviz_visualization
    depends_on:
      - ros-master
      - robot-control
    environment:
      - ROS_HOSTNAME=visualization
      - ROS_MASTER_URI=http://ros-master:11311
      - DISPLAY=:0
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - /dev/dri:/dev/dri
    networks:
      - robotics_net
    command: rviz

networks:
  robotics_net:
    driver: bridge
```

## Week Summary

This section provided comprehensive configuration templates for Physical AI and humanoid robotics systems. The templates covered ROS packages, launch files, URDF/Xacro models, Gazebo simulation, control systems, and deployment configurations. These standardized templates serve as starting points for various system components, ensuring consistency and reducing configuration errors in robotics development projects.