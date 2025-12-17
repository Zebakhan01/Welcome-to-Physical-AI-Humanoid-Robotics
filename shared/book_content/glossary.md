---
sidebar_position: 11
---

# Glossary

## A

**Actuator**: A component of a robot that converts control signals into physical motion. Examples include servo motors, stepper motors, and pneumatic cylinders.

**Artificial Intelligence (AI)**: The simulation of human intelligence processes by machines, especially computer systems. In robotics, AI enables perception, decision-making, and learning capabilities.

**Autonomous System**: A system that can operate independently without direct human intervention, making decisions based on sensor inputs and internal programming.

## B

**Behavior Tree**: A hierarchical tree structure used to create complex robot behaviors from simple, reusable tasks. Each node in the tree returns either success, failure, or running.

**Bounding Box**: A rectangular box that encloses an object in 3D space, used for collision detection and spatial queries.

## C

**Camera Calibration**: The process of determining the internal parameters of a camera (focal length, principal point, distortion coefficients) to enable accurate 3D reconstruction from 2D images.

**Center of Mass (COM)**: The point in a body where the total mass of the body is considered to be concentrated. Critical for balance and stability in humanoid robots.

**Collision Detection**: The computational problem of detecting the intersection of two or more objects, essential for safe robot operation.

**Control Theory**: A branch of engineering and mathematics that deals with the behavior of dynamical systems with inputs, and how their behavior is modified by feedback.

**Coordinate Frame**: A system of reference that defines positions and orientations in space, typically using a 3D coordinate system.

## D

**Deep Learning**: A subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data. Commonly used in robotics for perception and control.

**Denavit-Hartenberg (DH) Parameters**: A convention for defining coordinate frames on robotic linkages, used for forward and inverse kinematics calculations.

**Differential Drive**: A common mobile robot configuration using two independently driven wheels on a common axis, allowing for turning by driving the wheels at different speeds.

**Digital Twin**: A virtual replica of a physical system that can be used for simulation, analysis, and optimization.

## E

**Embodied AI**: Artificial intelligence that is integrated with a physical system, allowing the AI to interact with and learn from the physical world through a body.

**End Effector**: The device at the end of a robotic arm designed to interact with the environment, such as a gripper or tool.

**Euclidean Distance**: The straight-line distance between two points in Euclidean space, calculated using the Pythagorean theorem.

## F

**Forward Kinematics**: The use of kinematic equations to compute the position of the end-effector from specified values of joint parameters.

**Force Control**: A control strategy that regulates the forces exerted by a robot on its environment, often used in manipulation tasks.

**Fourier Transform**: A mathematical transform that decomposes functions depending on space or time into functions depending on spatial or temporal frequency.

## G

**Gazebo**: A 3D simulation environment for robotics that provides high-fidelity physics simulation and rendering capabilities.

**Geometric Transformation**: A function that changes the geometric properties of a space, such as translation, rotation, or scaling.

**Gripper**: A device at the end of a robot arm designed to grasp and hold objects.

## H

**Hardware-in-the-Loop (HIL)**: A testing technique that involves connecting actual hardware components to a simulation environment for testing purposes.

**Homing**: The process of establishing a known reference position for robot joints, typically using limit switches or encoders.

**Humanoid Robot**: A robot with a body structure similar to that of a human, typically having a head, torso, two arms, and two legs.

## I

**Inverse Kinematics (IK)**: The mathematical process of calculating the joint parameters needed to place the end-effector at a desired position and orientation.

**Inertial Measurement Unit (IMU)**: A device that measures and reports a body's specific force, angular rate, and sometimes the magnetic field surrounding the body.

**Iterative Learning Control (ILC)**: A control scheme that improves system performance by learning from repeated tasks.

## J

**Joint**: A connection between two or more links in a robotic system that allows relative motion between them.

**Joint Space**: The space defined by the joint angles of a robot, as opposed to Cartesian space which is defined by position and orientation.

## K

**Kalman Filter**: An algorithm that uses a series of measurements observed over time to estimate unknown variables, producing estimates that tend to be more accurate than those based on a single measurement alone.

**Kinematics**: The branch of mechanics concerned with the motion of objects without reference to the forces that cause the motion.

**Kinetic Chain**: A system of rigid bodies connected by joints that transmits motion, used to describe robot manipulator structures.

## L

**LIDAR (Light Detection and Ranging)**: A remote sensing method that uses light in the form of a pulsed laser to measure distances to objects.

**Localization**: The process of determining the position and orientation of a robot within a known or unknown environment.

**Lyapunov Stability**: A mathematical concept used to prove the stability of equilibrium points in dynamical systems.

## M

**Machine Learning**: A method of data analysis that automates analytical model building using algorithms that iteratively learn from data.

**Manipulation**: The branch of robotics dealing with the control of robot hands and arms to interact with objects in the environment.

**Mobile Robot**: A robot that is able to move around in its environment, as opposed to stationary robots.

**Monte Carlo Method**: A computational algorithm that relies on repeated random sampling to obtain numerical results, often used in robotics for localization and planning.

## N

**Navigation**: The process by which a robot determines and follows a path from its current location to a desired destination.

**Neural Network**: A computing system inspired by the biological neural networks that constitute animal brains, used in robotics for perception and control.

**Non-holonomic System**: A system whose state depends on the path taken to reach it, such as wheeled vehicles with rolling constraints.

## O

**Odometry**: The use of data from motion sensors to estimate change in position over time, commonly used for robot localization.

**Operational Space**: The space in which the end-effector of a robot operates, typically defined by position and orientation in Cartesian coordinates.

**Omnidirectional Drive**: A mobile robot configuration that can move in any direction regardless of its orientation, typically using specialized wheels.

## P

**Path Planning**: The computational process of finding a valid and optimal path from a start point to a goal point, avoiding obstacles.

**PID Controller**: A control loop feedback mechanism widely used in industrial control systems, consisting of Proportional, Integral, and Derivative terms.

**Point Cloud**: A set of data points in space, representing the external surface of an object or environment, typically generated by 3D scanners or LIDAR.

**Pose**: The position and orientation of an object in space, typically represented by a 3D position vector and a rotation matrix or quaternion.

## R

**Real-time System**: A system that must respond to inputs and compute results within a specified time constraint to ensure correct functioning.

**Robot Operating System (ROS)**: A flexible framework for writing robot software, providing a collection of tools, libraries, and conventions for robot development.

**ROS 2**: The second generation of the Robot Operating System, designed to be production-ready with improved security, real-time capabilities, and multi-robot support.

**Rigid Body**: An idealization of a solid body in which deformation is neglected, meaning the distance between any two points remains constant regardless of external forces.

## S

**Sensor Fusion**: The process of combining sensory data from multiple sources to achieve better accuracy and reliability than could be achieved by using a single sensor alone.

**Simultaneous Localization and Mapping (SLAM)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

**Singularity**: A configuration of a robot manipulator where it loses one or more degrees of freedom, typically occurring when joint axes align.

**State Estimation**: The process of estimating the internal state of a dynamic system from measurements that are corrupted by noise.

**State Machine**: A computational model used to design algorithms that can be in exactly one of a finite number of states at any given time.

**SwRIki**: A lightweight alternative to ROS that provides similar functionality with reduced overhead, suitable for resource-constrained systems.

## T

**Trajectory**: A time-parameterized path that specifies the position, velocity, and acceleration of a robot over time.

**Trapezoidal Motion Profile**: A motion control profile that accelerates to a maximum velocity, maintains that velocity, then decelerates, forming a trapezoidal velocity profile.

**Twist**: A 6D velocity vector combining linear and angular velocities, commonly used in robotics to represent spatial velocity.

## U

**Unity**: A cross-platform game engine that can be used for robotics simulation, providing high-quality graphics and physics simulation.

**URDF (Unified Robot Description Format)**: An XML format for representing a robot model, including kinematic and dynamic properties, visual appearance, and collision geometry.

**UV Mapping**: The process of projecting a 3D model to 2D space for texture mapping, though not directly related to robotics, it's used in simulation environments.

## V

**Variational Method**: A mathematical technique for finding functions that minimize or maximize certain functionals, used in optimal control and estimation.

**Virtual Reality (VR)**: A simulated experience that can be similar to or completely different from the real world, sometimes used for robot teleoperation and training.

**Vision System**: A system that uses cameras and image processing algorithms to provide robots with the ability to "see" and interpret their environment.

**Voronoi Diagram**: A partitioning of a plane into regions based on distance to points in a specific subset of the plane, used in path planning.

## W

**Waypoint**: A reference point on a path used for navigation, typically defined by coordinates in space.

**Wheel Odometry**: The use of wheel encoders to estimate the distance traveled by a wheeled robot, used for localization.

**Whole-Body Control**: A control approach that considers the entire robot body simultaneously, optimizing for multiple tasks and constraints.

## X

**Xacro**: An XML macro language for generating URDF files, allowing for parameterization and reuse of robot model components.

## Y

**Yaw**: The rotation of an object about its vertical axis, one of the three rotational degrees of freedom.

## Z

**Zero Moment Point (ZMP)**: A concept used in robotics and biomechanics to indicate the point on the ground where the moment caused by the ground reaction force is zero, critical for balance control in humanoid robots.

**Z-buffer**: A component of graphics processing that handles the depth coordinates in 3D scenes, used in simulation rendering.

---

This glossary provides definitions for key terms used throughout the Physical AI and Humanoid Robotics curriculum. Understanding these terms is essential for comprehending the concepts and technologies involved in the field.