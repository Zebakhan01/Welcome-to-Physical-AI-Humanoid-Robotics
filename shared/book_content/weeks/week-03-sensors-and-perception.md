---
sidebar_position: 3
---

# Week 3: Sensors and Perception

## Learning Objectives

By the end of this week, students will be able to:
- Understand different types of sensors used in robotics
- Explain the principles of robot perception
- Describe sensor fusion techniques
- Apply basic computer vision concepts to robotic systems

## Introduction to Robot Sensors

Robots rely on sensors to perceive their environment and internal state. Sensor data is crucial for navigation, manipulation, interaction, and safety. The quality and reliability of sensor data directly impact the robot's performance and capabilities.

### Sensor Classification

Sensors can be classified based on what they measure:

1. **Proprioceptive Sensors**: Measure the robot's internal state
2. **Exteroceptive Sensors**: Measure the external environment
3. **Interoceptive Sensors**: Measure the robot's internal health/status

## Proprioceptive Sensors

Proprioceptive sensors measure the robot's internal state, including joint positions, velocities, and forces.

### Position Sensors

**Encoders** are the most common position sensors in robotics:
- **Incremental Encoders**: Measure relative position changes
- **Absolute Encoders**: Provide absolute position information
- **Resolution**: Determined by counts per revolution

**Potentiometers** provide position feedback but are less common in modern robotics due to wear and limited resolution.

### Force and Torque Sensors

**Joint Torque Sensors** measure the torque applied at each joint, important for:
- Compliance control
- Collision detection
- Force control applications

**Six-Axis Force/Torque Sensors** measure forces in three directions and torques around three axes, typically mounted at the end-effector.

### Inertial Sensors

**Inertial Measurement Units (IMUs)** combine:
- Accelerometers: Measure linear acceleration
- Gyroscopes: Measure angular velocity
- Magnetometers: Measure magnetic field direction

IMUs are crucial for balance control in humanoid robots and navigation in mobile robots.

## Exteroceptive Sensors

Exteroceptive sensors measure the external environment, enabling robots to perceive their surroundings.

### Range Sensors

**Ultrasonic Sensors**:
- Principle: Measure time-of-flight of ultrasonic pulses
- Range: Typically 2cm to 4m
- Advantages: Simple, low-cost, work in various lighting conditions
- Limitations: Limited accuracy, affected by surface properties

**Infrared Sensors**:
- Principle: Measure reflected infrared light
- Range: Short to medium range
- Advantages: Small size, low power
- Limitations: Affected by ambient light, surface reflectivity

**LIDAR (Light Detection and Ranging)**:
- Principle: Measure time-of-flight of laser pulses
- Types: 2D (single plane), 3D (multiple planes)
- Advantages: High accuracy, reliable in various lighting
- Applications: Mapping, navigation, obstacle detection

### Vision Sensors

**Cameras** are the most versatile exteroceptive sensors:

**Monocular Cameras**:
- Provide 2D image data
- Depth estimation through motion or known object sizes
- Computationally efficient

**Stereo Cameras**:
- Two cameras to estimate depth through triangulation
- Provide 3D information
- More complex processing than monocular

**RGB-D Cameras**:
- Provide color (RGB) and depth information
- Examples: Microsoft Kinect, Intel RealSense
- Direct depth measurements

## Computer Vision for Robotics

Computer vision enables robots to interpret visual information from cameras.

### Image Processing Fundamentals

**Image Formation**:
- Pinhole camera model
- Intrinsic parameters: focal length, principal point, distortion
- Extrinsic parameters: camera position and orientation

**Feature Detection**:
- Corners, edges, blobs
- SIFT, SURF, ORB features
- Applications: object recognition, tracking, SLAM

### Object Detection and Recognition

**Traditional Approaches**:
- Template matching
- Feature-based recognition
- Geometric reasoning

**Deep Learning Approaches**:
- Convolutional Neural Networks (CNNs)
- Real-time object detection (YOLO, SSD)
- Semantic segmentation

### Visual SLAM

Simultaneous Localization and Mapping uses visual information to:
- Build a map of the environment
- Determine the robot's location within the map
- Enable navigation in unknown environments

## Sensor Fusion

Sensor fusion combines data from multiple sensors to improve perception accuracy and robustness.

### Why Sensor Fusion?

- **Redundancy**: Multiple sensors can verify measurements
- **Complementarity**: Different sensors provide different information
- **Robustness**: System continues to function if one sensor fails

### Fusion Techniques

**Kalman Filters**:
- Optimal fusion for linear systems with Gaussian noise
- Extended Kalman Filter (EKF) for nonlinear systems
- Unscented Kalman Filter (UKF) for better nonlinear approximation

**Particle Filters**:
- Handle non-Gaussian, nonlinear systems
- Represent probability distributions with samples
- Computationally more expensive than Kalman filters

**Bayesian Networks**:
- Graphical models for probabilistic reasoning
- Handle complex dependencies between sensor measurements

## Perception Challenges in Robotics

### Sensor Noise and Uncertainty

All sensors have inherent noise and uncertainty:
- **Systematic errors**: Biases, calibration errors
- **Random errors**: Noise, environmental effects
- **Outliers**: Spurious measurements

### Environmental Challenges

**Dynamic Environments**:
- Moving objects and people
- Changing lighting conditions
- Occlusions and clutter

**Adverse Conditions**:
- Poor lighting (dark, bright)
- Weather effects (rain, fog)
- Reflective or transparent surfaces

## Week Summary

This week explored the fundamental concepts of sensors and perception in robotics. We covered various types of sensors, their principles of operation, and how robots use sensor data to understand their environment and internal state. Sensor fusion techniques were discussed to improve perception accuracy and robustness.

The next week will focus on motion control, exploring how robots execute planned movements and respond to environmental interactions.