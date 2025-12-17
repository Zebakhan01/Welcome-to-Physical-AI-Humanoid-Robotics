---
sidebar_position: 5
---

# Terminology and Definitions

## Introduction to Robotics Terminology

This glossary provides comprehensive definitions for key terms used throughout the Physical AI & Humanoid Robotics curriculum. Understanding this terminology is essential for effective communication and comprehension of advanced robotics concepts. The definitions are organized by category and include both technical specifications and practical applications.

## Core Robotics Concepts

### Physical AI Terms

**Physical AI (PAI)**: Artificial intelligence systems that operate through physical embodiment, interacting directly with the physical world through sensors and actuators rather than operating in purely virtual environments.

**Embodied Intelligence**: Intelligence that emerges from the interaction between an agent and its physical environment, where the body's physical properties contribute to cognitive processes.

**Embodiment**: The principle that an agent's physical form and interaction with the environment are integral to its intelligence and behavior.

**Morphological Computation**: The idea that the physical properties of a robot's body can contribute to its computational processes, reducing the burden on its control system.

**Affordance**: The possibility of action provided by an object or environment to an agent, based on the agent's capabilities and the object's properties.

**Proprioception**: The sense of the relative position of one's own parts of the body and strength of effort being employed in movement.

**Exteroception**: Sensing of external environmental properties such as light, sound, texture, temperature, and chemical concentrations.

**Tactile Sensing**: The ability to sense touch, pressure, vibration, and texture through physical contact with objects or surfaces.

**Haptic Feedback**: The use of touch and motion sensation to provide information to a user, often through specialized interfaces or robotic systems.

### Humanoid Robotics Terms

**Humanoid Robot**: A robot with a body structure similar to that of a human, typically having a head, torso, two arms, and two legs.

**Bipedal Locomotion**: The act of walking on two legs, a key characteristic of humanoid robots.

**Center of Mass (CoM)**: The point in a body where the total mass of the body is considered to be concentrated, critical for balance and stability.

**Zero Moment Point (ZMP)**: A concept used in robotics and biomechanics to indicate the point on the ground where the moment caused by the ground reaction force is zero.

**Capture Point**: The point on the ground where a biped must step to come to a complete stop, used in humanoid balance control.

**Dynamic Balance**: Maintaining balance during motion, as opposed to static balance which is balance while stationary.

**Static Balance**: Maintaining balance while stationary, typically requiring the center of mass to be within the support polygon.

**Support Polygon**: The convex hull of all contact points with the ground, defining the area where the center of mass must be located for stability.

**Gait**: The pattern of movement of the limbs of legged robots, particularly the sequence of foot placements during walking.

**Stride**: The distance between successive placements of the same foot during walking.

**Step**: The distance between placements of opposite feet during walking.

**Double Support Phase**: The phase of walking when both feet are in contact with the ground.

**Single Support Phase**: The phase of walking when only one foot is in contact with the ground.

**Swing Phase**: The phase of walking when a foot is not in contact with the ground and is moving forward.

**Stance Phase**: The phase of walking when a foot is in contact with the ground and supporting the body.

## Kinematics and Dynamics

### Kinematics Terms

**Forward Kinematics**: The use of kinematic equations to compute the position of the end-effector from specified values of joint parameters.

**Inverse Kinematics (IK)**: The mathematical process of calculating the joint parameters needed to place the end-effector at a desired position and orientation.

**Joint Space**: The space defined by the joint angles of a robot, as opposed to Cartesian space which is defined by position and orientation.

**Operational Space**: The space in which the end-effector of a robot operates, typically defined by position and orientation in Cartesian coordinates.

**Degrees of Freedom (DOF)**: The number of independent parameters that define the configuration of a mechanical system.

**Workspace**: The set of all possible positions that the end-effector of a robot can reach.

**Configuration Space (C-space)**: The space of all possible configurations of a robot, defined by its joint variables.

**End Effector**: The device at the end of a robot arm designed to interact with the environment, such as a gripper or tool.

**Manipulability**: A measure of how easily a robot can move its end-effector in different directions.

**Jacobian Matrix**: A matrix that relates the joint velocities to the end-effector velocity in robotics.

**Singularity**: A configuration of a robot manipulator where it loses one or more degrees of freedom, typically occurring when joint axes align.

**Redundant Robot**: A robot with more degrees of freedom than required to perform a given task.

**Dexterity**: The ability of a robot to perform complex manipulation tasks with precision.

### Dynamics Terms

**Rigid Body Dynamics**: The study of the motion of rigid bodies under the influence of forces and torques.

**Inertial Properties**: Properties of a body that determine its resistance to changes in motion, including mass, center of mass, and moments of inertia.

**Centroidal Dynamics**: The dynamics of a robot described relative to its center of mass.

**Centroidal Momentum**: The momentum of a robot expressed in terms of its center of mass and centroidal frame.

**Articulated Body Algorithm**: An efficient recursive algorithm for computing the forward dynamics of tree-structured robots.

**Lagrangian Mechanics**: A reformulation of classical mechanics that uses the principle of least action.

**Newton-Euler Formulation**: A method for deriving the equations of motion for mechanical systems.

**Coriolis Forces**: Apparent forces that arise in rotating reference frames, important in robot dynamics.

**Centrifugal Forces**: Apparent forces that arise from the rotation of a reference frame.

**Generalized Coordinates**: A set of parameters that define the configuration of a system relative to a reference configuration.

**Generalized Forces**: Forces expressed in terms of generalized coordinates.

**Equations of Motion**: Mathematical equations that describe the behavior of a dynamic system in terms of its motion as a function of time.

**Holonomic Constraints**: Constraints that can be expressed as equations involving only the coordinates and time.

**Non-holonomic Constraints**: Constraints that cannot be integrated to eliminate variables, typically involving velocities.

## Control Systems

### Control Theory Terms

**Feedback Control**: A control system that uses the difference between desired and actual output to adjust the control input.

**Feedforward Control**: A control system that anticipates disturbances and adjusts the control input accordingly.

**Proportional-Integral-Derivative (PID) Control**: A control loop feedback mechanism widely used in industrial control systems.

**Proportional Gain (Kp)**: The gain applied to the error term in a PID controller.

**Integral Gain (Ki)**: The gain applied to the integral of the error in a PID controller.

**Derivative Gain (Kd)**: The gain applied to the derivative of the error in a PID controller.

**State Feedback Control**: A control system that uses measurements of all state variables for feedback.

**Observer**: A system that estimates the internal state of a physical system from measurements of its input and output.

**Kalman Filter**: An algorithm that uses a series of measurements observed over time to estimate unknown variables.

**Extended Kalman Filter (EKF)**: A nonlinear version of the Kalman filter that linearizes the system around the current estimate.

**Unscented Kalman Filter (UKF)**: A Kalman filter variant that uses a deterministic sampling approach.

**Particle Filter**: A set of Monte Carlo algorithms that uses a set of particles to represent the posterior distribution.

**Model Predictive Control (MPC)**: An advanced control method that uses a model of the system to predict future behavior.

**Linear Quadratic Regulator (LQR)**: An optimal control technique that minimizes a quadratic cost function.

**Optimal Control**: A control strategy that minimizes a specified cost function.

**Trajectory Tracking**: The control of a system to follow a desired trajectory.

**Path Following**: The control of a system to follow a geometric path without strict timing constraints.

**Adaptive Control**: A control method that adjusts its parameters based on changes in the system.

**Robust Control**: A control method designed to function properly with specified uncertainties in the system.

**Gain Scheduling**: A control method where controller parameters are adjusted based on operating conditions.

### Motion Control Terms

**Motion Planning**: The computational process of finding a valid path from a start point to a goal point.

**Trajectory Generation**: The process of creating a time-parameterized path with specified position, velocity, and acceleration profiles.

**Waypoint**: A reference point on a path used for navigation, typically defined by coordinates in space.

**Via Point**: A point that the robot must pass through during motion execution.

**Trapezoidal Profile**: A motion profile with constant acceleration, constant velocity, and constant deceleration phases.

**S-Curve Profile**: A motion profile with smooth acceleration and deceleration ramps.

**Minimum Jerk Trajectory**: A trajectory that minimizes the integral of the squared jerk (third derivative of position).

**Velocity Profile**: A graph showing how velocity changes over time during motion.

**Acceleration Profile**: A graph showing how acceleration changes over time during motion.

**Jerk**: The rate of change of acceleration; the third derivative of position with respect to time.

**Smooth Motion**: Motion characterized by continuous position, velocity, and acceleration.

**Motion Blending**: The technique of smoothly transitioning between motion segments.

**Look-ahead**: A motion control technique that anticipates upcoming motion segments to optimize current motion.

**Motion Interpolation**: The process of generating intermediate positions between specified waypoints.

## Perception and Sensing

### Sensor Technology Terms

**LiDAR (Light Detection and Ranging)**: A remote sensing method that uses light in the form of a pulsed laser to measure distances.

**RGB-D Camera**: A camera that captures both color (RGB) and depth (D) information simultaneously.

**Stereo Vision**: A computer vision technique that uses two cameras to determine depth information.

**Structured Light**: A 3D scanning method that projects a known light pattern onto a surface to measure its shape.

**Time-of-Flight (ToF)**: A method for measuring distance by timing the round trip of a light pulse.

**IMU (Inertial Measurement Unit)**: A device that measures and reports a body's specific force, angular rate, and sometimes the magnetic field.

**Gyroscope**: A device that measures or maintains orientation and angular velocity.

**Accelerometer**: A device that measures proper acceleration (the acceleration felt by the device).

**Magnetometer**: An instrument that measures magnetic field strength and direction.

**Force/Torque Sensor**: A sensor that measures the forces and torques applied to it.

**Tactile Sensor**: A sensor that measures information obtained through touch.

**Range Sensor**: A sensor that measures the distance to objects in its field of view.

**Ultrasonic Sensor**: A sensor that uses sound waves to measure distance.

**Infrared Sensor**: A sensor that detects infrared radiation to measure distance or temperature.

**Encoders**: Sensors that measure the position of rotating shafts, either absolute or incremental.

**Absolute Encoder**: An encoder that provides a unique position value for each shaft position.

**Incremental Encoder**: An encoder that provides position change information relative to a starting point.

**Visual-Inertial Odometry (VIO)**: The process of determining position and orientation using visual and inertial measurements.

**Simultaneous Localization and Mapping (SLAM)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location.

**Visual SLAM**: SLAM using visual sensors as the primary input.

**LIDAR SLAM**: SLAM using LIDAR sensors as the primary input.

### Computer Vision Terms

**Point Cloud**: A set of data points in space, representing the external surface of an object or environment.

**Feature Detection**: The process of finding distinctive points in an image that can be used for matching.

**Feature Matching**: The process of finding corresponding features between different images.

**Homography**: A transformation that maps points in one image to corresponding points in another image.

**Epipolar Geometry**: The geometry of stereo vision, describing the relationship between two cameras.

**Fundamental Matrix**: A 3x3 matrix that relates corresponding points in stereo images.

**Essential Matrix**: A 3x3 matrix that relates corresponding points in stereo images for calibrated cameras.

**Camera Calibration**: The process of determining the internal parameters of a camera.

**Intrinsic Parameters**: Camera parameters that relate the camera's coordinate system to the idealized image plane.

**Extrinsic Parameters**: Camera parameters that relate the camera's coordinate system to the world coordinate system.

**Radial Distortion**: A type of optical distortion where straight lines appear curved.

**Tangential Distortion**: A type of optical distortion caused by the lens not being perfectly aligned with the image plane.

**Perspective-n-Point (PnP)**: The problem of estimating the pose of an object given its 3D model and 2D image correspondences.

**Structure from Motion (SfM)**: A photogrammetric range imaging technique for estimating three-dimensional structures from two-dimensional image sequences.

**Bundle Adjustment**: A general technique for refining a visual reconstruction to produce jointly optimal structure and viewing parameter estimates.

**Optical Flow**: The pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.

**Visual Odometry**: The process of incrementally estimating the position and orientation of a robot using visual data.

**Semantic Segmentation**: The process of assigning a semantic label to each pixel in an image.

**Instance Segmentation**: The process of identifying and delineating each distinct object instance in an image.

**Object Detection**: The task of identifying and localizing objects in images or video.

**Object Recognition**: The task of identifying what objects are present in an image.

**Pose Estimation**: The process of determining the position and orientation of an object.

**Visual Servoing**: The process of controlling a robot using visual feedback.

## Artificial Intelligence and Machine Learning

### AI/ML Terms

**Deep Learning**: A subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations.

**Convolutional Neural Network (CNN)**: A class of deep neural networks most commonly applied to analyzing visual imagery.

**Recurrent Neural Network (RNN)**: A class of neural networks where connections between nodes form a directed graph along a temporal sequence.

**Long Short-Term Memory (LSTM)**: A type of RNN architecture that can learn long-term dependencies.

**Transformer Architecture**: A deep learning model that uses self-attention mechanisms to weigh the importance of different parts of the input data.

**Reinforcement Learning**: A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.

**Deep Reinforcement Learning**: The combination of deep learning and reinforcement learning.

**Q-Learning**: A model-free reinforcement learning algorithm to learn quality of actions telling an agent what action to take under what circumstances.

**Actor-Critic**: A family of reinforcement learning algorithms that combine value-based and policy-based methods.

**Policy Gradient**: A class of reinforcement learning algorithms that directly optimize the policy parameters.

**Proximal Policy Optimization (PPO)**: A policy gradient method that makes incremental improvements using a surrogate objective function.

**Trust Region Policy Optimization (TRPO)**: A reinforcement learning algorithm for policy optimization.

**Deep Q-Network (DQN)**: A reinforcement learning algorithm that combines Q-learning with deep neural networks.

**Imitation Learning**: A machine learning technique where an agent learns to perform tasks by observing and mimicking expert demonstrations.

**Behavioral Cloning**: A simple form of imitation learning that treats the problem as supervised learning.

**Generative Adversarial Network (GAN)**: A class of machine learning frameworks where two neural networks contest with each other.

**Variational Autoencoder (VAE)**: A generative model that learns to encode data into a latent space and decode it back.

**Domain Randomization**: A technique for training neural networks on simulated data by randomizing the simulation parameters.

**Sim-to-Real Transfer**: The process of transferring policies or models trained in simulation to real-world applications.

**Transfer Learning**: A machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

**Few-Shot Learning**: The ability to learn new concepts from a small number of examples.

**Meta-Learning**: Learning to learn; algorithms that learn how to quickly learn new tasks.

**Curriculum Learning**: The process of training a model on a curriculum of tasks arranged by increasing difficulty.

**Multi-Task Learning**: A method where a single model is trained to perform multiple related tasks.

### Vision-Language Models

**Vision-Language Model**: A model that processes both visual and textual information jointly.

**CLIP (Contrastive Language-Image Pretraining)**: A model that learns visual concepts from natural language supervision.

**DALL-E**: A model that generates images from textual descriptions.

**BLIP (Bootstrapping Language-Image Pre-training)**: A model for vision-language pre-training.

**Flamingo**: A multimodal few-shot learning model.

**Vision-Language-Action (VLA)**: Systems that integrate vision, language understanding, and physical action.

**Multimodal Learning**: Learning that involves multiple modalities of information.

**Cross-Modal Attention**: Attention mechanism that allows one modality to attend to another.

**Vision Transformers**: Transformer models adapted for computer vision tasks.

**Tokenization**: The process of converting continuous data into discrete tokens for processing.

**Embedding**: A dense vector representation of categorical variables.

**Attention Mechanism**: A mechanism that allows models to focus on relevant parts of the input.

## Simulation and Software

### Simulation Terms

**Gazebo**: A 3D simulation environment for robotics that provides high-fidelity physics simulation and rendering capabilities.

**Isaac Sim**: NVIDIA's next-generation simulation application for robotics and AI development.

**Unity Robotics**: The integration of Unity's game engine with robotics simulation and development tools.

**ROS (Robot Operating System)**: A flexible framework for writing robot software, providing a collection of tools, libraries, and conventions.

**ROS 2**: The second generation of the Robot Operating System, designed to be production-ready.

**URDF (Unified Robot Description Format)**: An XML format for representing a robot model, including kinematic and dynamic properties.

**SDF (Simulation Description Format)**: An XML format for describing objects and environments in simulation.

**Xacro**: An XML macro language for generating URDF files, allowing for parameterization and reuse of robot model components.

**Gazebo Plugin**: A shared library that extends the functionality of the Gazebo simulation environment.

**Simulation Fidelity**: The accuracy with which a simulation represents the real world.

**Physics Engine**: Software that simulates the behavior of physical objects.

**ODE (Open Dynamics Engine)**: An open-source physics engine used in robotics simulation.

**Bullet Physics**: A real-time physics engine used in robotics and gaming applications.

**NVIDIA PhysX**: A proprietary physics engine developed by NVIDIA.

**Real-Time Simulation**: Simulation that runs at the same rate as real time.

**Deterministic Simulation**: Simulation that produces the same results given the same initial conditions.

**Monte Carlo Simulation**: A computational algorithm that relies on repeated random sampling to obtain numerical results.

**Domain Randomization**: A technique for training neural networks on simulated data by randomizing the simulation parameters.

**Simulation-to-Reality Gap**: The difference in performance between simulation and real-world deployment.

**Digital Twin**: A virtual replica of a physical system that can be used for simulation, analysis, and optimization.

### Software Architecture Terms

**Node**: A process that performs computation in ROS.

**Topic**: A named bus over which nodes exchange messages in ROS.

**Service**: A synchronous request/reply communication pattern in ROS.

**Action**: A goal-oriented communication pattern in ROS with feedback and status.

**Message**: A data packet sent between nodes in ROS.

**Package**: A container for organizing ROS code and resources.

**Launch File**: An XML file that can start multiple nodes with configuration parameters.

**Bag File**: A file format in ROS for storing and playing back ROS message data.

**TF (Transforms)**: A package in ROS for tracking coordinate frame relationships over time.

**RViz**: A 3D visualization tool for displaying robot models and sensor data.

**RQt**: A Qt-based framework for creating GUIs in ROS.

**Middleware**: Software that provides common services and capabilities to applications beyond what's offered by the operating system.

**Pub/Sub Pattern**: A messaging pattern where publishers send messages without knowing who will receive them.

**Client/Server Pattern**: A communication pattern where clients request services from servers.

**Microservices Architecture**: An architectural style that structures an application as a collection of loosely coupled services.

**Event-Driven Architecture**: A software architecture pattern promoting the production, detection, consumption of, and reaction to events.

**API (Application Programming Interface)**: A set of rules that allows programs to communicate with each other.

**SDK (Software Development Kit)**: A collection of software tools and programs used by developers to create applications.

**Framework**: A platform for developing software applications that provides a foundation for developing software.

**Library**: A collection of precompiled routines that a program can use.

**Package Manager**: A collection of software tools that automates the process of installing, upgrading, configuring, and removing software packages.

## Hardware Components

### Actuator Terms

**Servo Motor**: A rotary or linear actuator that allows for precise control of angular or linear position, velocity and acceleration.

**Stepper Motor**: A brushless DC electric motor that divides a full rotation into a number of equal steps.

**DC Motor**: A direct current motor that converts electrical energy into mechanical energy.

**Brushless DC Motor**: A synchronous motor powered by DC electricity via a commutator and brushes.

**Series Elastic Actuator (SEA)**: An actuator that includes a spring in series with the motor and load, providing compliance.

**Variable Stiffness Actuator (VSA)**: An actuator that can actively adjust its stiffness characteristics.

**Pneumatic Actuator**: An actuator that uses compressed air to generate motion.

**Hydraulic Actuator**: An actuator that uses pressurized fluid to generate motion.

**Shape Memory Alloy (SMA)**: An alloy that "remembers" its original shape and returns to it when heated.

**Muscle Wire**: Another term for Shape Memory Alloy actuators.

**Piezoelectric Actuator**: An actuator that uses piezoelectric materials to generate motion.

**Voice Coil Actuator**: An actuator that uses a coil in a magnetic field to generate linear motion.

**Gear Ratio**: The ratio of the number of rotations of the input gear to the output gear.

**Backlash**: The clearance or lost motion in a mechanism caused by gaps between parts.

**Torque**: The tendency of a force to cause rotation around an axis.

**Speed**: The rate of rotation of a motor, typically measured in RPM.

**Power**: The rate of doing work, measured in watts.

**Efficiency**: The ratio of useful output to total input in a system.

### Sensor Hardware Terms

**MEMS (Micro-Electro-Mechanical Systems)**: Miniaturized mechanical and electro-mechanical elements that are made using the techniques of microfabrication.

**Accelerometer**: A device that measures proper acceleration (the acceleration felt by the device).

**Gyroscope**: A device that measures or maintains orientation and angular velocity.

**Magnetometer**: An instrument that measures magnetic field strength and direction.

**Barometer**: An instrument that measures atmospheric pressure.

**Thermistor**: A type of resistor whose resistance varies significantly with temperature.

**Photodiode**: A semiconductor device that converts light into current using the photoelectric effect.

**CMOS Image Sensor**: An image sensor consisting of an integrated circuit containing an array of pixel sensors.

**CCD Image Sensor**: A device for moving electrical charge, usually from within the device to an area where the charge can be manipulated.

**Hall Effect Sensor**: A sensor that varies its output voltage in response to a magnetic field.

**Strain Gauge**: A device used to measure strain on an object.

**Load Cell**: A transducer that converts a force such as tension, compression, pressure, or torque into an electrical signal.

**Rotary Encoder**: An electro-mechanical device that converts the angular position or motion of a shaft to an analog or digital code.

**Linear Encoder**: A sensor, transducer or readhead paired with a scale that encodes position.

## Control and Planning

### Motion Planning Terms

**Path Planning**: The computational problem of finding a valid and optimal path from a start point to a goal point.

**Motion Planning**: The problem of finding a sequence of valid configurations that moves an object from a start configuration to a goal configuration.

**Configuration Space (C-space)**: The space of all possible configurations of a robot.

**Free Space**: The portion of configuration space not occupied by obstacles.

**Obstacle Space**: The portion of configuration space occupied by obstacles.

**Roadmap**: A graph representation of the connectivity of the free space.

**Cell Decomposition**: A method of dividing the configuration space into simple regions.

**Sampling-Based Planning**: Motion planning algorithms that sample the configuration space.

**Probabilistic Roadmap (PRM)**: A motion planning approach that constructs a graph of possible paths.

**Rapidly-exploring Random Tree (RRT)**: A motion planning algorithm designed for problems involving obstacles.

**RRT***: An asymptotically optimal variant of the RRT algorithm.

**A* Algorithm**: A graph traversal and path search algorithm that is widely used due to its completeness and optimality.

**Dijkstra's Algorithm**: An algorithm for finding the shortest paths between nodes in a graph.

**Potential Fields**: A motion planning approach that treats the robot as a particle moving under the influence of artificial forces.

**Bug Algorithms**: Simple motion planning algorithms that follow walls or obstacles.

**Visibility Graph**: A graph of intervisible locations, used in motion planning.

**Voronoi Diagram**: A partitioning of a plane into regions based on distance to points in a specific subset of the plane.

**Trajectory Optimization**: The process of finding a trajectory that minimizes a cost function while satisfying constraints.

**Optimal Control**: A control strategy that minimizes a specified cost function.

**Model Predictive Control (MPC)**: An advanced control method that uses a model of the system to predict future behavior.

### Manipulation Terms

**Grasping**: The process of securely holding an object with a robotic hand or gripper.

**Grasp Planning**: The process of determining how to grasp an object.

**Force Closure**: A grasp that can resist any external wrench applied to the object.

**Form Closure**: A grasp that constrains all possible motions of the object.

**Antipodal Grasp**: A grasp where the contact points are on opposite sides of the object.

**Power Grasp**: A grasp that focuses on stability and strength rather than dexterity.

**Precision Grasp**: A grasp that focuses on fine manipulation rather than strength.

**Pinch Grasp**: A grasp using thumb and finger(s) to hold an object.

**Cylindrical Grasp**: A grasp where the fingers wrap around a cylindrical object.

**Spherical Grasp**: A grasp where the hand conforms to a spherical object.

**Manipulation**: The branch of robotics dealing with the control of robot hands and arms to interact with objects.

**In-Hand Manipulation**: The process of repositioning an object within the hand without releasing it.

**Regrasping**: The process of changing the grasp configuration of an object.

**Dexterous Manipulation**: Fine manipulation tasks requiring high precision and dexterity.

**Tool Use**: The use of external objects as tools to extend the robot's capabilities.

**Bimanual Manipulation**: Manipulation using two hands/arms.

**Dual-Arm Coordination**: The coordination of two robotic arms for manipulation tasks.

**Assembly**: The process of joining components together to form a complete product.

**Disassembly**: The process of separating components from an assembled product.

**Insertion**: The process of placing one object into or onto another object.

## Safety and Ethics

### Safety Terms

**Safety-Critical System**: A system whose failure could result in human injury or death.

**Fail-Safe**: A design philosophy that ensures a system remains safe even in the event of component failure.

**Emergency Stop**: A safety mechanism that shuts down equipment when activated.

**Safety Integrity Level (SIL)**: A measure of the safety-related performance of a safety function.

**Functional Safety**: The part of the overall safety relating to the EUC and its control system.

**Risk Assessment**: The process of identifying hazards and analyzing their associated risks.

**Hazard**: A potential source of harm.

**Risk**: The combination of the probability of occurrence of harm and the severity of that harm.

**Safety Factor**: A multiplier applied to imposed loads or a divider applied to the strength of materials.

**Redundancy**: The duplication of critical components to increase reliability and safety.

**Fault Tolerance**: The property that enables a system to continue operating properly in the event of a fault.

**Graceful Degradation**: The ability of a system to maintain limited functionality even when portions of the system have failed.

**Safety Monitor**: A system component that continuously monitors the safety of the overall system.

**Safety Controller**: A dedicated controller responsible for safety-related functions.

**Safety PLC**: A programmable logic controller designed for safety-related applications.

**Safety Circuit**: An electrical circuit designed to ensure safe operation of machinery.

**Interlock**: A device that prevents an operation from occurring unless certain conditions are met.

**Guard**: A physical barrier that prevents access to danger zones.

**Light Curtain**: A photo-electric presence detection device used to safeguard personnel.

**Laser Scanner**: A safety device that creates a protective field using laser beams.

### Ethical Terms

**Robot Ethics**: The branch of ethics that attempts to understand the moral implications of robotics.

**Asimov's Laws**: A fictional set of rules governing the behavior of robots, proposed by Isaac Asimov.

**Human-Robot Interaction (HRI)**: The study of interactions between humans and robots.

**Social Robot**: A robot that interacts with humans in a socially acceptable manner.

**Anthropomorphism**: The attribution of human characteristics or behavior to non-human entities.

**Robot Rights**: Theoretical rights that might be granted to artificial beings.

**AI Alignment**: The challenge of ensuring AI systems behave in accordance with human values and intentions.

**Bias in AI**: Systematic errors in AI systems that lead to unfair outcomes.

**Transparency**: The quality of making AI decision-making processes understandable to humans.

**Accountability**: The responsibility for AI system behavior and decisions.

**Privacy**: The protection of personal information in AI systems.

**Surveillance**: The monitoring of behavior, activities, or other changing information.

**Autonomy**: The ability of a system to operate independently.

**Human Oversight**: Human supervision of autonomous systems.

**Meaningful Human Control**: The concept that humans should retain meaningful control over autonomous systems.

## Performance and Evaluation

### Performance Metrics

**Task Completion Rate**: The percentage of tasks successfully completed by a robot.

**Success Rate**: The proportion of attempts that result in successful task completion.

**Failure Rate**: The proportion of attempts that result in task failure.

**Success Criteria**: The specific conditions that define a successful task completion.

**Performance Metric**: A quantitative measure of system performance.

**Benchmark**: A standard against which performance is measured.

**Accuracy**: The closeness of a measured value to the true value.

**Precision**: The closeness of repeated measurements to each other.

**Recall**: The fraction of relevant instances that are retrieved.

**F1 Score**: The harmonic mean of precision and recall.

**Mean Squared Error (MSE)**: The average of the squares of the errors.

**Root Mean Square Error (RMSE)**: The square root of the MSE.

**Mean Absolute Error (MAE)**: The average of the absolute errors.

**Throughput**: The amount of work done per unit time.

**Latency**: The time delay between input and output.

**Jitter**: The variation in latency over time.

**Bandwidth**: The maximum rate of data transfer across a path.

**Real-Time Performance**: The ability to respond to inputs within specified time constraints.

**Deterministic Behavior**: Behavior that is predictable and repeatable.

**Repeatability**: The ability to reproduce the same results under identical conditions.

**Reliability**: The probability that a system will perform its intended function without failure.

**Availability**: The proportion of time a system is operational.

**Maintainability**: The ease with which a system can be maintained.

**Scalability**: The ability of a system to handle increased workload.

**Robustness**: The ability of a system to function correctly in the presence of variations and disturbances.

**Fault Tolerance**: The ability of a system to continue operating properly in the event of component failure.

**Resilience**: The ability of a system to recover from failures or disruptions.

**Stability**: The property of a system to return to equilibrium after a disturbance.

**Convergence**: The property of an algorithm to approach a solution.

**Optimality**: The property of achieving the best possible performance.

**Efficiency**: The ratio of useful output to total input.

**Throughput**: The amount of work done per unit time.

**Utilization**: The proportion of time a resource is actively being used.

**Bottleneck**: A point of congestion in a system that limits overall performance.

**Scalability**: The ability of a system to handle increased workload.

**Load Balancing**: The distribution of work across multiple computing resources.

**Parallel Processing**: The simultaneous execution of multiple processes.

**Concurrency**: The ability to execute multiple computations simultaneously.

**Throughput**: The rate of production of a system.

**Latency**: The time delay between input and output.

**Jitter**: The variation in latency over time.

**Throughput**: The amount of work done per unit time.

This comprehensive terminology guide covers the essential concepts, technologies, and methodologies used in Physical AI and Humanoid Robotics. Understanding these terms is fundamental to grasping the advanced concepts presented throughout the curriculum and effectively communicating within the robotics community.