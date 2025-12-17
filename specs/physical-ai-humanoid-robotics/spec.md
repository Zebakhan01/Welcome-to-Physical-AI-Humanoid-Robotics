# Physical AI & Humanoid Robotics - Textbook Specification

## Project Overview

This project creates a unified AI-native textbook titled "Physical AI & Humanoid Robotics" that follows a book-first, spec-driven architecture suitable for hackathon submission. The project consists of a Docusaurus-based textbook deployed on GitHub Pages with an embedded RAG chatbot built using FastAPI, OpenAI Agents, Qdrant Cloud, and Neon Serverless Postgres.

## Core Deliverables

### 1. Docusaurus-based Textbook
- Title: Physical AI & Humanoid Robotics
- Structured for teaching a full academic course
- Deployed on GitHub Pages
- Beginner-friendly but technically accurate content

### 2. Embedded RAG Chatbot
- Built using FastAPI
- Uses OpenAI Agents / ChatKit SDKs
- Uses Qdrant Cloud (vector database)
- Uses Neon Serverless Postgres (metadata & users)
- Answers questions ONLY from the book content
- Supports answering questions based on user-selected text

## Architectural Constraints

- Book is the single source of truth
- Chatbot cannot hallucinate outside the book
- Content must strictly follow the provided course outline
- Design supports future user personalization and Urdu translation
- No real robot control code required
- Project is educational, not an industrial robotics system

## Textbook Structure

### Folder Structure (JSON Tree)
```json
{
  "physical-ai-textbook": {
    "docs": {
      "intro": {
        "index.md": "Introduction to the textbook",
        "course-overview.md": "Overview of the Physical AI & Humanoid Robotics course",
        "prerequisites.md": "Prerequisites and setup requirements",
        "learning-path.md": "Recommended learning path and study tips"
      },
      "weeks": {
        "week-01-intro-physical-ai.md": "Introduction to Physical AI concepts and fundamentals",
        "week-02-robotics-fundamentals.md": "Core robotics concepts and mathematical foundations",
        "week-03-sensors-and-perception.md": "Robot sensors, perception, and computer vision basics",
        "week-04-motion-control.md": "Robot motion control and kinematics",
        "week-05-locomotion.md": "Locomotion principles and gait patterns",
        "week-06-manipulation.md": "Robotic manipulation and grasping concepts",
        "week-07-learning-for-robotics.md": "Machine learning applications in robotics",
        "week-08-vision-language-action.md": "Vision-Language-Action models for embodied AI",
        "week-09-ros-fundamentals.md": "ROS (Robot Operating System) fundamentals",
        "week-10-simulation-platforms.md": "Simulation platforms for robotics development",
        "week-11-hardware-integration.md": "Connecting software to physical hardware",
        "week-12-humanoid-architectures.md": "Humanoid robot architectures and design",
        "week-13-project-integration.md": "Capstone project: integrating all concepts"
      },
      "modules": {
        "ros-2": {
          "index.md": "ROS 2 introduction and setup",
          "ros-2-basics.md": "Basic ROS 2 concepts: nodes, topics, services",
          "ros-2-packages.md": "Creating and managing ROS 2 packages",
          "ros-2-launch.md": "ROS 2 launch files and system management",
          "ros-2-actions.md": "ROS 2 actions and complex behaviors",
          "ros-2-navigation.md": "Navigation stack and path planning"
        },
        "gazebo": {
          "index.md": "Gazebo simulation environment overview",
          "gazebo-models.md": "Creating and importing robot models in Gazebo",
          "gazebo-plugins.md": "Developing custom Gazebo plugins",
          "gazebo-environments.md": "Setting up simulation environments",
          "gazebo-integration.md": "Integrating Gazebo with ROS 2"
        },
        "unity": {
          "index.md": "Unity for robotics simulation introduction",
          "unity-setup.md": "Setting up Unity for robotics applications",
          "unity-physics.md": "Physics simulation in Unity for robotics",
          "unity-robot-modeling.md": "Creating robots in Unity",
          "unity-ros-bridge.md": "Unity-ROS bridge integration"
        },
        "nvidia-isaac": {
          "index.md": "NVIDIA Isaac robotics platform overview",
          "isaac-sim.md": "NVIDIA Isaac Sim for high-fidelity simulation",
          "isaac-app-framework.md": "Isaac application framework",
          "isaac-ros-bridge.md": "Isaac-ROS integration",
          "isaac-ai-modules.md": "AI modules in Isaac platform"
        },
        "vla": {
          "index.md": "Vision-Language-Action models overview",
          "vla-foundations.md": "Foundations of VLA models",
          "vla-architectures.md": "Common VLA architectures and implementations",
          "vla-training.md": "Training VLA models for robotics",
          "vla-deployment.md": "Deploying VLA models on robots",
          "vla-case-studies.md": "Real-world VLA applications in robotics"
        }
      },
      "capstone": {
        "index.md": "Capstone project introduction and requirements",
        "project-planning.md": "Planning your humanoid robotics project",
        "implementation-phase.md": "Implementation phase guidance",
        "testing-validation.md": "Testing and validation strategies",
        "presentation-guidelines.md": "Project presentation guidelines"
      },
      "hardware": {
        "index.md": "Hardware guide overview",
        "actuators-servos.md": "Types of actuators and servo motors",
        "sensors-overview.md": "Robot sensors: cameras, IMUs, LiDAR, etc.",
        "control-systems.md": "Robot control systems and microcontrollers",
        "power-management.md": "Power systems for humanoid robots",
        "assembly-guide.md": "Step-by-step assembly instructions",
        "troubleshooting.md": "Common hardware troubleshooting"
      },
      "appendix": {
        "index.md": "Appendix overview and reference materials",
        "commands.md": "Common commands and terminal operations",
        "setup-scripts.md": "Setup scripts for different platforms",
        "configurations.md": "Reference configurations and parameters",
        "urdf-tutorial.md": "URDF (Unified Robot Description Format) tutorial",
        "simulation-tips.md": "Advanced simulation tips and tricks"
      }
    },
    "src": {
      "components": {},
      "pages": {
        "index.js": "Homepage for the textbook website",
        "chatbot.jsx": "Embedded RAG chatbot interface",
        "search.jsx": "Advanced search functionality"
      },
      "css": {
        "custom.css": "Custom styling for the textbook"
      }
    },
    "static": {
      "img": {
        "diagrams": {},
        "simulations": {},
        "photos": {}
      }
    },
    "backend": {
      "api": {
        "chat": {
          "chat_routes.py": "Chat API endpoints",
          "message_processing.py": "Message processing and validation"
        },
        "rag": {
          "embedding_service.py": "Embedding generation service",
          "retrieval_service.py": "Content retrieval service",
          "vector_store.py": "Vector store integration (Qdrant)"
        },
        "auth": {
          "auth_routes.py": "Authentication endpoints",
          "user_management.py": "User management service"
        },
        "content": {
          "content_parser.py": "Textbook content parsing",
          "metadata_service.py": "Metadata management (Neon Postgres)"
        }
      },
      "models": {
        "user.py": "User data model",
        "conversation.py": "Conversation data model",
        "content_chunk.py": "Content chunk data model"
      },
      "utils": {
        "config.py": "Configuration management",
        "logger.py": "Logging utilities",
        "validators.py": "Data validation utilities"
      },
      "main.py": "FastAPI application entry point",
      "database.py": "Database connection management"
    },
    "docusaurus.config.js": "Docusaurus configuration file",
    "package.json": "Frontend dependencies",
    "requirements.txt": "Backend Python dependencies",
    "README.md": "Project documentation and setup guide"
  }
}
```

### Complete List of Chapter Files

#### Introduction Section
- `docs/intro/index.md`: Introduction to the textbook
  - Chapter title: Welcome to Physical AI & Humanoid Robotics
  - Learning intent: Introduces students to the exciting field of physical AI and humanoid robotics. Provides an overview of what they will learn throughout the course. Sets expectations for the journey ahead.

- `docs/intro/course-overview.md`: Overview of the Physical AI & Humanoid Robotics course
  - Chapter title: Course Structure and Learning Objectives
  - Learning intent: Outlines the course structure, learning objectives, and assessment methods. Explains how each module connects to create a cohesive learning experience. Highlights the importance of hands-on practice.

- `docs/intro/prerequisites.md`: Prerequisites and setup requirements
  - Chapter title: Prerequisites and Technical Setup
  - Learning intent: Details the technical prerequisites needed for the course. Provides step-by-step instructions for setting up the development environment. Ensures all students start with a consistent baseline.

- `docs/intro/learning-path.md`: Recommended learning path and study tips
  - Chapter title: Learning Path and Study Strategies
  - Learning intent: Offers guidance on how to approach the material effectively. Suggests study strategies tailored to robotics education. Recommends supplementary resources for deeper understanding.

#### Week-based Chapters (Week 1 to Week 13)
- `docs/weeks/week-01-intro-physical-ai.md`: Introduction to Physical AI concepts and fundamentals
  - Chapter title: Introduction to Physical AI
  - Learning intent: Establishes the foundation of physical AI as the intersection of artificial intelligence and real-world interaction. Explores the differences between traditional AI and embodied AI. Introduces the concept of robots as thinking, acting entities.

- `docs/weeks/week-02-robotics-fundamentals.md`: Core robotics concepts and mathematical foundations
  - Chapter title: Robotics Fundamentals and Mathematical Foundations
  - Learning intent: Covers essential robotics concepts including coordinate systems, transformations, and kinematics. Introduces the mathematical tools needed for robotics applications. Provides practical examples using simulation environments.

- `docs/weeks/week-03-sensors-and-perception.md`: Robot sensors, perception, and computer vision basics
  - Chapter title: Sensors and Perception Systems
  - Learning intent: Explores various robot sensors and their applications. Covers computer vision basics relevant to robotics. Demonstrates how robots perceive and interpret their environment.

- `docs/weeks/week-04-motion-control.md`: Robot motion control and kinematics
  - Chapter title: Motion Control and Kinematics
  - Learning intent: Explains forward and inverse kinematics for robot movement. Covers motion planning and trajectory generation. Demonstrates control algorithms for precise robot movement.

- `docs/weeks/week-05-locomotion.md`: Locomotion principles and gait patterns
  - Chapter title: Locomotion Principles and Gait Patterns
  - Learning intent: Studies different locomotion methods for mobile robots. Explores static and dynamic balance concepts. Examines gait patterns in legged robots and biological systems.

- `docs/weeks/week-06-manipulation.md`: Robotic manipulation and grasping concepts
  - Chapter title: Manipulation and Grasping
  - Learning intent: Covers robotic manipulation principles and grasp planning. Explores end-effector design and control. Discusses dexterity and fine motor control in robots.

- `docs/weeks/week-07-learning-for-robotics.md`: Machine learning applications in robotics
  - Chapter title: Machine Learning Applications in Robotics
  - Learning intent: Explores reinforcement learning, imitation learning, and other ML techniques for robotics. Shows how learning algorithms enable adaptive robot behavior. Discusses safety considerations in learned control systems.

- `docs/weeks/week-08-vision-language-action.md`: Vision-Language-Action models for embodied AI
  - Chapter title: Vision-Language-Action Models for Embodied AI
  - Learning intent: Introduces VLA models as a key technology for embodied AI. Explains how vision, language, and action are integrated in modern robotic systems. Demonstrates practical applications of VLA models.

- `docs/weeks/week-09-ros-fundamentals.md`: ROS (Robot Operating System) fundamentals
  - Chapter title: ROS Fundamentals
  - Learning intent: Introduces the Robot Operating System as a middleware for robotics development. Explains nodes, topics, services, and parameters. Provides hands-on experience with basic ROS tools.

- `docs/weeks/week-10-simulation-platforms.md`: Simulation platforms for robotics development
  - Chapter title: Simulation Platforms for Robotics
  - Learning intent: Compares different simulation platforms for robotics development. Covers physics engines, sensor simulation, and environment modeling. Emphasizes the importance of simulation in robotics development.

- `docs/weeks/week-11-hardware-integration.md`: Connecting software to physical hardware
  - Chapter title: Hardware Integration
  - Learning intent: Bridges the gap between simulation and real hardware. Covers hardware abstraction layers and device drivers. Discusses safety considerations when working with physical robots.

- `docs/weeks/week-12-humanoid-architectures.md`: Humanoid robot architectures and design
  - Chapter title: Humanoid Robot Architectures and Design
  - Learning intent: Explores the unique challenges of humanoid robot design. Discusses mechanical design, actuator selection, and control architectures. Examines examples of successful humanoid robots.

- `docs/weeks/week-13-project-integration.md`: Capstone project: integrating all concepts
  - Chapter title: Capstone Project - Integrating All Concepts
  - Learning intent: Challenges students to integrate all learned concepts into a comprehensive project. Encourages creative problem-solving and system-level thinking. Prepares students for advanced robotics work.

#### Module-based Chapters: ROS 2
- `docs/modules/ros-2/index.md`: ROS 2 introduction and setup
  - Chapter title: ROS 2 Introduction and Setup
  - Learning intent: Provides an overview of ROS 2 and its advantages over ROS 1. Guides through the installation and setup process. Introduces the core concepts and architecture of ROS 2.

- `docs/modules/ros-2/ros-2-basics.md`: Basic ROS 2 concepts: nodes, topics, services
  - Chapter title: ROS 2 Basics - Nodes, Topics, Services
  - Learning intent: Explains the fundamental communication patterns in ROS 2. Demonstrates how to create nodes and establish communication. Provides practical exercises for hands-on learning.

- `docs/modules/ros-2/ros-2-packages.md`: Creating and managing ROS 2 packages
  - Chapter title: Creating and Managing ROS 2 Packages
  - Learning intent: Teaches how to create, build, and manage ROS 2 packages. Covers package.xml and CMakeLists.txt configuration. Explains best practices for organizing ROS 2 projects.

- `docs/modules/ros-2/ros-2-launch.md`: ROS 2 launch files and system management
  - Chapter title: ROS 2 Launch Files and System Management
  - Learning intent: Explains how to use launch files to manage complex ROS 2 systems. Covers parameter management and system startup procedures. Demonstrates best practices for system deployment.

- `docs/modules/ros-2/ros-2-actions.md`: ROS 2 actions and complex behaviors
  - Chapter title: ROS 2 Actions and Complex Behaviors
  - Learning intent: Introduces ROS 2 actions for long-running tasks with feedback. Explains when to use actions versus services or topics. Demonstrates complex behavior implementation.

- `docs/modules/ros-2/ros-2-navigation.md`: Navigation stack and path planning
  - Chapter title: ROS 2 Navigation Stack and Path Planning
  - Learning intent: Explores the ROS 2 navigation stack for mobile robots. Covers SLAM, path planning, and obstacle avoidance. Provides practical examples of navigation system implementation.

#### Module-based Chapters: Gazebo
- `docs/modules/gazebo/index.md`: Gazebo simulation environment overview
  - Chapter title: Gazebo Simulation Environment Overview
  - Learning intent: Introduces Gazebo as a high-fidelity physics simulator. Explains its role in robotics development and testing. Demonstrates basic Gazebo usage and interface.

- `docs/modules/gazebo/gazebo-models.md`: Creating and importing robot models in Gazebo
  - Chapter title: Creating and Importing Robot Models in Gazebo
  - Learning intent: Teaches how to create robot models for Gazebo simulation. Covers URDF integration and physics properties. Provides examples of realistic robot models.

- `docs/modules/gazebo/gazebo-plugins.md`: Developing custom Gazebo plugins
  - Chapter title: Developing Custom Gazebo Plugins
  - Learning intent: Explains how to extend Gazebo functionality with custom plugins. Covers sensor plugins, controller plugins, and world plugins. Provides examples of useful plugin implementations.

- `docs/modules/gazebo/gazebo-environments.md`: Setting up simulation environments
  - Chapter title: Setting Up Simulation Environments
  - Learning intent: Shows how to create realistic simulation environments in Gazebo. Covers terrain generation, object placement, and lighting. Explains how environments affect robot behavior.

- `docs/modules/gazebo/gazebo-integration.md`: Integrating Gazebo with ROS 2
  - Chapter title: Integrating Gazebo with ROS 2
  - Learning intent: Demonstrates seamless integration between Gazebo and ROS 2. Explains how to control simulated robots using ROS 2. Provides complete examples of simulation workflows.

#### Module-based Chapters: Unity
- `docs/modules/unity/index.md`: Unity for robotics simulation introduction
  - Chapter title: Unity for Robotics Simulation Introduction
  - Learning intent: Introduces Unity as a robotics simulation platform. Explains its advantages for visualization and VR applications. Demonstrates the setup process for robotics applications.

- `docs/modules/unity/unity-setup.md`: Setting up Unity for robotics applications
  - Chapter title: Setting Up Unity for Robotics Applications
  - Learning intent: Guides through configuring Unity for robotics development. Covers necessary packages and tools for robotics simulation. Explains how to import robot models.

- `docs/modules/unity/unity-physics.md`: Physics simulation in Unity for robotics
  - Chapter title: Physics Simulation in Unity for Robotics
  - Learning intent: Explores Unity's physics engine for realistic robot simulation. Covers joint constraints, collisions, and contact forces. Demonstrates how to tune physics parameters for accuracy.

- `docs/modules/unity/unity-robot-modeling.md`: Creating robots in Unity
  - Chapter title: Creating Robots in Unity
  - Learning intent: Teaches how to build robot models in Unity. Covers kinematic chains, actuators, and sensors. Explains how to animate robot movements realistically.

- `docs/modules/unity/unity-ros-bridge.md`: Unity-ROS bridge integration
  - Chapter title: Unity-ROS Bridge Integration
  - Learning intent: Explains how to connect Unity simulations with ROS 2. Covers the Unity-ROS bridge protocols and implementation. Provides examples of bidirectional communication.

#### Module-based Chapters: NVIDIA Isaac
- `docs/modules/nvidia-isaac/index.md`: NVIDIA Isaac robotics platform overview
  - Chapter title: NVIDIA Isaac Robotics Platform Overview
  - Learning intent: Introduces NVIDIA Isaac as a comprehensive robotics platform. Explains its components and advantages for AI-powered robots. Demonstrates the setup process for Isaac applications.

- `docs/modules/nvidia-isaac/isaac-sim.md`: NVIDIA Isaac Sim for high-fidelity simulation
  - Chapter title: NVIDIA Isaac Sim for High-Fidelity Simulation
  - Learning intent: Explores Isaac Sim's capabilities for photorealistic simulation. Covers advanced features like RTX rendering and synthetic data generation. Demonstrates how to create complex simulation scenarios.

- `docs/modules/nvidia-isaac/isaac-app-framework.md`: Isaac application framework
  - Chapter title: Isaac Application Framework
  - Learning intent: Explains the Isaac application framework for building robotics applications. Covers codecs, nodes, and extensions. Provides templates for common robotics applications.

- `docs/modules/nvidia-isaac/isaac-ros-bridge.md`: Isaac-ROS integration
  - Chapter title: Isaac-ROS Integration
  - Learning intent: Demonstrates how to integrate Isaac with ROS 2 ecosystems. Explains the benefits of combining Isaac's AI capabilities with ROS 2's flexibility. Provides integration examples.

- `docs/modules/nvidia-isaac/isaac-ai-modules.md`: AI modules in Isaac platform
  - Chapter title: AI Modules in Isaac Platform
  - Learning intent: Explores Isaac's built-in AI modules for perception and control. Covers computer vision, planning, and learning modules. Demonstrates how to customize and train these modules.

#### Module-based Chapters: Vision-Language-Action (VLA)
- `docs/modules/vla/index.md`: Vision-Language-Action models overview
  - Chapter title: Vision-Language-Action Models Overview
  - Learning intent: Introduces VLA models as a breakthrough in embodied AI. Explains how vision, language, and action are unified in modern AI systems. Discusses the implications for robotics.

- `docs/modules/vla/vla-foundations.md`: Foundations of VLA models
  - Chapter title: Foundations of VLA Models
  - Learning intent: Explores the theoretical foundations of VLA models. Covers transformer architectures adapted for embodied tasks. Explains how multimodal representations enable robot understanding.

- `docs/modules/vla/vla-architectures.md`: Common VLA architectures and implementations
  - Chapter title: Common VLA Architectures and Implementations
  - Learning intent: Surveys leading VLA architectures and their implementations. Compares different approaches to multimodal fusion. Provides practical examples of VLA model deployment.

- `docs/modules/vla/vla-training.md`: Training VLA models for robotics
  - Chapter title: Training VLA Models for Robotics
  - Learning intent: Explains the data requirements and training procedures for VLA models. Covers dataset preparation and evaluation metrics. Discusses computational requirements and optimization strategies.

- `docs/modules/vla/vla-deployment.md`: Deploying VLA models on robots
  - Chapter title: Deploying VLA Models on Robots
  - Learning intent: Addresses the challenges of deploying VLA models on physical robots. Covers inference optimization and real-time performance requirements. Provides best practices for production deployment.

- `docs/modules/vla/vla-case-studies.md`: Real-world VLA applications in robotics
  - Chapter title: Real-World VLA Applications in Robotics
  - Learning intent: Examines successful deployments of VLA models in robotics. Analyzes case studies from research and industry. Discusses lessons learned and future directions.

#### Capstone Project Section
- `docs/capstone/index.md`: Capstone project introduction and requirements
  - Chapter title: Capstone Project Introduction and Requirements
  - Learning intent: Introduces the capstone project as the culmination of the course. Explains project requirements and evaluation criteria. Inspires students to tackle ambitious robotics challenges.

- `docs/capstone/project-planning.md`: Planning your humanoid robotics project
  - Chapter title: Planning Your Humanoid Robotics Project
  - Learning intent: Guides students through the project planning process. Covers requirement analysis, system design, and timeline planning. Emphasizes iterative development and risk management.

- `docs/capstone/implementation-phase.md`: Implementation phase guidance
  - Chapter title: Implementation Phase Guidance
  - Learning intent: Provides detailed guidance for the implementation phase. Covers integration challenges and debugging strategies. Offers advice on testing and validation approaches.

- `docs/capstone/testing-validation.md`: Testing and validation strategies
  - Chapter title: Testing and Validation Strategies
  - Learning intent: Explains comprehensive testing methodologies for robotics projects. Covers simulation testing, hardware-in-the-loop testing, and safety validation. Emphasizes the importance of systematic verification.

- `docs/capstone/presentation-guidelines.md`: Project presentation guidelines
  - Chapter title: Project Presentation Guidelines
  - Learning intent: Provides guidelines for presenting robotics projects effectively. Covers technical documentation, demonstration planning, and communication strategies. Helps students showcase their achievements.

#### Hardware & Lab Guide
- `docs/hardware/index.md`: Hardware guide overview
  - Chapter title: Hardware Guide Overview
  - Learning intent: Introduces the hardware components commonly used in robotics. Explains how software concepts translate to physical implementations. Prepares students for hands-on lab experiences.

- `docs/hardware/actuators-servos.md`: Types of actuators and servo motors
  - Chapter title: Types of Actuators and Servo Motors
  - Learning intent: Surveys different types of actuators used in robotics. Explains the characteristics and applications of various motor types. Discusses selection criteria for different robotic applications.

- `docs/hardware/sensors-overview.md`: Robot sensors: cameras, IMUs, LiDAR, etc.
  - Chapter title: Robot Sensors: Cameras, IMUs, LiDAR, and More
  - Learning intent: Explores various sensors used in robotics applications. Explains how sensor data is processed and interpreted. Discusses sensor fusion techniques for improved perception.

- `docs/hardware/control-systems.md`: Robot control systems and microcontrollers
  - Chapter title: Robot Control Systems and Microcontrollers
  - Learning intent: Covers different control system architectures for robots. Explains the role of microcontrollers and single-board computers. Discusses real-time control requirements and implementation.

- `docs/hardware/power-management.md`: Power systems for humanoid robots
  - Chapter title: Power Systems for Humanoid Robots
  - Learning intent: Addresses the unique power challenges of humanoid robots. Explains battery selection, power distribution, and energy efficiency. Discusses thermal management and safety considerations.

- `docs/hardware/assembly-guide.md`: Step-by-step assembly instructions
  - Chapter title: Step-by-Step Assembly Instructions
  - Learning intent: Provides clear instructions for assembling robotic hardware. Emphasizes safety procedures and proper handling of components. Includes troubleshooting tips for common assembly issues.

- `docs/hardware/troubleshooting.md`: Common hardware troubleshooting
  - Chapter title: Common Hardware Troubleshooting
  - Learning intent: Equips students with troubleshooting skills for robotic hardware. Covers diagnostic procedures and systematic problem-solving approaches. Provides reference guides for common issues.

#### Glossary and Index
- `docs/glossary.md`: Comprehensive glossary of robotics terms
  - Chapter title: Glossary of Robotics Terms
  - Learning intent: Provides definitions for key robotics terminology. Organized alphabetically for easy reference. Includes both technical and conceptual terms used throughout the textbook.

- `docs/index.md`: Comprehensive index of topics and concepts
  - Chapter title: Index of Topics and Concepts
  - Learning intent: Offers a comprehensive index for quick topic lookup. Organized by subject area and concept relationships. Enables efficient navigation of the textbook content.

#### Appendix
- `docs/appendix/index.md`: Appendix overview and reference materials
  - Chapter title: Appendix Overview and Reference Materials
  - Learning intent: Introduces the appendix as a collection of reference materials. Explains how to use the various reference sections effectively. Points to supplementary resources for continued learning.

- `docs/appendix/commands.md`: Common commands and terminal operations
  - Chapter title: Common Commands and Terminal Operations
  - Learning intent: Provides a comprehensive reference for terminal commands. Covers Linux, ROS, Git, and other essential command-line tools. Includes examples and best practices for command usage.

- `docs/appendix/setup-scripts.md`: Setup scripts for different platforms
  - Chapter title: Setup Scripts for Different Platforms
  - Learning intent: Offers automated setup scripts for various platforms and configurations. Explains how to customize scripts for specific environments. Provides troubleshooting guidance for setup issues.

- `docs/appendix/configurations.md`: Reference configurations and parameters
  - Chapter title: Reference Configurations and Parameters
  - Learning intent: Contains reference configurations for common robotics setups. Explains the meaning and impact of key parameters. Provides templates for custom configurations.

- `docs/appendix/urdf-tutorial.md`: URDF (Unified Robot Description Format) tutorial
  - Chapter title: URDF Tutorial - Unified Robot Description Format
  - Learning intent: Provides a comprehensive tutorial on URDF for robot modeling. Explains all URDF elements and their relationships. Includes practical examples and best practices.

- `docs/appendix/simulation-tips.md`: Advanced simulation tips and tricks
  - Chapter title: Advanced Simulation Tips and Tricks
  - Learning intent: Shares expert knowledge for effective simulation use. Covers performance optimization and advanced features. Explains how to create realistic and efficient simulation environments.

## Quality Bar

### Educational Standards
- Content is beginner-friendly while maintaining technical accuracy
- Clear separation between concepts, simulations, and hardware
- Written for AI-native learning with future agent integration
- Each chapter includes learning objectives, key concepts, and practical exercises

### Technical Standards
- Code examples are tested and functional
- All simulation environments are properly configured
- RAG system properly indexes and retrieves textbook content
- Chatbot responses are grounded in textbook content only

### Accessibility Standards
- Content supports future Urdu translation
- Material is structured for personalized learning paths
- Visual elements include proper alt text and descriptions
- Navigation supports keyboard and screen reader accessibility

## Success Criteria

- Students can build and deploy a complete robotics application
- RAG chatbot accurately answers questions based solely on textbook content
- All simulation environments function correctly
- Content meets academic standards for a full semester course
- Project is suitable for hackathon submission and demonstration