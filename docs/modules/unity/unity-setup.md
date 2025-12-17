---
sidebar_position: 2
---

# Unity Setup

## Introduction to Unity for Robotics

Unity is a powerful cross-platform game engine that has gained significant traction in robotics simulation due to its high-fidelity graphics, robust physics engine, and flexible architecture. Unlike traditional robotics simulators, Unity provides photorealistic rendering capabilities, making it ideal for training perception systems and creating immersive simulation environments.

## Unity Installation and Configuration

### System Requirements

Before installing Unity, ensure your system meets the following requirements:

**Minimum Requirements:**
- Operating System: Windows 10 (64-bit), macOS 10.14+, or Ubuntu 18.04+
- CPU: SSE2 instruction set support
- RAM: 4 GB
- GPU: DX10 (shader model 4.0) capabilities

**Recommended Requirements for Robotics Simulation:**
- Operating System: Windows 10/11 (64-bit) or Ubuntu 20.04+
- CPU: Multi-core processor (Intel i7 or AMD Ryzen)
- RAM: 16 GB or more
- GPU: Dedicated graphics card with 4GB+ VRAM (NVIDIA GTX 1060 or better)
- Storage: SSD with 50GB+ free space

### Unity Hub Installation

Unity Hub is the recommended way to manage Unity installations:

1. Download Unity Hub from the official Unity website
2. Install Unity Hub with default settings
3. Sign in with a Unity ID (free account) to access additional assets and services
4. Use Unity Hub to install specific Unity versions and manage projects

### Unity Editor Installation

For robotics applications, consider installing Unity versions that offer:
- Long-term support (LTS) for stability
- High-fidelity rendering pipeline support
- XR capabilities for immersive simulation

Recommended Unity version: 2022.3 LTS or newer

## Unity Project Setup for Robotics

### Creating a New Project

1. Open Unity Hub and click "New Project"
2. Select the "3D (Built-in Render Pipeline)" template (simpler for robotics simulation)
3. Name your project (e.g., "RoboticsSimulation")
4. Choose a project location
5. Click "Create Project"

### Project Structure

Organize your Unity robotics project with the following structure:

```
RoboticsSimulation/
├── Assets/
│   ├── Scripts/           # C# scripts for robot control and simulation
│   ├── Models/            # 3D models for robots and environment
│   ├── Materials/         # Material definitions
│   ├── Textures/          # Texture files
│   ├── Prefabs/           # Reusable robot and environment objects
│   ├── Scenes/            # Simulation environments
│   ├── Plugins/           # External libraries and ROS integration
│   └── PhysicsMaterials/  # Custom physics materials
├── Packages/              # Package dependencies
└── ProjectSettings/       # Project configuration
```

## Essential Unity Components for Robotics

### Physics Engine Configuration

Unity uses the NVIDIA PhysX physics engine. For robotics simulation:

1. Go to Edit → Project Settings → Physics
2. Configure the following settings:

```
Gravity: (0, -9.81, 0) - Standard Earth gravity
Default Material: Create custom materials for robot components
Bounce Threshold: 2 - Minimum impact speed for bounce
Sleep Threshold: 0.005 - Energy threshold for sleeping rigidbodies
```

### Physics Material Setup

Create custom physics materials for robot components:

```csharp
// Example: Creating a physics material for robot wheels
using UnityEngine;

public class RobotPhysicsMaterials : MonoBehaviour
{
    [Header("Wheel Material")]
    public PhysicMaterial wheelMaterial;

    [Header("Gripper Material")]
    public PhysicMaterial gripperMaterial;

    [Header("Ground Material")]
    public PhysicMaterial groundMaterial;

    void Start()
    {
        // Configure wheel material for good traction
        if(wheelMaterial != null)
        {
            wheelMaterial.staticFriction = 0.8f;
            wheelMaterial.dynamicFriction = 0.7f;
            wheelMaterial.bounciness = 0.1f;
        }

        // Configure gripper material for object manipulation
        if(gripperMaterial != null)
        {
            gripperMaterial.staticFriction = 0.9f;
            gripperMaterial.dynamicFriction = 0.8f;
            gripperMaterial.bounciness = 0.0f;
        }
    }
}
```

## Unity Settings for Robotics Simulation

### Quality Settings

Optimize for robotics simulation performance:

1. Go to Edit → Project Settings → Quality
2. Adjust settings based on your target hardware:
   - For real-time simulation: Medium to High quality
   - For training data generation: Ultra quality
   - For headless operation: Fastest quality

### Time Settings

Configure time management for simulation:

1. Go to Edit → Project Settings → Time
2. Set Fixed Timestep for physics simulation:
   - For real-time: 0.02 (50 FPS physics)
   - For accuracy: 0.01 (100 FPS physics)
   - For performance: 0.033 (30 FPS physics)

```csharp
// Example: Dynamic timestep adjustment based on simulation mode
using UnityEngine;

public class SimulationTimeManager : MonoBehaviour
{
    [Header("Time Settings")]
    public float realTimeTimestep = 0.02f;
    public float fastTimestep = 0.001f;
    public bool useFastSimulation = false;

    void Start()
    {
        SetSimulationMode(useFastSimulation);
    }

    public void SetSimulationMode(bool fastMode)
    {
        useFastSimulation = fastMode;
        Time.fixedDeltaTime = fastMode ? fastTimestep : realTimeTimestep;
        Time.timeScale = fastMode ? 10.0f : 1.0f; // Speed up simulation time
    }
}
```

### Graphics Settings

For robotics simulation, consider:

1. Go to Edit → Project Settings → Graphics
2. Configure Tier Settings for different platforms
3. Set up appropriate rendering pipelines based on needs

## Robotics-Specific Unity Packages

### Installing Required Packages

Use the Unity Package Manager (Window → Package Manager) to install:

1. **Universal Render Pipeline (URP)** or **High Definition Render Pipeline (HDRP)** for advanced rendering
2. **ProBuilder** for rapid environment prototyping
3. **ProGrids** for precise object placement
4. **Cinemachine** for camera control and visualization

### Package Manager Setup

```json
// manifest.json (Packages/manifest.json)
{
  "dependencies": {
    "com.unity.burst": "1.8.4",
    "com.unity.collections": "2.1.4",
    "com.unity.inputsystem": "1.5.1",
    "com.unity.mathematics": "1.2.6",
    "com.unity.physics": "1.1.1",
    "com.unity.render-pipelines.universal": "14.0.8"
  }
}
```

## ROS Integration Setup

### Unity Robotics Package

The Unity Robotics package provides essential tools for ROS integration:

1. In Package Manager, click the "+" button
2. Select "Add package from git URL"
3. Enter: `com.unity.robotics.ros-tcp-connector`
4. Also install: `com.unity.robotics.urdf-importer`

### ROS TCP Connector Configuration

```csharp
// Example: Setting up ROS TCP connector
using Unity.Robotics.ROSTCPConnector;

public class ROSConnectionManager : MonoBehaviour
{
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public bool autoConnect = true;

    private ROSConnection rosConnection;

    void Start()
    {
        if (autoConnect)
        {
            ConnectToROS();
        }
    }

    public void ConnectToROS()
    {
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.Connect(rosIPAddress, rosPort);
        Debug.Log($"Connecting to ROS at {rosIPAddress}:{rosPort}");
    }

    public void DisconnectFromROS()
    {
        if (rosConnection != null)
        {
            rosConnection.Disconnect();
        }
    }
}
```

## Scene Setup for Robotics

### Basic Scene Configuration

Create a basic scene structure for robotics simulation:

1. Remove default objects (Main Camera, Directional Light)
2. Add a realistic ground plane
3. Configure lighting for the environment
4. Set up initial robot spawn location

```csharp
// Example: Basic robotics scene setup script
using UnityEngine;

public class RoboticsSceneSetup : MonoBehaviour
{
    [Header("Environment Setup")]
    public Transform robotSpawnPoint;
    public Material groundMaterial;
    public Light mainLight;

    [Header("Simulation Parameters")]
    public float simulationSpeed = 1.0f;

    void Start()
    {
        SetupEnvironment();
        ConfigurePhysics();
        InitializeSimulation();
    }

    void SetupEnvironment()
    {
        // Create or configure ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = Vector3.zero;
        ground.GetComponent<Renderer>().material = groundMaterial;

        // Configure lighting
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.intensity = 1.0f;
            mainLight.transform.rotation = Quaternion.Euler(50, -120, 0);
        }
    }

    void ConfigurePhysics()
    {
        // Set global physics properties
        Physics.gravity = new Vector3(0, -9.81f, 0);
    }

    void InitializeSimulation()
    {
        Time.timeScale = simulationSpeed;
        Debug.Log("Robotics simulation initialized");
    }
}
```

## Performance Optimization

### Graphics Optimization for Simulation

For efficient robotics simulation:

1. Use appropriate Level of Detail (LOD) systems
2. Implement occlusion culling for large environments
3. Use texture atlasing for similar materials
4. Optimize polygon counts for real-time performance

### Physics Optimization

1. Use appropriate collision shapes (simpler than visual meshes)
2. Configure rigidbody interpolation for smooth motion
3. Use appropriate mass values for realistic physics
4. Implement object pooling for frequently instantiated objects

## Testing and Validation

### Basic Setup Validation

Create a simple test to validate your Unity setup:

```csharp
// Example: Setup validation script
using UnityEngine;

public class SetupValidation : MonoBehaviour
{
    void Start()
    {
        ValidatePhysics();
        ValidateGraphics();
        ValidateInput();
    }

    void ValidatePhysics()
    {
        Debug.Log($"Physics engine: PhysX");
        Debug.Log($"Gravity: {Physics.gravity}");
        Debug.Log($"Fixed Timestep: {Time.fixedDeltaTime}");
    }

    void ValidateGraphics()
    {
        Debug.Log($"Graphics API: {SystemInfo.graphicsDeviceName}");
        Debug.Log($"Shader Level: {SystemInfo.graphicsShaderLevel}");
    }

    void ValidateInput()
    {
        Debug.Log($"Input system available: {Input.mousePresent}");
    }
}
```

## Week Summary

This section covered the essential setup procedures for configuring Unity as a robotics simulation platform. We explored system requirements, project structure, physics configuration, and the initial steps needed to prepare Unity for robotics applications. Proper setup is crucial for achieving realistic and performant robotics simulations.