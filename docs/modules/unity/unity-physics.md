---
sidebar_position: 3
---

# Unity Physics

## Introduction to Unity Physics for Robotics

Unity's physics engine, based on NVIDIA PhysX, provides realistic simulation of physical interactions that are essential for robotics applications. Understanding and properly configuring Unity's physics system is crucial for creating accurate robot simulations, from simple wheeled robots to complex manipulators and humanoid systems.

## Physics Engine Fundamentals

### Core Physics Components

Unity's physics system revolves around several key components:

**Rigidbody**: The primary component for physics simulation
- Controls an object's position and rotation through physics
- Responds to forces, gravity, and collisions
- Essential for any object that should move realistically

**Colliders**: Define the shape of an object for physics interactions
- Determine what the object collides with
- Don't need to match the visual mesh exactly
- Can be compound (multiple colliders per object)

**Physics Materials**: Control surface properties like friction and bounciness
- Define how objects interact when they collide
- Critical for realistic robot-environment interactions

### Rigidbody Configuration for Robotics

```csharp
// Example: Robot component with physics setup
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class RobotPhysics : MonoBehaviour
{
    [Header("Physics Configuration")]
    public float mass = 10f;
    public Vector3 centerOfMassOffset = Vector3.zero;
    public bool useGravity = true;
    public bool isKinematic = false;

    [Header("Movement Constraints")]
    public bool freezePositionX = false;
    public bool freezePositionY = false;
    public bool freezePositionZ = false;
    public bool freezeRotationX = false;
    public bool freezeRotationY = false;
    public bool freezeRotationZ = false;

    private Rigidbody rb;

    void Start()
    {
        SetupRigidbody();
    }

    void SetupRigidbody()
    {
        rb = GetComponent<Rigidbody>();

        // Configure basic properties
        rb.mass = mass;
        rb.useGravity = useGravity;
        rb.isKinematic = isKinematic;

        // Set center of mass if needed (for stability)
        if (centerOfMassOffset != Vector3.zero)
        {
            rb.centerOfMass = centerOfMassOffset;
        }

        // Apply constraints
        rb.constraints = RigidbodyConstraints.None;

        if (freezePositionX) rb.constraints |= RigidbodyConstraints.FreezePositionX;
        if (freezePositionY) rb.constraints |= RigidbodyConstraints.FreezePositionY;
        if (freezePositionZ) rb.constraints |= RigidbodyConstraints.FreezePositionZ;
        if (freezeRotationX) rb.constraints |= RigidbodyConstraints.FreezeRotationX;
        if (freezeRotationY) rb.constraints |= RigidbodyConstraints.FreezeRotationY;
        if (freezeRotationZ) rb.constraints |= RigidbodyConstraints.FreezeRotationZ;
    }

    // Apply forces to the robot
    public void ApplyForce(Vector3 force, ForceMode mode = ForceMode.Force)
    {
        if (rb != null && !isKinematic)
        {
            rb.AddForce(force, mode);
        }
    }

    // Apply torque to the robot
    public void ApplyTorque(Vector3 torque, ForceMode mode = ForceMode.Force)
    {
        if (rb != null && !isKinematic)
        {
            rb.AddTorque(torque, mode);
        }
    }
}
```

## Collision Detection and Response

### Collider Types and Usage

Different collider types serve different purposes in robotics simulation:

**Primitive Colliders** (Box, Sphere, Capsule):
- Fastest performance
- Good for simple shapes
- Ideal for basic robot bodies

**Mesh Collider**:
- Accurate collision based on mesh geometry
- Higher computational cost
- Use for complex static environments

**Terrain Collider**:
- Optimized for terrain interactions
- Essential for outdoor robot simulation

```csharp
// Example: Robot with multiple colliders for accurate physics
using UnityEngine;

public class RobotColliders : MonoBehaviour
{
    [Header("Collider Configuration")]
    public PhysicMaterial robotMaterial;
    public PhysicMaterial wheelMaterial;

    [Header("Wheel Colliders")]
    public Transform[] wheelTransforms;
    public float wheelRadius = 0.1f;
    public float wheelWidth = 0.05f;

    void Start()
    {
        SetupColliders();
    }

    void SetupColliders()
    {
        // Create main body collider
        SetupMainBodyCollider();

        // Create wheel colliders
        SetupWheelColliders();
    }

    void SetupMainBodyCollider()
    {
        // Use a box collider for the main body
        BoxCollider bodyCollider = gameObject.AddComponent<BoxCollider>();
        bodyCollider.center = Vector3.zero;
        bodyCollider.size = new Vector3(0.5f, 0.3f, 0.4f); // Adjust based on robot size
        bodyCollider.material = robotMaterial;
    }

    void SetupWheelColliders()
    {
        foreach (Transform wheel in wheelTransforms)
        {
            if (wheel != null)
            {
                // Add sphere collider for wheels
                SphereCollider wheelCollider = wheel.gameObject.AddComponent<SphereCollider>();
                wheelCollider.radius = wheelRadius;
                wheelCollider.center = Vector3.zero;
                wheelCollider.material = wheelMaterial;
            }
        }
    }
}
```

### Physics Materials for Robotics

Physics materials define surface properties that affect robot interactions:

```csharp
// Example: Creating physics materials for different robot components
using UnityEngine;

[CreateAssetMenu(fileName = "RobotPhysicsMaterials", menuName = "Robotics/Physics Materials")]
public class RobotPhysicsMaterials : ScriptableObject
{
    [Header("Wheel Materials")]
    public PhysicMaterial highFrictionWheels;  // For good traction
    public PhysicMaterial lowFrictionWheels;   // For swerve wheels

    [Header("Gripper Materials")]
    public PhysicMaterial gripperMaterial;     // For object manipulation
    public PhysicMaterial passiveGripper;      // For passive grippers

    [Header("Environment Materials")]
    public PhysicMaterial smoothFloor;         // For indoor environments
    public PhysicMaterial roughTerrain;        // For outdoor environments
    public PhysicMaterial rubberMat;           // For rubberized surfaces

    void OnValidate()
    {
        ConfigureMaterials();
    }

    void ConfigureMaterials()
    {
        if (highFrictionWheels != null)
        {
            highFrictionWheels.staticFriction = 0.8f;
            highFrictionWheels.dynamicFriction = 0.7f;
            highFrictionWheels.bounciness = 0.1f;
        }

        if (lowFrictionWheels != null)
        {
            lowFrictionWheels.staticFriction = 0.1f;
            lowFrictionWheels.dynamicFriction = 0.05f;
            lowFrictionWheels.bounciness = 0.0f;
        }

        if (gripperMaterial != null)
        {
            gripperMaterial.staticFriction = 0.9f;
            gripperMaterial.dynamicFriction = 0.8f;
            gripperMaterial.bounciness = 0.0f;
        }
    }
}
```

## Advanced Physics Concepts

### Joint Systems for Robot Articulation

Unity's joint components enable realistic robot articulation:

```csharp
// Example: Robot arm with joints
using UnityEngine;

public class RobotArmJoints : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ConfigurableJoint[] joints;
    public float[] jointLimits; // Limits for each joint in degrees

    [Header("Motor Configuration")]
    public bool[] useMotors;
    public float[] motorForces;
    public float[] targetPositions;

    void Start()
    {
        SetupJoints();
    }

    void SetupJoints()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                ConfigureJoint(joints[i], i);
            }
        }
    }

    void ConfigureJoint(ConfigurableJoint joint, int jointIndex)
    {
        // Configure joint limits
        if (jointIndex < jointLimits.Length)
        {
            SoftJointLimit limit = new SoftJointLimit();
            limit.limit = jointLimits[jointIndex];
            joint.lowAngularXLimit = limit;
            joint.highAngularXLimit = limit;
        }

        // Configure motor if needed
        if (jointIndex < useMotors.Length && useMotors[jointIndex])
        {
            JointDrive drive = new JointDrive();
            drive.mode = JointDriveMode.Position;
            drive.positionSpring = 1000f; // Stiffness
            drive.positionDamper = 100f;  // Damping
            drive.maximumForce = jointIndex < motorForces.Length ? motorForces[jointIndex] : 100f;

            joint.slerpDrive = drive;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
        }
    }

    public void SetJointTarget(int jointIndex, float targetAngle)
    {
        if (jointIndex >= 0 && jointIndex < joints.Length && joints[jointIndex] != null)
        {
            targetPositions[jointIndex] = targetAngle;
            // Update joint target position
            joints[jointIndex].targetRotation = Quaternion.Euler(targetAngle, 0, 0);
        }
    }
}
```

### Continuous Collision Detection

For fast-moving robots or precise collision detection:

```csharp
// Example: Continuous collision detection for fast-moving robots
using UnityEngine;

public class CCDController : MonoBehaviour
{
    [Header("CCD Configuration")]
    public bool useCCD = false;
    public float velocityThreshold = 2.0f;
    public Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        ConfigureCCD();
    }

    void ConfigureCCD()
    {
        if (rb != null)
        {
            rb.collisionDetectionMode = useCCD ?
                CollisionDetectionMode.ContinuousDynamic :
                CollisionDetectionMode.Discrete;
        }
    }

    void FixedUpdate()
    {
        if (rb != null && useCCD)
        {
            // Switch CCD mode based on velocity
            float speed = rb.velocity.magnitude;
            if (speed > velocityThreshold)
            {
                rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            }
            else
            {
                rb.collisionDetectionMode = CollisionDetectionMode.Discrete;
            }
        }
    }
}
```

## Physics Performance Optimization

### Efficient Physics Simulation

```csharp
// Example: Physics optimization manager
using UnityEngine;

public class PhysicsOptimizationManager : MonoBehaviour
{
    [Header("Performance Settings")]
    public int maxSubsteps = 8;
    public float sleepThreshold = 0.005f;
    public float defaultContactOffset = 0.01f;

    [Header("Simulation Quality")]
    public bool useInterpolation = true;
    public float interpolationDelay = 0.01f;

    void Start()
    {
        ConfigurePhysicsSettings();
    }

    void ConfigurePhysicsSettings()
    {
        // Set global physics parameters
        Physics.defaultContactOffset = defaultContactOffset;
        Physics.sleepThreshold = sleepThreshold;

        // Configure interpolation for smooth motion
        ConfigureInterpolation();
    }

    void ConfigureInterpolation()
    {
        // Apply interpolation settings to all rigidbodies in the scene
        Rigidbody[] allRigidbodies = FindObjectsOfType<Rigidbody>();

        foreach (Rigidbody rb in allRigidbodies)
        {
            rb.interpolation = useInterpolation ?
                RigidbodyInterpolation.Interpolate :
                RigidbodyInterpolation.None;
        }
    }

    // Dynamic performance adjustment based on simulation needs
    public void AdjustPhysicsQuality(float qualityFactor)
    {
        // Adjust based on quality factor (0.0 to 1.0)
        Physics.defaultContactOffset = defaultContactOffset * qualityFactor;
        Time.fixedDeltaTime = Mathf.Lerp(0.033f, 0.001f, qualityFactor);
    }
}
```

## Specialized Physics for Robotics

### Wheeled Vehicle Physics

```csharp
// Example: Wheeled robot physics
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class WheeledRobotPhysics : MonoBehaviour
{
    [Header("Wheel Configuration")]
    public WheelCollider[] wheelColliders;
    public Transform[] wheelMeshes; // Visual wheels
    public float maxMotorTorque = 100f;
    public float maxSteeringAngle = 30f;

    [Header("Vehicle Properties")]
    public float mass = 1000f;
    public float centerOfMassHeight = 0.5f;

    private Rigidbody rb;

    void Start()
    {
        SetupVehiclePhysics();
    }

    void SetupVehiclePhysics()
    {
        rb = GetComponent<Rigidbody>();
        rb.mass = mass;
        rb.centerOfMass = new Vector3(0, centerOfMassHeight, 0);

        foreach (WheelCollider wheel in wheelColliders)
        {
            ConfigureWheel(wheel);
        }
    }

    void ConfigureWheel(WheelCollider wheel)
    {
        wheel.mass = 10f;
        wheel.radius = 0.4f;
        wheel.suspensionDistance = 0.1f;

        JointSpring spring = new JointSpring();
        spring.spring = 20000f;
        spring.damper = 5000f;
        spring.targetPosition = 0.5f;
        wheel.suspensionSpring = spring;
    }

    public void SetMotorTorque(float torque)
    {
        foreach (WheelCollider wheel in wheelColliders)
        {
            if (IsDriveWheel(wheel))
            {
                wheel.motorTorque = torque;
            }
        }
    }

    public void SetSteering(float steering)
    {
        for (int i = 0; i < wheelColliders.Length; i++)
        {
            if (IsSteerWheel(wheelColliders[i]))
            {
                wheelColliders[i].steerAngle = steering * maxSteeringAngle;
            }
        }
    }

    bool IsDriveWheel(WheelCollider wheel)
    {
        // Determine if this wheel is a drive wheel
        // Implementation depends on your robot configuration
        return true;
    }

    bool IsSteerWheel(WheelCollider wheel)
    {
        // Determine if this wheel is a steer wheel
        // Implementation depends on your robot configuration
        return true;
    }

    void Update()
    {
        // Update wheel mesh positions to match wheel colliders
        UpdateWheelMeshes();
    }

    void UpdateWheelMeshes()
    {
        for (int i = 0; i < wheelColliders.Length; i++)
        {
            if (wheelMeshes[i] != null)
            {
                WheelCollider wheel = wheelColliders[i];
                Vector3 position;
                Quaternion rotation;
                wheel.GetWorldPose(out position, out rotation);

                wheelMeshes[i].position = position;
                wheelMeshes[i].rotation = rotation;
            }
        }
    }
}
```

## Physics Debugging and Validation

### Physics Debugging Tools

```csharp
// Example: Physics debugging visualization
using UnityEngine;

public class PhysicsDebugger : MonoBehaviour
{
    [Header("Debug Settings")]
    public bool showForces = true;
    public bool showColliders = true;
    public bool showVelocity = true;
    public Color forceColor = Color.red;
    public Color velocityColor = Color.blue;

    void OnDrawGizmos()
    {
        if (showForces)
        {
            DrawForceVectors();
        }

        if (showColliders)
        {
            DrawColliderBounds();
        }

        if (showVelocity)
        {
            DrawVelocityVectors();
        }
    }

    void DrawForceVectors()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // Draw current force vector
            Vector3 force = rb.velocity * rb.mass; // Approximate force
            Gizmos.color = forceColor;
            Gizmos.DrawRay(transform.position, force.normalized * 2f);
        }
    }

    void DrawColliderBounds()
    {
        Collider[] colliders = GetComponents<Collider>();
        foreach (Collider col in colliders)
        {
            Gizmos.color = Color.yellow;
            if (col is BoxCollider)
            {
                BoxCollider box = col as BoxCollider;
                Gizmos.matrix = Matrix4x4.TRS(box.bounds.center, transform.rotation, Vector3.one);
                Gizmos.DrawWireCube(Vector3.zero, box.size);
            }
            else if (col is SphereCollider)
            {
                SphereCollider sphere = col as SphereCollider;
                Gizmos.DrawWireSphere(sphere.bounds.center, sphere.radius);
            }
        }
    }

    void DrawVelocityVectors()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            Gizmos.color = velocityColor;
            Gizmos.DrawRay(transform.position, rb.velocity * 0.5f);
        }
    }
}
```

## Week Summary

This section covered Unity's physics engine in depth, focusing on its application to robotics simulation. We explored rigidbody configuration, collision detection, joint systems, performance optimization, and specialized physics concepts for different types of robots. Proper physics configuration is essential for creating realistic and stable robot simulations in Unity.