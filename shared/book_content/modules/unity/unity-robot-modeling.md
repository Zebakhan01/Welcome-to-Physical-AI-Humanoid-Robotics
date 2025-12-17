---
sidebar_position: 4
---

# Unity Robot Modeling

## Introduction to Robot Modeling in Unity

Creating realistic and functional robot models in Unity requires a combination of 3D modeling skills, physics understanding, and Unity-specific knowledge. Robot models in Unity serve dual purposes: visual representation for simulation and physical interaction through Unity's physics system. This section covers the complete process of creating, importing, and configuring robot models for simulation.

## Robot Model Structure and Hierarchy

### Proper Robot Model Hierarchy

A well-structured robot model hierarchy is essential for proper physics simulation and control:

```
Robot_Base (Rigidbody, Colliders)
├── Chassis (Visual + Collision)
├── Joint_1 (ConfigurableJoint)
│   └── Link_1 (Rigidbody, Colliders)
│       ├── Joint_2 (ConfigurableJoint)
│       │   └── Link_2 (Rigidbody, Colliders)
│       │       └── EndEffector (Rigidbody, Colliders)
├── Wheel_Left (Rigidbody, WheelCollider)
├── Wheel_Right (Rigidbody, WheelCollider)
└── Sensors (Various sensor components)
```

### Example: Hierarchical Robot Model Setup

```csharp
// Robot hierarchy manager
using UnityEngine;

public class RobotHierarchyManager : MonoBehaviour
{
    [Header("Robot Components")]
    public Transform chassis;
    public Transform[] links;
    public Transform[] wheels;
    public Transform[] sensors;

    [Header("Physics Configuration")]
    public PhysicMaterial defaultMaterial;
    public float robotMass = 10f;

    void Start()
    {
        SetupRobotHierarchy();
    }

    void SetupRobotHierarchy()
    {
        // Configure chassis as main rigidbody
        SetupChassis();

        // Configure links with individual rigidbodies
        SetupLinks();

        // Configure wheels with wheel colliders
        SetupWheels();

        // Configure sensors
        SetupSensors();
    }

    void SetupChassis()
    {
        if (chassis != null)
        {
            Rigidbody chassisRb = chassis.gameObject.AddComponent<Rigidbody>();
            chassisRb.mass = robotMass;
            chassisRb.useGravity = true;

            // Add collision for chassis
            if (chassis.GetComponent<Collider>() == null)
            {
                chassis.gameObject.AddComponent<BoxCollider>();
            }
        }
    }

    void SetupLinks()
    {
        foreach (Transform link in links)
        {
            if (link != null)
            {
                Rigidbody linkRb = link.gameObject.AddComponent<Rigidbody>();
                linkRb.mass = 1f; // Lighter than chassis
                linkRb.useGravity = true;

                // Add collision for link
                if (link.GetComponent<Collider>() == null)
                {
                    link.gameObject.AddComponent<BoxCollider>();
                }
            }
        }
    }

    void SetupWheels()
    {
        foreach (Transform wheel in wheels)
        {
            if (wheel != null)
            {
                // For wheeled robots, we might use WheelCollider
                // or regular colliders depending on simulation needs
                if (wheel.GetComponent<Collider>() == null)
                {
                    SphereCollider sphereCol = wheel.gameObject.AddComponent<SphereCollider>();
                    sphereCol.radius = 0.1f;
                    sphereCol.material = defaultMaterial;
                }
            }
        }
    }

    void SetupSensors()
    {
        foreach (Transform sensor in sensors)
        {
            if (sensor != null)
            {
                // Sensors typically don't need physics
                // but may need specific components
                ConfigureSensor(sensor);
            }
        }
    }

    void ConfigureSensor(Transform sensor)
    {
        // Add specific sensor components based on sensor type
        // This could be camera, lidar, or other sensor scripts
    }
}
```

## Importing Robot Models

### Import Settings for Robotics

When importing 3D models for robotics simulation, consider these Unity import settings:

```csharp
// Example: Robot model import configuration
using UnityEngine;

[CreateAssetMenu(fileName = "RobotImportConfig", menuName = "Robotics/Import Configuration")]
public class RobotImportConfig : ScriptableObject
{
    [Header("Model Import Settings")]
    public bool useRiggedImport = false; // Disable for non-animated robots
    public bool importBlendShapes = false; // Usually not needed for robots
    public bool importCameras = false;
    public bool importLights = false;
    public bool importMaterials = true;

    [Header("Animation Settings")]
    public bool importAnimations = false; // Use Unity physics instead
    public bool importConstraints = false;
    public bool importVisibility = false;

    [Header("Rig Settings")]
    public bool useHumanoid = false; // Only for humanoid robots
    public bool hasTranslationDoF = false;

    [Header("Mesh Settings")]
    public bool useFileUnits = true;
    public bool optimizeMeshPolygons = true;
    public bool optimizeMeshVertices = true;
    public bool importBlendShapes = false;
    public bool addColliders = false; // We'll add manually
    public bool useSRGBMaterialColor = true;

    [Header("Normals and Tangents")]
    public ModelImporterNormals normalImportMode = ModelImporterNormals.Import;
    public ModelImporterTangents tangentImportMode = ModelImporterTangents.CalculateMikk;
}
```

### Robot Model Optimization

```csharp
// Robot model optimization script
using UnityEngine;

public class RobotModelOptimizer : MonoBehaviour
{
    [Header("Optimization Settings")]
    public bool optimizeForSimulation = true;
    public bool removeUnusedMeshes = true;
    public bool combineMeshes = true;
    public float simplificationThreshold = 0.01f;

    [Header("LOD Configuration")]
    public bool useLOD = false;
    public int lodCount = 3;
    public float[] lodDistances = { 10f, 30f, 60f };

    void Start()
    {
        OptimizeRobotModel();
    }

    void OptimizeRobotModel()
    {
        if (optimizeForSimulation)
        {
            OptimizeForPhysics();
        }

        if (removeUnusedMeshes)
        {
            RemoveUnusedMeshes();
        }

        if (combineMeshes)
        {
            CombineMeshes();
        }

        if (useLOD)
        {
            SetupLODSystem();
        }
    }

    void OptimizeForPhysics()
    {
        // Reduce polygon count for physics colliders
        // Use simpler shapes than visual meshes
        Collider[] colliders = GetComponentsInChildren<Collider>();

        foreach (Collider col in colliders)
        {
            // Simplify collision geometry where possible
            SimplifyCollider(col);
        }
    }

    void SimplifyCollider(Collider collider)
    {
        // Example: Convert mesh colliders to simpler primitives where possible
        if (collider is MeshCollider)
        {
            MeshCollider meshCol = collider as MeshCollider;
            if (IsSimpleShape(meshCol.sharedMesh))
            {
                // Replace with primitive collider
                DestroyImmediate(meshCol);
                AddSimpleCollider(collider.transform);
            }
        }
    }

    bool IsSimpleShape(Mesh mesh)
    {
        // Simplistic check for simple shapes
        // In practice, this would be more sophisticated
        return mesh.triangles.Length < 100; // Very simple mesh
    }

    void AddSimpleCollider(Transform transform)
    {
        // Add appropriate primitive collider
        // This is a simplified example
        BoxCollider boxCol = transform.gameObject.AddComponent<BoxCollider>();
        // Configure based on mesh bounds
    }

    void RemoveUnusedMeshes()
    {
        // Remove meshes that aren't needed for simulation
        MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>();

        foreach (MeshFilter filter in meshFilters)
        {
            if (ShouldRemoveMesh(filter))
            {
                DestroyImmediate(filter);
            }
        }
    }

    bool ShouldRemoveMesh(MeshFilter filter)
    {
        // Determine if mesh should be removed
        // This could be based on tags, names, or other criteria
        return false; // Placeholder
    }

    void CombineMeshes()
    {
        // Combine static meshes to improve performance
        // Only for non-moving parts of the robot
        StaticBatchingUtility.Combine(gameObject);
    }

    void SetupLODSystem()
    {
        // Create LOD group for the robot
        LODGroup lodGroup = gameObject.AddComponent<LODGroup>();

        LOD[] lods = new LOD[lodCount];
        for (int i = 0; i < lodCount; i++)
        {
            Renderer[] renderers = GetLODRenderers(i);
            lods[i] = new LOD(1.0f - (i * (1.0f / lodCount)), renderers);
        }

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    Renderer[] GetLODRenderers(int lodLevel)
    {
        // Return appropriate renderers for each LOD level
        // Implementation depends on your specific robot structure
        return new Renderer[0]; // Placeholder
    }
}
```

## Robot Joint Configuration

### Configurable Joint Setup for Robot Arms

```csharp
// Robot arm joint configuration
using UnityEngine;

public class RobotArmJointConfig : MonoBehaviour
{
    [System.Serializable]
    public class JointConfig
    {
        public Transform jointTransform;
        public JointType jointType;
        public float minLimit = -90f;
        public float maxLimit = 90f;
        public float spring = 1000f;
        public float damper = 100f;
        public float maxForce = 100f;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    [Header("Joint Configuration")]
    public JointConfig[] jointConfigs;

    [Header("Control Settings")]
    public bool useMotors = true;
    public float motorForce = 100f;
    public float targetPosition = 0f;

    void Start()
    {
        ConfigureJoints();
    }

    void ConfigureJoints()
    {
        foreach (JointConfig config in jointConfigs)
        {
            if (config.jointTransform != null)
            {
                ConfigurableJoint joint = ConfigureJoint(config);
                if (useMotors)
                {
                    SetupJointMotor(joint, config);
                }
            }
        }
    }

    ConfigurableJoint ConfigureJoint(JointConfig config)
    {
        ConfigurableJoint joint = config.jointTransform.gameObject.AddComponent<ConfigurableJoint>();

        // Set joint configuration based on type
        switch (config.jointType)
        {
            case JointType.Revolute:
                ConfigureRevoluteJoint(joint, config);
                break;
            case JointType.Prismatic:
                ConfigurePrismaticJoint(joint, config);
                break;
            case JointType.Fixed:
                ConfigureFixedJoint(joint, config);
                break;
        }

        return joint;
    }

    void ConfigureRevoluteJoint(ConfigurableJoint joint, JointConfig config)
    {
        // Set angular limits
        SoftJointLimit lowLimit = new SoftJointLimit();
        lowLimit.limit = config.minLimit * Mathf.Deg2Rad;
        joint.lowAngularXLimit = lowLimit;

        SoftJointLimit highLimit = new SoftJointLimit();
        highLimit.limit = config.maxLimit * Mathf.Deg2Rad;
        joint.highAngularXLimit = highLimit;

        // Set drive for motor control
        JointDrive drive = new JointDrive();
        drive.positionSpring = config.spring;
        drive.positionDamper = config.damper;
        drive.maximumForce = config.maxForce;
        joint.slerpDrive = drive;

        // Set rotation drive mode
        joint.rotationDriveMode = RotationDriveMode.XYAndZ;
    }

    void ConfigurePrismaticJoint(ConfigurableJoint joint, JointConfig config)
    {
        // For prismatic joints, we need to limit linear motion
        joint.linearLimit = new SoftJointLimit { limit = config.maxLimit };
        joint.lowAngularXLimit = new SoftJointLimit { limit = 0 };
        joint.highAngularXLimit = new SoftJointLimit { limit = 0 };
        joint.angularXDrive = new JointDrive();
    }

    void ConfigureFixedJoint(ConfigurableJoint joint, JointConfig config)
    {
        // Set all limits to 0 for fixed joint
        joint.linearLimit = new SoftJointLimit { limit = 0 };
        joint.lowAngularXLimit = new SoftJointLimit { limit = 0 };
        joint.highAngularXLimit = new SoftJointLimit { limit = 0 };
        joint.angularXDrive = new JointDrive();
    }

    void SetupJointMotor(ConfigurableJoint joint, JointConfig config)
    {
        JointDrive drive = joint.slerpDrive;
        drive.positionSpring = config.spring;
        drive.positionDamper = config.damper;
        drive.maximumForce = config.maxForce;
        joint.slerpDrive = drive;
    }

    public void SetJointTarget(int jointIndex, float targetAngle)
    {
        if (jointIndex >= 0 && jointIndex < jointConfigs.Length)
        {
            ConfigurableJoint joint = jointConfigs[jointIndex].jointTransform.GetComponent<ConfigurableJoint>();
            if (joint != null)
            {
                joint.targetRotation = Quaternion.Euler(targetAngle, 0, 0);
            }
        }
    }
}
```

## Robot Sensor Integration

### Sensor Mounting and Configuration

```csharp
// Robot sensor mounting system
using UnityEngine;

public class RobotSensorMounting : MonoBehaviour
{
    [System.Serializable]
    public class SensorMount
    {
        public Transform mountPoint;
        public SensorType sensorType;
        public string sensorName;
        public Vector3 offset = Vector3.zero;
        public Vector3 rotation = Vector3.zero;
    }

    public enum SensorType
    {
        Camera,
        LIDAR,
        IMU,
        ForceTorque,
        GPS,
        Compass
    }

    [Header("Sensor Mounting")]
    public SensorMount[] sensorMounts;

    [Header("Sensor Configuration")]
    public float sensorUpdateRate = 30f;

    void Start()
    {
        MountSensors();
    }

    void MountSensors()
    {
        foreach (SensorMount mount in sensorMounts)
        {
            if (mount.mountPoint != null)
            {
                GameObject sensorGO = CreateSensor(mount);
                ConfigureSensorMount(mount, sensorGO);
            }
        }
    }

    GameObject CreateSensor(SensorMount mount)
    {
        GameObject sensorGO = new GameObject(mount.sensorName);

        // Position and orient the sensor
        sensorGO.transform.SetParent(mount.mountPoint);
        sensorGO.transform.localPosition = mount.offset;
        sensorGO.transform.localRotation = Quaternion.Euler(mount.rotation);

        // Add appropriate sensor component based on type
        switch (mount.sensorType)
        {
            case SensorType.Camera:
                AddCameraSensor(sensorGO);
                break;
            case SensorType.LIDAR:
                AddLIDARSensor(sensorGO);
                break;
            case SensorType.IMU:
                AddIMUSensor(sensorGO);
                break;
            case SensorType.ForceTorque:
                AddForceTorqueSensor(sensorGO);
                break;
            case SensorType.GPS:
                AddGPSSensor(sensorGO);
                break;
            case SensorType.Compass:
                AddCompassSensor(sensorGO);
                break;
        }

        return sensorGO;
    }

    void AddCameraSensor(GameObject sensorGO)
    {
        Camera cam = sensorGO.AddComponent<Camera>();
        cam.fieldOfView = 60f;
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = 100f;
        cam.depth = 0;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
    }

    void AddLIDARSensor(GameObject sensorGO)
    {
        // Custom LIDAR implementation or use existing asset
        // This would typically involve raycasting in multiple directions
        LIDARSensor lidar = sensorGO.AddComponent<LIDARSensor>();
        lidar.numberOfRays = 360;
        lidar.range = 10f;
        lidar.updateRate = sensorUpdateRate;
    }

    void AddIMUSensor(GameObject sensorGO)
    {
        IMUSensor imu = sensorGO.AddComponent<IMUSensor>();
        imu.updateRate = sensorUpdateRate;
    }

    void ConfigureSensorMount(SensorMount mount, GameObject sensorGO)
    {
        // Apply any additional configuration based on mount properties
        // This could include sensor-specific parameters
    }
}

// Example sensor implementations
public class LIDARSensor : MonoBehaviour
{
    public int numberOfRays = 360;
    public float range = 10f;
    public float updateRate = 10f;
    public LayerMask detectionMask = -1;

    private float updateInterval;
    private float lastUpdate;

    void Start()
    {
        updateInterval = 1f / updateRate;
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            ScanEnvironment();
            lastUpdate = Time.time;
        }
    }

    void ScanEnvironment()
    {
        float angleStep = 360f / numberOfRays;
        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            Ray ray = new Ray(transform.position, transform.TransformDirection(direction));

            if (Physics.Raycast(ray, out RaycastHit hit, range, detectionMask))
            {
                // Process distance reading
                float distance = hit.distance;
                // Publish to ROS or store for processing
            }
        }
    }
}

public class IMUSensor : MonoBehaviour
{
    public float updateRate = 100f;
    public bool includeAccelerometer = true;
    public bool includeGyroscope = true;
    public bool includeMagnetometer = true;

    private float updateInterval;
    private float lastUpdate;

    void Start()
    {
        updateInterval = 1f / updateRate;
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            ReadIMU();
            lastUpdate = Time.time;
        }
    }

    void ReadIMU()
    {
        if (includeAccelerometer)
        {
            Vector3 acceleration = GetAcceleration();
            // Process acceleration data
        }

        if (includeGyroscope)
        {
            Vector3 angularVelocity = GetAngularVelocity();
            // Process angular velocity data
        }

        if (includeMagnetometer)
        {
            Vector3 magneticField = GetMagneticField();
            // Process magnetic field data
        }
    }

    Vector3 GetAcceleration()
    {
        // Get acceleration from physics or simulate
        Rigidbody rb = GetComponentInParent<Rigidbody>();
        if (rb != null)
        {
            return rb.velocity; // Simplified - would need proper acceleration calculation
        }
        return Vector3.zero;
    }

    Vector3 GetAngularVelocity()
    {
        Rigidbody rb = GetComponentInParent<Rigidbody>();
        if (rb != null)
        {
            return rb.angularVelocity;
        }
        return Vector3.zero;
    }

    Vector3 GetMagneticField()
    {
        // Simulate magnetic field reading
        return Vector3.zero; // Placeholder
    }
}
```

## Robot Materials and Textures

### Robot-Specific Materials

```csharp
// Robot material configuration
using UnityEngine;

[CreateAssetMenu(fileName = "RobotMaterials", menuName = "Robotics/Material Configuration")]
public class RobotMaterials : ScriptableObject
{
    [Header("Visual Materials")]
    public Material chassisMaterial;
    public Material wheelMaterial;
    public Material sensorMaterial;
    public Material jointMaterial;

    [Header("Physics Materials")]
    public PhysicMaterial wheelPhysicsMaterial;
    public PhysicMaterial gripperPhysicsMaterial;
    public PhysicMaterial bodyPhysicsMaterial;

    [Header("Material Properties")]
    public Color chassisColor = Color.gray;
    public Color wheelColor = Color.black;
    public Color sensorColor = Color.blue;

    public void ApplyMaterials(GameObject robot)
    {
        ApplyVisualMaterials(robot);
        ApplyPhysicsMaterials(robot);
    }

    void ApplyVisualMaterials(GameObject robot)
    {
        Renderer[] renderers = robot.GetComponentsInChildren<Renderer>();

        foreach (Renderer renderer in renderers)
        {
            if (renderer.name.Contains("chassis", System.StringComparison.OrdinalIgnoreCase))
            {
                renderer.material = chassisMaterial ?? CreateDefaultMaterial(chassisColor);
            }
            else if (renderer.name.Contains("wheel", System.StringComparison.OrdinalIgnoreCase))
            {
                renderer.material = wheelMaterial ?? CreateDefaultMaterial(wheelColor);
            }
            else if (renderer.name.Contains("sensor", System.StringComparison.OrdinalIgnoreCase))
            {
                renderer.material = sensorMaterial ?? CreateDefaultMaterial(sensorColor);
            }
            else if (renderer.name.Contains("joint", System.StringComparison.OrdinalIgnoreCase))
            {
                renderer.material = jointMaterial ?? CreateDefaultMaterial(Color.red);
            }
        }
    }

    void ApplyPhysicsMaterials(GameObject robot)
    {
        Collider[] colliders = robot.GetComponentsInChildren<Collider>();

        foreach (Collider collider in colliders)
        {
            if (collider.name.Contains("wheel", System.StringComparison.OrdinalIgnoreCase))
            {
                collider.material = wheelPhysicsMaterial;
            }
            else if (collider.name.Contains("gripper", System.StringComparison.OrdinalIgnoreCase))
            {
                collider.material = gripperPhysicsMaterial;
            }
            else
            {
                collider.material = bodyPhysicsMaterial;
            }
        }
    }

    Material CreateDefaultMaterial(Color color)
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        return mat;
    }
}
```

## Week Summary

This section covered the complete process of creating, importing, and configuring robot models in Unity. We explored proper hierarchy setup, import optimization, joint configuration, sensor integration, and material application. Creating well-structured robot models is fundamental to achieving realistic and functional robotics simulations in Unity.