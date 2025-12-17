---
sidebar_position: 5
---

# Unity-ROS Bridge

## Introduction to Unity-ROS Integration

The Unity-ROS bridge enables bidirectional communication between Unity simulation environments and ROS-based robotic systems. This integration allows Unity to serve as a high-fidelity simulation platform while leveraging ROS's extensive ecosystem of tools, algorithms, and packages. The bridge facilitates the transfer of sensor data, control commands, and state information between Unity and ROS.

## ROS TCP Connector Setup

### Installing ROS TCP Connector

The Unity Robotics package provides the core functionality for ROS integration:

1. Open Unity Package Manager (Window → Package Manager)
2. Click the "+" button and select "Add package from git URL"
3. Enter: `com.unity.robotics.ros-tcp-connector`
4. Install the package

### Basic ROS Connection

```csharp
// Example: Basic ROS connection setup
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class BasicROSConnection : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public bool autoConnect = true;

    private ROSConnection ros;

    void Start()
    {
        if (autoConnect)
        {
            ConnectToROS();
        }
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Connect(rosIPAddress, rosPort);
        Debug.Log($"Connecting to ROS at {rosIPAddress}:{rosPort}");
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Disconnect();
        }
    }
}
```

## Message Publishing and Subscribing

### Publishing Custom Messages

```csharp
// Example: Publishing robot joint states
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnityEngine;

public class JointStatePublisher : MonoBehaviour
{
    [Header("Joint State Settings")]
    public string topicName = "/joint_states";
    public float publishRate = 30f; // Hz
    public Transform[] jointTransforms;
    public string[] jointNames;

    private ROSConnection ros;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;
        lastPublishTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishJointStates();
            lastPublishTime = Time.time;
        }
    }

    void PublishJointStates()
    {
        var jointState = new JointStateMsg
        {
            name = jointNames,
            position = new double[jointTransforms.Length],
            velocity = new double[jointTransforms.Length],
            effort = new double[jointTransforms.Length]
        };

        // Get current joint positions (simplified - would need proper joint angles)
        for (int i = 0; i < jointTransforms.Length; i++)
        {
            if (jointTransforms[i] != null)
            {
                jointState.position[i] = jointTransforms[i].localEulerAngles.y * Mathf.Deg2Rad; // Example for revolute joints
                jointState.velocity[i] = 0; // Would calculate from previous positions
                jointState.effort[i] = 0;   // Would come from joint forces
            }
        }

        jointState.header = new Messages.Standard.HeaderMsg
        {
            stamp = new Messages.Standard.TimeMsg
            {
                sec = (int)Time.time,
                nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9)
            },
            frame_id = "base_link"
        };

        ros.Publish(topicName, jointState);
    }
}
```

### Subscribing to ROS Topics

```csharp
// Example: Subscribing to velocity commands
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using UnityEngine;

public class VelocityCommandSubscriber : MonoBehaviour
{
    [Header("Velocity Command Settings")]
    public string topicName = "/cmd_vel";
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    private ROSConnection ros;
    private Rigidbody rb;
    private Vector3 linearVelocity;
    private Vector3 angularVelocity;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponent<Rigidbody>();

        // Subscribe to the velocity command topic
        ros.Subscribe<TwistMsg>(topicName, ReceiveVelocityCommand);
    }

    void ReceiveVelocityCommand(TwistMsg velocityCmd)
    {
        // Convert ROS Twist message to Unity vectors
        linearVelocity = new Vector3(
            (float)velocityCmd.linear.x,
            (float)velocityCmd.linear.y,
            (float)velocityCmd.linear.z
        );

        angularVelocity = new Vector3(
            (float)velocityCmd.angular.x,
            (float)velocityCmd.angular.y,
            (float)velocityCmd.angular.z
        );

        // Apply velocity limits
        linearVelocity = Vector3.ClampMagnitude(linearVelocity, maxLinearVelocity);
        angularVelocity = Vector3.ClampMagnitude(angularVelocity, maxAngularVelocity);
    }

    void FixedUpdate()
    {
        if (rb != null)
        {
            // Apply linear velocity
            rb.velocity = linearVelocity;

            // Apply angular velocity
            rb.angularVelocity = angularVelocity;
        }
    }
}
```

## Advanced ROS Integration Patterns

### Custom Message Types

```csharp
// Example: Creating and using custom ROS messages
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

// Custom message class (would typically be auto-generated)
[System.Serializable]
public class RobotStatusMsg
{
    public string robot_name;
    public float battery_level;
    public bool is_moving;
    public float[] joint_positions;
}

public class RobotStatusPublisher : MonoBehaviour
{
    [Header("Robot Status Settings")]
    public string topicName = "/robot_status";
    public float publishRate = 1f; // Hz
    public string robotName = "UnityRobot";

    private ROSConnection ros;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;
        lastPublishTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishRobotStatus();
            lastPublishTime = Time.time;
        }
    }

    void PublishRobotStatus()
    {
        var status = new RobotStatusMsg
        {
            robot_name = robotName,
            battery_level = GetBatteryLevel(),
            is_moving = IsRobotMoving(),
            joint_positions = GetJointPositions()
        };

        // Publish using a custom publisher method
        ros.Publish(topicName, status);
    }

    float GetBatteryLevel()
    {
        // Simulate battery level (in real implementation, this would be more sophisticated)
        return 100f - (Time.time * 0.01f); // Decreasing over time
    }

    bool IsRobotMoving()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            return rb.velocity.magnitude > 0.01f;
        }
        return false;
    }

    float[] GetJointPositions()
    {
        // Get positions of all joints in the robot
        ConfigurableJoint[] joints = GetComponentsInChildren<ConfigurableJoint>();
        float[] positions = new float[joints.Length];

        for (int i = 0; i < joints.Length; i++)
        {
            // This is a simplified example - actual joint position extraction
            // would depend on joint type and configuration
            positions[i] = joints[i].connectedAnchor.magnitude;
        }

        return positions;
    }
}
```

### Service Calls

```csharp
// Example: Making ROS service calls
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using UnityEngine;

public class ROSServiceCaller : MonoBehaviour
{
    [Header("Service Settings")]
    public string serviceName = "/set_parameters";
    public float timeout = 5.0f;

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    public void CallService()
    {
        var request = new SetParametersRequestMsg();
        // Configure request parameters

        ros.CallService<SetParametersRequestMsg, SetParametersResponseMsg>(
            serviceName,
            request,
            OnServiceResponse,
            OnServiceError,
            timeout
        );
    }

    void OnServiceResponse(SetParametersResponseMsg response)
    {
        Debug.Log("Service call successful: " + response.successful);
        // Handle successful response
    }

    void OnServiceError(string error)
    {
        Debug.LogError("Service call failed: " + error);
        // Handle service call error
    }
}
```

## Sensor Data Integration

### Camera Data Publishing

```csharp
// Example: Publishing camera images to ROS
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnityEngine;

public class CameraPublisher : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera cameraComponent;
    public string topicName = "/camera/image_raw";
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float publishRate = 30f;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        SetupCamera();
        publishInterval = 1f / publishRate;
        lastPublishTime = 0f;
    }

    void SetupCamera()
    {
        if (cameraComponent == null)
            cameraComponent = GetComponent<Camera>();

        // Create render texture for the camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cameraComponent.targetTexture = renderTexture;

        // Create texture2D for reading pixels
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishCameraImage();
            lastPublishTime = Time.time;
        }
    }

    void PublishCameraImage()
    {
        // Set the active render texture and read pixels
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS image message
        var imageMsg = new ImageMsg
        {
            header = new Messages.Standard.HeaderMsg
            {
                stamp = new Messages.Standard.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9)
                },
                frame_id = "camera_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel (RGB)
            data = texture2D.GetRawTextureData<byte>()
        };

        ros.Publish(topicName, imageMsg);
    }
}
```

### LIDAR Data Publishing

```csharp
// Example: Publishing LIDAR data to ROS
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnityEngine;

public class LIDARPublisher : MonoBehaviour
{
    [Header("LIDAR Settings")]
    public string topicName = "/scan";
    public int numberOfRays = 360;
    public float range = 10f;
    public float publishRate = 10f;
    public LayerMask detectionMask = -1;

    private ROSConnection ros;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;
        lastPublishTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishLIDARData();
            lastPublishTime = Time.time;
        }
    }

    void PublishLIDARData()
    {
        float[] ranges = new float[numberOfRays];
        float angleStep = 360f / numberOfRays;

        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            Ray ray = new Ray(transform.position, transform.TransformDirection(direction));

            if (Physics.Raycast(ray, out RaycastHit hit, range, detectionMask))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = range; // No obstacle detected within range
            }
        }

        var laserScan = new LaserScanMsg
        {
            header = new Messages.Standard.HeaderMsg
            {
                stamp = new Messages.Standard.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time - Mathf.Floor(Time.time)) * 1e9)
                },
                frame_id = "lidar_frame"
            },
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = (2 * Mathf.PI) / numberOfRays,
            time_increment = 0,
            scan_time = 1f / publishRate,
            range_min = 0.1f,
            range_max = range,
            ranges = ranges,
            intensities = new float[numberOfRays] // All zeros for now
        };

        ros.Publish(topicName, laserScan);
    }
}
```

## Performance Optimization

### Efficient Message Handling

```csharp
// Example: Optimized message handling system
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class OptimizedMessageHandler : MonoBehaviour
{
    [Header("Optimization Settings")]
    public float messageBufferSize = 10f; // seconds
    public bool enableRateLimiting = true;
    public float maxPublishRate = 30f;

    private ROSConnection ros;
    private System.Collections.Generic.Queue<System.Action> messageQueue;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        messageQueue = new System.Collections.Generic.Queue<System.Action>();
        lastPublishTime = 0f;
    }

    void Update()
    {
        ProcessMessageQueue();
    }

    void ProcessMessageQueue()
    {
        // Process queued messages
        while (messageQueue.Count > 0)
        {
            var messageAction = messageQueue.Dequeue();
            messageAction.Invoke();
        }
    }

    public void QueueMessage<T>(string topic, T message)
    {
        if (enableRateLimiting && Time.time - lastPublishTime < 1f / maxPublishRate)
        {
            return; // Rate limit exceeded
        }

        System.Action messageAction = () =>
        {
            ros.Publish(topic, message);
            lastPublishTime = Time.time;
        };

        messageQueue.Enqueue(messageAction);
    }

    public void SubscribeWithFilter<T>(string topic, System.Action<T> callback, float maxRate = 30f)
    {
        float minInterval = 1f / maxRate;
        float lastReceiveTime = 0f;

        ros.Subscribe<T>(topic, (msg) =>
        {
            if (Time.time - lastReceiveTime >= minInterval)
            {
                callback(msg);
                lastReceiveTime = Time.time;
            }
        });
    }
}
```

## Error Handling and Connection Management

### Robust Connection Management

```csharp
// Example: Robust ROS connection manager
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class RobustROSConnection : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public float connectionRetryDelay = 5f;
    public int maxConnectionRetries = 5;

    [Header("Connection Status")]
    public bool isConnected = false;
    public int connectionRetryCount = 0;

    private ROSConnection ros;
    private float lastConnectionAttempt;
    private bool connectionAttempted;

    void Start()
    {
        InitializeConnection();
    }

    void InitializeConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.OnConnected += OnROSConnected;
        ros.OnDisconnected += OnROSDisconnected;
        ros.OnConnectionFailed += OnROSConnectionFailed;

        AttemptConnection();
    }

    void AttemptConnection()
    {
        if (connectionRetryCount >= maxConnectionRetries)
        {
            Debug.LogError("Max connection retries reached. Please check ROS connection.");
            return;
        }

        if (Time.time - lastConnectionAttempt >= connectionRetryDelay)
        {
            Debug.Log($"Attempting to connect to ROS ({connectionRetryCount + 1}/{maxConnectionRetries})");
            ros.Connect(rosIPAddress, rosPort);
            lastConnectionAttempt = Time.time;
            connectionAttempted = true;
        }
    }

    void OnROSConnected()
    {
        isConnected = true;
        connectionRetryCount = 0;
        Debug.Log("Successfully connected to ROS");
        OnConnectionEstablished();
    }

    void OnROSDisconnected()
    {
        isConnected = false;
        Debug.Log("Disconnected from ROS");
        OnConnectionLost();
    }

    void OnROSConnectionFailed()
    {
        isConnected = false;
        connectionRetryCount++;
        Debug.LogWarning($"ROS connection failed. Attempt {connectionRetryCount}/{maxConnectionRetries}");

        // Schedule next connection attempt
        Invoke("AttemptConnection", connectionRetryDelay);
    }

    void OnConnectionEstablished()
    {
        // Override in derived classes to handle successful connection
        Debug.Log("Custom connection established logic");
    }

    void OnConnectionLost()
    {
        // Override in derived classes to handle connection loss
        Debug.Log("Custom connection lost logic");

        // Attempt to reconnect
        Invoke("AttemptConnection", connectionRetryDelay);
    }

    void Update()
    {
        if (!isConnected && !connectionAttempted)
        {
            AttemptConnection();
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.OnConnected -= OnROSConnected;
            ros.OnDisconnected -= OnROSDisconnected;
            ros.OnConnectionFailed -= OnROSConnectionFailed;
            ros.Disconnect();
        }
    }
}
```

## Best Practices for Unity-ROS Integration

### Architecture Patterns

1. **Separation of Concerns**: Keep ROS communication separate from Unity game logic
2. **Asynchronous Operations**: Use Unity's async patterns for ROS operations
3. **Message Buffers**: Implement buffering for high-frequency messages
4. **Error Recovery**: Build robust error handling and reconnection logic

### Performance Considerations

- Use appropriate publish rates for different sensor types
- Implement message filtering and compression where possible
- Consider using Unity's Job System for message processing
- Profile and optimize for your specific use case

## Week Summary

This section covered the Unity-ROS bridge integration, including connection setup, message publishing/subscribing, sensor data integration, and performance optimization. The Unity-ROS bridge enables powerful simulation capabilities while leveraging ROS's extensive robotics ecosystem, making it an essential tool for robotics development and testing.