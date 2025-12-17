"""
Comprehensive integration test for NVIDIA Isaac module with the main application
"""
import asyncio
import json
from fastapi.testclient import TestClient

# Import the main app
from backend.main import app

def test_isaac_endpoints():
    """Test all Isaac module endpoints through the main application"""
    client = TestClient(app)

    print("Testing Isaac module integration with main application...")

    # Test Isaac health check
    print("\n1. Testing Isaac health check...")
    response = client.get("/api/isaac/health")
    print(f"Health check status: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Health data: {health_data}")
    else:
        print(f"Health check failed: {response.text}")

    # Test Isaac app launch
    print("\n2. Testing Isaac app launch...")
    app_config = {
        "app_name": "test_app",
        "enable_gui": False,
        "headless": True
    }
    response = client.post("/api/isaac/apps/launch", json=app_config)
    print(f"App launch status: {response.status_code}")
    if response.status_code == 200:
        app_data = response.json()
        print(f"App data: {app_data}")
    else:
        print(f"App launch failed: {response.text}")

    # Test Isaac Sim environment creation
    print("\n3. Testing Isaac Sim environment creation...")
    env_config = {
        "name": "test_environment",
        "gravity": [0.0, 0.0, -9.81],
        "physics_dt": 0.016667
    }
    response = client.post("/api/isaac/sim/environments", json=env_config)
    print(f"Environment creation status: {response.status_code}")
    if response.status_code == 200:
        env_data = response.json()
        print(f"Environment data: {env_data}")
    else:
        print(f"Environment creation failed: {response.text}")

    # Test Isaac Sim robot spawning
    print("\n4. Testing Isaac Sim robot spawning...")
    robot_config = {
        "name": "test_robot",
        "urdf_path": "/Isaac/Robots/Franka/franka_alt_fingers.usd",
        "position": [0.0, 0.0, 0.0]
    }
    response = client.post("/api/isaac/sim/robots", json=robot_config)
    print(f"Robot spawning status: {response.status_code}")
    if response.status_code == 200:
        robot_data = response.json()
        print(f"Robot data: {robot_data}")
    else:
        print(f"Robot spawning failed: {response.text}")

    # Test Isaac AI detection
    print("\n5. Testing Isaac AI detection...")
    detection_request = {
        "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "model_pathname": "/Isaac/Models/detectnet/resnet18_detector.pth",
        "confidence_threshold": 0.5
    }
    response = client.post("/api/isaac/ai/detectnet/process", json=detection_request)
    print(f"Detection status: {response.status_code}")
    if response.status_code == 200:
        detection_data = response.json()
        print(f"Detection results: {len(detection_data['detections'])} detections")
    else:
        print(f"Detection failed: {response.text}")

    # Test Isaac AI navigation planning
    print("\n6. Testing Isaac AI navigation planning...")
    nav_request = {
        "start_pose": [0.0, 0.0, 0.0],
        "goal_pose": [1.0, 1.0, 0.0]
    }
    response = client.post("/api/isaac/ai/navigation/plan", json=nav_request)
    print(f"Navigation planning status: {response.status_code}")
    if response.status_code == 200:
        nav_data = response.json()
        print(f"Navigation path: {len(nav_data['path'])} waypoints")
    else:
        print(f"Navigation planning failed: {response.text}")

    # Test Isaac ROS Bridge message conversion
    print("\n7. Testing Isaac ROS Bridge message conversion...")
    ros_message = {
        "isaac_message_type": "sensor_data",
        "ros_message_type": "sensor_msgs/Image",
        "data": {"test": "data"},
        "frame_id": "camera_frame"
    }
    response = client.post("/api/isaac/ros_bridge/convert_message", json=ros_message)
    print(f"ROS Bridge conversion status: {response.status_code}")
    if response.status_code == 200:
        ros_data = response.json()
        print(f"ROS Bridge result: {ros_data}")
    else:
        print(f"ROS Bridge conversion failed: {response.text}")

    # Test Isaac app status
    print("\n8. Testing Isaac app status...")
    response = client.get("/api/isaac/apps/status")
    print(f"App status check status: {response.status_code}")
    if response.status_code == 200:
        status_data = response.json()
        print(f"App statuses: {status_data}")
    else:
        print(f"App status check failed: {response.text}")

    print("\nAll Isaac module integration tests completed!")

if __name__ == "__main__":
    test_isaac_endpoints()