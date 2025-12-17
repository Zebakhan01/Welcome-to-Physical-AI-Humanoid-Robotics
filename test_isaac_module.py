"""
Test script for NVIDIA Isaac module integration
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.api.nvidia_isaac.isaac_service import IsaacSimService, IsaacROSBridgeService, IsaacAIService
from backend.api.nvidia_isaac.models import (
    IsaacEnvironmentConfig, IsaacRobotConfig, IsaacDetectionRequest,
    IsaacNavigationRequest, IsaacManipulationRequest
)
from backend.utils.isaac_integration_utils import IsaacIntegrationUtils

async def test_isaac_module():
    """Test the Isaac module functionality"""
    print("Testing NVIDIA Isaac Module Integration...")

    # Test Isaac Sim Service
    print("\n1. Testing Isaac Sim Service...")
    sim_service = IsaacSimService()
    print(f"Isaac Sim available: {sim_service.is_available()}")

    # Test environment creation
    env_config = IsaacEnvironmentConfig(name="test_env", gravity=[0, 0, -9.81])
    try:
        env_result = await sim_service.create_environment(env_config)
        print(f"Environment created: {env_result['id']}")
    except Exception as e:
        print(f"Environment creation failed: {e}")

    # Test robot spawning
    robot_config = IsaacRobotConfig(name="test_robot", urdf_path="/test/robot.urdf")
    try:
        robot_result = await sim_service.spawn_robot(robot_config)
        print(f"Robot spawned: {robot_result['id']}")
    except Exception as e:
        print(f"Robot spawning failed: {e}")

    # Test Isaac ROS Bridge Service
    print("\n2. Testing Isaac ROS Bridge Service...")
    ros_service = IsaacROSBridgeService()
    print(f"Isaac ROS Bridge available: {ros_service.is_available()}")

    # Test Isaac AI Service
    print("\n3. Testing Isaac AI Service...")
    ai_service = IsaacAIService()
    print(f"Isaac AI available: {ai_service.is_available()}")

    # Test object detection
    detection_request = IsaacDetectionRequest(
        image_data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        model_pathname="/Isaac/Models/detectnet/resnet18_detector.pth",
        confidence_threshold=0.5
    )
    try:
        detection_result = await ai_service.detect_objects(detection_request)
        print(f"Detection completed with {len(detection_result.detections)} detections")
    except Exception as e:
        print(f"Detection failed: {e}")

    # Test navigation planning
    nav_request = IsaacNavigationRequest(
        start_pose=[0.0, 0.0, 0.0],
        goal_pose=[1.0, 1.0, 0.0]
    )
    try:
        nav_result = await ai_service.plan_navigation(nav_request)
        print(f"Navigation path planned with {len(nav_result.path)} waypoints")
    except Exception as e:
        print(f"Navigation planning failed: {e}")

    # Test manipulation planning
    manip_request = IsaacManipulationRequest(
        task_type="move_to_pose",
        target_pose={
            "position": [0.5, 0.5, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        }
    )
    try:
        manip_result = await ai_service.plan_manipulation(manip_request)
        print(f"Manipulation task planned successfully: {manip_result.success}")
    except Exception as e:
        print(f"Manipulation planning failed: {e}")

    # Test integration utilities
    print("\n4. Testing Isaac Integration Utilities...")
    integration_utils = IsaacIntegrationUtils()

    # Test ROS integration
    ros_data = {"topic": "/test", "message": "test_data"}
    try:
        ros_integration = await integration_utils.integrate_with_ros(ros_data)
        print("ROS integration successful")
    except Exception as e:
        print(f"ROS integration failed: {e}")

    # Test sensor integration
    sensor_data = {"sensor_type": "camera", "data": "test_image"}
    try:
        sensor_integration = await integration_utils.integrate_with_sensors(sensor_data)
        print("Sensor integration successful")
    except Exception as e:
        print(f"Sensor integration failed: {e}")

    # Test VLA integration
    vla_input = {"vision": "image_data", "language": "command", "action": "move"}
    try:
        vla_integration = await integration_utils.integrate_with_vla(vla_input)
        print("VLA integration successful")
    except Exception as e:
        print(f"VLA integration failed: {e}")

    # Test learning integration
    learning_data = {"dataset": "test_data", "algorithm": "rl"}
    try:
        learning_integration = await integration_utils.integrate_with_learning(learning_data)
        print("Learning integration successful")
    except Exception as e:
        print(f"Learning integration failed: {e}")

    # Test task simulation
    task_config = {
        "task_type": "navigation",
        "start_pose": [0.0, 0.0, 0.0],
        "goal_pose": [2.0, 2.0, 0.0],
        "environment_name": "test_navigation_env"
    }
    try:
        task_result = await integration_utils.simulate_robot_task(task_config)
        print(f"Task simulation completed: {task_result['task_type']}")
    except Exception as e:
        print(f"Task simulation failed: {e}")

    print("\nAll Isaac module tests completed!")

if __name__ == "__main__":
    asyncio.run(test_isaac_module())