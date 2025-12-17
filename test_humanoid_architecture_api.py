#!/usr/bin/env python3
"""
Test script for humanoid architecture API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_create_platform():
    """Test platform creation endpoint"""
    print("Testing Platform Creation Endpoint...")

    request_data = {
        "name": "test_nao_robot",
        "platform_type": "nao"
    }

    response = client.post("/api/humanoid/create-platform", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Robot name: {response_data.get('robot_name')}")
        print(f"Platform type: {response_data.get('platform_type')}")
        print(f"Total DOF: {response_data.get('total_dof')}")
        assert response_data["success"] == True
        assert response_data["platform_type"] == "nao"
        print("PASS: Platform Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Platform Creation test failed with status {response.status_code}\n")


def test_list_platforms():
    """Test list platforms endpoint"""
    print("Testing List Platforms Endpoint...")

    response = client.get("/api/humanoid/platforms")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Platforms: {response_data.get('platforms', [])}")
        assert response_data["success"] == True
        print("PASS: List Platforms test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: List Platforms test failed with status {response.status_code}\n")


def test_list_robots():
    """Test list robots endpoint"""
    print("Testing List Robots Endpoint...")

    response = client.get("/api/humanoid/robots")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Robots: {response_data}")
        assert isinstance(response_data, list)
        print("PASS: List Robots test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: List Robots test failed with status {response.status_code}\n")


def test_update_state():
    """Test state update endpoint"""
    print("Testing State Update Endpoint...")

    # First create a robot
    create_data = {
        "name": "state_test_robot",
        "platform_type": "nao"
    }
    client.post("/api/humanoid/create-platform", json=create_data)

    # Update state with sensor data
    request_data = {
        "robot_name": "state_test_robot",
        "sensor_data": {
            "imu": {"orientation": [0, 0, 0, 1], "angular_velocity": [0, 0, 0], "linear_acceleration": [0, 0, 9.81]},
            "ft_sensors": {"left_foot": [0, 0, -50, 0, 0, 0], "right_foot": [0, 0, -50, 0, 0, 0]}
        }
    }

    response = client.post("/api/humanoid/update-state", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Update success: {response_data.get('success')}")
        print(f"Is safe: {response_data.get('is_safe')}")
        assert response_data["success"] == True
        print("PASS: State Update test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: State Update test failed with status {response.status_code}\n")


def test_get_state():
    """Test get state endpoint"""
    print("Testing Get State Endpoint...")

    response = client.get("/api/humanoid/state?robot_name=state_test_robot")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Robot name: {response_data.get('robot_name')}")
        print(f"Joint states count: {len(response_data.get('joint_states', {}))}")
        print(f"Is safe: {response_data.get('is_safe')}")
        assert response_data["success"] == True
        print("PASS: Get State test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Get State test failed with status {response.status_code}\n")


def test_set_control_mode():
    """Test set control mode endpoint"""
    print("Testing Set Control Mode Endpoint...")

    request_data = {
        "robot_name": "state_test_robot",
        "mode": "walking"
    }

    response = client.post("/api/humanoid/set-control-mode", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Set mode success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Set Control Mode test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Set Control Mode test failed with status {response.status_code}\n")


def test_compute_commands():
    """Test compute commands endpoint"""
    print("Testing Compute Commands Endpoint...")

    response = client.post("/api/humanoid/compute-commands?robot_name=state_test_robot")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Compute success: {response_data.get('success')}")
        print(f"Joint commands count: {len(response_data.get('joint_commands', {}))}")
        assert response_data["success"] == True
        print("PASS: Compute Commands test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Compute Commands test failed with status {response.status_code}\n")


def test_add_task():
    """Test add task endpoint"""
    print("Testing Add Task Endpoint...")

    request_data = {
        "robot_name": "state_test_robot",
        "task_type": "move_arm",
        "priority": 10,
        "constraints": {"position": [0.5, 0.5, 0.5]}
    }

    response = client.post("/api/humanoid/add-task", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Add task success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Add Task test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Add Task test failed with status {response.status_code}\n")


def test_evaluate_performance():
    """Test evaluate performance endpoint"""
    print("Testing Evaluate Performance Endpoint...")

    request_data = {
        "robot_name": "state_test_robot",
        "task": "walking"
    }

    response = client.post("/api/humanoid/evaluate-performance", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Evaluation success: {response_data.get('success')}")
        if response_data["success"]:
            print(f"Metrics keys: {list(response_data.get('metrics', {}).keys())}")
        assert response_data["success"] == True
        print("PASS: Evaluate Performance test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Evaluate Performance test failed with status {response.status_code}\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Humanoid Architecture API Tests\n")

    test_create_platform()
    test_list_platforms()
    test_list_robots()
    test_update_state()
    test_get_state()
    test_set_control_mode()
    test_compute_commands()
    test_add_task()
    test_evaluate_performance()

    print("All humanoid architecture API tests completed!")


if __name__ == "__main__":
    run_api_tests()