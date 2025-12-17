#!/usr/bin/env python3
"""
Test script for learning API endpoints
"""
import asyncio
import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_rl_training():
    """Test RL training endpoint"""
    print("Testing RL Training Endpoint...")

    request_data = {
        "algorithm": "dqn",
        "state_dim": 10,
        "action_dim": 4,
        "episodes": 5,
        "environment_config": {}
    }

    response = client.post("/api/learning/train-rl", json=request_data)
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] == True
    assert "metrics" in response_data
    print("PASS: RL Training test passed\n")


def test_imitation_learning():
    """Test imitation learning endpoint"""
    print("Testing Imitation Learning Endpoint...")

    # Create sample demonstrations
    demonstrations = []
    for i in range(2):
        demo = {
            "states": [
                {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "joint_angles": [0.0, 0.0, 0.0],
                    "joint_velocities": [0.0, 0.0, 0.0],
                    "sensor_readings": [0.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0]
                }
            ],
            "actions": [
                {
                    "joint_commands": [0.0, 0.0],
                    "velocity_commands": [0.0, 0.0, 0.0],
                    "gripper_commands": [0.0]
                }
            ],
            "rewards": [1.0]
        }
        demonstrations.append(demo)

    request_data = {
        "demonstrations": demonstrations,
        "state_dim": 23,  # Based on the state vector size (3+4+3+3+4+3+3)
        "action_dim": 3,
        "epochs": 5
    }

    response = client.post("/api/learning/train-imitation", json=request_data)
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] == True
    print("PASS: Imitation Learning test passed\n")


def test_reward_function():
    """Test reward function endpoint"""
    print("Testing Reward Function Endpoint...")

    request_data = {
        "current_position": [0.0, 0.0, 0.0],
        "target_position": [1.0, 1.0, 1.0],
        "obstacles": [[0.5, 0.5, 0.5]],
        "function_type": "reach_target"
    }

    response = client.post("/api/learning/compute-reward", json=request_data)
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] == True
    assert isinstance(response_data["reward"], (int, float))
    print("PASS: Reward Function test passed\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Learning API Tests\n")

    test_rl_training()
    test_imitation_learning()
    test_reward_function()

    print("All learning API tests completed successfully!")


if __name__ == "__main__":
    run_api_tests()