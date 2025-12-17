#!/usr/bin/env python3
"""
Test script for simulation API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_create_world():
    """Test world creation endpoint"""
    print("Testing World Creation Endpoint...")

    request_data = {
        "name": "test_world",
        "description": "Test simulation world",
        "physics_engine": "bullet"
    }

    response = client.post("/api/simulation/create-world", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"World ID: {response_data.get('world_id')}")
        print(f"World name: {response_data.get('name')}")
        assert response_data["success"] == True
        assert "world_id" in response_data
        print("PASS: World Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: World Creation test failed with status {response.status_code}\n")


def test_load_robot():
    """Test robot loading endpoint"""
    print("Testing Robot Loading Endpoint...")

    # First create a world
    world_request = {
        "name": "robot_test_world",
        "description": "World for robot testing",
        "physics_engine": "bullet"
    }
    world_response = client.post("/api/simulation/create-world", json=world_request)
    world_id = world_response.json()["world_id"] if world_response.status_code == 200 else None

    if world_id:
        # Now load a robot
        request_data = {
            "world_id": world_id,
            "name": "test_robot",
            "urdf_path": "/path/to/test_robot.urdf",
            "initial_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

        response = client.post("/api/simulation/load-robot", json=request_data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Robot ID: {response_data.get('robot_id')}")
            print(f"Robot name: {response_data.get('name')}")
            assert response_data["success"] == True
            print("PASS: Robot Loading test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Robot Loading test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create world for robot loading test\n")


def test_add_sensor():
    """Test sensor addition endpoint"""
    print("Testing Sensor Addition Endpoint...")

    # First create a world
    world_request = {
        "name": "sensor_test_world",
        "description": "World for sensor testing",
        "physics_engine": "bullet"
    }
    world_response = client.post("/api/simulation/create-world", json=world_request)
    world_id = world_response.json()["world_id"] if world_response.status_code == 200 else None

    if world_id:
        # Load a robot
        robot_request = {
            "world_id": world_id,
            "name": "sensor_robot",
            "urdf_path": "/path/to/sensor_robot.urdf",
            "initial_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        robot_response = client.post("/api/simulation/load-robot", json=robot_request)
        robot_id = robot_response.json()["robot_id"] if robot_response.status_code == 200 else None

        if robot_id:
            # Add a sensor to the robot
            request_data = {
                "robot_id": robot_id,
                "sensor_type": "camera",
                "name": "robot_camera",
                "position": [0.1, 0.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "parameters": {
                    "width": 640,
                    "height": 480
                }
            }

            response = client.post("/api/simulation/add-sensor", json=request_data)
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Sensor ID: {response_data.get('sensor_id')}")
                assert response_data["success"] == True
                print("PASS: Sensor Addition test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Sensor Addition test failed with status {response.status_code}\n")
        else:
            print("FAIL: Could not load robot for sensor addition test\n")
    else:
        print("FAIL: Could not create world for sensor addition test\n")


def test_simulation_control():
    """Test simulation control endpoint"""
    print("Testing Simulation Control Endpoint...")

    # First create a world
    world_request = {
        "name": "control_test_world",
        "description": "World for control testing",
        "physics_engine": "bullet"
    }
    world_response = client.post("/api/simulation/create-world", json=world_request)
    world_id = world_response.json()["world_id"] if world_response.status_code == 200 else None

    if world_id:
        # Test starting simulation
        request_data = {
            "world_id": world_id,
            "action": "start"
        }

        response = client.post("/api/simulation/control-simulation", json=request_data)
        print(f"Start response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Start success: {response_data.get('success')}")
            assert response_data["success"] == True

            # Test pausing simulation
            request_data["action"] = "pause"
            response = client.post("/api/simulation/control-simulation", json=request_data)
            print(f"Pause response status: {response.status_code}")

            # Test resuming simulation
            if response.status_code == 200:
                request_data["action"] = "resume"
                response = client.post("/api/simulation/control-simulation", json=request_data)
                print(f"Resume response status: {response.status_code}")

                # Test stepping simulation
                if response.status_code == 200:
                    request_data["action"] = "step"
                    response = client.post("/api/simulation/control-simulation", json=request_data)
                    print(f"Step response status: {response.status_code}")

                    # Test stopping simulation
                    if response.status_code == 200:
                        request_data["action"] = "stop"
                        response = client.post("/api/simulation/control-simulation", json=request_data)
                        print(f"Stop response status: {response.status_code}")

                        if response.status_code == 200:
                            print("PASS: Simulation Control test passed\n")
                        else:
                            print(f"Stop Response: {response.json()}")
                            print(f"FAIL: Stop simulation failed with status {response.status_code}\n")
                    else:
                        print(f"Step Response: {response.json()}")
                        print(f"FAIL: Step simulation failed with status {response.status_code}\n")
                else:
                    print(f"Resume Response: {response.json()}")
                    print(f"FAIL: Resume simulation failed with status {response.status_code}\n")
            else:
                print(f"Pause Response: {response.json()}")
                print(f"FAIL: Pause simulation failed with status {response.status_code}\n")
        else:
            print(f"Start Response: {response.json()}")
            print(f"FAIL: Start simulation failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create world for simulation control test\n")


def test_get_simulation_state():
    """Test getting simulation state"""
    print("Testing Get Simulation State Endpoint...")

    # First create a world
    world_request = {
        "name": "state_test_world",
        "description": "World for state testing",
        "physics_engine": "bullet"
    }
    world_response = client.post("/api/simulation/create-world", json=world_request)
    world_id = world_response.json()["world_id"] if world_response.status_code == 200 else None

    if world_id:
        # Start simulation first
        control_request = {
            "world_id": world_id,
            "action": "start"
        }
        client.post("/api/simulation/control-simulation", json=control_request)

        # Step simulation to generate some state
        control_request["action"] = "step"
        client.post("/api/simulation/control-simulation", json=control_request)

        # Get simulation state
        response = client.get(f"/api/simulation/simulation-state?world_id={world_id}")
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Simulation time: {response_data.get('time')}")
            print(f"Running: {response_data.get('running')}")
            print(f"Robot states count: {len(response_data.get('robot_states', []))}")
            assert response_data["success"] == True
            print("PASS: Get Simulation State test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Get Simulation State test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create world for simulation state test\n")


def test_domain_randomization():
    """Test domain randomization endpoints"""
    print("Testing Domain Randomization Endpoints...")

    # Add a domain randomization parameter
    request_data = {
        "parameter_name": "test_gravity",
        "parameter_type": "float",
        "min_value": -12.0,
        "max_value": -6.0
    }

    response = client.post("/api/simulation/domain-randomization", json=request_data)
    print(f"Add parameter response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Add parameter success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Get randomization values
        response = client.get("/api/simulation/randomization-values")
        print(f"Get values response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Randomization parameters: {list(response_data.get('parameters', {}).keys())}")
            assert response_data["success"] == True

            # Randomize the parameter via query parameter
            response = client.post(f"/api/simulation/randomize-parameter?parameter_name=test_gravity")
            print(f"Randomize parameter response status: {response.status_code}")

            if response.status_code == 200:
                print("PASS: Domain Randomization test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Randomize parameter failed with status {response.status_code}\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Get randomization values failed with status {response.status_code}\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Add domain randomization parameter failed with status {response.status_code}\n")


def test_list_operations():
    """Test list operations for environments, worlds, and robots"""
    print("Testing List Operations...")

    # List environments
    response = client.get("/api/simulation/environments")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Environments: {len(response_data)}")

    # List worlds
    response = client.get("/api/simulation/worlds")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Worlds: {len(response_data)}")

    # List robots
    response = client.get("/api/simulation/robots")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Robots: {len(response_data)}")

    print("PASS: List Operations test passed\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Simulation API Tests\n")

    test_create_world()
    test_load_robot()
    test_add_sensor()
    test_simulation_control()
    test_get_simulation_state()
    test_domain_randomization()
    test_list_operations()

    print("All simulation API tests completed!")


if __name__ == "__main__":
    run_api_tests()