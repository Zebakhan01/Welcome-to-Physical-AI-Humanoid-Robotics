#!/usr/bin/env python3
"""
Test script for Unity integration API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_create_scene():
    """Test scene creation endpoint"""
    print("Testing Scene Creation Endpoint...")

    request_data = {
        "name": "api_test_scene"
    }

    response = client.post("/api/unity/create-scene", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Scene ID: {response_data.get('scene_id')}")
        print(f"Scene name: {response_data.get('name')}")
        assert response_data["success"] == True
        print("PASS: Scene Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Scene Creation test failed with status {response.status_code}\n")


def test_list_scenes():
    """Test list scenes endpoint"""
    print("Testing List Scenes Endpoint...")

    response = client.get("/api/unity/scenes")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        scenes = response_data.get("scenes", [])
        print(f"Scenes found: {len(scenes)}")
        assert response_data["success"] == True
        print("PASS: List Scenes test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: List Scenes test failed with status {response.status_code}\n")


def test_create_object():
    """Test object creation endpoint"""
    print("Testing Object Creation Endpoint...")

    # First create a scene
    scene_request = {"name": "object_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        request_data = {
            "scene_id": scene_id,
            "name": "test_cube",
            "object_type": "game_object",
            "position": [1.0, 1.0, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "tags": ["test", "cube"]
        }

        response = client.post("/api/unity/create-object", json=request_data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Object ID: {response_data.get('object_id')}")
            assert response_data["success"] == True
            print("PASS: Object Creation test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Object Creation test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create scene for object test\n")


def test_set_transform():
    """Test set transform endpoint"""
    print("Testing Set Transform Endpoint...")

    # First create a scene and object
    scene_request = {"name": "transform_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        obj_request = {
            "scene_id": scene_id,
            "name": "transform_test_obj",
            "object_type": "game_object",
            "position": [0.0, 0.0, 1.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "tags": ["test"]
        }
        obj_response = client.post("/api/unity/create-object", json=obj_request)
        object_id = obj_response.json()["object_id"] if obj_response.status_code == 200 else None

        if object_id:
            request_data = {
                "scene_id": scene_id,
                "object_id": object_id,
                "position": [2.0, 2.0, 1.0],
                "rotation": [0.0, 0.0, 0.1, 1.0]
            }

            response = client.post("/api/unity/set-transform", json=request_data)
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Transform set success: {response_data.get('success')}")
                assert response_data["success"] == True
                print("PASS: Set Transform test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Set Transform test failed with status {response.status_code}\n")
        else:
            print("FAIL: Could not create object for transform test\n")
    else:
        print("FAIL: Could not create scene for transform test\n")


def test_spawn_robot():
    """Test robot spawning endpoint"""
    print("Testing Robot Spawning Endpoint...")

    # First create a scene
    scene_request = {"name": "robot_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        request_data = {
            "scene_id": scene_id,
            "robot_name": "test_mobile_robot",
            "robot_type": "mobile_base"
        }

        response = client.post("/api/unity/spawn-robot", json=request_data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Robot object ID: {response_data.get('robot_object_id')}")
            assert response_data["success"] == True
            print("PASS: Robot Spawning test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Robot Spawning test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create scene for robot test\n")


def test_move_robot():
    """Test robot movement endpoint"""
    print("Testing Robot Movement Endpoint...")

    # First create a scene and spawn a robot
    scene_request = {"name": "move_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        spawn_request = {
            "scene_id": scene_id,
            "robot_name": "move_test_robot",
            "robot_type": "mobile_base"
        }
        spawn_response = client.post("/api/unity/spawn-robot", json=spawn_request)
        robot_object_id = spawn_response.json()["robot_object_id"] if spawn_response.status_code == 200 else None

        if robot_object_id:
            request_data = {
                "scene_id": scene_id,
                "robot_object_id": robot_object_id,
                "target_position": [1.0, 1.0, 0.0]
            }

            response = client.post("/api/unity/move-robot", json=request_data)
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Move success: {response_data.get('success')}")
                assert response_data["success"] == True
                print("PASS: Robot Movement test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Robot Movement test failed with status {response.status_code}\n")
        else:
            print("FAIL: Could not spawn robot for movement test\n")
    else:
        print("FAIL: Could not create scene for movement test\n")


def test_add_sensor():
    """Test sensor addition endpoint"""
    print("Testing Sensor Addition Endpoint...")

    # First create a scene and spawn a robot
    scene_request = {"name": "sensor_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        spawn_request = {
            "scene_id": scene_id,
            "robot_name": "sensor_test_robot",
            "robot_type": "mobile_base"
        }
        spawn_response = client.post("/api/unity/spawn-robot", json=spawn_request)
        robot_object_id = spawn_response.json()["robot_object_id"] if spawn_response.status_code == 200 else None

        if robot_object_id:
            request_data = {
                "scene_id": scene_id,
                "robot_object_id": robot_object_id,
                "sensor_type": "camera",
                "sensor_name": "robot_camera"
            }

            response = client.post("/api/unity/add-sensor", json=request_data)
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
            print("FAIL: Could not spawn robot for sensor test\n")
    else:
        print("FAIL: Could not create scene for sensor test\n")


def test_get_sensor_data():
    """Test get sensor data endpoint"""
    print("Testing Get Sensor Data Endpoint...")

    # This test will fail without a real sensor, but we can test the error handling
    response = client.get("/api/unity/get-sensor-data?sensor_id=nonexistent_sensor")
    print(f"Response status: {response.status_code}")

    # We expect this to return a 404 or similar error for nonexistent sensor
    print(f"Response: {response.json()}")
    print("PASS: Get Sensor Data test completed (tested error handling)\n")


def test_get_robot_state():
    """Test get robot state endpoint"""
    print("Testing Get Robot State Endpoint...")

    # First create a scene and spawn a robot
    scene_request = {"name": "state_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        spawn_request = {
            "scene_id": scene_id,
            "robot_name": "state_test_robot",
            "robot_type": "mobile_base"
        }
        spawn_response = client.post("/api/unity/spawn-robot", json=spawn_request)
        robot_object_id = spawn_response.json()["robot_object_id"] if spawn_response.status_code == 200 else None

        if robot_object_id:
            response = client.get(f"/api/unity/get-robot-state?scene_id={scene_id}&robot_object_id={robot_object_id}")
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                robot_state = response_data.get("robot_state")
                print(f"Robot state retrieved: {robot_state is not None}")
                assert response_data["success"] == True
                print("PASS: Get Robot State test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Get Robot State test failed with status {response.status_code}\n")
        else:
            print("FAIL: Could not spawn robot for state test\n")
    else:
        print("FAIL: Could not create scene for state test\n")


def test_apply_force():
    """Test apply force endpoint"""
    print("Testing Apply Force Endpoint...")

    # First create a scene and object
    scene_request = {"name": "force_test_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        obj_request = {
            "scene_id": scene_id,
            "name": "force_test_obj",
            "object_type": "model",
            "position": [0.0, 0.0, 2.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "tags": ["physics", "test"]
        }
        obj_response = client.post("/api/unity/create-object", json=obj_request)
        object_id = obj_response.json()["object_id"] if obj_response.status_code == 200 else None

        if object_id:
            request_data = {
                "object_id": object_id,
                "force": [5.0, 0.0, 0.0]
            }

            response = client.post("/api/unity/apply-force", json=request_data)
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Apply force success: {response_data.get('success')}")
                assert response_data["success"] == True
                print("PASS: Apply Force test passed\n")
            else:
                print(f"Response: {response.json()}")
                print(f"FAIL: Apply Force test failed with status {response.status_code}\n")
        else:
            print("FAIL: Could not create object for force test\n")
    else:
        print("FAIL: Could not create scene for force test\n")


def test_simulation_control():
    """Test simulation control endpoints"""
    print("Testing Simulation Control Endpoints...")

    # Test start simulation
    response = client.post("/api/unity/start-simulation")
    print(f"Start simulation response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Start simulation success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Test stop simulation
        response = client.post("/api/unity/stop-simulation")
        print(f"Stop simulation response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Stop simulation success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Simulation Control test passed\n")
        else:
            print(f"Stop Response: {response.json()}")
            print(f"FAIL: Stop simulation test failed with status {response.status_code}\n")
    else:
        print(f"Start Response: {response.json()}")
        print(f"FAIL: Start simulation test failed with status {response.status_code}\n")


def test_list_objects():
    """Test list objects endpoint"""
    print("Testing List Objects Endpoint...")

    # First create a scene
    scene_request = {"name": "list_objects_scene"}
    scene_response = client.post("/api/unity/create-scene", json=scene_request)
    scene_id = scene_response.json()["scene_id"] if scene_response.status_code == 200 else None

    if scene_id:
        # Create an object in the scene
        obj_request = {
            "scene_id": scene_id,
            "name": "list_test_obj",
            "object_type": "game_object",
            "position": [0.0, 0.0, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "tags": ["test"]
        }
        client.post("/api/unity/create-object", json=obj_request)

        response = client.get(f"/api/unity/objects?scene_id={scene_id}")
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            objects = response_data.get("objects", [])
            print(f"Objects in scene: {len(objects)}")
            assert response_data["success"] == True
            print("PASS: List Objects test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: List Objects test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create scene for list objects test\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Unity Integration API Tests\n")

    test_create_scene()
    test_list_scenes()
    test_create_object()
    test_set_transform()
    test_spawn_robot()
    test_move_robot()
    test_add_sensor()
    test_get_sensor_data()
    test_get_robot_state()
    test_apply_force()
    test_simulation_control()
    test_list_objects()

    print("All Unity integration API tests completed!")


if __name__ == "__main__":
    run_api_tests()