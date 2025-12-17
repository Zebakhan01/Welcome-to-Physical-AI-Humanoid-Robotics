#!/usr/bin/env python3
"""
Test script for ROS2 API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_create_node():
    """Test node creation endpoint"""
    print("Testing Node Creation Endpoint...")

    request_data = {
        "name": "api_test_node",
        "namespace": "/api_test"
    }

    response = client.post("/api/ros2/create-node", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Node ID: {response_data.get('node_id')}")
        print(f"Node name: {response_data.get('node_name')}")
        assert response_data["success"] == True
        print("PASS: Node Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Node Creation test failed with status {response.status_code}\n")


def test_lifecycle_transitions():
    """Test lifecycle transition endpoints"""
    print("Testing Lifecycle Transition Endpoints...")

    # First create a node
    create_data = {"name": "lifecycle_test_node", "namespace": "/lc_test"}
    create_response = client.post("/api/ros2/create-node", json=create_data)
    node_id = create_response.json()["node_id"] if create_response.status_code == 200 else None

    if node_id:
        # Test configure
        request_data = {"node_id": node_id, "action": "configure"}
        response = client.post("/api/ros2/lifecycle-transition", json=request_data)
        print(f"Configure response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Configure success: {response_data.get('success')}")
            assert response_data["success"] == True

            # Test activate
            request_data["action"] = "activate"
            response = client.post("/api/ros2/lifecycle-transition", json=request_data)
            print(f"Activate response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Activate success: {response_data.get('success')}")
                assert response_data["success"] == True
                print("PASS: Lifecycle Transitions test passed\n")
            else:
                print(f"Activate Response: {response.json()}")
                print(f"FAIL: Activate failed with status {response.status_code}\n")
        else:
            print(f"Configure Response: {response.json()}")
            print(f"FAIL: Configure failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create node for lifecycle test\n")


def test_create_publisher():
    """Test publisher creation endpoint"""
    print("Testing Publisher Creation Endpoint...")

    # First create a node
    create_data = {"name": "pub_test_node", "namespace": "/pub_test"}
    create_response = client.post("/api/ros2/create-node", json=create_data)
    node_id = create_response.json()["node_id"] if create_response.status_code == 200 else None

    if node_id:
        request_data = {
            "node_id": node_id,
            "topic": "/test_publisher_topic",
            "message_type": "std_msgs/String",
            "qos_profile": {
                "reliability": "reliable",
                "durability": "volatile",
                "history": "keep_last",
                "depth": 10
            }
        }

        response = client.post("/api/ros2/create-publisher", json=request_data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Create publisher success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Publisher Creation test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Publisher Creation test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create node for publisher test\n")


def test_create_subscriber():
    """Test subscriber creation endpoint"""
    print("Testing Subscriber Creation Endpoint...")

    # First create a node
    create_data = {"name": "sub_test_node", "namespace": "/sub_test"}
    create_response = client.post("/api/ros2/create-node", json=create_data)
    node_id = create_response.json()["node_id"] if create_response.status_code == 200 else None

    if node_id:
        request_data = {
            "node_id": node_id,
            "topic": "/test_subscriber_topic",
            "message_type": "std_msgs/String",
            "qos_profile": {
                "reliability": "reliable",
                "durability": "volatile",
                "history": "keep_last",
                "depth": 10
            }
        }

        response = client.post("/api/ros2/create-subscriber", json=request_data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Create subscriber success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Subscriber Creation test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Subscriber Creation test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create node for subscriber test\n")


def test_publish_message():
    """Test message publishing endpoint"""
    print("Testing Message Publishing Endpoint...")

    request_data = {
        "topic": "/test_publish_topic",
        "message": {"data": "Hello from API test", "id": 123}
    }

    response = client.post("/api/ros2/publish", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Publish success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Message Publishing test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Message Publishing test failed with status {response.status_code}\n")


def test_parameter_operations():
    """Test parameter operations endpoints"""
    print("Testing Parameter Operations Endpoints...")

    # First create a node
    create_data = {"name": "param_api_test_node", "namespace": "/param_test"}
    create_response = client.post("/api/ros2/create-node", json=create_data)
    node_id = create_response.json()["node_id"] if create_response.status_code == 200 else None

    if node_id:
        # Test declare parameter
        declare_data = {
            "node_id": node_id,
            "name": "test_api_param",
            "default_value": 42,
            "description": "Test parameter from API",
            "read_only": False
        }

        response = client.post("/api/ros2/declare-parameter", json=declare_data)
        print(f"Declare parameter response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Declare parameter success: {response_data.get('success')}")
            assert response_data["success"] == True

            # Test set parameter
            set_data = {
                "node_id": node_id,
                "name": "test_api_param",
                "value": 100
            }

            response = client.post("/api/ros2/set-parameter", json=set_data)
            print(f"Set parameter response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                print(f"Set parameter success: {response_data.get('success')}")
                assert response_data["success"] == True

                # Test get parameter
                response = client.get(f"/api/ros2/get-parameter?node_id={node_id}&name=test_api_param")
                print(f"Get parameter response status: {response.status_code}")

                if response.status_code == 200:
                    response_data = response.json()
                    print(f"Retrieved parameter: {response_data.get('name')} = {response_data.get('value')}")
                    assert response_data["success"] == True
                    assert response_data["value"] == 100
                    print("PASS: Parameter Operations test passed\n")
                else:
                    print(f"Get parameter Response: {response.json()}")
                    print(f"FAIL: Get parameter failed with status {response.status_code}\n")
            else:
                print(f"Set parameter Response: {response.json()}")
                print(f"FAIL: Set parameter failed with status {response.status_code}\n")
        else:
            print(f"Declare parameter Response: {response.json()}")
            print(f"FAIL: Declare parameter failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create node for parameter test\n")


def test_node_info():
    """Test node information endpoint"""
    print("Testing Node Information Endpoint...")

    # First create a node
    create_data = {"name": "info_test_node", "namespace": "/info_test"}
    create_response = client.post("/api/ros2/create-node", json=create_data)
    node_id = create_response.json()["node_id"] if create_response.status_code == 200 else None

    if node_id:
        response = client.get(f"/api/ros2/node-info?node_id={node_id}")
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Node name: {response_data.get('node_name')}")
            print(f"Namespace: {response_data.get('namespace')}")
            print(f"Lifecycle state: {response_data.get('lifecycle_state')}")
            print(f"Publishers: {len(response_data.get('publishers', []))}")
            assert response_data["success"] == True
            print("PASS: Node Information test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Node Information test failed with status {response.status_code}\n")
    else:
        print("FAIL: Could not create node for info test\n")


def test_list_operations():
    """Test list operations endpoints"""
    print("Testing List Operations Endpoints...")

    # List nodes
    response = client.get("/api/ros2/list-nodes")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Found {len(response_data.get('nodes', []))} nodes")
        assert response_data["success"] == True

    # List topics
    response = client.get("/api/ros2/list-topics")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Found {len(response_data.get('topics', {}))} topics")
        assert response_data["success"] == True

    print("PASS: List Operations test passed\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting ROS2 API Tests\n")

    test_create_node()
    test_lifecycle_transitions()
    test_create_publisher()
    test_create_subscriber()
    test_publish_message()
    test_parameter_operations()
    test_node_info()
    test_list_operations()

    print("All ROS2 API tests completed!")


if __name__ == "__main__":
    run_api_tests()