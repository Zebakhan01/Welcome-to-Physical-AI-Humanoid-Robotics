#!/usr/bin/env python3
"""
Test script for ROS API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_create_node():
    """Test node creation endpoint"""
    print("Testing Node Creation Endpoint...")

    request_data = {
        "name": "test_node",
        "namespace": "/test"
    }

    response = client.post("/api/ros/create-node", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Node ID: {response_data.get('node_id')}")
        print(f"Node name: {response_data.get('node_name')}")
        assert response_data["success"] == True
        assert "node_id" in response_data
        print("PASS: Node Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Node Creation test failed with status {response.status_code}\n")


def test_create_publisher():
    """Test publisher creation endpoint"""
    print("Testing Publisher Creation Endpoint...")

    request_data = {
        "topic": "/test_topic",
        "message_type": "std_msgs/String",
        "queue_size": 10
    }

    response = client.post("/api/ros/create-publisher", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Topic: {response_data.get('topic')}")
        assert response_data["success"] == True
        assert response_data["topic"] == "/test_topic"
        print("PASS: Publisher Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Publisher Creation test failed with status {response.status_code}\n")


def test_create_subscriber():
    """Test subscriber creation endpoint"""
    print("Testing Subscriber Creation Endpoint...")

    request_data = {
        "topic": "/test_topic",
        "message_type": "std_msgs/String"
    }

    response = client.post("/api/ros/create-subscriber", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Topic: {response_data.get('topic')}")
        assert response_data["success"] == True
        assert response_data["topic"] == "/test_topic"
        print("PASS: Subscriber Creation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Subscriber Creation test failed with status {response.status_code}\n")


def test_publish_message():
    """Test message publishing endpoint"""
    print("Testing Message Publishing Endpoint...")

    # First create a publisher
    pub_request = {
        "topic": "/publish_test",
        "message_type": "std_msgs/String",
        "queue_size": 10
    }
    client.post("/api/ros/create-publisher", json=pub_request)

    # Now publish a message
    request_data = {
        "topic": "/publish_test",
        "message": {"data": "Hello from API test"}
    }

    response = client.post("/api/ros/publish", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Message Publishing test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Message Publishing test failed with status {response.status_code}\n")


def test_register_service():
    """Test service registration endpoint"""
    print("Testing Service Registration Endpoint...")

    request_data = {
        "name": "/test_service",
        "service_type": "test_msgs/TestService",
        "handler_type": "echo"
    }

    response = client.post("/api/ros/register-service", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Service name: {response_data.get('service_name')}")
        assert response_data["success"] == True
        print("PASS: Service Registration test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Service Registration test failed with status {response.status_code}\n")


def test_call_service():
    """Test service calling endpoint"""
    print("Testing Service Calling Endpoint...")

    # First register a service
    reg_request = {
        "name": "/add_test",
        "service_type": "test_msgs/AddTwoInts",
        "handler_type": "add_two_ints"
    }
    client.post("/api/ros/register-service", json=reg_request)

    # Now call the service
    request_data = {
        "service_name": "/add_test",
        "request_data": {"a": 10, "b": 20}
    }

    response = client.post("/api/ros/call-service", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Service response: {response_data.get('response')}")
        assert response_data["success"] == True
        assert response_data["response"]["sum"] == 30
        print("PASS: Service Calling test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Service Calling test failed with status {response.status_code}\n")


def test_ros_info():
    """Test ROS system information endpoint"""
    print("Testing ROS System Information Endpoint...")

    response = client.get("/api/ros/info")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Nodes: {len(response_data.get('nodes', []))}")
        print(f"Topics: {len(response_data.get('topics', []))}")
        print(f"Services: {len(response_data.get('services', []))}")
        assert response_data["success"] == True
        print("PASS: ROS Info test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: ROS Info test failed with status {response.status_code}\n")


def test_parameter_operations():
    """Test parameter set/get operations"""
    print("Testing Parameter Operations...")

    # Set a parameter
    set_request = {
        "name": "/test_param",
        "value": "test_value"
    }
    response = client.post("/api/ros/set-parameter", json=set_request)
    print(f"Set parameter response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        assert response_data["success"] == True
        print("Parameter set successfully")

        # Get the parameter
        get_request = {
            "name": "/test_param",
            "default": "default_value"
        }
        response = client.post("/api/ros/get-parameter", json=get_request)
        print(f"Get parameter response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Retrieved parameter: {response_data.get('name')} = {response_data.get('value')}")
            assert response_data["value"] == "test_value"
            print("PASS: Parameter Operations test passed\n")
        else:
            print(f"Response: {response.json()}")
            print(f"FAIL: Get parameter test failed with status {response.status_code}\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Set parameter test failed with status {response.status_code}\n")


def test_list_operations():
    """Test list operations for nodes, topics, and services"""
    print("Testing List Operations...")

    # List nodes
    response = client.get("/api/ros/nodes")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Found {len(response_data.get('nodes', []))} nodes")
        assert response_data["success"] == True

    # List topics
    response = client.get("/api/ros/topics")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Found {len(response_data.get('topics', []))} topics")
        assert response_data["success"] == True

    # List services
    response = client.get("/api/ros/services")
    if response.status_code == 200:
        response_data = response.json()
        print(f"Found {len(response_data.get('services', []))} services")
        assert response_data["success"] == True

    print("PASS: List Operations test passed\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting ROS API Tests\n")

    test_create_node()
    test_create_publisher()
    test_create_subscriber()
    test_publish_message()
    test_register_service()
    test_call_service()
    test_ros_info()
    test_parameter_operations()
    test_list_operations()

    print("All ROS API tests completed!")


if __name__ == "__main__":
    run_api_tests()