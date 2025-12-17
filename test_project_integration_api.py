#!/usr/bin/env python3
"""
Test script for project integration API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_integrate_component():
    """Test component integration endpoint"""
    print("Testing Component Integration Endpoint...")

    request_data = {
        "component_id": "test_vision_component",
        "name": "Test Vision Component",
        "component_type": "perception",
        "dependencies": ["sensor_input"]
    }

    response = client.post("/api/integration/integrate-component", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Integration success: {response_data.get('success')}")
        print(f"Message: {response_data.get('message')}")
        assert response_data["success"] == True
        print("PASS: Component Integration test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Component Integration test failed with status {response.status_code}\n")


def test_setup_data_flow():
    """Test data flow setup endpoint"""
    print("Testing Data Flow Setup Endpoint...")

    # First integrate another component
    comp_request = {
        "component_id": "test_control_component",
        "name": "Test Control Component",
        "component_type": "control"
    }
    client.post("/api/integration/integrate-component", json=comp_request)

    # Setup data flow
    request_data = {
        "source": "test_vision_component",
        "target": "test_control_component",
        "data_type": "object_detection",
        "frequency": 30.0,
        "bandwidth": 1000.0
    }

    response = client.post("/api/integration/setup-data-flow", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Setup success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Data Flow Setup test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Data Flow Setup test failed with status {response.status_code}\n")


def test_system_metrics():
    """Test system metrics endpoint"""
    print("Testing System Metrics Endpoint...")

    response = client.get("/api/integration/system-metrics")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Total components: {response_data.get('total_components')}")
        print(f"Operational components: {response_data.get('operational_components')}")
        print(f"Health score: {response_data.get('system_health_score')}")
        assert response_data["success"] == True
        print("PASS: System Metrics test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: System Metrics test failed with status {response.status_code}\n")


def test_validate_system():
    """Test system validation endpoint"""
    print("Testing System Validation Endpoint...")

    response = client.post("/api/integration/validate-system")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Validation success: {response_data.get('overall_success')}")
        print(f"Test results count: {len(response_data.get('test_results', {}))}")
        print(f"Errors: {len(response_data.get('errors', []))}")
        assert response_data["success"] == True
        print("PASS: System Validation test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: System Validation test failed with status {response.status_code}\n")


def test_health_report():
    """Test health report endpoint"""
    print("Testing Health Report Endpoint...")

    response = client.get("/api/integration/health-report")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Report timestamp: {response_data.get('timestamp')}")
        print(f"System metrics keys: {list(response_data.get('system_metrics', {}).keys())}")
        print(f"Component status count: {len(response_data.get('component_status', {}))}")
        assert response_data["success"] == True
        print("PASS: Health Report test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Health Report test failed with status {response.status_code}\n")


def test_data_flow_analysis():
    """Test data flow analysis endpoint"""
    print("Testing Data Flow Analysis Endpoint...")

    response = client.get("/api/integration/data-flow-analysis")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Total flows: {response_data.get('total_flows')}")
        print(f"Flows by type: {response_data.get('by_type')}")
        print(f"Components involved: {len(response_data.get('components_involved'))}")
        assert response_data["success"] == True
        print("PASS: Data Flow Analysis test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Data Flow Analysis test failed with status {response.status_code}\n")


def test_start_stop_integration():
    """Test start/stop integration endpoints"""
    print("Testing Start/Stop Integration Endpoints...")

    # Start integration
    response = client.post("/api/integration/start-integration")
    print(f"Start response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Start success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Stop integration
        response = client.post("/api/integration/stop-integration")
        print(f"Stop response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Stop success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Start/Stop Integration test passed\n")
        else:
            print(f"Stop Response: {response.json()}")
            print(f"FAIL: Stop integration failed with status {response.status_code}\n")
    else:
        print(f"Start Response: {response.json()}")
        print(f"FAIL: Start integration failed with status {response.status_code}\n")


def test_message_bus_operations():
    """Test message bus operations"""
    print("Testing Message Bus Operations...")

    # Subscribe to a topic
    sub_request = {"topic": "test_integration_topic"}
    response = client.post("/api/integration/subscribe", json=sub_request)
    print(f"Subscribe response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Subscribe success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Publish to the topic
        pub_request = {
            "topic": "test_integration_topic",
            "message": {"test_key": "test_value", "timestamp": 12345}
        }
        response = client.post("/api/integration/publish", json=pub_request)
        print(f"Publish response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Publish success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Message Bus Operations test passed\n")
        else:
            print(f"Publish Response: {response.json()}")
            print(f"FAIL: Publish failed with status {response.status_code}\n")
    else:
        print(f"Subscribe Response: {response.json()}")
        print(f"FAIL: Subscribe failed with status {response.status_code}\n")


def test_list_components():
    """Test list components endpoint"""
    print("Testing List Components Endpoint...")

    response = client.get("/api/integration/components")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Components found: {len(response_data)}")
        if response_data:
            print(f"First component: {response_data[0].get('name') if response_data else 'None'}")
        print("PASS: List Components test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: List Components test failed with status {response.status_code}\n")


def test_identify_bottlenecks():
    """Test identify bottlenecks endpoint"""
    print("Testing Identify Bottlenecks Endpoint...")

    response = client.get("/api/integration/bottlenecks")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Bottlenecks found: {len(response_data)}")
        print(f"Bottleneck list: {response_data}")
        assert isinstance(response_data, list)
        print("PASS: Identify Bottlenecks test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Identify Bottlenecks test failed with status {response.status_code}\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Project Integration API Tests\n")

    test_integrate_component()
    test_setup_data_flow()
    test_system_metrics()
    test_validate_system()
    test_health_report()
    test_data_flow_analysis()
    test_start_stop_integration()
    test_message_bus_operations()
    test_list_components()
    test_identify_bottlenecks()

    print("All project integration API tests completed!")


if __name__ == "__main__":
    run_api_tests()