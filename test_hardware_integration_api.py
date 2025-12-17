#!/usr/bin/env python3
"""
Test script for hardware integration API endpoints
"""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_register_component():
    """Test component registration endpoint"""
    print("Testing Component Registration Endpoint...")

    # Register a network-based component instead of serial to avoid pyserial dependency
    request_data = {
        "component_id": "test_sensor_1",
        "component_type": "sensor",
        "protocol": "ethernet",
        "connection_params": {
            "host": "127.0.0.1",
            "port": 8080,
            "sensor_type": "imu"
        }
    }

    response = client.post("/api/hardware/register-component", json=request_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Success: {response_data.get('success')}")
        print(f"Message: {response_data.get('message')}")
        assert response_data["success"] == True
        print("PASS: Component Registration test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Component Registration test failed with status {response.status_code}\n")


def test_list_components():
    """Test list components endpoint"""
    print("Testing List Components Endpoint...")

    response = client.get("/api/hardware/components")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Components found: {len(response_data)}")
        assert isinstance(response_data, list)
        print("PASS: List Components test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: List Components test failed with status {response.status_code}\n")


def test_safety_status():
    """Test safety status endpoint"""
    print("Testing Safety Status Endpoint...")

    response = client.get("/api/hardware/safety-status")
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"System safe: {response_data.get('safe')}")
        print(f"Violations: {len(response_data.get('violations', []))}")
        print(f"Emergency stop active: {response_data.get('emergency_stop_active')}")
        assert response_data["success"] == True
        print("PASS: Safety Status test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Safety Status test failed with status {response.status_code}\n")


def test_emergency_stop():
    """Test emergency stop endpoint"""
    print("Testing Emergency Stop Endpoint...")

    # Activate emergency stop
    request_data = {"activate": True}
    response = client.post("/api/hardware/emergency-stop", json=request_data)
    print(f"Activate response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Activate success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Check safety status (should be unsafe now)
        status_response = client.get("/api/hardware/safety-status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            assert status_data["safe"] == False, "System should be unsafe when emergency stop is active"
            print(f"System safe after emergency stop: {status_data['safe']}")

            # Deactivate emergency stop
            request_data = {"activate": False}
            response = client.post("/api/hardware/emergency-stop", json=request_data)
            print(f"Deactivate response status: {response.status_code}")

            if response.status_code == 200:
                # Check safety status again (should be safe)
                status_response = client.get("/api/hardware/safety-status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    assert status_data["safe"] == True, "System should be safe after emergency stop is deactivated"
                    print(f"System safe after emergency stop release: {status_data['safe']}")
                    print("PASS: Emergency Stop test passed\n")
                else:
                    print(f"Status check after deactivate failed: {status_response.json()}")
                    print(f"FAIL: Could not verify emergency stop deactivation\n")
            else:
                print(f"Deactivate Response: {response.json()}")
                print(f"FAIL: Emergency stop deactivation failed with status {response.status_code}\n")
        else:
            print(f"Status check failed: {status_response.json()}")
            print(f"FAIL: Could not verify emergency stop activation\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Emergency stop activation failed with status {response.status_code}\n")


def test_calibration():
    """Test calibration endpoint"""
    print("Testing Calibration Endpoint...")

    calibration_data = {
        "component_id": "test_sensor_1",
        "calibration_data": {
            "offset": 0.0,
            "scale": 1.0,
            "temperature_compensation": 0.001
        }
    }

    response = client.post("/api/hardware/calibrate", json=calibration_data)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Calibration success: {response_data.get('success')}")
        assert response_data["success"] == True
        print("PASS: Calibration test passed\n")
    else:
        print(f"Response: {response.json()}")
        print(f"FAIL: Calibration test failed with status {response.status_code}\n")


def test_monitoring():
    """Test monitoring start/stop endpoints"""
    print("Testing Monitoring Start/Stop Endpoints...")

    # Start monitoring
    response = client.post("/api/hardware/start-monitoring")
    print(f"Start monitoring response status: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Start monitoring success: {response_data.get('success')}")
        assert response_data["success"] == True

        # Stop monitoring
        response = client.post("/api/hardware/stop-monitoring")
        print(f"Stop monitoring response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Stop monitoring success: {response_data.get('success')}")
            assert response_data["success"] == True
            print("PASS: Monitoring test passed\n")
        else:
            print(f"Stop Response: {response.json()}")
            print(f"FAIL: Stop monitoring failed with status {response.status_code}\n")
    else:
        print(f"Start Response: {response.json()}")
        print(f"FAIL: Start monitoring failed with status {response.status_code}\n")


def test_component_status():
    """Test component status endpoint"""
    print("Testing Component Status Endpoint...")

    # This will fail because we haven't registered a real component that can provide status
    # But we can test the error handling
    response = client.get("/api/hardware/component-status?component_id=nonexistent_component")
    print(f"Response status: {response.status_code}")

    # We expect this to return a 404 since the component doesn't exist
    if response.status_code == 404:
        print("PASS: Component Status test passed (correctly returned 404 for nonexistent component)\n")
    else:
        print(f"Response: {response.json()}")
        print(f"Note: Component Status returned status {response.status_code}\n")


def run_api_tests():
    """Run all API tests"""
    print("Starting Hardware Integration API Tests\n")

    test_register_component()
    test_list_components()
    test_safety_status()
    test_emergency_stop()
    test_calibration()
    test_monitoring()
    test_component_status()

    print("All hardware integration API tests completed!")


if __name__ == "__main__":
    run_api_tests()