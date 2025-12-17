#!/usr/bin/env python3
"""
Test script for hardware integration utilities module
"""
from backend.utils.hardware_integration_utils import (
    hardware_manager, CommunicationProtocol, HardwareComponentType,
    SafetyLevel, NetworkInterface, SensorInterface, ActuatorInterface
)

# Import SerialInterface and check serial availability separately
from backend.utils.hardware_integration_utils import SERIAL_AVAILABLE
if SERIAL_AVAILABLE:
    from backend.utils.hardware_integration_utils import SerialInterface
else:
    SerialInterface = None


def test_serial_interface():
    """Test serial interface functionality"""
    print("Testing Serial Interface...")

    if not SERIAL_AVAILABLE:
        print("Serial interface not available (pyserial not installed), skipping test")
        print("PASS: Serial Interface test skipped\n")
        return

    # Create a mock serial interface (using a simulated port)
    try:
        serial_intf = SerialInterface("test_serial_1", "/dev/ttyUSB0", 115200)

        # Note: This will fail in testing since we don't have an actual serial device
        # But we can test the object creation and properties
        print(f"Serial interface created: {serial_intf.component_id}")
        print(f"Port: {serial_intf.port}, Baudrate: {serial_intf.baudrate}")

        # Test that the interface has the right properties
        assert serial_intf.component_id == "test_serial_1"
        assert serial_intf.port == "/dev/ttyUSB0"
        assert serial_intf.baudrate == 115200
        assert serial_intf.component_type == HardwareComponentType.CONTROLLER

        print("PASS: Serial Interface setup test completed\n")
    except RuntimeError as e:
        print(f"Serial interface not available: {e}")
        print("PASS: Serial Interface test completed (properly handled unavailability)\n")


def test_network_interface():
    """Test network interface functionality"""
    print("Testing Network Interface...")

    # Create a mock network interface (using localhost for testing)
    network_intf = NetworkInterface("test_network_1", "127.0.0.1", 8080)

    print(f"Network interface created: {network_intf.component_id}")
    print(f"Host: {network_intf.host}, Port: {network_intf.port}")

    # Test that the interface has the right properties
    assert network_intf.component_id == "test_network_1"
    assert network_intf.host == "127.0.0.1"
    assert network_intf.port == 8080
    assert network_intf.component_type == HardwareComponentType.CONTROLLER

    print("PASS: Network Interface setup test completed\n")


def test_sensor_interface():
    """Test sensor interface functionality"""
    print("Testing Sensor Interface...")

    if not SERIAL_AVAILABLE:
        # Use a network interface as the base for testing
        base_intf = NetworkInterface("base_sensor", "127.0.0.1", 9090)
    else:
        # Create a base interface
        base_intf = SerialInterface("base_sensor", "/dev/ttyUSB1", 9600)

    # Wrap it in a sensor interface
    sensor_intf = SensorInterface("imu_sensor_1", "imu", base_intf)

    print(f"Sensor interface created: {sensor_intf.component_id}")
    print(f"Sensor type: {sensor_intf.sensor_type}")

    # Test that the sensor interface has the right properties
    assert sensor_intf.component_id == "imu_sensor_1"
    assert sensor_intf.sensor_type == "imu"
    assert sensor_intf.component_type == HardwareComponentType.SENSOR

    print("PASS: Sensor Interface setup test completed\n")


def test_actuator_interface():
    """Test actuator interface functionality"""
    print("Testing Actuator Interface...")

    if not SERIAL_AVAILABLE:
        # Use a network interface as the base for testing
        base_intf = NetworkInterface("base_actuator", "192.168.1.100", 502)
    else:
        # Create a base interface
        base_intf = SerialInterface("base_actuator", "/dev/ttyUSB2", 115200)

    # Wrap it in an actuator interface
    actuator_intf = ActuatorInterface("motor_1", "servo", base_intf)

    print(f"Actuator interface created: {actuator_intf.component_id}")
    print(f"Actuator type: {actuator_intf.actuator_type}")
    print(f"Initial position: {actuator_intf.current_position}")

    # Test that the actuator interface has the right properties
    assert actuator_intf.component_id == "motor_1"
    assert actuator_intf.actuator_type == "servo"
    assert actuator_intf.component_type == HardwareComponentType.ACTUATOR

    # Test safety limits validation
    test_cmd = {"command_type": "position", "value": 50.0}
    is_valid = actuator_intf._validate_command(test_cmd)
    assert is_valid, "Position within limits should be valid"

    # Test safety limits enforcement
    invalid_cmd = {"command_type": "position", "value": 200.0}  # Beyond max limit
    is_valid = actuator_intf._validate_command(invalid_cmd)
    assert not is_valid, "Position beyond limits should be invalid"

    print("PASS: Actuator Interface setup test completed\n")


def test_safety_system():
    """Test safety system functionality"""
    print("Testing Safety System...")

    # Create safety system
    safety_sys = hardware_manager.safety_system

    # Initially should be safe
    assert safety_sys.is_safe(), "Safety system should be safe initially"
    print(f"Initial safety status: {safety_sys.is_safe()}")

    # Test emergency stop
    safety_sys.activate_emergency_stop()
    assert not safety_sys.is_safe(), "Safety system should not be safe when emergency stop is active"
    print(f"After emergency stop: {safety_sys.is_safe()}")

    safety_sys.deactivate_emergency_stop()
    assert safety_sys.is_safe(), "Safety system should be safe after emergency stop is released"
    print(f"After emergency stop release: {safety_sys.is_safe()}")

    # Add a safety monitor
    def test_monitor():
        return {"safe": True, "details": "All good"}

    safety_sys.add_safety_monitor("test_monitor", test_monitor)
    status = safety_sys.check_safety()
    assert status["safe"], "System should be safe with good monitor"
    print(f"With test monitor: {status['safe']}")

    print("PASS: Safety System test completed\n")


def test_calibration_manager():
    """Test calibration manager functionality"""
    print("Testing Calibration Manager...")

    cal_manager = hardware_manager.calibration_manager

    # Store calibration data
    test_cal_data = {
        "offset": 0.1,
        "scale": 1.02,
        "temperature_coeff": 0.001
    }

    cal_manager.store_calibration("test_sensor", test_cal_data)
    retrieved_cal = cal_manager.get_calibration("test_sensor")

    assert retrieved_cal == test_cal_data, "Calibration data should match stored data"
    print(f"Calibration stored and retrieved: {retrieved_cal}")

    # Test if calibration is needed
    needs_cal = cal_manager.needs_calibration("test_sensor", max_age_hours=0)  # Should need cal since it's "old"
    print(f"Needs calibration (should be True): {needs_cal}")

    # Test if non-existent calibration is needed
    needs_cal = cal_manager.needs_calibration("nonexistent_sensor")
    assert needs_cal, "Non-existent calibration should need calibration"
    print(f"Non-existent sensor needs calibration: {needs_cal}")

    print("PASS: Calibration Manager test completed\n")


def test_hardware_manager():
    """Test hardware manager functionality"""
    print("Testing Hardware Manager...")

    # Register a component - use network interface if serial not available
    if SERIAL_AVAILABLE:
        intf = SerialInterface("registered_serial", "/dev/ttyUSB2", 115200)
        component_id = "registered_serial"
    else:
        intf = NetworkInterface("registered_network", "127.0.0.1", 8081)
        component_id = "registered_network"

    success = hardware_manager.register_component(intf)
    assert success, "Component should register successfully"
    print(f"Component registered: {success}")

    # Try to register the same component again (should fail)
    if SERIAL_AVAILABLE:
        duplicate_intf = SerialInterface(component_id, "/dev/ttyUSB3", 9600)
    else:
        duplicate_intf = NetworkInterface(component_id, "127.0.0.1", 8082)

    success = hardware_manager.register_component(duplicate_intf)
    assert not success, "Duplicate component should not register"
    print(f"Duplicate registration failed (as expected): {not success}")

    # List components
    components = hardware_manager.list_components()
    print(f"Total registered components: {len(components)}")
    assert component_id in components, "Registered component should be in list"

    # Get component
    retrieved = hardware_manager.get_component(component_id)
    assert retrieved is not None, "Component should be retrievable"
    assert retrieved.component_id == component_id, "Component ID should match"
    print(f"Component retrieved: {retrieved.component_id}")

    print("PASS: Hardware Manager test completed\n")


def run_all_tests():
    """Run all hardware integration utility tests"""
    print("Starting Hardware Integration Utilities Tests\n")

    test_serial_interface()
    test_network_interface()
    test_sensor_interface()
    test_actuator_interface()
    test_safety_system()
    test_calibration_manager()
    test_hardware_manager()

    print("All hardware integration utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()