#!/usr/bin/env python3
"""
Test script for project integration utilities module
"""
from backend.utils.project_integration_utils import (
    integration_manager, ComponentType, IntegrationStatus,
    MessageBus, ComponentManager, PerformanceProfiler, SystemValidator
)


def test_message_bus():
    """Test message bus functionality"""
    print("Testing Message Bus...")

    message_bus = MessageBus()

    # Test subscription and publishing
    received_messages = []

    def test_callback(message):
        received_messages.append(message)

    message_bus.subscribe("test_topic", test_callback)

    # Publish a message
    test_message = {"data": "test message", "id": 123}
    message_bus.publish("test_topic", test_message)

    print(f"Published message: {test_message}")
    print(f"Received messages: {received_messages}")

    assert len(received_messages) == 1, "Message should be received"
    assert received_messages[0] == test_message, "Received message should match published message"

    # Test message history
    history = message_bus.get_topic_history("test_topic")
    assert len(history) == 1, "History should contain the message"
    assert history[0]["message"] == test_message, "History message should match"

    print("PASS: Message Bus test completed\n")


def test_component_manager():
    """Test component manager functionality"""
    print("Testing Component Manager...")

    message_bus = MessageBus()
    comp_manager = ComponentManager(message_bus)

    # Register components
    success1 = comp_manager.register_component(
        "vision_component",
        "Vision System",
        ComponentType.PERCEPTION,
        ["sensor_input"]
    )

    success2 = comp_manager.register_component(
        "planning_component",
        "Path Planner",
        ComponentType.PLANNING,
        ["vision_component"]
    )

    print(f"Vision component registered: {success1}")
    print(f"Planning component registered: {success2}")

    assert success1 and success2, "Both components should register successfully"

    # Update component status
    comp_manager.update_component_status("vision_component", IntegrationStatus.OPERATIONAL)
    comp_manager.update_component_status("planning_component", IntegrationStatus.OPERATIONAL)

    # Update component resources
    comp_manager.update_component_resources("vision_component", {"cpu": 25.0, "memory": 40.0})

    # Get component info
    vision_info = comp_manager.get_component_info("vision_component")
    assert vision_info is not None, "Component should be retrievable"
    assert vision_info.status == IntegrationStatus.OPERATIONAL, "Status should be operational"
    assert vision_info.resources["cpu"] == 25.0, "CPU usage should be updated"

    # Add data flow
    comp_manager.add_data_flow("vision_component", "planning_component", "image_data", 30.0, 1000.0)
    print(f"Data flows count: {len(comp_manager.data_flows)}")

    assert len(comp_manager.data_flows) == 1, "Should have one data flow"

    print("PASS: Component Manager test completed\n")


def test_performance_profiler():
    """Test performance profiler functionality"""
    print("Testing Performance Profiler...")

    profiler = PerformanceProfiler()

    # Test profiling a simple function
    def test_function():
        # Simulate some work
        result = sum(i * i for i in range(1000))
        return result

    # Start and stop profiling
    profiler.start_profiling("test_function")
    result = test_function()
    profile_results = profiler.stop_profiling("test_function")

    print(f"Profile results keys: {list(profile_results.keys())}")
    print(f"Function result: {result}")

    assert "top_functions" in profile_results, "Profile should contain top functions"
    assert "stats" in profile_results, "Profile should contain stats"

    # Test bottleneck identification
    from backend.utils.project_integration_utils import IntegrationMetrics
    fake_metrics = IntegrationMetrics(
        total_components=5,
        operational_components=5,
        system_cpu=85.0,
        system_memory=90.0,
        network_usage=10.0,
        data_throughput=100.0,
        average_latency=10.0,
        error_rate=0.05,
        bottleneck_components=[],
        system_health_score=0.8
    )

    bottlenecks = profiler.identify_bottlenecks(fake_metrics)
    print(f"Identified bottlenecks: {bottlenecks}")

    # Should identify high CPU and memory usage as bottlenecks
    assert len(bottlenecks) >= 2, "Should identify CPU and memory bottlenecks"

    print("PASS: Performance Profiler test completed\n")


def test_system_validator():
    """Test system validator functionality"""
    print("Testing System Validator...")

    validator = SystemValidator()

    # Add a test validation rule
    def always_passes():
        return True

    def always_fails():
        return False

    validator.add_validation_rule("pass_test", always_passes, "This should pass")
    validator.add_validation_rule("fail_test", always_fails, "This should fail")

    # Run validation tests
    results = validator.run_validation_tests()

    print(f"Overall success: {results['overall_success']}")
    print(f"Test results: {results['test_results']}")
    print(f"Errors: {results['errors']}")

    # The overall result should be False because one test fails
    assert results["overall_success"] == False, "Overall success should be False due to failing test"
    assert results["test_results"]["pass_test"] == True, "Pass test should pass"
    assert results["test_results"]["fail_test"] == False, "Fail test should fail"
    assert len(results["errors"]) > 0, "Should have errors from failing test"

    print("PASS: System Validator test completed\n")


def test_integration_manager():
    """Test integration manager functionality"""
    print("Testing Integration Manager...")

    # Integrate several components
    success1 = integration_manager.integrate_component(
        "kinematics_module",
        "Kinematics Solver",
        ComponentType.PLANNING
    )

    success2 = integration_manager.integrate_component(
        "vision_module",
        "Computer Vision",
        ComponentType.PERCEPTION
    )

    success3 = integration_manager.integrate_component(
        "control_module",
        "Motion Controller",
        ComponentType.CONTROL
    )

    print(f"Components integrated: {success1}, {success2}, {success3}")

    assert success1 and success2 and success3, "All components should integrate successfully"

    # Setup data flows
    integration_manager.setup_data_flow("vision_module", "kinematics_module", "object_data", 10.0)
    integration_manager.setup_data_flow("kinematics_module", "control_module", "motion_plan", 50.0)

    # Start integration
    integration_manager.start_integration()
    print("Integration started")

    # Get system metrics
    metrics = integration_manager.get_system_metrics()
    print(f"System metrics - Total components: {metrics.total_components}")
    print(f"Operational components: {metrics.operational_components}")
    print(f"Health score: {metrics.system_health_score}")

    assert metrics.total_components == 3, "Should have 3 total components"
    assert metrics.operational_components == 3, "All components should be operational"

    # Run validation
    validation_results = integration_manager.run_system_validation()
    print(f"Validation success: {validation_results['overall_success']}")

    # Get health report
    health_report = integration_manager.get_system_health_report()
    print(f"Health report generated with {len(health_report['component_status'])} components")

    # Get data flow analysis
    flow_analysis = integration_manager.get_data_flow_analysis()
    print(f"Data flow analysis - Total flows: {flow_analysis['total_flows']}")
    print(f"Components involved: {len(flow_analysis['components_involved'])}")

    # Stop integration
    integration_manager.stop_integration()
    print("Integration stopped")

    assert flow_analysis["total_flows"] >= 2, "Should have at least 2 data flows"

    print("PASS: Integration Manager test completed\n")


def test_system_health_report():
    """Test system health report functionality"""
    print("Testing System Health Report...")

    # Integrate a few components
    integration_manager.integrate_component("test_comp_1", "Test Component 1", ComponentType.COMMUNICATION)
    integration_manager.integrate_component("test_comp_2", "Test Component 2", ComponentType.SENSING)

    # Start integration to get operational status
    integration_manager.start_integration()

    # Get health report
    report = integration_manager.get_system_health_report()

    print(f"Report timestamp: {report['timestamp']}")
    print(f"System metrics keys: {list(report['system_metrics'].keys())}")
    print(f"Component status count: {len(report['component_status'])}")

    # Verify structure of report
    assert "timestamp" in report, "Report should have timestamp"
    assert "system_metrics" in report, "Report should have system metrics"
    assert "component_status" in report, "Report should have component status"
    assert "validation_results" in report, "Report should have validation results"
    assert "bottlenecks" in report, "Report should have bottlenecks"
    assert "message_bus_stats" in report, "Report should have message bus stats"

    # Check that component status contains expected information
    for comp_id, comp_info in report['component_status'].items():
        assert "name" in comp_info, f"Component {comp_id} should have name"
        assert "type" in comp_info, f"Component {comp_id} should have type"
        assert "status" in comp_info, f"Component {comp_id} should have status"

    integration_manager.stop_integration()

    print("PASS: System Health Report test completed\n")


def run_all_tests():
    """Run all project integration utility tests"""
    print("Starting Project Integration Utilities Tests\n")

    test_message_bus()
    test_component_manager()
    test_performance_profiler()
    test_system_validator()
    test_integration_manager()
    test_system_health_report()

    print("All project integration utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()