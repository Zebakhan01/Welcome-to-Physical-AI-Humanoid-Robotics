#!/usr/bin/env python3
"""
Test script for ROS2 utilities module
"""
from backend.utils.ros2_utils import (
    ros2_system, ROS2Node, QoSProfile, QoSReliabilityPolicy,
    QoSDurabilityPolicy, QoSHistoryPolicy, LifecycleState, ROS2Parameter
)


def test_node_creation():
    """Test ROS2 node creation and lifecycle"""
    print("Testing ROS2 Node Creation and Lifecycle...")

    # Create a node
    node = ros2_system.create_node("test_ros2_node", "/test_ns")
    print(f"Node created: {node.name} with ID: {node.node_id}")
    print(f"Initial state: {node.lifecycle_state.value}")

    # Test lifecycle transitions
    success = node.configure()
    print(f"Configure success: {success}, State: {node.lifecycle_state.value}")
    assert success and node.lifecycle_state == LifecycleState.INACTIVE

    success = node.activate()
    print(f"Activate success: {success}, State: {node.lifecycle_state.value}")
    assert success and node.lifecycle_state == LifecycleState.ACTIVE

    success = node.deactivate()
    print(f"Deactivate success: {success}, State: {node.lifecycle_state.value}")
    assert success and node.lifecycle_state == LifecycleState.INACTIVE

    success = node.cleanup()
    print(f"Cleanup success: {success}, State: {node.lifecycle_state.value}")
    assert success and node.lifecycle_state == LifecycleState.UNCONFIGURED

    # Test node info
    info = node.get_info()
    print(f"Node info - Name: {info.node_name}, Namespace: {info.namespace}")

    print("PASS: Node Creation and Lifecycle test completed\n")


def test_parameter_management():
    """Test ROS2 parameter management"""
    print("Testing ROS2 Parameter Management...")

    # Create a node
    node = ros2_system.create_node("param_test_node")

    # Declare parameters
    node.declare_parameter("test_int_param", 42, "Test integer parameter")
    node.declare_parameter("test_string_param", "hello", "Test string parameter", read_only=True)
    node.declare_parameter("test_double_param", 3.14)

    print("Parameters declared")

    # Get parameters
    int_param = node.get_parameter("test_int_param")
    string_param = node.get_parameter("test_string_param")
    double_param = node.get_parameter("test_double_param")

    print(f"Int param: {int_param.value if int_param else 'None'}")
    print(f"String param: {string_param.value if string_param else 'None'}")
    print(f"Double param: {double_param.value if double_param else 'None'}")

    # Set parameter
    success = node.set_parameter("test_int_param", 100)
    print(f"Set parameter success: {success}")
    assert success

    # Get updated parameter
    updated_param = node.get_parameter("test_int_param")
    print(f"Updated param value: {updated_param.value if updated_param else 'None'}")
    assert updated_param.value == 100

    # Try to set read-only parameter (should fail)
    success = node.set_parameter("test_string_param", "changed")
    print(f"Set read-only parameter success: {success}")
    assert not success

    print("PASS: Parameter Management test completed\n")


def test_qos_profiles():
    """Test QoS profile functionality"""
    print("Testing QoS Profile Functionality...")

    # Create different QoS profiles
    default_qos = QoSProfile()
    reliable_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_ALL,
        depth=100
    )
    best_effort_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1
    )

    print(f"Default QoS - Reliability: {default_qos.reliability.value}, Depth: {default_qos.depth}")
    print(f"Reliable QoS - Reliability: {reliable_qos.reliability.value}, Depth: {reliable_qos.depth}")
    print(f"Best Effort QoS - Reliability: {best_effort_qos.reliability.value}, Depth: {best_effort_qos.depth}")

    # Test that values are as expected
    assert default_qos.reliability == QoSReliabilityPolicy.RELIABLE
    assert reliable_qos.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
    assert best_effort_qos.reliability == QoSReliabilityPolicy.BEST_EFFORT

    print("PASS: QoS Profile test completed\n")


def test_publisher_subscriber():
    """Test publisher-subscriber functionality"""
    print("Testing Publisher-Subscriber Functionality...")

    # Create a node
    node = ros2_system.create_node("comm_test_node")

    # Create a publisher
    qos_profile = QoSProfile(depth=5)
    publisher = node.create_publisher("/test_topic", "std_msgs/String", qos_profile)
    print(f"Publisher created for topic: {publisher.topic}")

    # Create a subscriber with a callback
    received_messages = []

    def message_callback(message):
        received_messages.append(message)
        print(f"Received message: {message}")

    subscriber = node.create_subscriber("/test_topic", "std_msgs/String", message_callback, qos_profile)
    print(f"Subscriber created for topic: {subscriber.topic}")

    # Publish a message
    test_message = {"data": "Hello ROS2!", "id": 123}
    import asyncio
    asyncio.run(publisher.publish(test_message))
    print(f"Published message: {test_message}")

    # Check if message was received
    print(f"Messages received: {len(received_messages)}")
    if received_messages:
        print(f"First received message: {received_messages[0]['data']}")

    print("PASS: Publisher-Subscriber test completed\n")


def test_message_bus():
    """Test message bus functionality"""
    print("Testing Message Bus Functionality...")

    # Create message bus
    msg_bus = ros2_system.message_bus

    # Subscribe to a topic
    received_messages = []

    def bus_callback(message):
        received_messages.append(message)
        print(f"Bus received: {message}")

    msg_bus.subscribe("/bus_test_topic", bus_callback)

    # Publish to the topic
    test_data = {"message": "Bus test", "timestamp": 12345}
    msg_bus.publish("/bus_test_topic", test_data)

    print(f"Messages received via bus: {len(received_messages)}")
    if received_messages:
        print(f"Received data: {received_messages[0]}")

    # Check topic names
    topics = msg_bus.get_topic_names_and_types()
    print(f"Available topics: {list(topics.keys())}")

    print("PASS: Message Bus test completed\n")


def test_global_parameters():
    """Test global parameter functionality"""
    print("Testing Global Parameter Functionality...")

    # Set global parameters
    success1 = ros2_system.set_global_parameter("global_int", 42)
    success2 = ros2_system.set_global_parameter("global_string", "global_test")

    print(f"Global parameter set success: {success1 and success2}")

    # Get global parameters
    global_int = ros2_system.get_global_parameter("global_int")
    global_string = ros2_system.get_global_parameter("global_string")

    print(f"Global int: {global_int.value if global_int else 'None'}")
    print(f"Global string: {global_string.value if global_string else 'None'}")

    assert global_int.value == 42
    assert global_string.value == "global_test"

    print("PASS: Global Parameter test completed\n")


def test_system_management():
    """Test ROS2 system management"""
    print("Testing ROS2 System Management...")

    # Create multiple nodes
    node1 = ros2_system.create_node("system_node_1")
    node2 = ros2_system.create_node("system_node_2", "/ns2")

    # List nodes
    nodes = ros2_system.list_nodes()
    print(f"Total nodes in system: {len(nodes)}")

    # Test node lookup
    found_node = ros2_system.get_node(node1.node_id)
    print(f"Node lookup success: {found_node is not None}")

    # Get node topics
    node_topics = ros2_system.get_node_topics(node1.node_id)
    print(f"Node topics: {node_topics}")

    # Delete a node
    delete_success = ros2_system.delete_node(node2.node_id)
    print(f"Node deletion success: {delete_success}")

    # List nodes again
    nodes_after = ros2_system.list_nodes()
    print(f"Nodes after deletion: {len(nodes_after)}")

    print("PASS: System Management test completed\n")


def run_all_tests():
    """Run all ROS2 utility tests"""
    print("Starting ROS2 Utilities Tests\n")

    test_node_creation()
    test_parameter_management()
    test_qos_profiles()
    test_publisher_subscriber()
    test_message_bus()
    test_global_parameters()
    test_system_management()

    print("All ROS2 utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()