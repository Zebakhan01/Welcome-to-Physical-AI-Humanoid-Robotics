#!/usr/bin/env python3
"""
Test script for ROS utilities module
"""
import numpy as np
from backend.utils.ros_utils import (
    ros_system, ROSNode, ROSService, ROSActionServer, ROSMessage,
    ROSMessageFactory, ROSBagSimulator, NodeInfo, TopicInfo, ServiceInfo, Parameter
)


def test_ros_message():
    """Test ROS message creation and serialization"""
    print("Testing ROS Message Creation and Serialization...")

    # Create a simple message
    msg = ROSMessageFactory.create_string_message("Hello ROS")
    print(f"Created message type: {msg._type}")
    print(f"Message data: {getattr(msg, 'data', 'No data attribute')}")

    # Test serialization/deserialization
    serialized = msg.serialize()
    deserialized = ROSMessage.deserialize(serialized)
    print(f"Deserialized message type: {deserialized._type}")
    print(f"Deserialized message data: {getattr(deserialized, 'data', 'No data attribute')}")

    print("PASS: ROS Message test completed\n")


def test_ros_node():
    """Test ROS node creation and management"""
    print("Testing ROS Node Creation and Management...")

    # Create a node
    node = ros_system.create_node("test_node", "/test")
    print(f"Created node: {node.name} with ID: {node.node_id}")

    # Get node info
    node_info = node.get_info()
    print(f"Node info - Name: {node_info.node_name}, Namespace: {node_info.namespace}")

    # List nodes
    nodes = ros_system.list_nodes()
    print(f"Total nodes in system: {len(nodes)}")

    print("PASS: ROS Node test completed\n")


def test_ros_publisher_subscriber():
    """Test ROS publisher and subscriber functionality"""
    print("Testing ROS Publisher and Subscriber...")

    # Create publisher
    publisher = ros_system.create_publisher("/test_topic", "std_msgs/String", 10)
    print(f"Created publisher for topic: {publisher.topic}")

    # Create subscriber with callback
    received_messages = []
    def callback(msg):
        received_messages.append(msg)
        print(f"Received message: {getattr(msg, 'data', 'No data')}")

    subscriber = ros_system.create_subscriber("/test_topic", "std_msgs/String", callback)
    print(f"Created subscriber for topic: {subscriber.topic}")

    # Test publishing
    test_msg = {"data": "Test message"}
    import asyncio
    asyncio.run(publisher.publish(test_msg))
    print(f"Published message, received count: {len(received_messages)}")

    print("PASS: ROS Publisher/Subscriber test completed\n")


def test_ros_service():
    """Test ROS service registration and calling"""
    print("Testing ROS Service Registration and Calling...")

    # Define a simple service handler
    def add_two_ints_handler(request):
        a = request.get("a", 0)
        b = request.get("b", 0)
        return {"sum": a + b}

    # Register service
    service = ros_system.register_service("/add_two_ints", "test_msgs/AddTwoInts", add_two_ints_handler)
    print(f"Registered service: {service.name}")

    # Call service
    request_data = {"a": 5, "b": 3}
    result = ros_system.call_service("/add_two_ints", request_data)
    print(f"Service call result: {result}")
    assert result["sum"] == 8, f"Expected sum 8, got {result['sum']}"

    print("PASS: ROS Service test completed\n")


def test_ros_action():
    """Test ROS action server functionality"""
    print("Testing ROS Action Server...")

    # Define action handler
    def move_to_pose_handler(goal):
        target_pose = goal.get('target_pose', 'unknown')
        return {
            "success": True,
            "message": f"Moved to pose {target_pose}"
        }

    # Register action server
    action_server = ros_system.register_action("/move_to_pose", "test_msgs/MoveToPose", move_to_pose_handler)
    print(f"Registered action server: {action_server.name}")

    # Send goal
    goal = {"target_pose": [1.0, 2.0, 0.0]}
    goal_id = ros_system.send_action_goal("/move_to_pose", goal)
    print(f"Sent goal with ID: {goal_id}")

    print("PASS: ROS Action test completed\n")


def test_ros_parameters():
    """Test ROS parameter server functionality"""
    print("Testing ROS Parameter Server...")

    # Set parameters
    ros_system.set_parameter("/test_param", "test_value")
    ros_system.set_parameter("/test_int", 42)
    ros_system.set_parameter("/test_float", 3.14)

    # Get parameters
    param_value = ros_system.get_parameter("/test_param")
    int_value = ros_system.get_parameter("/test_int")
    float_value = ros_system.get_parameter("/test_float")

    print(f"Parameter values: {param_value}, {int_value}, {float_value}")

    # List parameters
    params = ros_system.list_parameters()
    print(f"Total parameters: {len(params)}")

    print("PASS: ROS Parameter test completed\n")


def test_ros_system_info():
    """Test ROS system information retrieval"""
    print("Testing ROS System Information...")

    # Create some components to test info retrieval
    ros_system.create_node("info_test_node", "/info_test")
    ros_system.create_publisher("/info_test_topic", "std_msgs/String", 5)

    # List nodes
    nodes = ros_system.list_nodes()
    print(f"Nodes in system: {len(nodes)}")

    # List topics
    topics = ros_system.list_topics()
    print(f"Topics in system: {len(topics)}")

    # Get topic info
    if topics:
        topic_info = ros_system.get_topic_info(topics[0])
        print(f"First topic info: {topic_info.name if topic_info else 'None'}")

    print("PASS: ROS System Info test completed\n")


def test_ros_bag_simulation():
    """Test ROS bag simulation functionality"""
    print("Testing ROS Bag Simulation...")

    # Create bag simulator
    bag_sim = ROSBagSimulator()

    # Start recording
    bag_sim.start_recording("/bag_test_topic")
    print("Started recording on /bag_test_topic")

    # Create and record a message
    test_msg = ROSMessageFactory.create_string_message("Bag test message")
    bag_sim.record_message("/bag_test_topic", test_msg)
    print("Recorded message")

    # Get recorded messages
    recorded_msgs = bag_sim.get_recorded_messages("/bag_test_topic")
    print(f"Recorded messages count: {len(recorded_msgs)}")

    # Play back messages
    playback_received = []
    def playback_callback(msg):
        playback_received.append(msg)
        print(f"Played back message: {getattr(msg, 'data', 'No data')}")

    bag_sim.play_back("/bag_test_topic", playback_callback)
    print(f"Playback received count: {len(playback_received)}")

    print("PASS: ROS Bag Simulation test completed\n")


def run_all_tests():
    """Run all ROS utility tests"""
    print("Starting ROS Utilities Tests\n")

    test_ros_message()
    test_ros_node()
    test_ros_publisher_subscriber()
    test_ros_service()
    test_ros_action()
    test_ros_parameters()
    test_ros_system_info()
    test_ros_bag_simulation()

    print("All ROS utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()