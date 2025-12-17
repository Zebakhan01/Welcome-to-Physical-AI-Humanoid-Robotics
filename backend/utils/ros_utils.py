import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime
import base64
import pickle


class ROSMessageType(Enum):
    """Types of ROS messages"""
    STRING = "std_msgs/String"
    INT32 = "std_msgs/Int32"
    FLOAT32 = "std_msgs/Float32"
    BOOL = "std_msgs/Bool"
    HEADER = "std_msgs/Header"
    POSE = "geometry_msgs/Pose"
    TWIST = "geometry_msgs/Twist"
    POINT = "geometry_msgs/Point"
    QUATERNION = "geometry_msgs/Quaternion"
    JOINT_STATES = "sensor_msgs/JointState"
    LASER_SCAN = "sensor_msgs/LaserScan"
    IMAGE = "sensor_msgs/Image"


@dataclass
class ROSMessage:
    """Base ROS message structure"""
    _type: str
    _connection_header: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def serialize(self) -> bytes:
        """Serialize the message to bytes"""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes):
        """Deserialize message from bytes"""
        return pickle.loads(data)


@dataclass
class NodeInfo:
    """Information about a ROS node"""
    node_id: str
    node_name: str
    namespace: str
    pid: int
    machine: str
    uri: str
    connections: List[str]
    timestamp: float


@dataclass
class TopicInfo:
    """Information about a ROS topic"""
    name: str
    message_type: str
    publishers: List[str]  # List of node names
    subscribers: List[str]  # List of node names
    connections: int
    timestamp: float


@dataclass
class ServiceInfo:
    """Information about a ROS service"""
    name: str
    service_type: str
    provider: str  # Node name providing the service
    uri: str
    timestamp: float


@dataclass
class Parameter:
    """ROS parameter"""
    name: str
    value: Any
    type: str  # 'int', 'float', 'string', 'bool', 'list', 'dict'
    timestamp: float


class ROSNode:
    """Simulated ROS Node"""

    def __init__(self, name: str, namespace: str = "/"):
        self.name = name
        self.namespace = namespace
        self.node_id = f"{namespace}{name}_{uuid.uuid4().hex[:8]}"
        self.pid = id(self)  # Simulate process ID
        self.machine = "localhost"
        self.uri = f"http://localhost:12345/{self.node_id}"
        self.active = True
        self.subscribers = {}  # topic -> callback
        self.publishers = {}   # topic -> message queue
        self.services = {}     # service_name -> service handler
        self.actions = {}      # action_name -> action server
        self.parameters = {}   # parameter_name -> value

    def get_info(self) -> NodeInfo:
        """Get node information"""
        return NodeInfo(
            node_id=self.node_id,
            node_name=self.name,
            namespace=self.namespace,
            pid=self.pid,
            machine=self.machine,
            uri=self.uri,
            connections=len(self.subscribers) + len(self.publishers),
            timestamp=time.time()
        )


class ROSService:
    """Simulated ROS Service"""

    def __init__(self, name: str, service_type: str, handler: Callable):
        self.name = name
        self.service_type = service_type
        self.handler = handler
        self.active = True
        self.timestamp = time.time()

    async def call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the service"""
        return await self.handler(request) if asyncio.iscoroutinefunction(self.handler) else self.handler(request)


class ROSActionServer:
    """Simulated ROS Action Server"""

    def __init__(self, name: str, action_type: str, execute_callback: Callable):
        self.name = name
        self.action_type = action_type
        self.execute_callback = execute_callback
        self.active = True
        self.goals = {}  # goal_id -> goal_state
        self.feedback_publishers = {}
        self.result_publishers = {}
        self.timestamp = time.time()

    async def send_goal(self, goal: Dict[str, Any]) -> str:
        """Send a goal to the action server"""
        goal_id = str(uuid.uuid4())
        self.goals[goal_id] = {
            "goal": goal,
            "status": "ACTIVE",
            "timestamp": time.time()
        }
        # Start execution in background
        asyncio.create_task(self._execute_goal(goal_id, goal))
        return goal_id

    async def _execute_goal(self, goal_id: str, goal: Dict[str, Any]):
        """Execute the goal in background"""
        try:
            # Call the execute callback
            result = await self.execute_callback(goal) if asyncio.iscoroutinefunction(self.execute_callback) else self.execute_callback(goal)
            self.goals[goal_id]["status"] = "SUCCESS"
            self.goals[goal_id]["result"] = result
        except Exception as e:
            self.goals[goal_id]["status"] = "ABORTED"
            self.goals[goal_id]["error"] = str(e)


class ROSParameterServer:
    """Simulated ROS Parameter Server"""

    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.callbacks: List[Callable] = []

    def set_parameter(self, name: str, value: Any):
        """Set a parameter"""
        param_type = self._get_type(value)
        param = Parameter(
            name=name,
            value=value,
            type=param_type,
            timestamp=time.time()
        )
        self.parameters[name] = param
        self._notify_callbacks(name, value)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter"""
        param = self.parameters.get(name)
        return param.value if param else default

    def has_parameter(self, name: str) -> bool:
        """Check if parameter exists"""
        return name in self.parameters

    def delete_parameter(self, name: str):
        """Delete a parameter"""
        if name in self.parameters:
            del self.parameters[name]

    def list_parameters(self) -> List[str]:
        """List all parameter names"""
        return list(self.parameters.keys())

    def add_change_callback(self, callback: Callable):
        """Add callback for parameter changes"""
        self.callbacks.append(callback)

    def _get_type(self, value: Any) -> str:
        """Get parameter type as string"""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        else:
            return "unknown"

    def _notify_callbacks(self, name: str, value: Any):
        """Notify callbacks of parameter change"""
        for callback in self.callbacks:
            try:
                callback(name, value)
            except Exception:
                pass  # Ignore callback errors


class ROSServiceRegistry:
    """Registry for ROS services"""

    def __init__(self):
        self.services: Dict[str, ROSService] = {}

    def register_service(self, name: str, service_type: str, handler: Callable) -> ROSService:
        """Register a new service"""
        service = ROSService(name, service_type, handler)
        self.services[name] = service
        return service

    def get_service(self, name: str) -> Optional[ROSService]:
        """Get a service by name"""
        return self.services.get(name)

    async def call_service(self, name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a service by name"""
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Service '{name}' not found")
        return await service.call(request)

    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self.services.keys())


class ROSActionRegistry:
    """Registry for ROS actions"""

    def __init__(self):
        self.actions: Dict[str, ROSActionServer] = {}

    def register_action(self, name: str, action_type: str, execute_callback: Callable) -> ROSActionServer:
        """Register a new action server"""
        action_server = ROSActionServer(name, action_type, execute_callback)
        self.actions[name] = action_server
        return action_server

    def get_action(self, name: str) -> Optional[ROSActionServer]:
        """Get an action server by name"""
        return self.actions.get(name)

    def send_goal(self, name: str, goal: Dict[str, Any]) -> str:
        """Send a goal to an action server"""
        action_server = self.get_action(name)
        if not action_server:
            raise ValueError(f"Action server '{name}' not found")
        return asyncio.run(action_server.send_goal(goal))

    def list_actions(self) -> List[str]:
        """List all registered action servers"""
        return list(self.actions.keys())


class ROSSubscriber:
    """ROS Subscriber"""

    def __init__(self, topic: str, message_type: str, callback: Callable):
        self.topic = topic
        self.message_type = message_type
        self.callback = callback
        self.active = True
        self.queue_size = 10
        self.timestamp = time.time()

    def receive_message(self, message: ROSMessage):
        """Receive and process a message"""
        if self.active:
            # Call the callback with the message data
            if asyncio.iscoroutinefunction(self.callback):
                asyncio.run(self.callback(message))
            else:
                self.callback(message)


class ROSPublisher:
    """ROS Publisher"""

    def __init__(self, topic: str, message_type: str, queue_size: int = 10):
        self.topic = topic
        self.message_type = message_type
        self.queue_size = queue_size
        self.active = True
        self.message_queue = asyncio.Queue(maxsize=queue_size)
        self.subscribers = []  # List of subscriber callbacks
        self.seq = 0
        self.timestamp = time.time()

    async def publish(self, message: Union[ROSMessage, Dict[str, Any]]):
        """Publish a message"""
        if not self.active:
            return

        # If it's a dict, convert to ROSMessage
        if isinstance(message, dict):
            ros_msg = ROSMessage(_type=self.message_type)
            for key, value in message.items():
                setattr(ros_msg, key, value)
        else:
            ros_msg = message

        # Update sequence number
        ros_msg.seq = self.seq
        self.seq += 1

        # Notify all subscribers
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ros_msg)
                else:
                    callback(ros_msg)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")

    def add_subscriber(self, callback: Callable):
        """Add a subscriber callback"""
        self.subscribers.append(callback)

    def remove_subscriber(self, callback: Callable):
        """Remove a subscriber callback"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class ROSSystem:
    """Main ROS System Simulator"""

    def __init__(self):
        self.nodes: Dict[str, ROSNode] = {}
        self.publishers: Dict[str, ROSPublisher] = {}
        self.subscribers: Dict[str, ROSSubscriber] = {}
        self.topics: Dict[str, TopicInfo] = {}
        self.services = ROSServiceRegistry()
        self.actions = ROSActionRegistry()
        self.parameters = ROSParameterServer()
        self.master_uri = "http://localhost:11311"
        self.active = True
        self.timestamp = time.time()

    def create_node(self, name: str, namespace: str = "/") -> ROSNode:
        """Create a new ROS node"""
        node = ROSNode(name, namespace)
        self.nodes[node.node_id] = node
        return node

    def get_node(self, node_id: str) -> Optional[ROSNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[str]:
        """List all node IDs"""
        return list(self.nodes.keys())

    def create_publisher(self, topic: str, message_type: str, queue_size: int = 10) -> ROSPublisher:
        """Create a publisher for a topic"""
        if topic not in self.publishers:
            publisher = ROSPublisher(topic, message_type, queue_size)
            self.publishers[topic] = publisher

            # Update topic info
            self.topics[topic] = TopicInfo(
                name=topic,
                message_type=message_type,
                publishers=[],
                subscribers=[],
                connections=0,
                timestamp=time.time()
            )

        return self.publishers[topic]

    def create_subscriber(self, topic: str, message_type: str, callback: Callable) -> ROSSubscriber:
        """Create a subscriber for a topic"""
        if topic not in self.subscribers:
            subscriber = ROSSubscriber(topic, message_type, callback)
            self.subscribers[topic] = subscriber

            # Update topic info
            if topic not in self.topics:
                self.topics[topic] = TopicInfo(
                    name=topic,
                    message_type=message_type,
                    publishers=[],
                    subscribers=[],
                    connections=0,
                    timestamp=time.time()
                )

            # Connect publisher and subscriber
            if topic in self.publishers:
                self.publishers[topic].add_subscriber(callback)

        return self.subscribers[topic]

    def get_publisher(self, topic: str) -> Optional[ROSPublisher]:
        """Get a publisher for a topic"""
        return self.publishers.get(topic)

    def get_subscriber(self, topic: str) -> Optional[ROSSubscriber]:
        """Get a subscriber for a topic"""
        return self.subscribers.get(topic)

    def list_topics(self) -> List[str]:
        """List all topics"""
        return list(self.topics.keys())

    def get_topic_info(self, topic: str) -> Optional[TopicInfo]:
        """Get information about a topic"""
        return self.topics.get(topic)

    def register_service(self, name: str, service_type: str, handler: Callable) -> ROSService:
        """Register a service"""
        return self.services.register_service(name, service_type, handler)

    async def call_service(self, name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a service"""
        return await self.services.call_service(name, request)

    def register_action(self, name: str, action_type: str, execute_callback: Callable) -> ROSActionServer:
        """Register an action server"""
        return self.actions.register_action(name, action_type, execute_callback)

    def send_action_goal(self, name: str, goal: Dict[str, Any]) -> str:
        """Send a goal to an action server"""
        return self.actions.send_goal(name, goal)

    def set_parameter(self, name: str, value: Any):
        """Set a parameter"""
        self.parameters.set_parameter(name, value)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter"""
        return self.parameters.get_parameter(name, default)

    def has_parameter(self, name: str) -> bool:
        """Check if parameter exists"""
        return self.parameters.has_parameter(name)

    def list_parameters(self) -> List[str]:
        """List all parameters"""
        return self.parameters.list_parameters()

    async def spin_once(self):
        """Process one iteration of the ROS event loop"""
        # This would handle message processing, service calls, etc.
        # For simulation, we'll just sleep briefly
        await asyncio.sleep(0.01)

    async def spin(self):
        """Run the ROS event loop"""
        while self.active:
            await self.spin_once()

    def shutdown(self):
        """Shutdown the ROS system"""
        self.active = False
        for node in self.nodes.values():
            node.active = False


class ROSMessageFactory:
    """Factory for creating ROS messages"""

    @staticmethod
    def create_string_message(data: str) -> ROSMessage:
        """Create a String message"""
        msg = ROSMessage(_type=ROSMessageType.STRING.value)
        msg.data = data
        return msg

    @staticmethod
    def create_int32_message(data: int) -> ROSMessage:
        """Create an Int32 message"""
        msg = ROSMessage(_type=ROSMessageType.INT32.value)
        msg.data = data
        return msg

    @staticmethod
    def create_float32_message(data: float) -> ROSMessage:
        """Create a Float32 message"""
        msg = ROSMessage(_type=ROSMessageType.FLOAT32.value)
        msg.data = data
        return msg

    @staticmethod
    def create_bool_message(data: bool) -> ROSMessage:
        """Create a Bool message"""
        msg = ROSMessage(_type=ROSMessageType.BOOL.value)
        msg.data = data
        return msg

    @staticmethod
    def create_pose_message(position: Dict[str, float], orientation: Dict[str, float]) -> ROSMessage:
        """Create a Pose message"""
        msg = ROSMessage(_type=ROSMessageType.POSE.value)
        msg.position = position  # {"x": 0.0, "y": 0.0, "z": 0.0}
        msg.orientation = orientation  # {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        return msg

    @staticmethod
    def create_twist_message(linear: Dict[str, float], angular: Dict[str, float]) -> ROSMessage:
        """Create a Twist message"""
        msg = ROSMessage(_type=ROSMessageType.TWIST.value)
        msg.linear = linear  # {"x": 0.0, "y": 0.0, "z": 0.0}
        msg.angular = angular  # {"x": 0.0, "y": 0.0, "z": 0.0}
        return msg

    @staticmethod
    def create_joint_states_message(name: List[str], position: List[float],
                                  velocity: List[float], effort: List[float]) -> ROSMessage:
        """Create a JointState message"""
        msg = ROSMessage(_type=ROSMessageType.JOINT_STATES.value)
        msg.name = name
        msg.position = position
        msg.velocity = velocity
        msg.effort = effort
        return msg


class ROSBagSimulator:
    """Simulates ROS bag functionality for recording/replaying messages"""

    def __init__(self):
        self.recordings: Dict[str, List[Dict[str, Any]]] = {}
        self.is_recording = False
        self.current_recording = []

    def start_recording(self, topic: str):
        """Start recording messages on a topic"""
        self.is_recording = True
        self.current_recording = []
        self.recordings[topic] = self.current_recording

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.current_recording = []

    def record_message(self, topic: str, message: ROSMessage):
        """Record a message if recording is active for this topic"""
        if self.is_recording and topic in self.recordings:
            self.recordings[topic].append({
                "timestamp": time.time(),
                "topic": topic,
                "message": message.serialize()
            })

    def get_recorded_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Get recorded messages for a topic"""
        return self.recordings.get(topic, [])

    def play_back(self, topic: str, callback: Callable):
        """Play back recorded messages to a callback"""
        messages = self.get_recorded_messages(topic)
        for record in messages:
            msg = ROSMessage.deserialize(record["message"])
            callback(msg)
            time.sleep(0.1)  # Simulate timing


# Global ROS system instance
ros_system = ROSSystem()