import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime
import threading


class QoSReliabilityPolicy(Enum):
    """ROS2 Quality of Service Reliability Policies"""
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class QoSDurabilityPolicy(Enum):
    """ROS2 Quality of Service Durability Policies"""
    VOLATILE = "volatile"
    TRANSIENT_LOCAL = "transient_local"


class QoSHistoryPolicy(Enum):
    """ROS2 Quality of Service History Policies"""
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"


class LifecycleState(Enum):
    """ROS2 Lifecycle Node States"""
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


@dataclass
class QoSProfile:
    """Quality of Service profile for ROS2 communications"""
    reliability: QoSReliabilityPolicy = QoSReliabilityPolicy.RELIABLE
    durability: QoSDurabilityPolicy = QoSDurabilityPolicy.VOLATILE
    history: QoSHistoryPolicy = QoSHistoryPolicy.KEEP_LAST
    depth: int = 10
    deadline: float = 0.0  # seconds
    lifespan: float = 0.0  # seconds
    liveliness_lease_duration: float = 0.0  # seconds


@dataclass
class ROS2NodeInfo:
    """Information about a ROS2 node"""
    node_id: str
    node_name: str
    namespace: str
    lifecycle_state: LifecycleState
    timestamp: datetime
    publishers: List[str]
    subscribers: List[str]
    services: List[str]
    actions: List[str]


@dataclass
class ROS2Parameter:
    """ROS2 Parameter"""
    name: str
    value: Any
    type: str
    description: str = ""
    read_only: bool = False


class ROS2Node:
    """Simulated ROS2 Node with lifecycle management"""

    def __init__(self, name: str, namespace: str = "/"):
        self.name = name
        self.namespace = namespace
        self.node_id = f"{namespace}{name}_{uuid.uuid4().hex[:8]}"
        self.lifecycle_state = LifecycleState.UNCONFIGURED
        self.parameters: Dict[str, ROS2Parameter] = {}
        self.publishers: Dict[str, 'ROS2Publisher'] = {}
        self.subscribers: Dict[str, 'ROS2Subscriber'] = {}
        self.services: Dict[str, 'ROS2Service'] = {}
        self.actions: Dict[str, 'ROS2ActionServer'] = {}
        self.timer_callbacks: List[Dict[str, Any]] = []
        self.timer_thread = None
        self.timer_running = False
        self.parameter_callbacks: List[Callable] = []

    def get_info(self) -> ROS2NodeInfo:
        """Get node information"""
        return ROS2NodeInfo(
            node_id=self.node_id,
            node_name=self.name,
            namespace=self.namespace,
            lifecycle_state=self.lifecycle_state,
            timestamp=datetime.now(),
            publishers=list(self.publishers.keys()),
            subscribers=list(self.subscribers.keys()),
            services=list(self.services.keys()),
            actions=list(self.actions.keys())
        )

    def configure(self) -> bool:
        """Configure the lifecycle node"""
        if self.lifecycle_state == LifecycleState.UNCONFIGURED:
            self.lifecycle_state = LifecycleState.INACTIVE
            return True
        return False

    def activate(self) -> bool:
        """Activate the lifecycle node"""
        if self.lifecycle_state == LifecycleState.INACTIVE:
            self.lifecycle_state = LifecycleState.ACTIVE
            return True
        return False

    def deactivate(self) -> bool:
        """Deactivate the lifecycle node"""
        if self.lifecycle_state == LifecycleState.ACTIVE:
            self.lifecycle_state = LifecycleState.INACTIVE
            return True
        return False

    def cleanup(self) -> bool:
        """Cleanup the lifecycle node"""
        if self.lifecycle_state in [LifecycleState.INACTIVE, LifecycleState.ACTIVE]:
            self.lifecycle_state = LifecycleState.UNCONFIGURED
            return True
        return False

    def shutdown(self) -> bool:
        """Shutdown the lifecycle node"""
        self.lifecycle_state = LifecycleState.FINALIZED
        return True

    def declare_parameter(self, name: str, default_value: Any = None,
                         description: str = "", read_only: bool = False) -> bool:
        """Declare a parameter for the node"""
        param_type = type(default_value).__name__ if default_value is not None else "unknown"
        self.parameters[name] = ROS2Parameter(
            name=name,
            value=default_value,
            type=param_type,
            description=description,
            read_only=read_only
        )
        return True

    def get_parameter(self, name: str) -> Optional[ROS2Parameter]:
        """Get a parameter value"""
        return self.parameters.get(name)

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value"""
        if name in self.parameters:
            param = self.parameters[name]
            if not param.read_only:
                param.value = value
                param.type = type(value).__name__

                # Trigger parameter callbacks
                for callback in self.parameter_callbacks:
                    try:
                        callback([param])
                    except Exception:
                        pass  # Continue with other callbacks

                return True
        return False

    def add_on_set_parameters_callback(self, callback: Callable):
        """Add a callback for parameter changes"""
        self.parameter_callbacks.append(callback)

    def create_publisher(self, topic: str, message_type: str, qos_profile: QoSProfile = None) -> 'ROS2Publisher':
        """Create a publisher for the node"""
        if qos_profile is None:
            qos_profile = QoSProfile()

        publisher = ROS2Publisher(topic, message_type, qos_profile)
        self.publishers[topic] = publisher
        return publisher

    def create_subscriber(self, topic: str, message_type: str, callback: Callable,
                         qos_profile: QoSProfile = None) -> 'ROS2Subscriber':
        """Create a subscriber for the node"""
        if qos_profile is None:
            qos_profile = QoSProfile()

        subscriber = ROS2Subscriber(topic, message_type, callback, qos_profile)
        self.subscribers[topic] = subscriber

        # Register with global message bus
        ros2_system.message_bus.subscribe(topic, callback)

        return subscriber

    def create_timer(self, period_sec: float, callback: Callable):
        """Create a timer that calls callback periodically"""
        timer_id = str(uuid.uuid4())
        timer_info = {
            "id": timer_id,
            "period": period_sec,
            "callback": callback,
            "next_call": time.time() + period_sec
        }
        self.timer_callbacks.append(timer_info)

        # Start timer thread if not already running
        if not self.timer_thread or not self.timer_thread.is_alive():
            self._start_timer_thread()

    def _start_timer_thread(self):
        """Start the timer execution thread"""
        if not self.timer_running:
            self.timer_running = True
            self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
            self.timer_thread.start()

    def _timer_loop(self):
        """Timer execution loop"""
        while self.timer_running:
            current_time = time.time()
            for timer in self.timer_callbacks:
                if current_time >= timer["next_call"]:
                    try:
                        timer["callback"]()
                    except Exception:
                        pass  # Continue with other timers
                    timer["next_call"] = current_time + timer["period"]
            time.sleep(0.01)  # Small delay to prevent busy waiting

    def destroy_node(self):
        """Clean up node resources"""
        self.timer_running = False
        if self.timer_thread:
            self.timer_thread.join(timeout=1.0)

        # Unsubscribe from all topics
        for topic in self.subscribers:
            ros2_system.message_bus.unsubscribe(topic, self.subscribers[topic].callback)


class ROS2Publisher:
    """ROS2 Publisher with QoS support"""

    def __init__(self, topic: str, message_type: str, qos_profile: QoSProfile):
        self.topic = topic
        self.message_type = message_type
        self.qos_profile = qos_profile
        self.sequence_number = 0
        self.last_publish_time = 0.0

    async def publish(self, message: Union[Dict[str, Any], str]):
        """Publish a message to the topic"""
        # Add metadata to message
        enriched_message = {
            "data": message,
            "timestamp": time.time(),
            "sequence": self.sequence_number,
            "publisher_id": id(self)
        }

        # Publish through message bus
        ros2_system.message_bus.publish(self.topic, enriched_message)

        self.sequence_number += 1
        self.last_publish_time = time.time()

    def get_info(self) -> Dict[str, Any]:
        """Get publisher information"""
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "qos_profile": {
                "reliability": self.qos_profile.reliability.value,
                "durability": self.qos_profile.durability.value,
                "history": self.qos_profile.history.value,
                "depth": self.qos_profile.depth
            },
            "sequence_number": self.sequence_number
        }


class ROS2Subscriber:
    """ROS2 Subscriber with QoS support"""

    def __init__(self, topic: str, message_type: str, callback: Callable, qos_profile: QoSProfile):
        self.topic = topic
        self.message_type = message_type
        self.callback = callback
        self.qos_profile = qos_profile
        self.message_count = 0
        self.last_message_time = 0.0

    def process_message(self, message: Dict[str, Any]):
        """Process an incoming message"""
        self.message_count += 1
        self.last_message_time = time.time()

        # Call the user callback with the actual data
        try:
            self.callback(message.get("data"))
        except Exception:
            pass  # Don't let callback errors affect message processing

    def get_info(self) -> Dict[str, Any]:
        """Get subscriber information"""
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "qos_profile": {
                "reliability": self.qos_profile.reliability.value,
                "durability": self.qos_profile.durability.value,
                "history": self.qos_profile.history.value,
                "depth": self.qos_profile.depth
            },
            "message_count": self.message_count
        }


class ROS2Service:
    """ROS2 Service Server"""

    def __init__(self, name: str, service_type: str, callback: Callable):
        self.name = name
        self.service_type = service_type
        self.callback = callback
        self.request_count = 0
        self.response_count = 0

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a service request"""
        self.request_count += 1
        try:
            response = self.callback(request)
            self.response_count += 1
            return {
                "success": True,
                "response": response,
                "request_id": self.request_count
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": self.request_count
            }


class ROS2ActionServer:
    """ROS2 Action Server"""

    def __init__(self, name: str, action_type: str, execute_callback: Callable):
        self.name = name
        self.action_type = action_type
        self.execute_callback = execute_callback
        self.active_goals: Dict[str, Dict[str, Any]] = {}
        self.completed_goals: Dict[str, Dict[str, Any]] = {}

    async def handle_goal(self, goal: Dict[str, Any]) -> Dict[str, str]:
        """Handle a new action goal"""
        goal_id = str(uuid.uuid4())

        goal_info = {
            "goal": goal,
            "status": "accepted",
            "start_time": time.time(),
            "execute_task": None
        }

        self.active_goals[goal_id] = goal_info

        # Execute the goal in the background
        goal_info["execute_task"] = asyncio.create_task(
            self._execute_goal(goal_id, goal)
        )

        return {
            "goal_id": goal_id,
            "status": "accepted"
        }

    async def _execute_goal(self, goal_id: str, goal: Dict[str, Any]):
        """Execute a goal in the background"""
        try:
            result = await self.execute_callback(goal)
            self.active_goals[goal_id]["status"] = "succeeded"

            # Move to completed goals
            completed_info = self.active_goals.pop(goal_id)
            completed_info["result"] = result
            completed_info["end_time"] = time.time()

            self.completed_goals[goal_id] = completed_info

        except Exception as e:
            self.active_goals[goal_id]["status"] = "failed"
            self.active_goals[goal_id]["error"] = str(e)

    def get_result(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed goal"""
        return self.completed_goals.get(goal_id)

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an active goal"""
        if goal_id in self.active_goals:
            goal_info = self.active_goals[goal_id]
            if goal_info["execute_task"]:
                goal_info["execute_task"].cancel()
            goal_info["status"] = "canceled"

            # Move to completed with cancellation
            completed_info = self.active_goals.pop(goal_id)
            completed_info["result"] = {"canceled": True}
            completed_info["end_time"] = time.time()

            self.completed_goals[goal_id] = completed_info
            return True
        return False


class MessageBus:
    """Message bus for ROS2-style communication"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history = 100

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(callback)
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
            except ValueError:
                pass  # Callback not found

    def publish(self, topic: str, message: Any):
        """Publish a message to a topic"""
        # Add to message history
        if topic not in self.message_history:
            self.message_history[topic] = []

        self.message_history[topic].append({
            "timestamp": time.time(),
            "message": message
        })

        if len(self.message_history[topic]) > self.max_history:
            self.message_history[topic].pop(0)

        # Notify all subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception:
                    pass  # Continue with other callbacks

    def get_topic_names_and_types(self) -> Dict[str, List[str]]:
        """Get all topics and their types"""
        topics = {}
        for topic, subscribers in self.subscribers.items():
            # In a real system, we'd determine types from message definitions
            topics[topic] = ["unknown_type"]  # Placeholder
        return topics


class ROS2System:
    """Main ROS2 System Manager"""

    def __init__(self):
        self.nodes: Dict[str, ROS2Node] = {}
        self.message_bus = MessageBus()
        self.parameters: Dict[str, ROS2Parameter] = {}
        self.global_parameter_callbacks: List[Callable] = []

    def create_node(self, name: str, namespace: str = "/") -> ROS2Node:
        """Create a new ROS2 node"""
        node = ROS2Node(name, namespace)
        self.nodes[node.node_id] = node
        return node

    def get_node(self, node_id: str) -> Optional[ROS2Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[ROS2NodeInfo]:
        """List all nodes"""
        return [node.get_info() for node in self.nodes.values()]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.destroy_node()  # Clean up node resources
            del self.nodes[node_id]
            return True
        return False

    def get_topic_names_and_types(self) -> Dict[str, List[str]]:
        """Get all topics and their types"""
        return self.message_bus.get_topic_names_and_types()

    def get_node_topics(self, node_id: str) -> Optional[Dict[str, List[str]]]:
        """Get topics associated with a node"""
        node = self.get_node(node_id)
        if node:
            return {
                "publishers": node.publishers.keys(),
                "subscribers": node.subscribers.keys()
            }
        return None

    def set_global_parameter(self, name: str, value: Any) -> bool:
        """Set a global parameter"""
        param_type = type(value).__name__
        self.parameters[name] = ROS2Parameter(
            name=name,
            value=value,
            type=param_type
        )

        # Trigger global parameter callbacks
        for callback in self.global_parameter_callbacks:
            try:
                callback([self.parameters[name]])
            except Exception:
                pass  # Continue with other callbacks

        return True

    def get_global_parameter(self, name: str) -> Optional[ROS2Parameter]:
        """Get a global parameter"""
        return self.parameters.get(name)

    def add_global_parameter_callback(self, callback: Callable):
        """Add a callback for global parameter changes"""
        self.global_parameter_callbacks.append(callback)

    def call_service(self, service_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a service across all nodes"""
        # Find the service in any node
        for node in self.nodes.values():
            if service_name in node.services:
                service = node.services[service_name]
                return asyncio.run(service.handle_request(request))

        return {
            "success": False,
            "error": f"Service '{service_name}' not found"
        }

    def send_action_goal(self, action_name: str, goal: Dict[str, Any]) -> Dict[str, str]:
        """Send a goal to an action server"""
        # Find the action server in any node
        for node in self.nodes.values():
            if action_name in node.actions:
                action_server = node.actions[action_name]
                return asyncio.run(action_server.handle_goal(goal))

        return {
            "error": f"Action server '{action_name}' not found"
        }


# Global ROS2 system instance
ros2_system = ROS2System()