from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from backend.utils.ros2_utils import (
    ros2_system, ROS2Node, QoSProfile, QoSReliabilityPolicy,
    QoSDurabilityPolicy, QoSHistoryPolicy, LifecycleState, ROS2Parameter
)
from backend.utils.logger import logger


router = APIRouter()


class CreateNodeRequest(BaseModel):
    name: str
    namespace: str = "/"


class CreateNodeResponse(BaseModel):
    node_id: str
    node_name: str
    success: bool


class LifecycleTransitionRequest(BaseModel):
    node_id: str
    action: str  # configure, activate, deactivate, cleanup, shutdown


class LifecycleTransitionResponse(BaseModel):
    success: bool
    message: str


class CreatePublisherRequest(BaseModel):
    node_id: str
    topic: str
    message_type: str
    qos_profile: Optional[Dict[str, Any]] = None


class CreatePublisherResponse(BaseModel):
    success: bool
    message: str


class CreateSubscriberRequest(BaseModel):
    node_id: str
    topic: str
    message_type: str
    qos_profile: Optional[Dict[str, Any]] = None


class CreateSubscriberResponse(BaseModel):
    success: bool
    message: str


class PublishMessageRequest(BaseModel):
    topic: str
    message: Dict[str, Any]


class PublishMessageResponse(BaseModel):
    success: bool
    message: str


class DeclareParameterRequest(BaseModel):
    node_id: str
    name: str
    default_value: Any = None
    description: str = ""
    read_only: bool = False


class DeclareParameterResponse(BaseModel):
    success: bool
    message: str


class SetParameterRequest(BaseModel):
    node_id: str
    name: str
    value: Any


class SetParameterResponse(BaseModel):
    success: bool
    message: str


class GetParameterResponse(BaseModel):
    name: str
    value: Any
    type: str
    success: bool


class ListNodesResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    success: bool


class GetNodeInfoResponse(BaseModel):
    node_id: str
    node_name: str
    namespace: str
    lifecycle_state: str
    publishers: List[str]
    subscribers: List[str]
    services: List[str]
    actions: List[str]
    success: bool


class ListNodesResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    success: bool


class ListTopicsResponse(BaseModel):
    topics: Dict[str, List[str]]  # topic -> [types]
    success: bool


class CallServiceRequest(BaseModel):
    service_name: str
    request_data: Dict[str, Any]


class CallServiceResponse(BaseModel):
    success: bool
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SendActionGoalRequest(BaseModel):
    action_name: str
    goal: Dict[str, Any]


class SendActionGoalResponse(BaseModel):
    success: bool
    goal_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


@router.post("/create-node", response_model=CreateNodeResponse)
async def create_node(request: CreateNodeRequest):
    """
    Create a new ROS2 node
    """
    try:
        node = ros2_system.create_node(request.name, request.namespace)

        response = CreateNodeResponse(
            node_id=node.node_id,
            node_name=node.name,
            success=True
        )

        logger.info(f"Node created: {node.name} with ID {node.node_id}")

        return response

    except Exception as e:
        logger.error(f"Error creating node: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating node: {str(e)}")


@router.post("/lifecycle-transition", response_model=LifecycleTransitionResponse)
async def lifecycle_transition(request: LifecycleTransitionRequest):
    """
    Perform a lifecycle transition on a node
    """
    try:
        node = ros2_system.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")

        success = False
        message = ""

        if request.action == "configure":
            success = node.configure()
            message = f"Node {request.node_id} configured" if success else f"Failed to configure node {request.node_id}"
        elif request.action == "activate":
            success = node.activate()
            message = f"Node {request.node_id} activated" if success else f"Failed to activate node {request.node_id}"
        elif request.action == "deactivate":
            success = node.deactivate()
            message = f"Node {request.node_id} deactivated" if success else f"Failed to deactivate node {request.node_id}"
        elif request.action == "cleanup":
            success = node.cleanup()
            message = f"Node {request.node_id} cleaned up" if success else f"Failed to clean up node {request.node_id}"
        elif request.action == "shutdown":
            success = node.shutdown()
            message = f"Node {request.node_id} shut down" if success else f"Failed to shut down node {request.node_id}"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

        response = LifecycleTransitionResponse(
            success=success,
            message=message
        )

        logger.info(f"Lifecycle transition: {request.action} for node {request.node_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error performing lifecycle transition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing lifecycle transition: {str(e)}")


@router.post("/create-publisher", response_model=CreatePublisherResponse)
async def create_publisher(request: CreatePublisherRequest):
    """
    Create a publisher for a node
    """
    try:
        node = ros2_system.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")

        # Create QoS profile if provided
        qos_profile = None
        if request.qos_profile:
            reliability = QoSReliabilityPolicy(request.qos_profile.get("reliability", "reliable"))
            durability = QoSDurabilityPolicy(request.qos_profile.get("durability", "volatile"))
            history = QoSHistoryPolicy(request.qos_profile.get("history", "keep_last"))

            qos_profile = QoSProfile(
                reliability=reliability,
                durability=durability,
                history=history,
                depth=request.qos_profile.get("depth", 10)
            )

        publisher = node.create_publisher(request.topic, request.message_type, qos_profile)

        response = CreatePublisherResponse(
            success=True,
            message=f"Publisher created for topic {request.topic} on node {request.node_id}"
        )

        logger.info(f"Publisher created: {request.topic} on node {request.node_id}")

        return response

    except Exception as e:
        logger.error(f"Error creating publisher: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating publisher: {str(e)}")


@router.post("/create-subscriber", response_model=CreateSubscriberResponse)
async def create_subscriber(request: CreateSubscriberRequest):
    """
    Create a subscriber for a node
    """
    try:
        node = ros2_system.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")

        # Create a dummy callback for the subscriber
        def dummy_callback(msg):
            logger.info(f"Received message on {request.topic}: {msg}")

        # Create QoS profile if provided
        qos_profile = None
        if request.qos_profile:
            reliability = QoSReliabilityPolicy(request.qos_profile.get("reliability", "reliable"))
            durability = QoSDurabilityPolicy(request.qos_profile.get("durability", "volatile"))
            history = QoSHistoryPolicy(request.qos_profile.get("history", "keep_last"))

            qos_profile = QoSProfile(
                reliability=reliability,
                durability=durability,
                history=history,
                depth=request.qos_profile.get("depth", 10)
            )

        subscriber = node.create_subscriber(request.topic, request.message_type, dummy_callback, qos_profile)

        response = CreateSubscriberResponse(
            success=True,
            message=f"Subscriber created for topic {request.topic} on node {request.node_id}"
        )

        logger.info(f"Subscriber created: {request.topic} on node {request.node_id}")

        return response

    except Exception as e:
        logger.error(f"Error creating subscriber: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating subscriber: {str(e)}")


@router.post("/publish", response_model=PublishMessageResponse)
async def publish_message(request: PublishMessageRequest):
    """
    Publish a message to a topic
    """
    try:
        # Use the message bus directly to publish
        ros2_system.message_bus.publish(request.topic, request.message)

        response = PublishMessageResponse(
            success=True,
            message=f"Published message to topic {request.topic}"
        )

        logger.info(f"Message published to topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error publishing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error publishing message: {str(e)}")


@router.post("/declare-parameter", response_model=DeclareParameterResponse)
async def declare_parameter(request: DeclareParameterRequest):
    """
    Declare a parameter for a node
    """
    try:
        node = ros2_system.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")

        success = node.declare_parameter(
            request.name,
            request.default_value,
            request.description,
            request.read_only
        )

        if success:
            response = DeclareParameterResponse(
                success=True,
                message=f"Parameter {request.name} declared on node {request.node_id}"
            )
        else:
            response = DeclareParameterResponse(
                success=False,
                message=f"Failed to declare parameter {request.name} on node {request.node_id}"
            )

        logger.info(f"Parameter declared: {request.name} on node {request.node_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error declaring parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error declaring parameter: {str(e)}")


@router.post("/set-parameter", response_model=SetParameterResponse)
async def set_parameter(request: SetParameterRequest):
    """
    Set a parameter value for a node
    """
    try:
        node = ros2_system.get_node(request.node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")

        success = node.set_parameter(request.name, request.value)

        if success:
            response = SetParameterResponse(
                success=True,
                message=f"Parameter {request.name} set on node {request.node_id}"
            )
        else:
            response = SetParameterResponse(
                success=False,
                message=f"Failed to set parameter {request.name} on node {request.node_id}"
            )

        logger.info(f"Parameter set: {request.name} on node {request.node_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error setting parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting parameter: {str(e)}")


@router.get("/get-parameter", response_model=GetParameterResponse)
async def get_parameter(node_id: str, name: str):
    """
    Get a parameter value from a node
    """
    try:
        node = ros2_system.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        param = node.get_parameter(name)
        if param is None:
            raise HTTPException(status_code=404, detail=f"Parameter {name} not found on node {node_id}")

        response = GetParameterResponse(
            name=param.name,
            value=param.value,
            type=param.type,
            success=True
        )

        logger.info(f"Parameter retrieved: {name} from node {node_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting parameter: {str(e)}")


@router.get("/node-info", response_model=GetNodeInfoResponse)
async def get_node_info(node_id: str):
    """
    Get information about a node
    """
    try:
        node = ros2_system.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        info = node.get_info()

        response = GetNodeInfoResponse(
            node_id=info.node_id,
            node_name=info.node_name,
            namespace=info.namespace,
            lifecycle_state=info.lifecycle_state.value,
            publishers=info.publishers,
            subscribers=info.subscribers,
            services=info.services,
            actions=info.actions,
            success=True
        )

        logger.info(f"Node info retrieved for: {node_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting node info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting node info: {str(e)}")


@router.get("/list-nodes", response_model=ListNodesResponse)
async def list_nodes():
    """
    List all nodes in the system
    """
    try:
        node_infos = ros2_system.list_nodes()

        nodes_list = []
        for info in node_infos:
            nodes_list.append({
                "node_id": info.node_id,
                "node_name": info.node_name,
                "namespace": info.namespace,
                "lifecycle_state": info.lifecycle_state.value,
                "publishers": info.publishers,
                "subscribers": info.subscribers,
                "services": info.services,
                "actions": info.actions
            })

        response = ListNodesResponse(
            nodes=nodes_list,
            success=True
        )

        logger.info(f"Listed {len(nodes_list)} nodes")

        return response

    except Exception as e:
        logger.error(f"Error listing nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing nodes: {str(e)}")


@router.get("/list-topics", response_model=ListTopicsResponse)
async def list_topics():
    """
    List all topics in the system
    """
    try:
        topics = ros2_system.get_topic_names_and_types()

        response = ListTopicsResponse(
            topics=topics,
            success=True
        )

        logger.info(f"Listed {len(topics)} topics")

        return response

    except Exception as e:
        logger.error(f"Error listing topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing topics: {str(e)}")


@router.post("/call-service", response_model=CallServiceResponse)
async def call_service(request: CallServiceRequest):
    """
    Call a service
    """
    try:
        result = ros2_system.call_service(request.service_name, request.request_data)

        response = CallServiceResponse(
            success=result.get("success", False),
            response=result.get("response"),
            error=result.get("error")
        )

        logger.info(f"Service called: {request.service_name}, success: {result.get('success', False)}")

        return response

    except Exception as e:
        logger.error(f"Error calling service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling service: {str(e)}")


@router.post("/send-action-goal", response_model=SendActionGoalResponse)
async def send_action_goal(request: SendActionGoalRequest):
    """
    Send a goal to an action server
    """
    try:
        result = ros2_system.send_action_goal(request.action_name, request.goal)

        response = SendActionGoalResponse(
            success="error" not in result,
            goal_id=result.get("goal_id"),
            status=result.get("status"),
            error=result.get("error")
        )

        logger.info(f"Action goal sent: {request.action_name}, success: 'error' not in result")

        return response

    except Exception as e:
        logger.error(f"Error sending action goal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending action goal: {str(e)}")