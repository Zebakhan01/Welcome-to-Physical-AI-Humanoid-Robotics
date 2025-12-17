from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Callable
import asyncio
import json
from backend.utils.ros_utils import (
    ros_system, ROSNode, ROSService, ROSActionServer, ROSMessage,
    ROSMessageFactory, ROSBagSimulator, NodeInfo, TopicInfo, ServiceInfo, Parameter
)
from backend.utils.logger import logger

router = APIRouter()


class CreateNodeRequest(BaseModel):
    name: str
    namespace: str = "/"


class CreateNodeResponse(BaseModel):
    node_id: str
    node_name: str
    uri: str
    success: bool


class CreatePublisherRequest(BaseModel):
    topic: str
    message_type: str
    queue_size: int = 10


class CreatePublisherResponse(BaseModel):
    topic: str
    success: bool


class CreateSubscriberRequest(BaseModel):
    topic: str
    message_type: str


class CreateSubscriberResponse(BaseModel):
    topic: str
    success: bool


class PublishMessageRequest(BaseModel):
    topic: str
    message: Dict[str, Any]


class PublishMessageResponse(BaseModel):
    success: bool
    message: str


class CallServiceRequest(BaseModel):
    service_name: str
    request_data: Dict[str, Any]


class CallServiceResponse(BaseModel):
    success: bool
    response: Dict[str, Any]


class RegisterServiceRequest(BaseModel):
    name: str
    service_type: str
    # For simulation, we'll define common service handlers
    handler_type: str  # "echo", "add_two_ints", "set_bool", etc.


class RegisterServiceResponse(BaseModel):
    service_name: str
    success: bool


class ROSInfoResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    services: List[str]
    parameters: List[str]
    success: bool


class SetParameterRequest(BaseModel):
    name: str
    value: Any


class SetParameterResponse(BaseModel):
    success: bool


class GetParameterRequest(BaseModel):
    name: str
    default: Optional[Any] = None


class GetParameterResponse(BaseModel):
    name: str
    value: Any
    type: str
    success: bool


class RegisterActionRequest(BaseModel):
    name: str
    action_type: str
    # For simulation, we'll define common action handlers
    handler_type: str  # "move_to_pose", "follow_joint_trajectory", etc.


class RegisterActionResponse(BaseModel):
    action_name: str
    success: bool


class SendActionGoalRequest(BaseModel):
    action_name: str
    goal: Dict[str, Any]


class SendActionGoalResponse(BaseModel):
    goal_id: str
    success: bool


class GetActionResultRequest(BaseModel):
    action_name: str
    goal_id: str


class GetActionResultResponse(BaseModel):
    goal_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    success: bool


# Store service handlers for simulation
service_handlers: Dict[str, Callable] = {
    "echo": lambda req: req,  # Echo service - returns the same request
    "add_two_ints": lambda req: {"sum": req.get("a", 0) + req.get("b", 0)},
    "set_bool": lambda req: {"success": True, "message": "Bool set successfully"},
    "trigger": lambda req: {"success": True, "message": "Triggered successfully"}
}


@router.post("/create-node", response_model=CreateNodeResponse)
async def create_node(request: CreateNodeRequest):
    """
    Create a new ROS node
    """
    try:
        node = ros_system.create_node(request.name, request.namespace)

        response = CreateNodeResponse(
            node_id=node.node_id,
            node_name=node.name,
            uri=node.uri,
            success=True
        )

        logger.info(f"Node created: {node.name} with ID {node.node_id}")

        return response

    except Exception as e:
        logger.error(f"Error creating node: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating node: {str(e)}")


@router.post("/create-publisher", response_model=CreatePublisherResponse)
async def create_publisher(request: CreatePublisherRequest):
    """
    Create a publisher for a topic
    """
    try:
        publisher = ros_system.create_publisher(request.topic, request.message_type, request.queue_size)

        response = CreatePublisherResponse(
            topic=request.topic,
            success=True
        )

        logger.info(f"Publisher created for topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error creating publisher: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating publisher: {str(e)}")


@router.post("/create-subscriber", response_model=CreateSubscriberResponse)
async def create_subscriber(request: CreateSubscriberRequest):
    """
    Create a subscriber for a topic
    """
    try:
        # For this simulation, we'll just create a placeholder
        # In a real implementation, this would register a callback function
        def dummy_callback(msg):
            logger.info(f"Received message on {request.topic}: {msg}")

        subscriber = ros_system.create_subscriber(request.topic, request.message_type, dummy_callback)

        response = CreateSubscriberResponse(
            topic=request.topic,
            success=True
        )

        logger.info(f"Subscriber created for topic: {request.topic}")

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
        publisher = ros_system.get_publisher(request.topic)
        if not publisher:
            raise HTTPException(status_code=404, detail=f"Publisher for topic '{request.topic}' not found")

        # Publish the message
        await publisher.publish(request.message)

        response = PublishMessageResponse(
            success=True,
            message=f"Published to topic {request.topic}"
        )

        logger.info(f"Message published to topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error publishing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error publishing message: {str(e)}")


@router.post("/call-service", response_model=CallServiceResponse)
async def call_service(request: CallServiceRequest):
    """
    Call a ROS service
    """
    try:
        result = await ros_system.call_service(request.service_name, request.request_data)

        response = CallServiceResponse(
            success=True,
            response=result
        )

        logger.info(f"Service {request.service_name} called successfully")

        return response

    except Exception as e:
        logger.error(f"Error calling service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling service: {str(e)}")


@router.post("/register-service", response_model=RegisterServiceResponse)
async def register_service(request: RegisterServiceRequest):
    """
    Register a new ROS service
    """
    try:
        # Get the appropriate handler function
        handler = service_handlers.get(request.handler_type)
        if not handler:
            raise HTTPException(status_code=400, detail=f"Unknown handler type: {request.handler_type}")

        service = ros_system.register_service(request.name, request.service_type, handler)

        response = RegisterServiceResponse(
            service_name=request.name,
            success=True
        )

        logger.info(f"Service registered: {request.name}")

        return response

    except Exception as e:
        logger.error(f"Error registering service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering service: {str(e)}")


@router.get("/info", response_model=ROSInfoResponse)
async def get_ros_info():
    """
    Get information about the ROS system
    """
    try:
        # Get node information
        nodes_info = []
        for node_id, node in ros_system.nodes.items():
            node_info = node.get_info()
            nodes_info.append({
                "node_id": node_info.node_id,
                "node_name": node_info.node_name,
                "namespace": node_info.namespace,
                "uri": node_info.uri
            })

        # Get topic information
        topics_info = []
        for topic_name, topic_info in ros_system.topics.items():
            topics_info.append({
                "name": topic_info.name,
                "message_type": topic_info.message_type,
                "publishers": topic_info.publishers,
                "subscribers": topic_info.subscribers,
                "connections": topic_info.connections
            })

        response = ROSInfoResponse(
            nodes=nodes_info,
            topics=topics_info,
            services=ros_system.services.list_services(),
            parameters=ros_system.list_parameters(),
            success=True
        )

        logger.info("ROS system info retrieved")

        return response

    except Exception as e:
        logger.error(f"Error getting ROS info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting ROS info: {str(e)}")


@router.post("/set-parameter", response_model=SetParameterResponse)
async def set_parameter(request: SetParameterRequest):
    """
    Set a ROS parameter
    """
    try:
        ros_system.set_parameter(request.name, request.value)

        response = SetParameterResponse(
            success=True
        )

        logger.info(f"Parameter set: {request.name}")

        return response

    except Exception as e:
        logger.error(f"Error setting parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting parameter: {str(e)}")


@router.post("/get-parameter", response_model=GetParameterResponse)
async def get_parameter(request: GetParameterRequest):
    """
    Get a ROS parameter
    """
    try:
        value = ros_system.get_parameter(request.name, request.default)
        param_type = type(value).__name__ if value is not None else "unknown"

        response = GetParameterResponse(
            name=request.name,
            value=value,
            type=param_type,
            success=value is not None or request.default is not None
        )

        logger.info(f"Parameter retrieved: {request.name}")

        return response

    except Exception as e:
        logger.error(f"Error getting parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting parameter: {str(e)}")


@router.post("/register-action", response_model=RegisterActionResponse)
async def register_action(request: RegisterActionRequest):
    """
    Register a new ROS action server
    """
    try:
        # Define action handlers for simulation
        action_handlers = {
            "move_to_pose": lambda goal: {
                "success": True,
                "message": f"Moved to pose {goal.get('target_pose', 'unknown')}"
            },
            "follow_joint_trajectory": lambda goal: {
                "success": True,
                "message": "Trajectory executed successfully"
            },
            "pick_and_place": lambda goal: {
                "success": True,
                "message": f"Pick and place task completed: {goal.get('object_name', 'unknown object')}"
            }
        }

        handler = action_handlers.get(request.handler_type)
        if not handler:
            raise HTTPException(status_code=400, detail=f"Unknown action handler type: {request.handler_type}")

        action_server = ros_system.register_action(request.name, request.action_type, handler)

        response = RegisterActionResponse(
            action_name=request.name,
            success=True
        )

        logger.info(f"Action server registered: {request.name}")

        return response

    except Exception as e:
        logger.error(f"Error registering action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering action: {str(e)}")


@router.post("/send-action-goal", response_model=SendActionGoalResponse)
async def send_action_goal(request: SendActionGoalRequest):
    """
    Send a goal to an action server
    """
    try:
        goal_id = ros_system.send_action_goal(request.action_name, request.goal)

        response = SendActionGoalResponse(
            goal_id=goal_id,
            success=True
        )

        logger.info(f"Action goal sent: {request.action_name}, goal_id: {goal_id}")

        return response

    except Exception as e:
        logger.error(f"Error sending action goal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending action goal: {str(e)}")


class ListServicesResponse(BaseModel):
    services: List[str]
    success: bool


@router.get("/services", response_model=ListServicesResponse)
async def list_services():
    """
    List all available services
    """
    try:
        services = ros_system.services.list_services()

        response = ListServicesResponse(
            services=services,
            success=True
        )

        logger.info(f"Listed {len(services)} services")

        return response

    except Exception as e:
        logger.error(f"Error listing services: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing services: {str(e)}")


class ListTopicsResponse(BaseModel):
    topics: List[Dict[str, str]]
    success: bool


@router.get("/topics", response_model=ListTopicsResponse)
async def list_topics():
    """
    List all available topics
    """
    try:
        topic_list = []
        for topic_name, topic_info in ros_system.topics.items():
            topic_list.append({
                "name": topic_info.name,
                "type": topic_info.message_type
            })

        response = ListTopicsResponse(
            topics=topic_list,
            success=True
        )

        logger.info(f"Listed {len(topic_list)} topics")

        return response

    except Exception as e:
        logger.error(f"Error listing topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing topics: {str(e)}")


class ListNodesResponse(BaseModel):
    nodes: List[Dict[str, str]]
    success: bool


@router.get("/nodes", response_model=ListNodesResponse)
async def list_nodes():
    """
    List all available nodes
    """
    try:
        node_list = []
        for node_id, node in ros_system.nodes.items():
            node_list.append({
                "name": node.name,
                "id": node.node_id,
                "namespace": node.namespace
            })

        response = ListNodesResponse(
            nodes=node_list,
            success=True
        )

        logger.info(f"Listed {len(node_list)} nodes")

        return response

    except Exception as e:
        logger.error(f"Error listing nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing nodes: {str(e)}")


class ROSBagRecordRequest(BaseModel):
    topic: str


class ROSBagRecordResponse(BaseModel):
    success: bool
    message: str


@router.post("/bag-record", response_model=ROSBagRecordResponse)
async def bag_record(request: ROSBagRecordRequest):
    """
    Start recording messages on a topic (simulated rosbag)
    """
    try:
        # This is a simplified simulation of rosbag functionality
        # In a real system, we'd need to hook into the message publishing system
        from backend.utils.ros_utils import ROSBagSimulator
        bag_sim = ROSBagSimulator()
        bag_sim.start_recording(request.topic)

        response = ROSBagRecordResponse(
            success=True,
            message=f"Started recording on topic {request.topic}"
        )

        logger.info(f"Started recording on topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error starting bag recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting bag recording: {str(e)}")


class ROSBagPlayRequest(BaseModel):
    topic: str


class ROSBagPlayResponse(BaseModel):
    success: bool
    message: str


@router.post("/bag-play", response_model=ROSBagPlayResponse)
async def bag_play(request: ROSBagPlayRequest):
    """
    Play back recorded messages from a topic (simulated rosbag)
    """
    try:
        # This is a simplified simulation of rosbag playback functionality
        from backend.utils.ros_utils import ROSBagSimulator
        bag_sim = ROSBagSimulator()

        def playback_callback(msg):
            logger.info(f"Playing back message: {msg}")

        bag_sim.play_back(request.topic, playback_callback)

        response = ROSBagPlayResponse(
            success=True,
            message=f"Played back messages on topic {request.topic}"
        )

        logger.info(f"Played back messages on topic: {request.topic}")

        return response

    except Exception as e:
        logger.error(f"Error playing back bag: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error playing back bag: {str(e)}")