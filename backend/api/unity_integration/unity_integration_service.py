from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from backend.utils.unity_integration_utils import (
    unity_integration_manager, UnityVector3, UnityQuaternion, UnityTransform,
    UnityObjectType, UnityComponentType, UnityPhysicsEngine
)
from backend.utils.logger import logger


router = APIRouter()


class CreateSceneRequest(BaseModel):
    name: str


class CreateSceneResponse(BaseModel):
    scene_id: str
    name: str
    success: bool
    message: str


class LoadSceneRequest(BaseModel):
    scene_id: str


class LoadSceneResponse(BaseModel):
    success: bool
    message: str


class CreateObjectRequest(BaseModel):
    scene_id: str
    name: str
    object_type: str  # game_object, model, sensor, actuator, light, camera, terrain
    position: List[float] = [0.0, 0.0, 0.0]
    rotation: List[float] = [0.0, 0.0, 0.0, 1.0]  # quaternion
    tags: List[str] = []


class CreateObjectResponse(BaseModel):
    object_id: str
    success: bool


class SetTransformRequest(BaseModel):
    scene_id: str
    object_id: str
    position: Optional[List[float]] = None  # [x, y, z]
    rotation: Optional[List[float]] = None  # [x, y, z, w] quaternion
    scale: Optional[List[float]] = None    # [x, y, z]


class SetTransformResponse(BaseModel):
    success: bool
    message: str


class AddComponentRequest(BaseModel):
    scene_id: str
    object_id: str
    component_type: str  # rigidbody, collider, mesh_renderer, transform, script
    component_data: Dict[str, Any]


class AddComponentResponse(BaseModel):
    success: bool
    message: str


class SpawnRobotRequest(BaseModel):
    scene_id: str
    robot_name: str
    robot_type: str = "mobile_base"  # mobile_base, manipulator, humanoid, etc.


class SpawnRobotResponse(BaseModel):
    robot_object_id: str
    success: bool
    message: str


class MoveRobotRequest(BaseModel):
    scene_id: str
    robot_object_id: str
    target_position: List[float]  # [x, y, z]


class MoveRobotResponse(BaseModel):
    success: bool
    message: str


class AddSensorRequest(BaseModel):
    scene_id: str
    robot_object_id: str
    sensor_type: str  # camera, lidar, imu
    sensor_name: str
    sensor_config: Optional[Dict[str, Any]] = None


class AddSensorResponse(BaseModel):
    sensor_id: str
    success: bool


class GetSensorDataRequest(BaseModel):
    sensor_id: str


class GetSensorDataResponse(BaseModel):
    sensor_data: Optional[Dict[str, Any]]
    success: bool
    message: Optional[str] = None


class GetRobotStateRequest(BaseModel):
    scene_id: str
    robot_object_id: str


class GetRobotStateResponse(BaseModel):
    robot_state: Optional[Dict[str, Any]]
    success: bool
    message: Optional[str] = None


class ApplyForceRequest(BaseModel):
    object_id: str
    force: List[float]  # [x, y, z]


class ApplyForceResponse(BaseModel):
    success: bool
    message: str


class StartSimulationResponse(BaseModel):
    success: bool
    message: str


class StopSimulationResponse(BaseModel):
    success: bool
    message: str


class ListScenesResponse(BaseModel):
    scenes: List[Dict[str, Any]]
    success: bool


class ListObjectsRequest(BaseModel):
    scene_id: str


class ListObjectsResponse(BaseModel):
    objects: List[Dict[str, Any]]
    success: bool


class ListRobotsRequest(BaseModel):
    scene_id: str


class ListRobotsResponse(BaseModel):
    robots: List[Dict[str, Any]]
    success: bool


class ListSensorsResponse(BaseModel):
    sensors: List[Dict[str, Any]]
    success: bool


@router.post("/create-scene", response_model=CreateSceneResponse)
async def create_scene(request: CreateSceneRequest):
    """
    Create a new Unity scene
    """
    try:
        result = unity_integration_manager.create_unity_environment(request.name)

        response = CreateSceneResponse(
            scene_id=result["scene_id"],
            name=result["name"],
            success=result["success"],
            message=result["message"]
        )

        logger.info(f"Unity scene created: {request.name} with ID {result['scene_id']}")

        return response

    except Exception as e:
        logger.error(f"Error creating Unity scene: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Unity scene: {str(e)}")


@router.post("/load-scene", response_model=LoadSceneResponse)
async def load_scene(request: LoadSceneRequest):
    """
    Load a Unity scene as the active scene
    """
    try:
        success = unity_integration_manager.scene_manager.load_scene(request.scene_id)

        if success:
            response = LoadSceneResponse(
                success=True,
                message=f"Scene {request.scene_id} loaded successfully"
            )
        else:
            response = LoadSceneResponse(
                success=False,
                message=f"Failed to load scene {request.scene_id}"
            )

        logger.info(f"Scene load attempt: {request.scene_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error loading scene: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading scene: {str(e)}")


@router.post("/create-object", response_model=CreateObjectResponse)
async def create_object(request: CreateObjectRequest):
    """
    Create an object in a Unity scene
    """
    try:
        # Validate position and rotation
        if len(request.position) != 3:
            raise HTTPException(status_code=400, detail="Position must be a 3-element vector [x, y, z]")

        if len(request.rotation) != 4:
            raise HTTPException(status_code=400, detail="Rotation must be a 4-element quaternion [x, y, z, w]")

        object_type = UnityObjectType(request.object_type.lower())

        object_id = unity_integration_manager.scene_manager.create_object(
            request.scene_id,
            request.name,
            object_type,
            tuple(request.position),
            tuple(request.rotation),
            request.tags
        )

        if object_id:
            response = CreateObjectResponse(
                object_id=object_id,
                success=True
            )
        else:
            raise HTTPException(status_code=404, detail=f"Scene {request.scene_id} not found")

        logger.info(f"Object created: {request.name} with ID {object_id} in scene {request.scene_id}")

        return response

    except ValueError as e:
        logger.error(f"Invalid object type: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid object type: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating object: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating object: {str(e)}")


@router.post("/set-transform", response_model=SetTransformResponse)
async def set_transform(request: SetTransformRequest):
    """
    Set the transform (position, rotation, scale) of an object
    """
    try:
        success = unity_integration_manager.scene_manager.set_transform(
            request.scene_id,
            request.object_id,
            request.position,
            request.rotation,
            request.scale
        )

        if success:
            response = SetTransformResponse(
                success=True,
                message=f"Transform updated for object {request.object_id}"
            )
        else:
            response = SetTransformResponse(
                success=False,
                message=f"Failed to update transform for object {request.object_id}"
            )

        logger.info(f"Transform set for object {request.object_id} in scene {request.scene_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error setting transform: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting transform: {str(e)}")


@router.post("/add-component", response_model=AddComponentResponse)
async def add_component(request: AddComponentRequest):
    """
    Add a component to an object in a scene
    """
    try:
        component_type = UnityComponentType(request.component_type.lower())

        success = unity_integration_manager.scene_manager.add_component(
            request.scene_id,
            request.object_id,
            component_type,
            request.component_data
        )

        if success:
            response = AddComponentResponse(
                success=True,
                message=f"Component {request.component_type} added to object {request.object_id}"
            )
        else:
            response = AddComponentResponse(
                success=False,
                message=f"Failed to add component {request.component_type} to object {request.object_id}"
            )

        logger.info(f"Component {request.component_type} added to object {request.object_id}, success: {success}")

        return response

    except ValueError as e:
        logger.error(f"Invalid component type: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid component type: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding component: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding component: {str(e)}")


@router.post("/spawn-robot", response_model=SpawnRobotResponse)
async def spawn_robot(request: SpawnRobotRequest):
    """
    Spawn a robot in a Unity environment
    """
    try:
        result = unity_integration_manager.spawn_robot_in_environment(
            request.scene_id,
            request.robot_name,
            request.robot_type
        )

        if result:
            response = SpawnRobotResponse(
                robot_object_id=result["robot_object_id"],
                success=result["success"],
                message=result["message"]
            )
        else:
            raise HTTPException(status_code=404, detail=f"Scene {request.scene_id} not found")

        logger.info(f"Robot spawned: {request.robot_name} in scene {request.scene_id}")

        return response

    except Exception as e:
        logger.error(f"Error spawning robot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error spawning robot: {str(e)}")


@router.post("/move-robot", response_model=MoveRobotResponse)
async def move_robot(request: MoveRobotRequest):
    """
    Move a robot to a target position
    """
    try:
        if len(request.target_position) != 3:
            raise HTTPException(status_code=400, detail="Target position must be a 3-element vector [x, y, z]")

        success = unity_integration_manager.move_robot(
            request.scene_id,
            request.robot_object_id,
            tuple(request.target_position)
        )

        if success:
            response = MoveRobotResponse(
                success=True,
                message=f"Robot {request.robot_object_id} moved to position {request.target_position}"
            )
        else:
            response = MoveRobotResponse(
                success=False,
                message=f"Failed to move robot {request.robot_object_id}"
            )

        logger.info(f"Robot {request.robot_object_id} moved to {request.target_position}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error moving robot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error moving robot: {str(e)}")


@router.post("/add-sensor", response_model=AddSensorResponse)
async def add_sensor(request: AddSensorRequest):
    """
    Add a sensor to a robot in the Unity environment
    """
    try:
        sensor_id = unity_integration_manager.add_sensor_to_robot(
            request.scene_id,
            request.robot_object_id,
            request.sensor_type,
            request.sensor_name
        )

        if sensor_id:
            response = AddSensorResponse(
                sensor_id=sensor_id,
                success=True
            )
        else:
            raise HTTPException(status_code=404, detail=f"Robot {request.robot_object_id} not found")

        logger.info(f"Sensor {request.sensor_type} added to robot {request.robot_object_id}")

        return response

    except Exception as e:
        logger.error(f"Error adding sensor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding sensor: {str(e)}")


@router.get("/get-sensor-data", response_model=GetSensorDataResponse)
async def get_sensor_data(sensor_id: str):
    """
    Get data from a sensor in the Unity environment
    """
    try:
        sensor_data = unity_integration_manager.get_sensor_data(sensor_id)

        if sensor_data:
            response = GetSensorDataResponse(
                sensor_data=sensor_data,
                success=True
            )
        else:
            response = GetSensorDataResponse(
                sensor_data=None,
                success=False,
                message=f"Sensor {sensor_id} not found or no data available"
            )

        logger.info(f"Sensor data retrieved for {sensor_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting sensor data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting sensor data: {str(e)}")


@router.get("/get-robot-state", response_model=GetRobotStateResponse)
async def get_robot_state(scene_id: str, robot_object_id: str):
    """
    Get the state of a robot in the Unity environment
    """
    try:
        robot_state = unity_integration_manager.get_robot_state(scene_id, robot_object_id)

        if robot_state:
            response = GetRobotStateResponse(
                robot_state=robot_state,
                success=True
            )
        else:
            response = GetRobotStateResponse(
                robot_state=None,
                success=False,
                message=f"Robot {robot_object_id} not found in scene {scene_id}"
            )

        logger.info(f"Robot state retrieved for {robot_object_id} in scene {scene_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting robot state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting robot state: {str(e)}")


@router.post("/apply-force", response_model=ApplyForceResponse)
async def apply_force(request: ApplyForceRequest):
    """
    Apply a force to an object in the Unity environment
    """
    try:
        if len(request.force) != 3:
            raise HTTPException(status_code=400, detail="Force must be a 3-element vector [x, y, z]")

        success = unity_integration_manager.scene_manager.physics_simulator.apply_force(
            request.object_id,
            tuple(request.force)
        )

        if success:
            response = ApplyForceResponse(
                success=True,
                message=f"Force {request.force} applied to object {request.object_id}"
            )
        else:
            response = ApplyForceResponse(
                success=False,
                message=f"Failed to apply force to object {request.object_id}"
            )

        logger.info(f"Force {request.force} applied to object {request.object_id}, success: {success}")

        return response

    except Exception as e:
        logger.error(f"Error applying force: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying force: {str(e)}")


@router.post("/start-simulation", response_model=StartSimulationResponse)
async def start_simulation():
    """
    Start the Unity simulation
    """
    try:
        unity_integration_manager.start_simulation()

        response = StartSimulationResponse(
            success=True,
            message="Unity simulation started successfully"
        )

        logger.info("Unity simulation started")

        return response

    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@router.post("/stop-simulation", response_model=StopSimulationResponse)
async def stop_simulation():
    """
    Stop the Unity simulation
    """
    try:
        unity_integration_manager.stop_simulation()

        response = StopSimulationResponse(
            success=True,
            message="Unity simulation stopped successfully"
        )

        logger.info("Unity simulation stopped")

        return response

    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")


@router.get("/scenes", response_model=ListScenesResponse)
async def list_scenes():
    """
    List all Unity scenes
    """
    try:
        scenes = unity_integration_manager.list_scenes()

        response = ListScenesResponse(
            scenes=scenes,
            success=True
        )

        logger.info(f"Listed {len(scenes)} Unity scenes")

        return response

    except Exception as e:
        logger.error(f"Error listing scenes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing scenes: {str(e)}")


@router.get("/objects", response_model=ListObjectsResponse)
async def list_objects(scene_id: str):
    """
    List all objects in a Unity scene
    """
    try:
        objects = unity_integration_manager.scene_manager.list_objects(scene_id)

        response = ListObjectsResponse(
            objects=objects,
            success=True
        )

        logger.info(f"Listed {len(objects)} objects in scene {scene_id}")

        return response

    except Exception as e:
        logger.error(f"Error listing objects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing objects: {str(e)}")


@router.get("/robots", response_model=ListRobotsResponse)
async def list_robots(scene_id: str):
    """
    List all robots in a Unity scene
    """
    try:
        robots = unity_integration_manager.list_robots(scene_id)

        response = ListRobotsResponse(
            robots=robots,
            success=True
        )

        logger.info(f"Listed {len(robots)} robots in scene {scene_id}")

        return response

    except Exception as e:
        logger.error(f"Error listing robots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing robots: {str(e)}")


@router.get("/sensors", response_model=ListSensorsResponse)
async def list_sensors():
    """
    List all sensors in the Unity system
    """
    try:
        sensors = unity_integration_manager.list_sensors()

        response = ListSensorsResponse(
            sensors=sensors,
            success=True
        )

        logger.info(f"Listed {len(sensors)} sensors")

        return response

    except Exception as e:
        logger.error(f"Error listing sensors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sensors: {str(e)}")


@router.get("/scene-info", response_model=Dict[str, Any])
async def get_scene_info(scene_id: str):
    """
    Get detailed information about a Unity scene
    """
    try:
        scene_info = unity_integration_manager.get_scene_info(scene_id)

        if scene_info:
            return scene_info
        else:
            raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    except Exception as e:
        logger.error(f"Error getting scene info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting scene info: {str(e)}")


@router.delete("/scene", response_model=Dict[str, bool])
async def delete_scene(scene_id: str):
    """
    Delete a Unity scene
    """
    try:
        success = unity_integration_manager.scene_manager.delete_scene(scene_id)

        if success:
            logger.info(f"Scene {scene_id} deleted successfully")
            return {"success": True}
        else:
            raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    except Exception as e:
        logger.error(f"Error deleting scene: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting scene: {str(e)}")


@router.delete("/object", response_model=Dict[str, bool])
async def delete_object(scene_id: str, object_id: str):
    """
    Delete an object from a Unity scene
    """
    try:
        success = unity_integration_manager.scene_manager.delete_object(scene_id, object_id)

        if success:
            logger.info(f"Object {object_id} deleted from scene {scene_id}")
            return {"success": True}
        else:
            raise HTTPException(status_code=404, detail=f"Object {object_id} not found in scene {scene_id}")

    except Exception as e:
        logger.error(f"Error deleting object: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting object: {str(e)}")