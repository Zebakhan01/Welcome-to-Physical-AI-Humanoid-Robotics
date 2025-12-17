from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from backend.utils.gazebo_utils import (
    gazebo_scene_manager, SDFWorld, SDFModel, GazeboPose, PhysicsEngine,
    ModelType, SDFParser, GazeboModel, GazeboWorld
)
from backend.utils.logger import logger


router = APIRouter()


class CreateWorldRequest(BaseModel):
    name: str
    physics_engine: str = "ode"  # ode, bullet, dart, simbody
    gravity: List[float] = [0.0, 0.0, -9.8]
    scene_properties: Optional[Dict[str, Any]] = None


class CreateWorldResponse(BaseModel):
    world_id: str
    success: bool
    message: str


class LoadWorldFromSDFRequest(BaseModel):
    sdf_xml: str


class LoadWorldFromSDFResponse(BaseModel):
    world_id: str
    success: bool
    message: str


class SpawnModelRequest(BaseModel):
    world_id: str
    name: str
    pose: List[float]  # [x, y, z, roll, pitch, yaw]
    model_type: str = "dynamic"  # static, dynamic, articulated
    sdf_xml: str


class SpawnModelResponse(BaseModel):
    model_id: str
    success: bool
    message: str


class LoadModelFromSDFRequest(BaseModel):
    world_id: str
    sdf_xml: str


class LoadModelFromSDFResponse(BaseModel):
    model_id: str
    success: bool
    message: str


class GetModelStateRequest(BaseModel):
    world_id: str
    model_id: str


class GetModelStateResponse(BaseModel):
    model_state: Optional[Dict[str, Any]]
    success: bool
    message: Optional[str] = None


class SetModelStateRequest(BaseModel):
    world_id: str
    model_id: str
    pose: Optional[List[float]] = None  # [x, y, z, roll, pitch, yaw]
    velocity: Optional[List[float]] = None  # [vx, vy, vz, wx, wy, wz]


class SetModelStateResponse(BaseModel):
    success: bool
    message: str


class ApplyForceRequest(BaseModel):
    world_id: str
    model_id: str
    force: List[float]  # [fx, fy, fz, tx, ty, tz]


class ApplyForceResponse(BaseModel):
    success: bool
    message: str


class DeleteModelRequest(BaseModel):
    world_id: str
    model_id: str


class DeleteModelResponse(BaseModel):
    success: bool
    message: str


class StartSimulationRequest(BaseModel):
    world_id: str


class StartSimulationResponse(BaseModel):
    success: bool
    message: str


class StopSimulationRequest(BaseModel):
    world_id: str


class StopSimulationResponse(BaseModel):
    success: bool
    message: str


class PauseSimulationRequest(BaseModel):
    world_id: str


class PauseSimulationResponse(BaseModel):
    success: bool
    message: str


class StepSimulationRequest(BaseModel):
    world_id: str
    step_time: float = 0.001


class StepSimulationResponse(BaseModel):
    success: bool
    message: str


class GetWorldStateRequest(BaseModel):
    world_id: str


class GetWorldStateResponse(BaseModel):
    world_state: Optional[Dict[str, Any]]
    success: bool
    message: Optional[str] = None


class ListWorldsResponse(BaseModel):
    worlds: List[Dict[str, Any]]
    success: bool


class ListModelsRequest(BaseModel):
    world_id: str


class ListModelsResponse(BaseModel):
    models: List[Dict[str, Any]]
    success: bool


class DeleteWorldRequest(BaseModel):
    world_id: str


class DeleteWorldResponse(BaseModel):
    success: bool
    message: str


@router.post("/create-world", response_model=CreateWorldResponse)
async def create_world(request: CreateWorldRequest):
    """
    Create a new Gazebo simulation world
    """
    try:
        # Validate physics engine
        try:
            physics_engine = PhysicsEngine(request.physics_engine.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid physics engine: {request.physics_engine}")

        # Validate gravity vector
        if len(request.gravity) != 3:
            raise HTTPException(status_code=400, detail="Gravity must be a 3-element vector [x, y, z]")

        # Create SDF world
        sdf_world = SDFWorld(
            name=request.name,
            physics_engine=physics_engine,
            gravity=tuple(request.gravity),
            models=[],
            lights=[],
            plugins=[],
            scene_properties=request.scene_properties or {}
        )

        # Create the world in the scene manager
        world_id = gazebo_scene_manager.create_world(sdf_world)

        response = CreateWorldResponse(
            world_id=world_id,
            success=True,
            message=f"World '{request.name}' created successfully with ID {world_id}"
        )

        logger.info(f"World created: {request.name} with ID {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error creating world: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating world: {str(e)}")


@router.post("/load-world-from-sdf", response_model=LoadWorldFromSDFResponse)
async def load_world_from_sdf(request: LoadWorldFromSDFRequest):
    """
    Load a Gazebo world from SDF XML content
    """
    try:
        # Parse the SDF XML
        sdf_world = SDFParser.parse_world_from_xml(request.sdf_xml)
        if not sdf_world:
            raise HTTPException(status_code=400, detail="Invalid SDF XML content")

        # Create the world in the scene manager
        world_id = gazebo_scene_manager.create_world(sdf_world)

        response = LoadWorldFromSDFResponse(
            world_id=world_id,
            success=True,
            message=f"World loaded from SDF successfully with ID {world_id}"
        )

        logger.info(f"World loaded from SDF with ID: {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error loading world from SDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading world from SDF: {str(e)}")


@router.post("/spawn-model", response_model=SpawnModelResponse)
async def spawn_model(request: SpawnModelRequest):
    """
    Spawn a model in a Gazebo world
    """
    try:
        # Validate pose
        if len(request.pose) != 6:
            raise HTTPException(status_code=400, detail="Pose must be a 6-element vector [x, y, z, roll, pitch, yaw]")

        # Validate model type
        try:
            model_type = ModelType(request.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")

        # Create pose object
        pose = GazeboPose(*request.pose)

        # Parse the model SDF if provided
        sdf_model = SDFParser.parse_model_from_xml(request.sdf_xml)
        if not sdf_model:
            raise HTTPException(status_code=400, detail="Invalid SDF XML content for model")

        # Override model properties with request values
        sdf_model.name = request.name
        sdf_model.pose = pose
        sdf_model.model_type = model_type

        # Spawn the model in the world
        model_id = gazebo_scene_manager.spawn_model(request.world_id, sdf_model)
        if not model_id:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        response = SpawnModelResponse(
            model_id=model_id,
            success=True,
            message=f"Model '{request.name}' spawned successfully with ID {model_id}"
        )

        logger.info(f"Model spawned: {request.name} with ID {model_id} in world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error spawning model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error spawning model: {str(e)}")


@router.post("/load-model-from-sdf", response_model=LoadModelFromSDFResponse)
async def load_model_from_sdf(request: LoadModelFromSDFRequest):
    """
    Load a model from SDF XML content into a world
    """
    try:
        # Parse the model SDF
        sdf_model = SDFParser.parse_model_from_xml(request.sdf_xml)
        if not sdf_model:
            raise HTTPException(status_code=400, detail="Invalid SDF XML content for model")

        # Spawn the model in the world
        model_id = gazebo_scene_manager.spawn_model(request.world_id, sdf_model)
        if not model_id:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        response = LoadModelFromSDFResponse(
            model_id=model_id,
            success=True,
            message=f"Model loaded from SDF successfully with ID {model_id}"
        )

        logger.info(f"Model loaded from SDF with ID: {model_id} in world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error loading model from SDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model from SDF: {str(e)}")


@router.get("/get-model-state", response_model=GetModelStateResponse)
async def get_model_state(world_id: str, model_id: str):
    """
    Get the state of a model in a world
    """
    try:
        model_state = gazebo_scene_manager.get_model_state(world_id, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found in world {world_id}")

        response = GetModelStateResponse(
            model_state=model_state,
            success=True
        )

        logger.info(f"Model state retrieved for {model_id} in world {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting model state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model state: {str(e)}")


@router.post("/set-model-state", response_model=SetModelStateResponse)
async def set_model_state(request: SetModelStateRequest):
    """
    Set the state of a model in a world
    """
    try:
        success = True
        message = f"Model {request.model_id} state updated in world {request.world_id}"

        # Set pose if provided
        if request.pose:
            if len(request.pose) != 6:
                raise HTTPException(status_code=400, detail="Pose must be a 6-element vector [x, y, z, roll, pitch, yaw]")
            pose = GazeboPose(*request.pose)
            success = gazebo_scene_manager.set_model_pose(request.world_id, request.model_id, pose)
            if not success:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found in world {request.world_id}")

        # Additional state updates can be added here

        response = SetModelStateResponse(
            success=True,
            message=message
        )

        logger.info(f"Model state set for {request.model_id} in world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error setting model state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting model state: {str(e)}")


@router.post("/apply-force", response_model=ApplyForceResponse)
async def apply_force(request: ApplyForceRequest):
    """
    Apply a force to a model in a world
    """
    try:
        if len(request.force) != 6:
            raise HTTPException(status_code=400, detail="Force must be a 6-element vector [fx, fy, fz, tx, ty, tz]")

        force = GazeboPose(*request.force)

        success = gazebo_scene_manager.apply_force_to_model(request.world_id, request.model_id, force)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found in world {request.world_id}")

        response = ApplyForceResponse(
            success=True,
            message=f"Force applied to model {request.model_id} in world {request.world_id}"
        )

        logger.info(f"Force applied to model {request.model_id} in world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error applying force: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying force: {str(e)}")


@router.post("/delete-model", response_model=DeleteModelResponse)
async def delete_model(request: DeleteModelRequest):
    """
    Delete a model from a world
    """
    try:
        success = gazebo_scene_manager.delete_model(request.world_id, request.model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found in world {request.world_id}")

        response = DeleteModelResponse(
            success=True,
            message=f"Model {request.model_id} deleted from world {request.world_id}"
        )

        logger.info(f"Model {request.model_id} deleted from world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


@router.post("/start-simulation", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """
    Start simulation for a world
    """
    try:
        world = gazebo_scene_manager.get_world(request.world_id)
        if not world:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        world.start_simulation()

        response = StartSimulationResponse(
            success=True,
            message=f"Simulation started for world {request.world_id}"
        )

        logger.info(f"Simulation started for world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@router.post("/stop-simulation", response_model=StopSimulationResponse)
async def stop_simulation(request: StopSimulationRequest):
    """
    Stop simulation for a world
    """
    try:
        world = gazebo_scene_manager.get_world(request.world_id)
        if not world:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        world.stop_simulation()

        response = StopSimulationResponse(
            success=True,
            message=f"Simulation stopped for world {request.world_id}"
        )

        logger.info(f"Simulation stopped for world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")


@router.post("/pause-simulation", response_model=PauseSimulationResponse)
async def pause_simulation(request: PauseSimulationRequest):
    """
    Pause simulation for a world
    """
    try:
        world = gazebo_scene_manager.get_world(request.world_id)
        if not world:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        world.pause_simulation()

        response = PauseSimulationResponse(
            success=True,
            message=f"Simulation paused for world {request.world_id}"
        )

        logger.info(f"Simulation paused for world {request.world_id}")

        return response

    except Exception as e:
        logger.error(f"Error pausing simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error pausing simulation: {str(e)}")


@router.post("/step-simulation", response_model=StepSimulationResponse)
async def step_simulation(request: StepSimulationRequest):
    """
    Step simulation for a world
    """
    try:
        world = gazebo_scene_manager.get_world(request.world_id)
        if not world:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        world.step_simulation(request.step_time)

        response = StepSimulationResponse(
            success=True,
            message=f"Simulation stepped for world {request.world_id} by {request.step_time}s"
        )

        logger.info(f"Simulation stepped for world {request.world_id} by {request.step_time}s")

        return response

    except Exception as e:
        logger.error(f"Error stepping simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stepping simulation: {str(e)}")


@router.get("/get-world-state", response_model=GetWorldStateResponse)
async def get_world_state(world_id: str):
    """
    Get the state of a world
    """
    try:
        world_state = gazebo_scene_manager.get_world_state(world_id)
        if not world_state:
            raise HTTPException(status_code=404, detail=f"World {world_id} not found")

        response = GetWorldStateResponse(
            world_state=world_state,
            success=True
        )

        logger.info(f"World state retrieved for {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting world state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting world state: {str(e)}")


@router.get("/list-worlds", response_model=ListWorldsResponse)
async def list_worlds():
    """
    List all worlds in the scene manager
    """
    try:
        worlds = gazebo_scene_manager.list_worlds()

        response = ListWorldsResponse(
            worlds=worlds,
            success=True
        )

        logger.info(f"Listed {len(worlds)} worlds")

        return response

    except Exception as e:
        logger.error(f"Error listing worlds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing worlds: {str(e)}")


@router.get("/list-models", response_model=ListModelsResponse)
async def list_models(world_id: str):
    """
    List all models in a world
    """
    try:
        world = gazebo_scene_manager.get_world(world_id)
        if not world:
            raise HTTPException(status_code=404, detail=f"World {world_id} not found")

        models = world.list_models()

        response = ListModelsResponse(
            models=models,
            success=True
        )

        logger.info(f"Listed {len(models)} models in world {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/delete-world", response_model=DeleteWorldResponse)
async def delete_world(request: DeleteWorldRequest):
    """
    Delete a world from the scene manager
    """
    try:
        success = gazebo_scene_manager.delete_world(request.world_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        response = DeleteWorldResponse(
            success=True,
            message=f"World {request.world_id} deleted successfully"
        )

        logger.info(f"World {request.world_id} deleted")

        return response

    except Exception as e:
        logger.error(f"Error deleting world: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting world: {str(e)}")