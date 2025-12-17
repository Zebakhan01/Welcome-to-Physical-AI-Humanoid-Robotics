from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from backend.utils.simulation_utils import (
    simulation_manager, SimulationPlatform, PhysicsEngine,
    SimulationWorld, RobotModel, SimulationSensor, SimulationState,
    SimulationMetrics, DomainRandomization
)
from backend.utils.logger import logger


router = APIRouter()


class CreateWorldRequest(BaseModel):
    name: str
    description: str = ""
    physics_engine: str = "bullet"  # Default to bullet physics


class CreateWorldResponse(BaseModel):
    world_id: str
    name: str
    success: bool


class LoadRobotRequest(BaseModel):
    world_id: str
    name: str
    urdf_path: str
    initial_pose: List[float]  # [x, y, z, roll, pitch, yaw]


class LoadRobotResponse(BaseModel):
    robot_id: str
    name: str
    success: bool


class AddSensorRequest(BaseModel):
    robot_id: str
    sensor_type: str  # camera, lidar, imu, etc.
    name: str = ""
    position: List[float] = [0.0, 0.0, 0.0]
    orientation: List[float] = [0.0, 0.0, 0.0, 1.0]  # quaternion
    parameters: Dict[str, Any] = {}


class AddSensorResponse(BaseModel):
    sensor_id: str
    success: bool


class SimulationControlRequest(BaseModel):
    world_id: str
    action: str  # start, stop, pause, resume, step, reset


class SimulationControlResponse(BaseModel):
    success: bool
    message: str


class SimulationStateResponse(BaseModel):
    world_id: str
    time: float
    paused: bool
    running: bool
    robot_states: Dict[str, Any]
    sensor_data: Dict[str, Any]
    physics_stats: Dict[str, float]
    success: bool


class SimulationMetricsResponse(BaseModel):
    real_time_factor: float
    update_rate: float
    cpu_usage: float
    memory_usage: float
    physics_step_time: float
    rendering_time: float
    simulation_time: float
    wall_time: float
    success: bool


class DomainRandomizationRequest(BaseModel):
    parameter_name: str
    parameter_type: str
    min_value: float
    max_value: float


class DomainRandomizationResponse(BaseModel):
    success: bool
    message: str


class GetRandomizationValuesResponse(BaseModel):
    parameters: Dict[str, float]
    success: bool


@router.post("/create-world", response_model=CreateWorldResponse)
async def create_world(request: CreateWorldRequest):
    """
    Create a new simulation world
    """
    try:
        # Get physics engine enum
        try:
            physics_engine = PhysicsEngine(request.physics_engine.upper())
        except ValueError:
            physics_engine = PhysicsEngine.BULLET

        # Create environment if it doesn't exist
        env_name = f"env_{request.name}"
        env = simulation_manager.get_environment(env_name)
        if not env:
            env = simulation_manager.create_environment(env_name)

        # Create the world
        world = env.create_world(request.name, request.description, physics_engine)

        response = CreateWorldResponse(
            world_id=world.id,
            name=world.name,
            success=True
        )

        logger.info(f"World created: {world.name} with ID {world.id}")

        return response

    except Exception as e:
        logger.error(f"Error creating world: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating world: {str(e)}")


@router.post("/load-robot", response_model=LoadRobotResponse)
async def load_robot(request: LoadRobotRequest):
    """
    Load a robot model into a simulation world
    """
    try:
        # Get the environment
        env = None
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            if any(world.id == request.world_id for world in env.worlds.values()):
                break

        if not env:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        # Load the robot
        initial_pose = tuple(request.initial_pose)
        robot = env.load_robot_model(request.world_id, request.urdf_path, request.name, initial_pose)

        response = LoadRobotResponse(
            robot_id=robot.id,
            name=robot.name,
            success=True
        )

        logger.info(f"Robot loaded: {robot.name} with ID {robot.id}")

        return response

    except Exception as e:
        logger.error(f"Error loading robot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading robot: {str(e)}")


@router.post("/add-sensor", response_model=AddSensorResponse)
async def add_sensor(request: AddSensorRequest):
    """
    Add a sensor to a robot in simulation
    """
    try:
        # Check if robot exists
        robot_exists = False
        env = None
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            if request.robot_id in env.robots:
                robot_exists = True
                break

        if not robot_exists:
            raise HTTPException(status_code=404, detail=f"Robot {request.robot_id} not found")

        # Prepare sensor configuration
        sensor_config = {
            "type": request.sensor_type,
            "name": request.name or f"{request.sensor_type}_sensor",
            "position": tuple(request.position),
            "orientation": tuple(request.orientation),
            "parameters": request.parameters,
            "topic": f"/sensor/{request.robot_id}/{request.sensor_type}"
        }

        # Add sensor to robot
        sensor = env.add_sensor_to_robot(request.robot_id, sensor_config)

        response = AddSensorResponse(
            sensor_id=sensor.id,
            success=True
        )

        logger.info(f"Sensor added to robot {request.robot_id}: {sensor.name}")

        return response

    except Exception as e:
        logger.error(f"Error adding sensor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding sensor: {str(e)}")


@router.post("/control-simulation", response_model=SimulationControlResponse)
async def control_simulation(request: SimulationControlRequest):
    """
    Control the simulation (start, stop, pause, etc.)
    """
    try:
        # Find the environment containing this world
        env = None
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            if request.world_id in env.worlds:
                break

        if not env:
            raise HTTPException(status_code=404, detail=f"World {request.world_id} not found")

        # Perform the requested action
        if request.action == "start":
            env.start_simulation()
            message = "Simulation started"
        elif request.action == "stop":
            env.stop_simulation()
            message = "Simulation stopped"
        elif request.action == "pause":
            env.pause_simulation()
            message = "Simulation paused"
        elif request.action == "resume":
            env.resume_simulation()
            message = "Simulation resumed"
        elif request.action == "step":
            env.step_simulation()
            message = "Simulation stepped forward"
        elif request.action == "reset":
            env.reset_simulation()
            message = "Simulation reset"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

        response = SimulationControlResponse(
            success=True,
            message=message
        )

        logger.info(f"Simulation control action '{request.action}' performed: {message}")

        return response

    except Exception as e:
        logger.error(f"Error controlling simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error controlling simulation: {str(e)}")


@router.get("/simulation-state", response_model=SimulationStateResponse)
async def get_simulation_state(world_id: str):
    """
    Get the current state of the simulation
    """
    try:
        # Find the environment containing this world
        env = None
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            if world_id in env.worlds:
                break

        if not env:
            raise HTTPException(status_code=404, detail=f"World {world_id} not found")

        state = env.get_current_state()
        if not state:
            raise HTTPException(status_code=404, detail="No simulation state available")

        response = SimulationStateResponse(
            world_id=state.world_id,
            time=state.time,
            paused=state.paused,
            running=state.running,
            robot_states=state.robot_states,
            sensor_data=state.sensor_data,
            physics_stats=state.physics_stats,
            success=True
        )

        logger.info(f"Simulation state retrieved for world {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting simulation state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting simulation state: {str(e)}")


@router.get("/simulation-metrics", response_model=SimulationMetricsResponse)
async def get_simulation_metrics(world_id: str):
    """
    Get simulation performance metrics
    """
    try:
        # Find the environment containing this world
        env = None
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            if world_id in env.worlds:
                break

        if not env:
            raise HTTPException(status_code=404, detail=f"World {world_id} not found")

        metrics = env.get_simulation_metrics()

        response = SimulationMetricsResponse(
            real_time_factor=metrics.real_time_factor,
            update_rate=metrics.update_rate,
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            physics_step_time=metrics.physics_step_time,
            rendering_time=metrics.rendering_time,
            simulation_time=metrics.simulation_time,
            wall_time=metrics.wall_time,
            success=True
        )

        logger.info(f"Simulation metrics retrieved for world {world_id}")

        return response

    except Exception as e:
        logger.error(f"Error getting simulation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting simulation metrics: {str(e)}")


@router.post("/domain-randomization", response_model=DomainRandomizationResponse)
async def add_domain_randomization(request: DomainRandomizationRequest):
    """
    Add a parameter for domain randomization
    """
    try:
        # Add the randomization parameter
        simulation_manager.domain_randomizer.add_randomization_parameter(
            request.parameter_name,
            request.parameter_type,
            request.min_value,
            request.max_value
        )

        response = DomainRandomizationResponse(
            success=True,
            message=f"Domain randomization parameter '{request.parameter_name}' added"
        )

        logger.info(f"Domain randomization parameter added: {request.parameter_name}")

        return response

    except Exception as e:
        logger.error(f"Error adding domain randomization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding domain randomization: {str(e)}")


@router.post("/randomize-parameter", response_model=DomainRandomizationResponse)
async def randomize_parameter(parameter_name: str):
    """
    Randomize a specific parameter
    """
    try:
        # Randomize the parameter
        new_value = simulation_manager.domain_randomizer.randomize_parameter(parameter_name)

        response = DomainRandomizationResponse(
            success=True,
            message=f"Parameter '{parameter_name}' randomized to {new_value}"
        )

        logger.info(f"Parameter randomized: {parameter_name} = {new_value}")

        return response

    except Exception as e:
        logger.error(f"Error randomizing parameter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error randomizing parameter: {str(e)}")


@router.get("/randomization-values", response_model=GetRandomizationValuesResponse)
async def get_randomization_values():
    """
    Get current values of all randomized parameters
    """
    try:
        values = simulation_manager.domain_randomizer.get_current_randomization_values()

        response = GetRandomizationValuesResponse(
            parameters=values,
            success=True
        )

        logger.info("Randomization values retrieved")

        return response

    except Exception as e:
        logger.error(f"Error getting randomization values: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting randomization values: {str(e)}")


@router.get("/environments", response_model=List[str])
async def list_environments():
    """
    List all available simulation environments
    """
    try:
        environments = simulation_manager.list_environments()

        logger.info(f"Listed {len(environments)} simulation environments")

        return environments

    except Exception as e:
        logger.error(f"Error listing environments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing environments: {str(e)}")


@router.get("/worlds", response_model=List[Dict[str, Any]])
async def list_worlds():
    """
    List all available simulation worlds
    """
    try:
        worlds_list = []
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            for world_id, world in env.worlds.items():
                worlds_list.append({
                    "id": world.id,
                    "name": world.name,
                    "description": world.description,
                    "physics_engine": world.physics_engine.value,
                    "created_at": world.created_at.isoformat()
                })

        logger.info(f"Listed {len(worlds_list)} simulation worlds")

        return worlds_list

    except Exception as e:
        logger.error(f"Error listing worlds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing worlds: {str(e)}")


@router.get("/robots", response_model=List[Dict[str, Any]])
async def list_robots():
    """
    List all robots in all simulation environments
    """
    try:
        robots_list = []
        for env_name in simulation_manager.list_environments():
            env = simulation_manager.get_environment(env_name)
            for robot_id, robot in env.robots.items():
                robots_list.append({
                    "id": robot.id,
                    "name": robot.name,
                    "urdf_path": robot.urdf_path,
                    "mass": robot.mass
                })

        logger.info(f"Listed {len(robots_list)} robots")

        return robots_list

    except Exception as e:
        logger.error(f"Error listing robots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing robots: {str(e)}")