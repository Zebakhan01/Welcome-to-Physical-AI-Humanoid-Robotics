from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
from backend.utils.humanoid_architecture_utils import (
    humanoid_manager, HumanoidPlatform, ActuatorType, ControlLevel,
    HumanoidController, JointConfiguration, BalanceState, HumanoidState
)
from backend.utils.logger import logger


router = APIRouter()


class CreatePlatformRequest(BaseModel):
    name: str
    platform_type: str  # asimo, atlas, nao, etc.


class CreatePlatformResponse(BaseModel):
    robot_name: str
    platform_type: str
    total_dof: int
    success: bool


class UpdateStateRequest(BaseModel):
    robot_name: str
    sensor_data: Dict[str, Any]


class UpdateStateResponse(BaseModel):
    success: bool
    balance_state: Optional[Dict[str, Any]] = None
    is_safe: bool = False


class ComputeCommandsResponse(BaseModel):
    success: bool
    joint_commands: Dict[str, Dict[str, float]]
    task_execution: List[Dict[str, Any]]
    constraints_satisfied: bool


class SetControlModeRequest(BaseModel):
    robot_name: str
    mode: str  # walking, standing, manipulation, idle


class SetControlModeResponse(BaseModel):
    success: bool
    message: str


class JointStateResponse(BaseModel):
    position: float
    velocity: float
    effort: float
    limits: List[float]
    stiffness: float
    damping: float


class GetStateResponse(BaseModel):
    robot_name: str
    joint_states: Dict[str, JointStateResponse]
    balance_state: Dict[str, Any]
    sensor_data: Dict[str, Any]
    timestamp: float
    is_safe: bool
    control_mode: str
    success: bool


class AddTaskRequest(BaseModel):
    robot_name: str
    task_type: str
    priority: int
    constraints: Dict[str, Any]


class AddTaskResponse(BaseModel):
    success: bool
    message: str


class EvaluatePerformanceRequest(BaseModel):
    robot_name: str
    task: str  # walking, manipulation, balance


class EvaluatePerformanceResponse(BaseModel):
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ListPlatformsResponse(BaseModel):
    platforms: List[str]
    success: bool


@router.post("/create-platform", response_model=CreatePlatformResponse)
async def create_platform(request: CreatePlatformRequest):
    """
    Create a new humanoid platform
    """
    try:
        platform_type = HumanoidPlatform(request.platform_type.lower())

        controller = humanoid_manager.create_platform(platform_type, request.name)

        response = CreatePlatformResponse(
            robot_name=request.name,
            platform_type=request.platform_type,
            total_dof=controller.body_plan.total_dof,
            success=True
        )

        logger.info(f"Platform created: {request.name} ({request.platform_type}) with {controller.body_plan.total_dof} DOF")

        return response

    except ValueError:
        logger.error(f"Invalid platform type: {request.platform_type}")
        raise HTTPException(status_code=400, detail=f"Invalid platform type: {request.platform_type}")
    except Exception as e:
        logger.error(f"Error creating platform: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating platform: {str(e)}")


@router.post("/update-state", response_model=UpdateStateResponse)
async def update_state(request: UpdateStateRequest):
    """
    Update the state of a humanoid robot with sensor data
    """
    try:
        robot = humanoid_manager.get_robot(request.robot_name)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot {request.robot_name} not found")

        # Update the robot state with sensor data
        updated_state = robot.update_state(request.sensor_data)

        response = UpdateStateResponse(
            success=True,
            balance_state={
                "com_position": updated_state.balance_state.com_position,
                "com_velocity": updated_state.balance_state.com_velocity,
                "support_polygon": updated_state.balance_state.support_polygon,
                "zmp_position": updated_state.balance_state.zmp_position,
                "stability_margin": updated_state.balance_state.stability_margin,
                "balance_mode": updated_state.balance_state.balance_mode,
                "double_support": updated_state.balance_state.double_support
            },
            is_safe=updated_state.is_safe
        )

        logger.info(f"State updated for robot {request.robot_name}, safe: {updated_state.is_safe}")

        return response

    except Exception as e:
        logger.error(f"Error updating state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating state: {str(e)}")


@router.post("/compute-commands", response_model=ComputeCommandsResponse)
async def compute_commands(robot_name: str):
    """
    Compute control commands for a humanoid robot
    """
    try:
        robot = humanoid_manager.get_robot(robot_name)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot {robot_name} not found")

        commands = robot.compute_control_commands()

        response = ComputeCommandsResponse(
            success=True,
            joint_commands=commands["joint_commands"],
            task_execution=commands["task_execution"],
            constraints_satisfied=commands["constraints_satisfied"]
        )

        logger.info(f"Control commands computed for robot {robot_name}")

        return response

    except Exception as e:
        logger.error(f"Error computing commands: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing commands: {str(e)}")


@router.post("/set-control-mode", response_model=SetControlModeResponse)
async def set_control_mode(request: SetControlModeRequest):
    """
    Set the control mode for a humanoid robot
    """
    try:
        robot = humanoid_manager.get_robot(request.robot_name)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot {request.robot_name} not found")

        robot.set_control_mode(request.mode)

        response = SetControlModeResponse(
            success=True,
            message=f"Control mode set to {request.mode} for robot {request.robot_name}"
        )

        logger.info(f"Control mode set to {request.mode} for robot {request.robot_name}")

        return response

    except Exception as e:
        logger.error(f"Error setting control mode: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting control mode: {str(e)}")


@router.get("/state", response_model=GetStateResponse)
async def get_state(robot_name: str):
    """
    Get the current state of a humanoid robot
    """
    try:
        robot = humanoid_manager.get_robot(robot_name)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot {robot_name} not found")

        state = robot.current_state

        # Convert joint states to the proper response format
        joint_states_dict = {}
        for joint_name, joint_state in state.joint_states.items():
            joint_states_dict[joint_name] = JointStateResponse(
                position=joint_state.position,
                velocity=joint_state.velocity,
                effort=joint_state.effort,
                limits=list(joint_state.limits),  # Convert tuple to list for JSON serialization
                stiffness=joint_state.stiffness,
                damping=joint_state.damping
            )

        response = GetStateResponse(
            robot_name=robot_name,
            joint_states=joint_states_dict,
            balance_state={
                "com_position": state.balance_state.com_position,
                "com_velocity": state.balance_state.com_velocity,
                "support_polygon": state.balance_state.support_polygon,
                "zmp_position": state.balance_state.zmp_position,
                "stability_margin": state.balance_state.stability_margin,
                "balance_mode": state.balance_state.balance_mode,
                "double_support": state.balance_state.double_support
            },
            sensor_data=state.sensor_data,
            timestamp=state.timestamp,
            is_safe=state.is_safe,
            control_mode=state.control_mode,
            success=True
        )

        logger.info(f"State retrieved for robot {robot_name}")

        return response

    except Exception as e:
        logger.error(f"Error getting state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting state: {str(e)}")


@router.post("/add-task", response_model=AddTaskResponse)
async def add_task(request: AddTaskRequest):
    """
    Add a task to the whole-body controller
    """
    try:
        robot = humanoid_manager.get_robot(request.robot_name)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot {request.robot_name} not found")

        robot.whole_body_controller.add_task(
            request.task_type,
            request.priority,
            request.constraints
        )

        response = AddTaskResponse(
            success=True,
            message=f"Task {request.task_type} added to robot {request.robot_name} with priority {request.priority}"
        )

        logger.info(f"Task {request.task_type} added to robot {request.robot_name}")

        return response

    except Exception as e:
        logger.error(f"Error adding task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding task: {str(e)}")


@router.post("/evaluate-performance", response_model=EvaluatePerformanceResponse)
async def evaluate_performance(request: EvaluatePerformanceRequest):
    """
    Evaluate the performance of a humanoid robot on a specific task
    """
    try:
        result = humanoid_manager.evaluate_performance(request.robot_name, request.task)

        if result["success"]:
            response = EvaluatePerformanceResponse(
                success=True,
                metrics=result["metrics"],
                message=None
            )
        else:
            response = EvaluatePerformanceResponse(
                success=False,
                metrics=None,
                message=result.get("error")
            )

        logger.info(f"Performance evaluation for {request.robot_name} on {request.task}: {result['success']}")

        return response

    except Exception as e:
        logger.error(f"Error evaluating performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating performance: {str(e)}")


@router.get("/platforms", response_model=ListPlatformsResponse)
async def list_platforms():
    """
    List all available humanoid platforms
    """
    try:
        platforms = humanoid_manager.list_platforms()

        response = ListPlatformsResponse(
            platforms=platforms,
            success=True
        )

        logger.info(f"Listed {len(platforms)} platforms")

        return response

    except Exception as e:
        logger.error(f"Error listing platforms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing platforms: {str(e)}")


@router.get("/robots", response_model=List[str])
async def list_robots():
    """
    List all active robots
    """
    try:
        robots = list(humanoid_manager.active_robots.keys())

        logger.info(f"Listed {len(robots)} active robots")

        return robots

    except Exception as e:
        logger.error(f"Error listing robots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing robots: {str(e)}")