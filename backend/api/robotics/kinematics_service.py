from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.robotics_utils import KinematicsCalculator, Pose, JointState
from backend.utils.logger import logger

router = APIRouter()

class ForwardKinematicsRequest(BaseModel):
    joint_angles: List[float]  # in radians
    link_lengths: List[float]  # in meters
    dh_parameters: Optional[List[Dict[str, float]]] = None  # Optional DH parameters

class ForwardKinematicsResponse(BaseModel):
    end_effector_pose: Optional[Pose]
    position: Dict[str, float]  # x, y, z coordinates
    joint_positions: List[Dict[str, float]]  # Position of each joint

class InverseKinematicsRequest(BaseModel):
    target_position: Dict[str, float]  # x, y, z coordinates
    link_lengths: List[float]  # in meters
    initial_guess: Optional[List[float]] = None

class InverseKinematicsResponse(BaseModel):
    joint_angles: List[List[float]]  # Multiple possible solutions
    success: bool
    message: str

class DHParameters(BaseModel):
    a: float  # link length
    alpha: float  # link twist
    d: float  # link offset
    theta: float  # joint angle

class DHTransformRequest(BaseModel):
    dh_params: List[DHParameters]

class DHTransformResponse(BaseModel):
    transforms: List[List[List[float]]]  # List of 4x4 transformation matrices
    final_pose: Optional[Pose]

@router.post("/forward-kinematics", response_model=ForwardKinematicsResponse)
async def calculate_forward_kinematics(request: ForwardKinematicsRequest):
    """
    Calculate forward kinematics for a robotic manipulator
    """
    try:
        if len(request.joint_angles) != len(request.link_lengths):
            raise HTTPException(
                status_code=400,
                detail="Number of joint angles must match number of link lengths"
            )

        # For now, implement 2-DOF planar manipulator calculation
        if len(request.joint_angles) >= 2:
            x, y = KinematicsCalculator.forward_kinematics_2dof(
                request.joint_angles[0],
                request.joint_angles[1],
                request.link_lengths[0],
                request.link_lengths[1]
            )

            # Calculate joint positions along the arm
            joint_positions = [
                {"x": 0.0, "y": 0.0, "z": 0.0}  # Base position
            ]

            # First joint position
            x1 = request.link_lengths[0] * np.cos(request.joint_angles[0])
            y1 = request.link_lengths[0] * np.sin(request.joint_angles[0])
            joint_positions.append({"x": x1, "y": y1, "z": 0.0})

            # End effector position
            joint_positions.append({"x": x, "y": y, "z": 0.0})

            response = ForwardKinematicsResponse(
                end_effector_pose=Pose(x=x, y=y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
                position={"x": x, "y": y, "z": 0.0},
                joint_positions=joint_positions
            )
        else:
            # For single joint or other configurations
            x = request.link_lengths[0] * np.cos(request.joint_angles[0]) if len(request.joint_angles) > 0 else 0.0
            y = request.link_lengths[0] * np.sin(request.joint_angles[0]) if len(request.joint_angles) > 0 else 0.0

            response = ForwardKinematicsResponse(
                end_effector_pose=Pose(x=x, y=y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
                position={"x": x, "y": y, "z": 0.0},
                joint_positions=[{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": x, "y": y, "z": 0.0}]
            )

        logger.info(f"Calculated forward kinematics for {len(request.joint_angles)} joints")

        return response

    except Exception as e:
        logger.error(f"Error calculating forward kinematics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating forward kinematics: {str(e)}")

@router.post("/inverse-kinematics", response_model=InverseKinematicsResponse)
async def calculate_inverse_kinematics(request: InverseKinematicsRequest):
    """
    Calculate inverse kinematics for a robotic manipulator
    """
    try:
        if len(request.link_lengths) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 link lengths required for inverse kinematics"
            )

        # For now, implement 2-DOF planar manipulator calculation
        solutions = KinematicsCalculator.inverse_kinematics_2dof(
            request.target_position["x"],
            request.target_position["y"],
            request.link_lengths[0],
            request.link_lengths[1]
        )

        if solutions:
            logger.info(f"Found {len(solutions)} solutions for inverse kinematics")
            return InverseKinematicsResponse(
                joint_angles=[list(sol) for sol in solutions],
                success=True,
                message=f"Successfully calculated {len(solutions)} possible joint configurations"
            )
        else:
            logger.info("No solutions found for inverse kinematics - target may be unreachable")
            return InverseKinematicsResponse(
                joint_angles=[],
                success=False,
                message="No valid joint configurations found - target may be unreachable"
            )

    except Exception as e:
        logger.error(f"Error calculating inverse kinematics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating inverse kinematics: {str(e)}")

@router.post("/dh-transform", response_model=DHTransformResponse)
async def calculate_dh_transform(request: DHTransformRequest):
    """
    Calculate Denavit-Hartenberg transformation matrices
    """
    try:
        transforms = []
        cumulative_transform = np.eye(4)

        for dh_param in request.dh_params:
            transform = KinematicsCalculator.dh_transform(
                dh_param.a, dh_param.alpha, dh_param.d, dh_param.theta
            )
            cumulative_transform = cumulative_transform @ transform
            transforms.append(transform.tolist())

        # Extract final pose from cumulative transformation
        final_transform = cumulative_transform
        x, y, z = final_transform[0, 3], final_transform[1, 3], final_transform[2, 3]

        # Extract orientation (simplified - in practice, this would be more complex)
        roll = np.arctan2(final_transform[2, 1], final_transform[2, 2])
        pitch = np.arcsin(-final_transform[2, 0])
        yaw = np.arctan2(final_transform[1, 0], final_transform[0, 0])

        final_pose = Pose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

        logger.info(f"Calculated DH transforms for {len(request.dh_params)} joints")

        return DHTransformResponse(
            transforms=transforms,
            final_pose=final_pose
        )

    except Exception as e:
        logger.error(f"Error calculating DH transforms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating DH transforms: {str(e)}")

class RobotMetricsRequest(BaseModel):
    joint_angles: List[float]
    link_lengths: List[float]

class RobotMetricsResponse(BaseModel):
    configuration: int
    total_length: float
    workspace_volume: float
    end_effector_position: Optional[Dict[str, float]]
    reachable_points: List[Dict[str, float]]

@router.post("/metrics", response_model=RobotMetricsResponse)
async def calculate_robot_metrics(request: RobotMetricsRequest):
    """
    Calculate various metrics for a robot configuration
    """
    try:
        from backend.utils.robotics_utils import calculate_robot_metrics

        metrics = calculate_robot_metrics(request.joint_angles, request.link_lengths)

        logger.info(f"Calculated metrics for robot with {len(request.joint_angles)} joints")

        return RobotMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error calculating robot metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating robot metrics: {str(e)}")