from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.robotics_utils import DynamicsCalculator, RobotController
from backend.utils.logger import logger

router = APIRouter()

class SimplePendulumRequest(BaseModel):
    theta: float  # Initial angle in radians
    theta_dot: float  # Initial angular velocity in rad/s
    length: float  # Pendulum length in meters
    mass: float = 1.0  # Mass in kg
    gravity: float = 9.81  # Gravity constant

class SimplePendulumResponse(BaseModel):
    theta_ddot: float  # Angular acceleration
    kinetic_energy: float
    potential_energy: float
    total_energy: float

class MassMatrixRequest(BaseModel):
    joint_angles: List[float]  # Current joint angles
    link_masses: Optional[List[float]] = None  # Mass of each link
    link_lengths: Optional[List[float]] = None  # Length of each link

class MassMatrixResponse(BaseModel):
    mass_matrix: List[List[float]]
    matrix_size: int

class PIDControlRequest(BaseModel):
    current_value: float
    setpoint: float
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    error_integral: float = 0.0
    prev_error: float = 0.0
    dt: float = 0.01

class PIDControlResponse(BaseModel):
    control_output: float
    updated_error_integral: float
    updated_prev_error: float

class RobotDynamicsRequest(BaseModel):
    joint_angles: List[float]
    joint_velocities: List[float]
    joint_torques: List[float]
    link_masses: Optional[List[float]] = None
    link_lengths: Optional[List[float]] = None

class RobotDynamicsResponse(BaseModel):
    joint_accelerations: List[float]
    kinetic_energy: float
    potential_energy: float
    total_energy: float

@router.post("/simple-pendulum", response_model=SimplePendulumResponse)
async def calculate_simple_pendulum(request: SimplePendulumRequest):
    """
    Calculate dynamics for a simple pendulum (simplified model of single link)
    """
    try:
        theta_ddot, kinetic_energy, potential_energy = DynamicsCalculator.simple_pendulum_dynamics(
            request.theta,
            request.theta_dot,
            request.length,
            request.mass,
            request.gravity
        )

        total_energy = kinetic_energy + potential_energy

        logger.info(f"Calculated simple pendulum dynamics with length={request.length}, mass={request.mass}")

        return SimplePendulumResponse(
            theta_ddot=theta_ddot,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            total_energy=total_energy
        )

    except Exception as e:
        logger.error(f"Error calculating simple pendulum dynamics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating simple pendulum dynamics: {str(e)}")

@router.post("/mass-matrix", response_model=MassMatrixResponse)
async def calculate_mass_matrix(request: MassMatrixRequest):
    """
    Calculate mass matrix for robotic system
    """
    try:
        # Use default values if not provided
        link_masses = request.link_masses or [1.0] * len(request.joint_angles)
        link_lengths = request.link_lengths or [1.0] * len(request.joint_angles)

        if len(request.joint_angles) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 joints required for mass matrix calculation"
            )

        # For now, calculate for 2-DOF system
        mass_matrix = DynamicsCalculator.mass_matrix_2dof(
            request.joint_angles[0],
            request.joint_angles[1],
            link_masses[0] if len(link_masses) > 0 else 1.0,
            link_masses[1] if len(link_masses) > 1 else 1.0,
            link_lengths[0] if len(link_lengths) > 0 else 1.0,
            link_lengths[1] if len(link_lengths) > 1 else 1.0
        )

        logger.info(f"Calculated mass matrix for {len(request.joint_angles)} DOF system")

        return MassMatrixResponse(
            mass_matrix=mass_matrix.tolist(),
            matrix_size=len(mass_matrix)
        )

    except Exception as e:
        logger.error(f"Error calculating mass matrix: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating mass matrix: {str(e)}")

@router.post("/pid-control", response_model=PIDControlResponse)
async def calculate_pid_control(request: PIDControlRequest):
    """
    Calculate PID control output
    """
    try:
        control_output, updated_error_integral, updated_prev_error = RobotController.pid_control(
            request.current_value,
            request.setpoint,
            request.kp,
            request.ki,
            request.kd,
            request.error_integral,
            request.prev_error,
            request.dt
        )

        logger.info(f"Calculated PID control with Kp={request.kp}, Ki={request.ki}, Kd={request.kd}")

        return PIDControlResponse(
            control_output=control_output,
            updated_error_integral=updated_error_integral,
            updated_prev_error=updated_prev_error
        )

    except Exception as e:
        logger.error(f"Error calculating PID control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating PID control: {str(e)}")

@router.post("/robot-dynamics", response_model=RobotDynamicsResponse)
async def calculate_robot_dynamics(request: RobotDynamicsRequest):
    """
    Calculate robot dynamics based on the general equation: M(q)q̈ + C(q, q̇)q̇ + g(q) = τ
    This is a simplified implementation
    """
    try:
        if len(request.joint_angles) != len(request.joint_velocities) or len(request.joint_angles) != len(request.joint_torques):
            raise HTTPException(
                status_code=400,
                detail="Joint angles, velocities, and torques must have the same length"
            )

        # Use default values if not provided
        link_masses = request.link_masses or [1.0] * len(request.joint_angles)
        link_lengths = request.link_lengths or [1.0] * len(request.joint_angles)

        # For a simplified 2-DOF system
        if len(request.joint_angles) >= 2:
            # Calculate mass matrix
            mass_matrix = DynamicsCalculator.mass_matrix_2dof(
                request.joint_angles[0],
                request.joint_angles[1],
                link_masses[0] if len(link_masses) > 0 else 1.0,
                link_masses[1] if len(link_masses) > 1 else 1.0,
                link_lengths[0] if len(link_lengths) > 0 else 1.0,
                link_lengths[1] if len(link_lengths) > 1 else 1.0
            )

            # For simplicity, we'll calculate accelerations directly from torques and mass matrix
            # In reality, this would also include Coriolis and gravity terms
            tau = np.array(request.joint_torques[:2])  # Use first 2 torques
            M_inv = np.linalg.inv(mass_matrix)
            accelerations = M_inv @ tau

            # Calculate energies (simplified)
            kinetic_energy = 0.5 * request.joint_velocities[0]**2 * link_masses[0] + \
                           0.5 * request.joint_velocities[1]**2 * link_masses[1]
            potential_energy = link_masses[0] * 9.81 * link_lengths[0] * (1 - np.cos(request.joint_angles[0])) + \
                             link_masses[1] * 9.81 * link_lengths[1] * (1 - np.cos(request.joint_angles[1]))
            total_energy = kinetic_energy + potential_energy

            # Create response with accelerations for all joints (pad with zeros if needed)
            all_accelerations = [float(acc) for acc in accelerations]
            if len(all_accelerations) < len(request.joint_angles):
                all_accelerations.extend([0.0] * (len(request.joint_angles) - len(all_accelerations)))

            logger.info(f"Calculated robot dynamics for {len(request.joint_angles)} DOF system")

            return RobotDynamicsResponse(
                joint_accelerations=all_accelerations,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy
            )
        else:
            # For single joint or other configurations
            # Calculate simple single-link dynamics
            link_mass = link_masses[0] if len(link_masses) > 0 else 1.0
            link_length = link_lengths[0] if len(link_lengths) > 0 else 1.0

            # For a simple single link, acceleration = torque / (mass * length^2)
            if link_mass * link_length**2 != 0:
                acceleration = request.joint_torques[0] / (link_mass * link_length**2)
            else:
                acceleration = 0.0

            # Calculate energies (simplified)
            kinetic_energy = 0.5 * link_mass * (link_length * request.joint_velocities[0])**2
            potential_energy = link_mass * 9.81 * link_length * (1 - np.cos(request.joint_angles[0]))
            total_energy = kinetic_energy + potential_energy

            all_accelerations = [acceleration]
            if len(all_accelerations) < len(request.joint_angles):
                all_accelerations.extend([0.0] * (len(request.joint_angles) - len(all_accelerations)))

            logger.info(f"Calculated simplified robot dynamics for single joint")

            return RobotDynamicsResponse(
                joint_accelerations=all_accelerations,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy
            )

    except Exception as e:
        logger.error(f"Error calculating robot dynamics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating robot dynamics: {str(e)}")

class SimulationTrajectoryRequest(BaseModel):
    initial_joints: List[float]
    target_joints: List[float]
    steps: int = 100

class SimulationTrajectoryResponse(BaseModel):
    trajectory: List[List[float]]
    step_count: int

@router.post("/simulate-motion", response_model=SimulationTrajectoryResponse)
async def simulate_robot_motion(request: SimulationTrajectoryRequest):
    """
    Simulate smooth motion from initial to target joint configuration
    """
    try:
        if len(request.initial_joints) != len(request.target_joints):
            raise HTTPException(
                status_code=400,
                detail="Initial and target joint configurations must have the same length"
            )

        from backend.utils.robotics_utils import simulate_robot_motion

        trajectory = simulate_robot_motion(
            request.initial_joints,
            request.target_joints,
            request.steps
        )

        logger.info(f"Simulated motion trajectory with {len(trajectory)} steps")

        return SimulationTrajectoryResponse(
            trajectory=trajectory,
            step_count=len(trajectory)
        )

    except Exception as e:
        logger.error(f"Error simulating robot motion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error simulating robot motion: {str(e)}")