from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.motion_control_utils import (
    PIDController, JointController, TrajectoryGenerator,
    OperationalSpaceController, ImpedanceController, Simulator,
    compute_forward_kinematics_2dof, compute_inverse_kinematics_2dof
)
from backend.utils.logger import logger

router = APIRouter()

class PIDControlRequest(BaseModel):
    setpoint: float
    measured_value: float
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.01

class PIDControlResponse(BaseModel):
    control_output: float
    error: float
    proportional_term: float
    integral_term: float
    derivative_term: float

class JointControlRequest(BaseModel):
    current_positions: List[float]
    desired_positions: List[float]
    current_velocities: Optional[List[float]] = None
    desired_velocities: Optional[List[float]] = None
    pid_gains: Optional[Dict[str, List[float]]] = None  # kp, ki, kd for each joint

class JointControlResponse(BaseModel):
    joint_torques: List[float]
    num_joints: int

class TrajectoryGenerationRequest(BaseModel):
    start_positions: List[float]
    end_positions: List[float]
    duration: float = 1.0
    dt: float = 0.01

class TrajectoryPointResponse(BaseModel):
    time: float
    positions: List[float]
    velocities: List[float]
    accelerations: List[float]

class TrajectoryGenerationResponse(BaseModel):
    trajectory: List[TrajectoryPointResponse]
    num_points: int

class OperationalSpaceControlRequest(BaseModel):
    current_pos: List[float]  # Cartesian position
    desired_pos: List[float]  # Cartesian position
    current_vel: Optional[List[float]] = None
    desired_vel: Optional[List[float]] = None
    jacobian: Optional[List[List[float]]] = None  # 2D matrix
    kp: float = 10.0
    kd: float = 2.0

class OperationalSpaceControlResponse(BaseModel):
    cartesian_force: List[float]
    joint_torques: Optional[List[float]]

class ImpedanceControlRequest(BaseModel):
    pos_error: List[float]
    vel_error: List[float]
    mass: float = 1.0
    damping: float = 10.0
    stiffness: float = 100.0

class ImpedanceControlResponse(BaseModel):
    impedance_force: List[float]

class SimulationRequest(BaseModel):
    initial_positions: List[float]
    torques: List[float]
    steps: int = 100
    dt: float = 0.01
    friction_coeff: float = 0.1

class SimulationResponse(BaseModel):
    positions: List[List[float]]
    velocities: List[List[float]]
    accelerations: List[List[float]]
    time_stamps: List[float]

class KinematicsRequest(BaseModel):
    joint_angles: List[float]
    link_lengths: List[float]
    method: str = "forward"  # "forward" or "inverse"

class KinematicsResponse(BaseModel):
    result: Optional[List[float]]
    success: bool
    message: str

@router.post("/pid-control", response_model=PIDControlResponse)
async def pid_control(request: PIDControlRequest):
    """
    Compute PID control output
    """
    try:
        # Create a temporary PID controller with the provided gains
        pid = PIDController(kp=request.kp, ki=request.ki, kd=request.kd, dt=request.dt)

        # For this endpoint, we'll calculate the terms manually to return detailed information
        error = request.setpoint - request.measured_value

        # Proportional term
        p_term = request.kp * error

        # Integral term (simplified for single calculation)
        i_term = request.ki * error * request.dt

        # Derivative term
        d_term = request.kd * (error / request.dt)  # Simplified derivative

        control_output = p_term + i_term + d_term

        response = PIDControlResponse(
            control_output=control_output,
            error=error,
            proportional_term=p_term,
            integral_term=i_term,
            derivative_term=d_term
        )

        logger.info(f"PID control computed: setpoint={request.setpoint}, measured={request.measured_value}")

        return response

    except Exception as e:
        logger.error(f"Error in PID control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in PID control: {str(e)}")

@router.post("/joint-control", response_model=JointControlResponse)
async def joint_control(request: JointControlRequest):
    """
    Compute joint torques using PID control
    """
    try:
        num_joints = len(request.current_positions)

        if len(request.desired_positions) != num_joints:
            raise HTTPException(
                status_code=400,
                detail="Current and desired positions must have the same length"
            )

        # Use default PID gains if not provided
        controller = JointController(num_joints)

        # If custom gains are provided, update the controllers
        if request.pid_gains:
            if 'kp' in request.pid_gains and len(request.pid_gains['kp']) == num_joints:
                for i, kp in enumerate(request.pid_gains['kp']):
                    controller.pid_controllers[i].kp = kp
            if 'ki' in request.pid_gains and len(request.pid_gains['ki']) == num_joints:
                for i, ki in enumerate(request.pid_gains['ki']):
                    controller.pid_controllers[i].ki = ki
            if 'kd' in request.pid_gains and len(request.pid_gains['kd']) == num_joints:
                for i, kd in enumerate(request.pid_gains['kd']):
                    controller.pid_controllers[i].kd = kd

        # Compute torques
        torques = controller.compute_joint_torques(
            request.current_positions,
            request.desired_positions,
            request.current_velocities or [0.0] * num_joints,
            request.desired_velocities or [0.0] * num_joints
        )

        response = JointControlResponse(
            joint_torques=torques,
            num_joints=num_joints
        )

        logger.info(f"Joint control computed for {num_joints} joints")

        return response

    except Exception as e:
        logger.error(f"Error in joint control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in joint control: {str(e)}")

@router.post("/generate-trajectory", response_model=TrajectoryGenerationResponse)
async def generate_trajectory(request: TrajectoryGenerationRequest):
    """
    Generate joint space trajectory
    """
    try:
        if len(request.start_positions) != len(request.end_positions):
            raise HTTPException(
                status_code=400,
                detail="Start and end positions must have the same length"
            )

        # Generate trajectory
        trajectory_points = TrajectoryGenerator.generate_multiple_joint_trajectory(
            request.start_positions,
            request.end_positions,
            request.duration,
            request.dt
        )

        # Convert to response format
        trajectory_response = []
        for point in trajectory_points:
            trajectory_response.append(TrajectoryPointResponse(
                time=point.time,
                positions=point.positions,
                velocities=point.velocities,
                accelerations=point.accelerations or [0.0] * len(point.positions)
            ))

        response = TrajectoryGenerationResponse(
            trajectory=trajectory_response,
            num_points=len(trajectory_response)
        )

        logger.info(f"Generated trajectory with {len(trajectory_response)} points for {len(request.start_positions)} joints")

        return response

    except Exception as e:
        logger.error(f"Error generating trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating trajectory: {str(e)}")

@router.post("/operational-space-control", response_model=OperationalSpaceControlResponse)
async def operational_space_control(request: OperationalSpaceControlRequest):
    """
    Compute operational space (Cartesian) control
    """
    try:
        if len(request.current_pos) != len(request.desired_pos):
            raise HTTPException(
                status_code=400,
                detail="Current and desired positions must have the same length"
            )

        # Create controller
        controller = OperationalSpaceController(
            num_joints=len(request.current_pos),  # This is approximate
            kp=request.kp,
            kd=request.kd
        )

        # Convert to numpy arrays
        current_pos = np.array(request.current_pos)
        desired_pos = np.array(request.desired_pos)
        current_vel = np.array(request.current_vel) if request.current_vel else np.zeros_like(current_pos)
        desired_vel = np.array(request.desired_vel) if request.desired_vel else np.zeros_like(desired_pos)

        # Compute Cartesian force
        cartesian_force = controller.compute_cartesian_control(
            current_pos, desired_pos, current_vel, desired_vel
        )

        # If Jacobian is provided, map to joint torques
        joint_torques = None
        if request.jacobian:
            jacobian = np.array(request.jacobian)
            joint_torques = controller.map_cartesian_to_joint(cartesian_force, jacobian).tolist()

        response = OperationalSpaceControlResponse(
            cartesian_force=cartesian_force.tolist(),
            joint_torques=joint_torques
        )

        logger.info(f"Operational space control computed for {len(request.current_pos)}D space")

        return response

    except Exception as e:
        logger.error(f"Error in operational space control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in operational space control: {str(e)}")

@router.post("/impedance-control", response_model=ImpedanceControlResponse)
async def impedance_control(request: ImpedanceControlRequest):
    """
    Compute impedance control force
    """
    try:
        # Create controller
        controller = ImpedanceController(
            mass=request.mass,
            damping=request.damping,
            stiffness=request.stiffness
        )

        # Convert to numpy arrays
        pos_error = np.array(request.pos_error)
        vel_error = np.array(request.vel_error)

        if len(pos_error) != len(vel_error):
            raise HTTPException(
                status_code=400,
                detail="Position and velocity error vectors must have the same length"
            )

        # Compute impedance force
        impedance_force = controller.compute_impedance_force(pos_error, vel_error)

        response = ImpedanceControlResponse(
            impedance_force=impedance_force.tolist()
        )

        logger.info(f"Impedance control computed for {len(request.pos_error)}D space")

        return response

    except Exception as e:
        logger.error(f"Error in impedance control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in impedance control: {str(e)}")

@router.post("/simulate-robot-motion", response_model=SimulationResponse)
async def simulate_motion(request: SimulationRequest):
    """
    Simulate robot motion with applied torques
    """
    try:
        if len(request.initial_positions) != len(request.torques):
            raise HTTPException(
                status_code=400,
                detail="Initial positions and torques must have the same length"
            )

        # Create simulator
        simulator = Simulator(len(request.initial_positions), request.dt)

        # Set initial conditions
        simulator.positions = np.array(request.initial_positions)
        simulator.velocities = np.zeros(len(request.initial_positions))
        simulator.accelerations = np.zeros(len(request.initial_positions))

        positions_history = [request.initial_positions[:]]
        velocities_history = [[0.0] * len(request.initial_positions)]
        accelerations_history = [[0.0] * len(request.initial_positions)]
        time_stamps = [0.0]

        # Run simulation
        for step in range(request.steps):
            # Apply the same torques for the entire simulation (simplified)
            pos, vel, acc = simulator.step(request.torques, request.friction_coeff)

            positions_history.append(pos[:])
            velocities_history.append(vel[:])
            accelerations_history.append(acc[:])
            time_stamps.append((step + 1) * request.dt)

        response = SimulationResponse(
            positions=positions_history,
            velocities=velocities_history,
            accelerations=accelerations_history,
            time_stamps=time_stamps
        )

        logger.info(f"Motion simulation completed for {request.steps} steps with {len(request.initial_positions)} joints")

        return response

    except Exception as e:
        logger.error(f"Error in motion simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in motion simulation: {str(e)}")

@router.post("/kinematics", response_model=KinematicsResponse)
async def kinematics_calculation(request: KinematicsRequest):
    """
    Perform forward or inverse kinematics calculation
    """
    try:
        if request.method.lower() == "forward":
            if len(request.joint_angles) != len(request.link_lengths):
                raise HTTPException(
                    status_code=400,
                    detail="Number of joint angles must match number of link lengths for forward kinematics"
                )

            # For 2-DOF system, use the utility function
            if len(request.joint_angles) == 2 and len(request.link_lengths) == 2:
                x, y = compute_forward_kinematics_2dof(
                    request.joint_angles[0],
                    request.joint_angles[1],
                    request.link_lengths[0],
                    request.link_lengths[1]
                )
                result = [x, y]
                success = True
                message = "Forward kinematics calculated successfully"
            else:
                # For more complex systems, this would need to be extended
                result = [0.0, 0.0]  # Placeholder
                success = False
                message = f"Forward kinematics not implemented for {len(request.joint_angles)} DOF system"

        elif request.method.lower() == "inverse":
            if len(request.joint_angles) != 2:  # Expected target position [x, y]
                raise HTTPException(
                    status_code=400,
                    detail="For inverse kinematics, provide target position [x, y] and link lengths"
                )

            if len(request.link_lengths) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Link lengths must be provided for inverse kinematics"
                )

            # Perform inverse kinematics
            solution = compute_inverse_kinematics_2dof(
                request.joint_angles[0],  # x
                request.joint_angles[1],  # y
                request.link_lengths[0],
                request.link_lengths[1]
            )

            if solution:
                result = [solution[0], solution[1]]
                success = True
                message = "Inverse kinematics calculated successfully"
            else:
                result = None
                success = False
                message = "Target position is not reachable"
        else:
            raise HTTPException(
                status_code=400,
                detail="Method must be either 'forward' or 'inverse'"
            )

        response = KinematicsResponse(
            result=result,
            success=success,
            message=message
        )

        logger.info(f"Kinematics calculation ({request.method}) completed: {message}")

        return response

    except Exception as e:
        logger.error(f"Error in kinematics calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in kinematics calculation: {str(e)}")

class ControlPerformanceRequest(BaseModel):
    control_type: str  # "pid", "joint", "operational", etc.
    parameters: Dict[str, Any]
    simulation_time: float = 5.0
    dt: float = 0.01

class ControlPerformanceResponse(BaseModel):
    performance_metrics: Dict[str, float]
    stability: bool
    settling_time: float
    overshoot: float
    steady_state_error: float

@router.post("/control-performance", response_model=ControlPerformanceResponse)
async def control_performance_analysis(request: ControlPerformanceRequest):
    """
    Analyze control system performance
    """
    try:
        # This is a simplified analysis - in a real system, this would run a more complex simulation
        # For now, we'll provide a basic analysis based on control type

        if request.control_type == "pid":
            # Basic PID performance metrics
            # These would be computed from a real simulation in practice
            performance_metrics = {
                "rise_time": 0.5,
                "settling_time": 1.2,
                "overshoot": 0.15,
                "steady_state_error": 0.02,
                "integral_square_error": 0.8
            }
            stability = True
        else:
            # Default metrics for other control types
            performance_metrics = {
                "rise_time": 0.8,
                "settling_time": 2.0,
                "overshoot": 0.1,
                "steady_state_error": 0.05,
                "integral_square_error": 1.2
            }
            stability = True

        response = ControlPerformanceResponse(
            performance_metrics=performance_metrics,
            stability=stability,
            settling_time=performance_metrics["settling_time"],
            overshoot=performance_metrics["overshoot"],
            steady_state_error=performance_metrics["steady_state_error"]
        )

        logger.info(f"Control performance analysis completed for {request.control_type} controller")

        return response

    except Exception as e:
        logger.error(f"Error in control performance analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in control performance analysis: {str(e)}")