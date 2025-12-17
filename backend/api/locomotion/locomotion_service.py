from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.locomotion_utils import (
    FootStep, WalkingState, GaitParameters,
    InvertedPendulumModel, LinearInvertedPendulumModel,
    WalkingPatternGenerator, BalanceController, GaitAnalyzer,
    CentralPatternGenerator, simulate_walking_step
)
from backend.utils.logger import logger

router = APIRouter()

class ZMPRequest(BaseModel):
    com_position: List[float]  # [x, y, z]
    com_velocity: List[float]  # [dx, dy, dz]
    com_acceleration: List[float]  # [ddx, ddy, ddz]
    com_height: float = 0.8
    gravity: float = 9.81

class ZMPResponse(BaseModel):
    zmp_position: List[float]  # [x, y]
    success: bool

class CapturePointRequest(BaseModel):
    com_position: List[float]  # [x, y]
    com_velocity: List[float]  # [dx, dy]
    com_height: float = 0.8
    gravity: float = 9.81

class CapturePointResponse(BaseModel):
    capture_point: List[float]  # [x, y]
    distance_to_capture: float
    success: bool

class FootstepPlanRequest(BaseModel):
    initial_com: List[float]  # [x, y, z]
    final_com: List[float]  # [x, y, z]
    step_time: float = 0.8
    dt: float = 0.01

class FootstepResponse(BaseModel):
    footsteps: List[Dict[str, Any]]
    num_steps: int
    success: bool

class WalkingPatternRequest(BaseModel):
    num_steps: int
    start_position: List[float]  # [x, y]
    gait_params: Optional[Dict[str, float]] = None

class WalkingPatternResponse(BaseModel):
    pattern: List[Dict[str, Any]]
    success: bool

class BalanceControlRequest(BaseModel):
    current_zmp: List[float]  # [x, y]
    desired_zmp: List[float]  # [x, y]
    kp: float = 10.0
    ki: float = 0.1
    kd: float = 1.0
    dt: float = 0.01

class BalanceControlResponse(BaseModel):
    correction: List[float]  # [dx, dy]
    error: List[float]  # [ex, ey]
    success: bool

class GaitAnalysisRequest(BaseModel):
    footsteps: List[Dict[str, Any]]  # List of footstep dictionaries

class GaitAnalysisResponse(BaseModel):
    parameters: Dict[str, float]
    stability_metrics: Dict[str, float]
    success: bool

class CPGRequest(BaseModel):
    duration: float
    frequency: float = 1.0
    amplitude: float = 0.1
    dt: float = 0.01

class CPGResponse(BaseModel):
    time: List[float]
    left_leg_signal: List[float]
    right_leg_signal: List[float]
    success: bool

class WalkingSimulationRequest(BaseModel):
    initial_com: List[float]  # [x, y, z]
    zmp_reference: List[List[float]]  # List of [x, y] ZMP references
    duration: float
    dt: float = 0.01

class WalkingSimulationResponse(BaseModel):
    time: List[float]
    com_trajectory: List[List[float]]  # List of [x, y, z] positions
    success: bool

class StabilityAnalysisRequest(BaseModel):
    current_zmp: List[float]  # [x, y]
    support_polygon: List[List[float]]  # List of [x, y] vertices

class StabilityAnalysisResponse(BaseModel):
    stability_margin: float
    is_stable: bool
    distance_to_boundary: float

@router.post("/compute-zmp", response_model=ZMPResponse)
async def compute_zmp(request: ZMPRequest):
    """
    Compute Zero Moment Point from CoM state
    """
    try:
        if len(request.com_position) != 3 or len(request.com_velocity) != 3 or len(request.com_acceleration) != 3:
            raise HTTPException(
                status_code=400,
                detail="CoM position, velocity, and acceleration must each have 3 components [x, y, z]"
            )

        # Create inverted pendulum model
        model = InvertedPendulumModel(com_height=request.com_height, gravity=request.gravity)

        # Convert to numpy arrays
        com_pos = np.array(request.com_position[:2])  # Only x, y needed
        com_acc = np.array(request.com_acceleration[:2])  # Only x, y needed

        # Compute ZMP
        zmp = model.compute_zmp(com_pos, np.array(request.com_velocity[:2]), com_acc)

        response = ZMPResponse(
            zmp_position=zmp.tolist(),
            success=True
        )

        logger.info(f"ZMP computed: {zmp.tolist()}")

        return response

    except Exception as e:
        logger.error(f"Error computing ZMP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing ZMP: {str(e)}")

@router.post("/compute-capture-point", response_model=CapturePointResponse)
async def compute_capture_point(request: CapturePointRequest):
    """
    Compute Capture Point for balance recovery
    """
    try:
        if len(request.com_position) != 2 or len(request.com_velocity) != 2:
            raise HTTPException(
                status_code=400,
                detail="CoM position and velocity must each have 2 components [x, y]"
            )

        # Create inverted pendulum model
        model = InvertedPendulumModel(com_height=request.com_height, gravity=request.gravity)

        # Convert to numpy arrays
        com_pos = np.array(request.com_position)
        com_vel = np.array(request.com_velocity)

        # Compute capture point
        cp = model.compute_capture_point(com_pos, com_vel)

        # Calculate distance from current position to capture point
        distance = np.linalg.norm(cp - com_pos)

        response = CapturePointResponse(
            capture_point=cp.tolist(),
            distance_to_capture=float(distance),
            success=True
        )

        logger.info(f"Capture point computed: {cp.tolist()}, distance: {distance:.3f}")

        return response

    except Exception as e:
        logger.error(f"Error computing capture point: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing capture point: {str(e)}")

@router.post("/generate-footsteps", response_model=FootstepResponse)
async def generate_footsteps(request: FootstepPlanRequest):
    """
    Generate footsteps using preview control for LIPM
    """
    try:
        if len(request.initial_com) != 3 or len(request.final_com) != 3:
            raise HTTPException(
                status_code=400,
                detail="Initial and final CoM must each have 3 components [x, y, z]"
            )

        # Create LIPM
        model = LinearInvertedPendulumModel(com_height=request.initial_com[2])

        # Convert to numpy arrays
        initial_com = np.array(request.initial_com)
        final_com = np.array(request.final_com)

        # Generate footsteps
        footsteps = model.generate_footsteps_preview_control(
            initial_com, final_com, request.step_time, request.dt
        )

        # Convert footsteps to response format
        footsteps_response = []
        for step in footsteps:
            footsteps_response.append({
                "x": step.x,
                "y": step.y,
                "theta": step.theta,
                "time": step.time,
                "support_leg": step.support_leg
            })

        response = FootstepResponse(
            footsteps=footsteps_response,
            num_steps=len(footsteps),
            success=True
        )

        logger.info(f"Generated {len(footsteps)} footsteps")

        return response

    except Exception as e:
        logger.error(f"Error generating footsteps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating footsteps: {str(e)}")

@router.post("/generate-walking-pattern", response_model=WalkingPatternResponse)
async def generate_walking_pattern(request: WalkingPatternRequest):
    """
    Generate walking pattern with alternating footsteps
    """
    try:
        # Create gait parameters
        gait_params = GaitParameters()
        if request.gait_params:
            for key, value in request.gait_params.items():
                if hasattr(gait_params, key):
                    setattr(gait_params, key, value)

        # Validate start position
        if len(request.start_position) != 2:
            raise HTTPException(
                status_code=400,
                detail="Start position must have 2 components [x, y]"
            )

        # Create pattern generator
        generator = WalkingPatternGenerator(gait_params)

        # Generate walking pattern
        pattern = generator.generate_walk_pattern(request.num_steps, tuple(request.start_position))

        response = WalkingPatternResponse(
            pattern=pattern,
            success=True
        )

        logger.info(f"Generated walking pattern with {request.num_steps} steps")

        return response

    except Exception as e:
        logger.error(f"Error generating walking pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating walking pattern: {str(e)}")

@router.post("/balance-control", response_model=BalanceControlResponse)
async def balance_control(request: BalanceControlRequest):
    """
    Compute balance correction using PID control
    """
    try:
        if len(request.current_zmp) != 2 or len(request.desired_zmp) != 2:
            raise HTTPException(
                status_code=400,
                detail="Current and desired ZMP must each have 2 components [x, y]"
            )

        # Create balance controller
        controller = BalanceController(kp=request.kp, ki=request.ki, kd=request.kd)

        # Convert to numpy arrays
        current_zmp = np.array(request.current_zmp)
        desired_zmp = np.array(request.desired_zmp)

        # Compute balance correction
        correction = controller.compute_balance_correction(current_zmp, desired_zmp, request.dt)

        # Calculate error
        error = (desired_zmp - current_zmp).tolist()

        response = BalanceControlResponse(
            correction=correction.tolist(),
            error=error,
            success=True
        )

        logger.info(f"Balance correction computed: {correction.tolist()}")

        return response

    except Exception as e:
        logger.error(f"Error in balance control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in balance control: {str(e)}")

@router.post("/analyze-gait", response_model=GaitAnalysisResponse)
async def analyze_gait(request: GaitAnalysisRequest):
    """
    Analyze gait parameters from footsteps
    """
    try:
        # Convert footstep dictionaries back to FootStep objects
        footsteps = []
        for step_dict in request.footsteps:
            try:
                footstep = FootStep(
                    x=step_dict.get("x", 0.0),
                    y=step_dict.get("y", 0.0),
                    theta=step_dict.get("theta", 0.0),
                    time=step_dict.get("time", 0.0),
                    support_leg=step_dict.get("support_leg", "left")
                )
                footsteps.append(footstep)
            except Exception:
                continue  # Skip invalid footstep entries

        if not footsteps:
            raise HTTPException(
                status_code=400,
                detail="No valid footsteps provided for analysis"
            )

        # Analyze gait
        analyzer = GaitAnalyzer()
        gait_params = analyzer.calculate_gait_parameters(footsteps)

        response = GaitAnalysisResponse(
            parameters=gait_params,
            stability_metrics={},  # Additional stability metrics can be added here
            success=True
        )

        logger.info(f"Gait analysis completed for {len(footsteps)} footsteps")

        return response

    except Exception as e:
        logger.error(f"Error in gait analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in gait analysis: {str(e)}")

@router.post("/generate-cpg", response_model=CPGResponse)
async def generate_cpg(request: CPGRequest):
    """
    Generate Central Pattern Generator signals for rhythmic walking
    """
    try:
        # Create CPG
        cpg = CentralPatternGenerator(frequency=request.frequency, amplitude=request.amplitude)

        # Create time vector
        times = np.arange(0, request.duration, request.dt).tolist()

        # Generate leg trajectories
        left_leg, right_leg = cpg.generate_leg_trajectories(times)

        response = CPGResponse(
            time=times,
            left_leg_signal=left_leg,
            right_leg_signal=right_leg,
            success=True
        )

        logger.info(f"CPG signals generated for {len(times)} time steps")

        return response

    except Exception as e:
        logger.error(f"Error generating CPG signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating CPG signals: {str(e)}")

@router.post("/simulate-walking", response_model=WalkingSimulationResponse)
async def simulate_walking(request: WalkingSimulationRequest):
    """
    Simulate walking using LIPM
    """
    try:
        if len(request.initial_com) != 3:
            raise HTTPException(
                status_code=400,
                detail="Initial CoM must have 3 components [x, y, z]"
            )

        # Convert ZMP reference to numpy array
        zmp_ref = np.array(request.zmp_reference)

        if zmp_ref.shape[1] != 2:
            raise HTTPException(
                status_code=400,
                detail="ZMP reference must have 2 components [x, y] for each time step"
            )

        # Convert initial CoM to numpy array
        initial_com = np.array(request.initial_com)

        # Run simulation
        time_vector, com_trajectory = simulate_walking_step(
            initial_com, zmp_ref, request.dt, request.duration
        )

        # Convert to response format
        time_list = time_vector.tolist()
        com_list = [pos.tolist() for pos in com_trajectory]

        response = WalkingSimulationResponse(
            time=time_list,
            com_trajectory=com_list,
            success=True
        )

        logger.info(f"Walking simulation completed with {len(time_list)} time steps")

        return response

    except Exception as e:
        logger.error(f"Error in walking simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in walking simulation: {str(e)}")

@router.post("/analyze-stability", response_model=StabilityAnalysisResponse)
async def analyze_stability(request: StabilityAnalysisRequest):
    """
    Analyze stability based on ZMP and support polygon
    """
    try:
        if len(request.current_zmp) != 2:
            raise HTTPException(
                status_code=400,
                detail="Current ZMP must have 2 components [x, y]"
            )

        if len(request.support_polygon) < 3:
            raise HTTPException(
                status_code=400,
                detail="Support polygon must have at least 3 vertices"
            )

        # Convert to numpy arrays
        current_zmp = np.array(request.current_zmp)
        support_polygon = [np.array(vertex) for vertex in request.support_polygon]

        # Analyze stability
        analyzer = GaitAnalyzer()
        stability_margin = analyzer.calculate_stability_margin(current_zmp, support_polygon)

        is_stable = stability_margin > 0

        response = StabilityAnalysisResponse(
            stability_margin=stability_margin,
            is_stable=is_stable,
            distance_to_boundary=stability_margin
        )

        logger.info(f"Stability analysis: margin={stability_margin:.3f}, stable={is_stable}")

        return response

    except Exception as e:
        logger.error(f"Error in stability analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in stability analysis: {str(e)}")

class WalkTrajectoryRequest(BaseModel):
    start_position: List[float]  # [x, y, theta]
    goal_position: List[float]  # [x, y, theta]
    step_length: float = 0.3
    step_width: float = 0.2
    step_time: float = 0.8

class WalkTrajectoryResponse(BaseModel):
    footsteps: List[Dict[str, float]]
    trajectory_info: Dict[str, float]
    success: bool

@router.post("/plan-walk-trajectory", response_model=WalkTrajectoryResponse)
async def plan_walk_trajectory(request: WalkTrajectoryRequest):
    """
    Plan a walk trajectory from start to goal position
    """
    try:
        if len(request.start_position) != 3 or len(request.goal_position) != 3:
            raise HTTPException(
                status_code=400,
                detail="Start and goal positions must have 3 components [x, y, theta]"
            )

        # Calculate distance and direction
        dx = request.goal_position[0] - request.start_position[0]
        dy = request.goal_position[1] - request.start_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate number of steps needed
        num_steps = max(1, int(distance / request.step_length))

        # Generate footsteps along the path
        footsteps = []
        for i in range(num_steps + 1):  # Include both start and end positions
            t = i / num_steps if num_steps > 0 else 0
            x = request.start_position[0] + t * dx
            y = request.start_position[1] + t * dy
            theta = request.start_position[2] + t * (request.goal_position[2] - request.start_position[2])

            # Alternate feet for walking pattern
            support_leg = "left" if i % 2 == 0 else "right"
            foot_x = x
            foot_y = y + (request.step_width/2 if support_leg == "right" else -request.step_width/2)

            footsteps.append({
                "step_number": i,
                "x": foot_x,
                "y": foot_y,
                "theta": theta,
                "time": i * request.step_time,
                "support_leg": support_leg
            })

        # Calculate trajectory info
        trajectory_info = {
            "total_distance": float(distance),
            "num_steps": num_steps,
            "estimated_time": num_steps * request.step_time,
            "step_length": request.step_length,
            "step_width": request.step_width
        }

        response = WalkTrajectoryResponse(
            footsteps=footsteps,
            trajectory_info=trajectory_info,
            success=True
        )

        logger.info(f"Walk trajectory planned: {num_steps} steps, {distance:.2f}m")

        return response

    except Exception as e:
        logger.error(f"Error planning walk trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error planning walk trajectory: {str(e)}")