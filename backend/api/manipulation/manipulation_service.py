from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.manipulation_utils import (
    GraspConfiguration, ManipulationState, ObjectProperties,
    GraspAnalyzer, ManipulationKinematics, ManipulationController,
    GraspPlanner, calculate_manipulability, compute_contact_jacobian
)
from backend.utils.logger import logger

router = APIRouter()

class GraspAnalysisRequest(BaseModel):
    finger_positions: List[List[float]]  # List of [x, y, z] positions
    contact_normals: List[List[float]]   # List of [x, y, z] normal vectors
    object_properties: Dict[str, Any]    # Object properties

class GraspAnalysisResponse(BaseModel):
    grasp_matrix: List[List[float]]
    grasp_quality: float
    force_closure: bool
    suggested_grasp_type: str
    success: bool

class ForwardKinematicsRequest(BaseModel):
    joint_angles: List[float]
    arm_dof: int = 7

class ForwardKinematicsResponse(BaseModel):
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [roll, pitch, yaw]
    success: bool

class JacobianRequest(BaseModel):
    joint_angles: List[float]
    arm_dof: int = 7

class JacobianResponse(BaseModel):
    jacobian: List[List[float]]
    manipulability: float
    success: bool

class ManipulationControlRequest(BaseModel):
    current_pos: List[float]  # [x, y, z]
    desired_pos: List[float]  # [x, y, z]
    current_vel: List[float]  # [vx, vy, vz]
    desired_vel: List[float]  # [vx, vy, vz]
    stiffness: List[float]    # [kx, ky, kz]
    damping: List[float]      # [dx, dy, dz]
    control_type: str = "impedance"  # "impedance", "hybrid", "admittance"

class ManipulationControlResponse(BaseModel):
    control_output: List[float]
    success: bool

class GraspPlanningRequest(BaseModel):
    object_properties: Dict[str, Any]
    object_pose: List[float]  # [x, y, z, roll, pitch, yaw]
    grasp_type: str = "auto"  # "auto", "cylindrical", "precision", etc.

class GraspPlanningResponse(BaseModel):
    grasp_configuration: Dict[str, Any]
    grasp_quality: float
    success: bool

class HybridControlRequest(BaseModel):
    desired_pos: List[float]  # [x, y, z]
    current_pos: List[float]  # [x, y, z]
    desired_force: List[float]  # [fx, fy, fz]
    current_force: List[float]  # [fx, fy, fz]
    selection_matrix: List[List[float]]  # 3x3 matrix

class HybridControlResponse(BaseModel):
    control_output: List[float]
    position_error: List[float]
    force_error: List[float]
    success: bool

class ContactJacobianRequest(BaseModel):
    contact_points: List[List[float]]  # List of [x, y, z] positions
    end_effector_pose: List[float]     # [x, y, z]

class ContactJacobianResponse(BaseModel):
    contact_jacobian: List[List[float]]
    success: bool

class MultiFingerCoordinationRequest(BaseModel):
    finger_positions: List[List[float]]  # List of [x, y, z] positions
    finger_forces: List[List[float]]     # List of [fx, fy, fz] forces
    coordination_type: str = "synergy"   # "synergy", "independent"

class MultiFingerCoordinationResponse(BaseModel):
    coordinated_positions: List[List[float]]
    coordinated_forces: List[List[float]]
    coordination_metrics: Dict[str, float]
    success: bool

class WholeBodyManipulationRequest(BaseModel):
    arm_joint_angles: List[float]
    object_pose: List[float]  # [x, y, z, roll, pitch, yaw]
    base_pose: List[float]    # [x, y, z, roll, pitch, yaw] for mobile base
    balance_constraint: bool = True

class WholeBodyManipulationResponse(BaseModel):
    optimized_joint_angles: List[float]
    com_adjustment: List[float]  # [x, y, z] adjustment to center of mass
    balance_metrics: Dict[str, float]
    success: bool

class InHandManipulationRequest(BaseModel):
    initial_grasp: Dict[str, Any]
    target_object_pose: List[float]  # [x, y, z, roll, pitch, yaw]
    manipulation_type: str = "rolling"  # "rolling", "sliding", "regrasp"

class InHandManipulationResponse(BaseModel):
    new_grasp_configuration: Dict[str, Any]
    manipulation_path: List[List[float]]  # List of intermediate poses
    success: bool

@router.post("/analyze-grasp", response_model=GraspAnalysisResponse)
async def analyze_grasp(request: GraspAnalysisRequest):
    """
    Analyze grasp stability and quality
    """
    try:
        # Convert to numpy arrays
        finger_positions = [np.array(pos) for pos in request.finger_positions]
        contact_normals = [np.array(norm) for norm in request.contact_normals]

        # Create object properties object
        obj_props = ObjectProperties(
            shape=request.object_properties.get("shape", "box"),
            dimensions=tuple(request.object_properties.get("dimensions", [0.1, 0.1, 0.1])),
            mass=request.object_properties.get("mass", 0.5),
            center_of_mass=tuple(request.object_properties.get("center_of_mass", [0, 0, 0])),
            friction_coeff=request.object_properties.get("friction_coeff", 0.8),
            fragility=request.object_properties.get("fragility", 0.1)
        )

        # Analyze grasp
        grasp_matrix = GraspAnalyzer.compute_grasp_matrix(finger_positions, contact_normals)
        grasp_quality = GraspAnalyzer.calculate_grasp_quality(grasp_matrix)

        # For 2D force closure check (simplified)
        # In practice, this would be a full 3D force closure analysis
        force_closure = GraspAnalyzer.check_force_closure_2d(
            [(pos[0], pos[1]) for pos in finger_positions],
            [(norm[0], norm[1]) for norm in contact_normals]
        )

        suggested_type = GraspAnalyzer.suggest_grasp_type(obj_props)

        response = GraspAnalysisResponse(
            grasp_matrix=grasp_matrix.tolist(),
            grasp_quality=grasp_quality,
            force_closure=force_closure,
            suggested_grasp_type=suggested_type,
            success=True
        )

        logger.info(f"Grasp analysis completed: quality={grasp_quality:.3f}, force_closure={force_closure}")

        return response

    except Exception as e:
        logger.error(f"Error in grasp analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in grasp analysis: {str(e)}")

@router.post("/forward-kinematics", response_model=ForwardKinematicsResponse)
async def forward_kinematics(request: ForwardKinematicsRequest):
    """
    Calculate forward kinematics for manipulator
    """
    try:
        # Create kinematics object
        kinematics = ManipulationKinematics(arm_dof=request.arm_dof)

        # Calculate forward kinematics
        position, orientation = kinematics.forward_kinematics(request.joint_angles)

        response = ForwardKinematicsResponse(
            position=position.tolist(),
            orientation=orientation.tolist(),
            success=True
        )

        logger.info(f"Forward kinematics: position={position.tolist()}")

        return response

    except Exception as e:
        logger.error(f"Error in forward kinematics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in forward kinematics: {str(e)}")

@router.post("/jacobian", response_model=JacobianResponse)
async def calculate_jacobian(request: JacobianRequest):
    """
    Calculate Jacobian matrix for manipulator
    """
    try:
        # Create kinematics object
        kinematics = ManipulationKinematics(arm_dof=request.arm_dof)

        # Calculate Jacobian
        jacobian = kinematics.jacobian(request.joint_angles)

        # Calculate manipulability
        manipulability = calculate_manipulability(jacobian)

        response = JacobianResponse(
            jacobian=jacobian.tolist(),
            manipulability=manipulability,
            success=True
        )

        logger.info(f"Jacobian calculated: manipulability={manipulability:.3f}")

        return response

    except Exception as e:
        logger.error(f"Error calculating Jacobian: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating Jacobian: {str(e)}")

@router.post("/manipulation-control", response_model=ManipulationControlResponse)
async def manipulation_control(request: ManipulationControlRequest):
    """
    Compute manipulation control output
    """
    try:
        # Create controller
        controller = ManipulationController()

        # Convert to numpy arrays
        current_pos = np.array(request.current_pos)
        desired_pos = np.array(request.desired_pos)
        current_vel = np.array(request.current_vel)
        desired_vel = np.array(request.desired_vel)
        stiffness = np.array(request.stiffness)
        damping = np.array(request.damping)

        if request.control_type == "impedance":
            # Compute impedance control
            control_output = controller.compute_impedance_control(
                desired_pos, current_pos, desired_vel, current_vel, stiffness, damping
            )
        elif request.control_type == "hybrid":
            # For hybrid control, use a simplified approach
            # This would be more complex in a real implementation
            pos_error = desired_pos - current_pos
            vel_error = desired_vel - current_vel
            control_output = stiffness * pos_error + damping * vel_error
        elif request.control_type == "admittance":
            # For admittance, compute force to position mapping
            applied_force = desired_pos - current_pos  # Simplified
            control_output = controller.compute_admittance_control(
                applied_force, 1.0, stiffness[0], damping[0]
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown control type: {request.control_type}")

        response = ManipulationControlResponse(
            control_output=control_output.tolist(),
            success=True
        )

        logger.info(f"Manipulation control computed using {request.control_type} control")

        return response

    except Exception as e:
        logger.error(f"Error in manipulation control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in manipulation control: {str(e)}")

@router.post("/plan-grasp", response_model=GraspPlanningResponse)
async def plan_grasp(request: GraspPlanningRequest):
    """
    Plan an appropriate grasp for an object
    """
    try:
        # Create object properties object
        obj_props = ObjectProperties(
            shape=request.object_properties.get("shape", "box"),
            dimensions=tuple(request.object_properties.get("dimensions", [0.1, 0.1, 0.1])),
            mass=request.object_properties.get("mass", 0.5),
            center_of_mass=tuple(request.object_properties.get("center_of_mass", [0, 0, 0])),
            friction_coeff=request.object_properties.get("friction_coeff", 0.8),
            fragility=request.object_properties.get("fragility", 0.1)
        )

        # Create grasp planner
        planner = GraspPlanner()

        # Plan grasp
        grasp_config = planner.plan_grasp(obj_props, request.object_pose, request.grasp_type)

        # Convert to response format
        grasp_dict = {
            "finger_positions": [list(pos) for pos in grasp_config.finger_positions],
            "finger_forces": [list(force) for force in grasp_config.finger_forces],
            "object_pose": list(grasp_config.object_pose),
            "grasp_type": grasp_config.grasp_type
        }

        response = GraspPlanningResponse(
            grasp_configuration=grasp_dict,
            grasp_quality=grasp_config.grasp_quality,
            success=True
        )

        logger.info(f"Grasp planned: type={grasp_config.grasp_type}, quality={grasp_config.grasp_quality:.3f}")

        return response

    except Exception as e:
        logger.error(f"Error in grasp planning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in grasp planning: {str(e)}")

@router.post("/hybrid-control", response_model=HybridControlResponse)
async def hybrid_position_force_control(request: HybridControlRequest):
    """
    Compute hybrid position/force control
    """
    try:
        # Create controller
        controller = ManipulationController()

        # Convert to numpy arrays
        desired_pos = np.array(request.desired_pos)
        current_pos = np.array(request.current_pos)
        desired_force = np.array(request.desired_force)
        current_force = np.array(request.current_force)
        selection_matrix = np.array(request.selection_matrix)

        # Compute hybrid control
        control_output, pos_error, force_error = controller.compute_hybrid_position_force_control(
            desired_pos, current_pos, desired_force, current_force, selection_matrix
        )

        response = HybridControlResponse(
            control_output=control_output.tolist(),
            position_error=pos_error.tolist(),
            force_error=force_error.tolist(),
            success=True
        )

        logger.info("Hybrid position/force control computed")

        return response

    except Exception as e:
        logger.error(f"Error in hybrid control: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in hybrid control: {str(e)}")

@router.post("/contact-jacobian", response_model=ContactJacobianResponse)
async def contact_jacobian(request: ContactJacobianRequest):
    """
    Compute contact Jacobian for manipulation
    """
    try:
        # Convert to numpy arrays
        contact_points = [np.array(point) for point in request.contact_points]
        end_effector_pose = np.array(request.end_effector_pose)

        # Compute contact Jacobian
        contact_jac = compute_contact_jacobian(contact_points, end_effector_pose)

        response = ContactJacobianResponse(
            contact_jacobian=contact_jac.tolist(),
            success=True
        )

        logger.info(f"Contact Jacobian computed for {len(contact_points)} contact points")

        return response

    except Exception as e:
        logger.error(f"Error computing contact Jacobian: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing contact Jacobian: {str(e)}")

@router.post("/multi-finger-coordination", response_model=MultiFingerCoordinationResponse)
async def multi_finger_coordination(request: MultiFingerCoordinationRequest):
    """
    Coordinate multiple fingers for manipulation
    """
    try:
        # Convert to numpy arrays
        finger_positions = [np.array(pos) for pos in request.finger_positions]
        finger_forces = [np.array(force) for force in request.finger_forces]

        if request.coordination_type == "synergy":
            # Apply synergistic coordination (simplified)
            # This would implement principal component analysis in a real system
            coordinated_positions = [pos.tolist() for pos in finger_positions]
            coordinated_forces = [force.tolist() for force in finger_forces]

            # Calculate coordination metrics
            coordination_metrics = {
                "synergy_index": 0.8,  # Simplified metric
                "force_distribution": float(np.mean([np.linalg.norm(f) for f in finger_forces]))
            }
        elif request.coordination_type == "independent":
            # Independent finger control
            coordinated_positions = [pos.tolist() for pos in finger_positions]
            coordinated_forces = [force.tolist() for force in finger_forces]

            coordination_metrics = {
                "independence_index": 0.9,  # Simplified metric
                "force_distribution": float(np.mean([np.linalg.norm(f) for f in finger_forces]))
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown coordination type: {request.coordination_type}")

        response = MultiFingerCoordinationResponse(
            coordinated_positions=coordinated_positions,
            coordinated_forces=coordinated_forces,
            coordination_metrics=coordination_metrics,
            success=True
        )

        logger.info(f"Multi-finger coordination computed using {request.coordination_type} approach")

        return response

    except Exception as e:
        logger.error(f"Error in multi-finger coordination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in multi-finger coordination: {str(e)}")

@router.post("/whole-body-manipulation", response_model=WholeBodyManipulationResponse)
async def whole_body_manipulation(request: WholeBodyManipulationRequest):
    """
    Plan whole-body manipulation considering balance
    """
    try:
        # For now, this is a simplified implementation
        # In a real system, this would involve complex whole-body optimization

        # Calculate CoM adjustment based on arm position
        # Simplified approach: adjust CoM to maintain balance during manipulation
        arm_contribution = np.array(request.arm_joint_angles[:3]) * 0.05  # Simplified model
        base_pos = np.array(request.base_pose[:3])

        # Calculate needed CoM adjustment
        com_adjustment = [float(-arm_contribution[0]*0.1), float(-arm_contribution[1]*0.1), 0.0]

        # Return optimized joint angles (in this simplified version, just return the input)
        optimized_angles = request.arm_joint_angles

        # Calculate balance metrics
        balance_metrics = {
            "com_stability": 0.8,  # Simplified metric
            "support_margin": 0.15,  # Distance to support polygon boundary
            "manipulability": 0.7   # Simplified manipulability measure
        }

        response = WholeBodyManipulationResponse(
            optimized_joint_angles=optimized_angles,
            com_adjustment=com_adjustment,
            balance_metrics=balance_metrics,
            success=True
        )

        logger.info("Whole-body manipulation plan computed")

        return response

    except Exception as e:
        logger.error(f"Error in whole-body manipulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in whole-body manipulation: {str(e)}")

@router.post("/in-hand-manipulation", response_model=InHandManipulationResponse)
async def in_hand_manipulation(request: InHandManipulationRequest):
    """
    Plan in-hand manipulation to reposition object within grasp
    """
    try:
        # Convert initial grasp to proper format
        initial_config = GraspConfiguration(
            finger_positions=[tuple(pos) for pos in request.initial_grasp["finger_positions"]],
            finger_forces=[tuple(force) for force in request.initial_grasp["finger_forces"]],
            object_pose=tuple(request.initial_grasp["object_pose"]),
            grasp_type=request.initial_grasp["grasp_type"],
            grasp_quality=request.initial_grasp.get("grasp_quality", 0.5)
        )

        # Plan manipulation path based on type
        manipulation_path = []

        if request.manipulation_type == "rolling":
            # Generate intermediate poses for rolling manipulation
            start_pose = np.array(initial_config.object_pose)
            target_pose = np.array(request.target_object_pose)

            # Create a simple linear interpolation of orientation changes
            for i in range(5):  # 5 intermediate steps
                t = i / 4  # 0 to 1
                intermediate_pose = (1 - t) * start_pose + t * target_pose
                manipulation_path.append(intermediate_pose.tolist())

        elif request.manipulation_type == "sliding":
            # Generate path for sliding manipulation
            start_pos = np.array(initial_config.object_pose[:3])
            target_pos = np.array(request.target_object_pose[:3])

            for i in range(5):
                t = i / 4
                intermediate_pos = (1 - t) * start_pos + t * target_pos
                # Keep orientation the same for sliding
                intermediate_pose = list(intermediate_pos) + list(initial_config.object_pose[3:])
                manipulation_path.append(intermediate_pose)

        elif request.manipulation_type == "regrasp":
            # For regrasp, generate a path that releases and regrasps
            # This is a simplified version
            manipulation_path = [request.initial_grasp["object_pose"], request.target_object_pose]

        else:
            raise HTTPException(status_code=400, detail=f"Unknown manipulation type: {request.manipulation_type}")

        # For now, return the target grasp configuration
        new_grasp_config = {
            "finger_positions": initial_config.finger_positions,  # Would be adjusted in real implementation
            "finger_forces": initial_config.finger_forces,       # Would be adjusted in real implementation
            "object_pose": request.target_object_pose,
            "grasp_type": initial_config.grasp_type,
            "grasp_quality": initial_config.grasp_quality
        }

        response = InHandManipulationResponse(
            new_grasp_configuration=new_grasp_config,
            manipulation_path=manipulation_path,
            success=True
        )

        logger.info(f"In-hand manipulation planned: type={request.manipulation_type}")

        return response

    except Exception as e:
        logger.error(f"Error in in-hand manipulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in in-hand manipulation: {str(e)}")

class GraspQualityAssessmentRequest(BaseModel):
    grasp_config: Dict[str, Any]
    object_properties: Dict[str, Any]

class GraspQualityAssessmentResponse(BaseModel):
    quality_score: float
    stability_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    success: bool

@router.post("/assess-grasp-quality", response_model=GraspQualityAssessmentResponse)
async def assess_grasp_quality(request: GraspQualityAssessmentRequest):
    """
    Assess the quality of a grasp configuration
    """
    try:
        # Create object properties
        obj_props = ObjectProperties(
            shape=request.object_properties.get("shape", "box"),
            dimensions=tuple(request.object_properties.get("dimensions", [0.1, 0.1, 0.1])),
            mass=request.object_properties.get("mass", 0.5),
            center_of_mass=tuple(request.object_properties.get("center_of_mass", [0, 0, 0])),
            friction_coeff=request.object_properties.get("friction_coeff", 0.8),
            fragility=request.object_properties.get("fragility", 0.1)
        )

        # Extract grasp configuration
        finger_positions = [tuple(pos) for pos in request.grasp_config["finger_positions"]]
        contact_normals = []  # We'll compute these based on finger positions and object center

        # Calculate contact normals pointing toward object center
        obj_center = request.grasp_config["object_pose"][:3]
        for pos in finger_positions:
            dx = obj_center[0] - pos[0]
            dy = obj_center[1] - pos[1]
            dz = obj_center[2] - pos[2]
            norm = (dx**2 + dy**2 + dz**2)**0.5
            if norm > 0:
                contact_normals.append((dx/norm, dy/norm, dz/norm))
            else:
                contact_normals.append((0, 0, 1))

        # Analyze grasp
        grasp_matrix = GraspAnalyzer.compute_grasp_matrix(finger_positions, contact_normals)
        quality_score = GraspAnalyzer.calculate_grasp_quality(grasp_matrix)

        # Calculate stability metrics
        stability_metrics = {
            "force_closure": GraspAnalyzer.check_force_closure_2d(
                [(p[0], p[1]) for p in finger_positions],
                [(n[0], n[1]) for n in contact_normals]
            ),
            "grasp_matrix_condition": float(np.linalg.cond(grasp_matrix)) if grasp_matrix.size > 0 else float('inf'),
            "friction_cone_adherence": min(1.0, obj_props.friction_coeff * 2)  # Simplified
        }

        # Generate improvement suggestions
        improvement_suggestions = []
        if quality_score < 0.5:
            improvement_suggestions.append("Consider increasing the number of contact points")
        if obj_props.friction_coeff < 0.5:
            improvement_suggestions.append("Object surface may be too slippery for stable grasp")
        if len(finger_positions) < 3:
            improvement_suggestions.append("Use at least 3 contact points for better stability")

        response = GraspQualityAssessmentResponse(
            quality_score=quality_score,
            stability_metrics=stability_metrics,
            improvement_suggestions=improvement_suggestions,
            success=True
        )

        logger.info(f"Grasp quality assessed: score={quality_score:.3f}")

        return response

    except Exception as e:
        logger.error(f"Error assessing grasp quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error assessing grasp quality: {str(e)}")