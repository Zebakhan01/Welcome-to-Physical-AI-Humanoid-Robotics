import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from math import sqrt, cos, sin, atan2, pi
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class GraspConfiguration:
    """Represents a grasp configuration"""
    finger_positions: List[Tuple[float, float, float]]  # Position of each finger [x, y, z]
    finger_forces: List[Tuple[float, float, float]]     # Force applied by each finger [fx, fy, fz]
    object_pose: Tuple[float, float, float, float, float, float]  # [x, y, z, roll, pitch, yaw]
    grasp_type: str  # "power", "precision", etc.
    grasp_quality: float  # Quality metric (0-1)

@dataclass
class ManipulationState:
    """Represents the current state of manipulation"""
    end_effector_pos: Tuple[float, float, float]  # [x, y, z]
    end_effector_orient: Tuple[float, float, float, float]  # [qx, qy, qz, qw] quaternion
    joint_angles: List[float]
    joint_velocities: List[float]
    grasp_configuration: Optional[GraspConfiguration]
    object_in_hand: bool

@dataclass
class ObjectProperties:
    """Properties of an object to be manipulated"""
    shape: str  # "cylinder", "sphere", "box", "complex"
    dimensions: Tuple[float, float, float]  # [x, y, z] for bounding box
    mass: float
    center_of_mass: Tuple[float, float, float]  # [x, y, z] relative to object origin
    friction_coeff: float
    fragility: float  # 0-1 scale, 1 being very fragile

class GraspAnalyzer:
    """Analyzes grasp stability and quality"""

    @staticmethod
    def compute_grasp_matrix(finger_positions: List[Tuple[float, float, float]],
                           contact_normals: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute grasp matrix for force analysis
        Each row represents a contact point with its normal and moment arm
        """
        # For each contact point, we have 6 constraints (3 forces + 3 moments)
        # This is a simplified representation
        grasp_matrix = []

        for pos, normal in zip(finger_positions, contact_normals):
            # Each contact contributes to the grasp matrix
            # Format: [fx, fy, fz, mx, my, mz] for each contact
            row = list(normal)  # Forces [fx, fy, fz]
            # Moments [mx, my, mz] = position cross force
            moment = np.cross(pos, normal)
            row.extend(moment.tolist())
            grasp_matrix.append(row)

        return np.array(grasp_matrix)

    @staticmethod
    def calculate_grasp_quality(grasp_matrix: np.ndarray) -> float:
        """
        Calculate grasp quality based on the grasp matrix
        Using the minimum singular value as a measure of grasp quality
        """
        if grasp_matrix.size == 0:
            return 0.0

        # Calculate singular values
        singular_values = np.linalg.svd(grasp_matrix, compute_uv=False)

        # Minimum singular value indicates grasp quality
        # Higher values indicate better force transmission
        min_sv = min(singular_values) if len(singular_values) > 0 else 0.0

        # Normalize to 0-1 range (arbitrary normalization)
        # In practice, this would be based on specific grasp requirements
        quality = min(1.0, max(0.0, min_sv))

        return quality

    @staticmethod
    def check_force_closure_2d(finger_positions: List[Tuple[float, float]],
                             contact_normals: List[Tuple[float, float]]) -> bool:
        """
        Check if a 2D grasp has force closure
        Simplified implementation for 2D case
        """
        if len(finger_positions) < 2:
            return False

        # For 2D force closure, we need at least 2 contacts with normals pointing inward
        # and the ability to resist forces in any direction
        # This is a simplified check

        # Check if we have at least 2 contacts on opposite sides
        # (This is a basic check - full force closure analysis is more complex)
        x_coords = [pos[0] for pos in finger_positions]
        y_coords = [pos[1] for pos in finger_positions]

        # If we have contacts on both sides of the object, it's more likely to have force closure
        return len(finger_positions) >= 2

    @staticmethod
    def suggest_grasp_type(object_props: ObjectProperties) -> str:
        """
        Suggest appropriate grasp type based on object properties
        """
        shape = object_props.shape
        dims = object_props.dimensions
        fragility = object_props.fragility

        # Analyze object dimensions to suggest grasp type
        max_dim = max(dims)
        min_dim = min(dims)

        if shape == "cylinder":
            if max_dim / min_dim > 2:  # Long cylinder
                return "cylindrical"
            else:  # Short cylinder
                return "spherical"
        elif shape == "sphere":
            return "spherical"
        elif shape == "box":
            if fragility > 0.7:  # Fragile objects
                return "precision"
            else:
                return "power"
        else:  # Complex shapes
            return "power"  # Default to power grasp for stability

class ManipulationKinematics:
    """Handles kinematics for manipulation tasks"""

    def __init__(self, arm_dof: int = 7):
        self.arm_dof = arm_dof
        # Define DH parameters for a typical robotic arm
        # These are example values - in reality, they'd match the specific robot
        self.dh_params = [
            {"a": 0, "alpha": pi/2, "d": 0.2, "theta": 0},      # Joint 1
            {"a": 0.4, "alpha": 0, "d": 0, "theta": 0},          # Joint 2
            {"a": 0, "alpha": pi/2, "d": 0, "theta": 0},         # Joint 3
            {"a": 0, "alpha": -pi/2, "d": 0.3, "theta": 0},      # Joint 4
            {"a": 0, "alpha": pi/2, "d": 0, "theta": 0},         # Joint 5
            {"a": 0, "alpha": 0, "d": 0.1, "theta": 0},          # Joint 6
            {"a": 0, "alpha": 0, "d": 0.1, "theta": 0},          # Joint 7 (hand)
        ]

    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Calculate Denavit-Hartenberg transformation matrix"""
        ct = cos(theta)
        st = sin(theta)
        ca = cos(alpha)
        sa = sin(alpha)

        transform = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

        return transform

    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward kinematics for the manipulator
        Returns: (position [x, y, z], orientation [roll, pitch, yaw])
        """
        if len(joint_angles) != self.arm_dof:
            raise ValueError(f"Expected {self.arm_dof} joint angles, got {len(joint_angles)}")

        # Update DH parameters with current joint angles
        current_dh = []
        for i, theta in enumerate(joint_angles):
            dh_param = self.dh_params[i].copy()
            dh_param["theta"] = theta
            current_dh.append(dh_param)

        # Calculate cumulative transformation
        cumulative_transform = np.eye(4)
        for dh in current_dh:
            transform = self.dh_transform(dh["a"], dh["alpha"], dh["d"], dh["theta"])
            cumulative_transform = cumulative_transform @ transform

        # Extract position and orientation
        position = cumulative_transform[:3, 3]

        # Extract orientation (convert rotation matrix to Euler angles)
        rotation_matrix = cumulative_transform[:3, :3]
        r = R.from_matrix(rotation_matrix)
        orientation = r.as_euler('xyz')  # roll, pitch, yaw

        return position, orientation

    def jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        Calculate geometric Jacobian for the manipulator
        """
        if len(joint_angles) != self.arm_dof:
            raise ValueError(f"Expected {self.arm_dof} joint angles, got {len(joint_angles)}")

        # Update DH parameters with current joint angles
        current_dh = []
        transforms = [np.eye(4)]  # Start with identity
        for i, theta in enumerate(joint_angles):
            dh_param = self.dh_params[i].copy()
            dh_param["theta"] = theta
            current_dh.append(dh_param)

            # Calculate transform for this joint
            transform = self.dh_transform(dh_param["a"], dh_param["alpha"], dh_param["d"], dh_param["theta"])
            transforms.append(transforms[-1] @ transform)

        # Calculate Jacobian
        # The Jacobian maps joint velocities to end-effector velocities
        jacobian = np.zeros((6, self.arm_dof))  # 6 DOF (3 pos + 3 rot), n joints

        # End-effector position from base
        end_pos = transforms[-1][:3, 3]

        for i in range(self.arm_dof):
            # Z-axis of joint i in base frame
            z_i = transforms[i][:3, 2]  # Third column is Z axis
            # Position of joint i in base frame
            p_i = transforms[i][:3, 3]

            # Linear velocity part: z_i × (end_pos - p_i)
            linear = np.cross(z_i, end_pos - p_i)
            # Angular velocity part: z_i
            angular = z_i

            jacobian[:3, i] = linear  # Linear velocity
            jacobian[3:, i] = angular  # Angular velocity

        return jacobian

class ManipulationController:
    """Controller for manipulation tasks"""

    def __init__(self, kp_pos: float = 10.0, ki_pos: float = 0.1, kd_pos: float = 1.0,
                 kp_force: float = 5.0, ki_force: float = 0.05, kd_force: float = 0.5):
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        self.kp_force = kp_force
        self.ki_force = ki_force
        self.kd_force = kd_force

        # PID state variables
        self.position_error_integral = np.zeros(3)
        self.position_error_prev = np.zeros(3)
        self.force_error_integral = np.zeros(3)
        self.force_error_prev = np.zeros(3)

    def compute_impedance_control(self, desired_pos: np.ndarray, current_pos: np.ndarray,
                                desired_vel: np.ndarray, current_vel: np.ndarray,
                                stiffness: np.ndarray, damping: np.ndarray) -> np.ndarray:
        """
        Compute impedance control output
        F = K * (x_d - x) + D * (v_d - v)
        """
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel

        # Impedance control law
        force = stiffness * pos_error + damping * vel_error

        return force

    def compute_hybrid_position_force_control(self, desired_pos: np.ndarray, current_pos: np.ndarray,
                                            desired_force: np.ndarray, current_force: np.ndarray,
                                            selection_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hybrid position/force control
        selection_matrix determines which DOFs are position-controlled vs force-controlled
        """
        # Position error
        pos_error = desired_pos - current_pos

        # Force error
        force_error = desired_force - current_force

        # Apply selection matrix to determine which errors to use
        # This is a simplified approach - in practice, this would be more complex
        pos_control = self.kp_pos * pos_error
        force_control = self.kp_force * force_error

        # Combine based on selection (for simplicity, we'll add them)
        # In real implementation, this would be more sophisticated
        control_output = pos_control + force_control

        return control_output, pos_error, force_error

    def compute_admittance_control(self, applied_force: np.ndarray,
                                 mass: float = 1.0, stiffness: float = 100.0,
                                 damping: float = 20.0) -> np.ndarray:
        """
        Compute admittance control: position response to applied force
        M*a + B*v + K*x = F
        For simplicity: x = F/K (static response)
        """
        # Static approximation: x = F/K
        displacement = applied_force / stiffness

        return displacement

class GraspPlanner:
    """Plans grasps for objects"""

    def __init__(self):
        pass

    def plan_cylindrical_grasp(self, object_props: ObjectProperties,
                             object_pose: Tuple[float, float, float, float, float, float]) -> GraspConfiguration:
        """
        Plan a cylindrical grasp for a cylindrical object
        """
        # Extract object pose [x, y, z, roll, pitch, yaw]
        obj_x, obj_y, obj_z = object_pose[0:3]

        # Plan grasp points around the cylinder
        # For a cylindrical grasp, fingers wrap around the cylinder
        radius = min(object_props.dimensions[0], object_props.dimensions[1]) / 2  # Approximate radius
        height = object_props.dimensions[2]

        # Place fingers around the cylinder
        finger_positions = []
        n_fingers = 3  # For a three-finger grasp
        for i in range(n_fingers):
            angle = 2 * pi * i / n_fingers  # Evenly spaced around cylinder
            x = obj_x + radius * cos(angle)
            y = obj_y + radius * sin(angle)
            z = obj_z  # At center height
            finger_positions.append((x, y, z))

        # Calculate contact normals (pointing toward center)
        contact_normals = []
        for pos in finger_positions:
            dx = obj_x - pos[0]
            dy = obj_y - pos[1]
            dz = 0  # For cylindrical grasp, mainly radial force
            norm = sqrt(dx*dx + dy*dy + dz*dz)
            if norm > 0:
                contact_normals.append((dx/norm, dy/norm, dz/norm))
            else:
                contact_normals.append((0, 0, 1))  # Default normal

        # Calculate finger forces (normal to surface, inward)
        finger_forces = []
        for normal in contact_normals:
            # Apply appropriate force magnitude based on object mass
            force_magnitude = object_props.mass * 9.81 * 0.5  # Safety factor
            finger_forces.append((normal[0] * force_magnitude,
                                normal[1] * force_magnitude,
                                normal[2] * force_magnitude))

        grasp_config = GraspConfiguration(
            finger_positions=finger_positions,
            finger_forces=finger_forces,
            object_pose=object_pose,
            grasp_type="cylindrical",
            grasp_quality=0.8  # Assumed quality for cylindrical grasp
        )

        return grasp_config

    def plan_precision_grasp(self, object_props: ObjectProperties,
                           object_pose: Tuple[float, float, float, float, float, float]) -> GraspConfiguration:
        """
        Plan a precision grasp for a small object
        """
        # Extract object pose [x, y, z, roll, pitch, yaw]
        obj_x, obj_y, obj_z = object_pose[0:3]

        # For precision grasp, use thumb and finger tips
        # Position thumb and fingers on opposite sides of object
        half_size = max(object_props.dimensions) / 4  # Approximate grasp span

        finger_positions = [
            (obj_x + half_size, obj_y, obj_z),  # Index finger
            (obj_x - half_size, obj_y, obj_z),  # Thumb
            (obj_x, obj_y + half_size, obj_z)   # Middle finger (support)
        ]

        # Contact normals pointing toward object center
        contact_normals = []
        for pos in finger_positions:
            dx = obj_x - pos[0]
            dy = obj_y - pos[1]
            dz = obj_z - pos[2]
            norm = sqrt(dx*dx + dy*dy + dz*dz)
            if norm > 0:
                contact_normals.append((dx/norm, dy/norm, dz/norm))
            else:
                contact_normals.append((0, 0, 1))

        # Calculate finger forces for precision grasp
        finger_forces = []
        for normal in contact_normals:
            # Apply appropriate force magnitude for precision grasp
            force_magnitude = object_props.mass * 9.81 * 0.3  # Lower force for precision
            finger_forces.append((normal[0] * force_magnitude,
                                normal[1] * force_magnitude,
                                normal[2] * force_magnitude))

        grasp_config = GraspConfiguration(
            finger_positions=finger_positions,
            finger_forces=finger_forces,
            object_pose=object_pose,
            grasp_type="precision",
            grasp_quality=0.7  # Assumed quality for precision grasp
        )

        return grasp_config

    def plan_grasp(self, object_props: ObjectProperties,
                   object_pose: Tuple[float, float, float, float, float, float],
                   grasp_type: str = "auto") -> GraspConfiguration:
        """
        Plan an appropriate grasp based on object properties
        """
        if grasp_type == "auto":
            suggested_type = GraspAnalyzer.suggest_grasp_type(object_props)
        else:
            suggested_type = grasp_type

        if suggested_type in ["cylindrical", "spherical"]:
            return self.plan_cylindrical_grasp(object_props, object_pose)
        elif suggested_type == "precision":
            return self.plan_precision_grasp(object_props, object_pose)
        else:
            # Default to power grasp
            return self.plan_cylindrical_grasp(object_props, object_pose)

def calculate_manipulability(jacobian: np.ndarray) -> float:
    """
    Calculate manipulability measure from Jacobian matrix
    Uses Yoshikawa's manipulability measure: sqrt(det(J*J^T))
    """
    if jacobian.shape[0] < jacobian.shape[1]:
        # For redundant manipulators, use pseudo-inverse approach
        J_pinv = np.linalg.pinv(jacobian)
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
    else:
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))

    # Return 0 if determinant is negative (unreachable configuration)
    return max(0.0, manipulability)

def compute_contact_jacobian(contact_points: List[Tuple[float, float, float]],
                          end_effector_pose: Tuple[float, float, float]) -> np.ndarray:
    """
    Compute Jacobian for contact points relative to end-effector
    """
    n_contacts = len(contact_points)
    contact_jacobian = np.zeros((3 * n_contacts, 6))  # 3 DOF per contact, 6 DOF end-effector

    end_pos = np.array(end_effector_pose)

    for i, contact_point in enumerate(contact_points):
        # Position of contact point relative to end-effector
        rel_pos = np.array(contact_point) - end_pos

        # For position-level contact Jacobian
        # The contact Jacobian relates end-effector motion to contact point motion
        contact_jacobian[3*i:3*i+3, :3] = np.eye(3)  # Linear motion
        # Angular motion creates linear velocity at contact point: v = omega x r
        skew_sym = np.array([
            [0, -rel_pos[2], rel_pos[1]],
            [rel_pos[2], 0, -rel_pos[0]],
            [-rel_pos[1], rel_pos[0], 0]
        ])
        contact_jacobian[3*i:3*i+3, 3:] = skew_sym

    return contact_jacobian