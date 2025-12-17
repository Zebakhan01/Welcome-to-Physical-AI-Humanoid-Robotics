import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from math import sin, cos, sqrt, atan2, pi
import time

@dataclass
class JointState:
    """Represents the state of a robotic joint"""
    position: float  # in radians
    velocity: float  # in rad/s
    effort: float    # in Nm

@dataclass
class ControlInput:
    """Represents control input to a robotic system"""
    joint_torques: List[float]
    cartesian_forces: Optional[List[float]] = None

@dataclass
class TrajectoryPoint:
    """Represents a point in a trajectory"""
    time: float
    positions: List[float]
    velocities: List[float]
    accelerations: Optional[List[float]] = None

class PIDController:
    """PID controller implementation"""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, dt: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.time()

    def update(self, setpoint: float, measured_value: float) -> float:
        """Update PID controller and return control output"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt < self.dt:
            dt = self.dt  # Use fixed time step if needed

        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Store values for next iteration
        self.previous_error = error
        self.last_time = current_time

        output = p_term + i_term + d_term
        return output

    def reset(self):
        """Reset the PID controller"""
        self.integral = 0.0
        self.previous_error = 0.0

class JointController:
    """Controller for individual robot joints"""

    def __init__(self, num_joints: int):
        self.num_joints = num_joints
        # Initialize PID controllers for each joint
        self.pid_controllers = [PIDController(kp=10.0, ki=0.1, kd=0.5) for _ in range(num_joints)]

    def compute_joint_torques(self, current_positions: List[float],
                            desired_positions: List[float],
                            current_velocities: List[float] = None,
                            desired_velocities: List[float] = None) -> List[float]:
        """Compute joint torques using PID control"""
        torques = []

        for i in range(self.num_joints):
            torque = self.pid_controllers[i].update(
                desired_positions[i],
                current_positions[i]
            )
            torques.append(torque)

        return torques

class TrajectoryGenerator:
    """Generate trajectories for robot motion"""

    @staticmethod
    def generate_cubic_trajectory(start_pos: float, end_pos: float,
                                start_vel: float = 0.0, end_vel: float = 0.0,
                                duration: float = 1.0, dt: float = 0.01) -> List[TrajectoryPoint]:
        """Generate cubic polynomial trajectory"""
        trajectory = []

        # Cubic polynomial coefficients
        a0 = start_pos
        a1 = start_vel
        a2 = (3 * (end_pos - start_pos) - 2 * start_vel * duration - end_vel * duration) / (duration**2)
        a3 = (2 * (start_pos - end_pos) + (start_vel + end_vel) * duration) / (duration**3)

        t = 0.0
        while t <= duration:
            # Position: s(t) = a0 + a1*t + a2*t^2 + a3*t^3
            pos = a0 + a1*t + a2*(t**2) + a3*(t**3)

            # Velocity: s'(t) = a1 + 2*a2*t + 3*a3*t^2
            vel = a1 + 2*a2*t + 3*a3*(t**2)

            # Acceleration: s''(t) = 2*a2 + 6*a3*t
            acc = 2*a2 + 6*a3*t

            trajectory.append(TrajectoryPoint(
                time=t,
                positions=[pos],
                velocities=[vel],
                accelerations=[acc]
            ))

            t += dt

        return trajectory

    @staticmethod
    def generate_multiple_joint_trajectory(start_positions: List[float],
                                         end_positions: List[float],
                                         duration: float = 1.0,
                                         dt: float = 0.01) -> List[TrajectoryPoint]:
        """Generate trajectory for multiple joints"""
        trajectory = []

        # Calculate coefficients for each joint
        coeffs = []
        for i in range(len(start_positions)):
            a0 = start_positions[i]
            a1 = 0.0  # Start with zero velocity
            a2 = (3 * (end_positions[i] - start_positions[i])) / (duration**2)
            a3 = (2 * (start_positions[i] - end_positions[i])) / (duration**3)
            coeffs.append((a0, a1, a2, a3))

        t = 0.0
        while t <= duration:
            positions = []
            velocities = []
            accelerations = []

            for a0, a1, a2, a3 in coeffs:
                pos = a0 + a1*t + a2*(t**2) + a3*(t**3)
                vel = a1 + 2*a2*t + 3*a3*(t**2)
                acc = 2*a2 + 6*a3*t

                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)

            trajectory.append(TrajectoryPoint(
                time=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations
            ))

            t += dt

        return trajectory

class JacobianCalculator:
    """Calculate Jacobian matrices for robot kinematics"""

    @staticmethod
    def calculate_2dof_jacobian(theta1: float, theta2: float,
                              l1: float, l2: float) -> np.ndarray:
        """Calculate Jacobian for 2-DOF planar manipulator"""
        # Jacobian matrix for 2-DOF planar manipulator
        # J = [∂x/∂θ1  ∂x/∂θ2]
        #     [∂y/∂θ1  ∂y/∂θ2]

        # x = l1*cos(θ1) + l2*cos(θ1 + θ2)
        # y = l1*sin(θ1) + l2*sin(θ1 + θ2)

        # ∂x/∂θ1 = -l1*sin(θ1) - l2*sin(θ1 + θ2)
        # ∂x/∂θ2 = -l2*sin(θ1 + θ2)
        # ∂y/∂θ1 = l1*cos(θ1) + l2*cos(θ1 + θ2)
        # ∂y/∂θ2 = l2*cos(θ1 + θ2)

        s1 = sin(theta1)
        s12 = sin(theta1 + theta2)
        c1 = cos(theta1)
        c12 = cos(theta1 + theta2)

        jacobian = np.array([
            [-l1*s1 - l2*s12, -l2*s12],  # dx/dtheta
            [l1*c1 + l2*c12, l2*c12]     # dy/dtheta
        ])

        return jacobian

class OperationalSpaceController:
    """Controller for operational space (Cartesian) control"""

    def __init__(self, num_joints: int, kp: float = 10.0, kd: float = 2.0):
        self.num_joints = num_joints
        self.kp = kp  # Position gain
        self.kd = kd  # Damping gain
        self.jacobian_calc = JacobianCalculator()

    def compute_cartesian_control(self, current_pos: np.ndarray,
                                desired_pos: np.ndarray,
                                current_vel: np.ndarray = None,
                                desired_vel: np.ndarray = None,
                                jacobian: np.ndarray = None) -> np.ndarray:
        """Compute Cartesian space control"""
        if current_vel is None:
            current_vel = np.zeros_like(current_pos)
        if desired_vel is None:
            desired_vel = np.zeros_like(desired_pos)

        # Position error
        pos_error = desired_pos - current_pos

        # Velocity error
        vel_error = desired_vel - current_vel

        # Cartesian force
        cartesian_force = self.kp * pos_error + self.kd * vel_error

        return cartesian_force

    def map_cartesian_to_joint(self, cartesian_force: np.ndarray,
                             jacobian: np.ndarray) -> np.ndarray:
        """Map Cartesian forces to joint torques using Jacobian transpose"""
        # τ = J^T * F
        joint_torques = jacobian.T @ cartesian_force
        return joint_torques

class ImpedanceController:
    """Impedance controller for compliant behavior"""

    def __init__(self, mass: float = 1.0, damping: float = 10.0, stiffness: float = 100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

    def compute_impedance_force(self, pos_error: np.ndarray,
                              vel_error: np.ndarray) -> np.ndarray:
        """Compute impedance control force: F = M*ẍ + B*ẋ + K*x"""
        # For simplicity, we'll use a basic impedance model
        # F = -K * pos_error - B * vel_error
        force = -self.stiffness * pos_error - self.damping * vel_error
        return force

class Simulator:
    """Simple robot simulator for motion control"""

    def __init__(self, num_joints: int, dt: float = 0.01):
        self.num_joints = num_joints
        self.dt = dt
        self.positions = np.zeros(num_joints)
        self.velocities = np.zeros(num_joints)
        self.accelerations = np.zeros(num_joints)

    def step(self, torques: List[float], friction_coeff: float = 0.1) -> Tuple[List[float], List[float], List[float]]:
        """Simulate one time step"""
        torques = np.array(torques)

        # Simple dynamics model: τ = I*α + friction*ω (simplified)
        # For simplicity, assume unit inertia
        friction_torques = friction_coeff * self.velocities
        accelerations = torques - friction_torques  # Simplified model

        # Update velocities and positions using Euler integration
        self.velocities += accelerations * self.dt
        self.positions += self.velocities * self.dt

        return (self.positions.tolist(),
                self.velocities.tolist(),
                accelerations.tolist())

def compute_forward_kinematics_2dof(theta1: float, theta2: float, l1: float, l2: float) -> Tuple[float, float]:
    """Compute forward kinematics for 2-DOF planar manipulator"""
    x = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
    y = l1 * sin(theta1) + l2 * sin(theta1 + theta2)
    return x, y

def compute_inverse_kinematics_2dof(x: float, y: float, l1: float, l2: float) -> Optional[Tuple[float, float]]:
    """Compute inverse kinematics for 2-DOF planar manipulator"""
    r = sqrt(x**2 + y**2)

    # Check if point is reachable
    if r > l1 + l2 or r < abs(l1 - l2):
        return None  # Point not reachable

    # Calculate angle of second link
    cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    # Ensure the value is within valid range for acos
    cos_theta2 = max(-1, min(1, cos_theta2))  # Clamp to [-1, 1] to avoid numerical errors
    sin_theta2 = sqrt(1 - cos_theta2**2)
    theta2 = atan2(sin_theta2, cos_theta2)

    # Calculate angle of first link
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = atan2(y, x) - atan2(k2, k1)

    return theta1, theta2