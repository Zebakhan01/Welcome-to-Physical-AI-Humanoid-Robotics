import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from math import sin, cos, sqrt, atan2, acos
import json

@dataclass
class JointState:
    """Represents the state of a robotic joint"""
    position: float  # in radians
    velocity: float  # in rad/s
    effort: float    # in Nm

@dataclass
class Pose:
    """Represents position and orientation in 3D space"""
    x: float
    y: float
    z: float
    roll: float  # in radians
    pitch: float  # in radians
    yaw: float    # in radians

class KinematicsCalculator:
    """Calculator for robotic kinematics"""

    @staticmethod
    def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Calculate Denavit-Hartenberg transformation matrix
        """
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

    @staticmethod
    def forward_kinematics_2dof(theta1: float, theta2: float, l1: float, l2: float) -> Tuple[float, float]:
        """
        Calculate forward kinematics for a 2-DOF planar manipulator
        """
        x = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
        y = l1 * sin(theta1) + l2 * sin(theta1 + theta2)
        return x, y

    @staticmethod
    def inverse_kinematics_2dof(x: float, y: float, l1: float, l2: float) -> List[Tuple[float, float]]:
        """
        Calculate inverse kinematics for a 2-DOF planar manipulator
        Returns list of possible solutions
        """
        # Calculate distance from origin to target
        r = sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > l1 + l2:
            return []  # Target is too far

        if r < abs(l1 - l2):
            return []  # Target is too close

        # Calculate angle of second link
        cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        sin_theta2 = sqrt(1 - cos_theta2**2)
        theta2 = atan2(sin_theta2, cos_theta2)

        # Calculate angle of first link
        k1 = l1 + l2 * cos_theta2
        k2 = l2 * sin_theta2
        theta1 = atan2(y, x) - atan2(k2, k1)

        # Second solution (elbow down)
        theta2_alt = -theta2
        k1_alt = l1 + l2 * cos(theta2_alt)
        k2_alt = l2 * sin(theta2_alt)
        theta1_alt = atan2(y, x) - atan2(k2_alt, k1_alt)

        return [(theta1, theta2), (theta1_alt, theta2_alt)]

class DynamicsCalculator:
    """Calculator for robotic dynamics"""

    @staticmethod
    def simple_pendulum_dynamics(theta: float, theta_dot: float, length: float, mass: float = 1.0, gravity: float = 9.81) -> Tuple[float, float, float]:
        """
        Calculate simple pendulum dynamics (simplified model of single link)
        Returns: (theta_ddot, kinetic_energy, potential_energy)
        """
        # Equation of motion for simple pendulum: theta_ddot = -(g/l) * sin(theta)
        theta_ddot = -(gravity / length) * sin(theta)

        # Kinetic energy: KE = 0.5 * m * v^2 = 0.5 * m * (l * theta_dot)^2
        kinetic_energy = 0.5 * mass * (length * theta_dot)**2

        # Potential energy: PE = m * g * h = m * g * l * (1 - cos(theta))
        potential_energy = mass * gravity * length * (1 - cos(theta))

        return theta_ddot, kinetic_energy, potential_energy

    @staticmethod
    def mass_matrix_2dof(theta1: float, theta2: float, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0) -> np.ndarray:
        """
        Calculate mass matrix for 2-DOF manipulator (simplified)
        """
        # This is a simplified version - in reality, this would be more complex
        # Based on the general form of robot dynamics: M(q)q̈ + C(q, q̇)q̇ + g(q) = τ
        s2 = sin(theta2)
        c2 = cos(theta2)

        m11 = m1 * l1**2 + m2 * (l1**2 + l2**2 + 2*l1*l2*c2) + 0.25  # Simplified
        m12 = m2 * (l2**2 + l1*l2*c2) + 0.25  # Simplified
        m21 = m12
        m22 = m2 * l2**2 + 0.25  # Simplified

        return np.array([[m11, m12], [m21, m22]])

class RobotController:
    """Simulates basic robot control systems"""

    @staticmethod
    def pid_control(current_value: float, setpoint: float, kp: float, ki: float, kd: float,
                    error_integral: float = 0.0, prev_error: float = 0.0, dt: float = 0.01) -> Tuple[float, float, float]:
        """
        Simple PID controller implementation
        Returns: (control_output, updated_error_integral, updated_prev_error)
        """
        error = setpoint - current_value

        # Proportional term
        p_term = kp * error

        # Integral term
        error_integral += error * dt
        i_term = ki * error_integral

        # Derivative term
        derivative = (error - prev_error) / dt
        d_term = kd * derivative

        control_output = p_term + i_term + d_term

        return control_output, error_integral, error

def calculate_robot_metrics(joint_angles: List[float], link_lengths: List[float]) -> Dict[str, Any]:
    """
    Calculate various metrics for a robot with given joint angles and link lengths
    """
    metrics = {
        "configuration": len(joint_angles),
        "total_length": sum(link_lengths),
        "workspace_volume": 0.0,  # Simplified calculation
        "reachable_points": []
    }

    # Calculate end effector position for different configurations
    if len(joint_angles) >= 2 and len(link_lengths) >= 2:
        x, y = KinematicsCalculator.forward_kinematics_2dof(
            joint_angles[0], joint_angles[1],
            link_lengths[0], link_lengths[1]
        )
        metrics["end_effector_position"] = {"x": x, "y": y}

    return metrics

# Example usage functions
def simulate_robot_motion(initial_joints: List[float], target_joints: List[float], steps: int = 100) -> List[List[float]]:
    """
    Simulate smooth motion from initial to target joint configuration
    """
    trajectory = []

    for i in range(steps + 1):
        t = i / steps  # Interpolation parameter from 0 to 1

        current_joints = []
        for j in range(len(initial_joints)):
            # Linear interpolation between initial and target
            joint_value = initial_joints[j] + t * (target_joints[j] - initial_joints[j])
            current_joints.append(joint_value)

        trajectory.append(current_joints)

    return trajectory