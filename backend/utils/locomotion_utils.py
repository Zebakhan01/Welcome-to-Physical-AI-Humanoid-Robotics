import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from math import sqrt, cos, sin, tan, pi, exp
import numpy as np
from scipy import signal
from scipy.integrate import odeint

@dataclass
class FootStep:
    """Represents a foot step in walking pattern"""
    x: float  # Position in x direction
    y: float  # Position in y direction (lateral)
    theta: float  # Orientation in radians
    time: float  # Time of step
    support_leg: str  # "left" or "right"

@dataclass
class WalkingState:
    """Represents the current state of walking"""
    com_x: float  # Center of mass x position
    com_y: float  # Center of mass y position
    com_z: float  # Center of mass z position (height)
    com_dx: float  # Center of mass velocity x
    com_dy: float  # Center of mass velocity y
    zmp_x: float  # Zero moment point x
    zmp_y: float  # Zero moment point y
    left_foot: Tuple[float, float, float]  # (x, y, theta)
    right_foot: Tuple[float, float, float]  # (x, y, theta)
    support_leg: str  # "left", "right", or "double"

@dataclass
class GaitParameters:
    """Parameters for walking gait"""
    step_length: float = 0.3  # Forward step length (m)
    step_width: float = 0.2   # Lateral step width (m)
    step_time: float = 0.8    # Time for each step (s)
    z_com: float = 0.8        # Constant CoM height (m)
    walking_speed: float = 0.5  # Desired walking speed (m/s)

class InvertedPendulumModel:
    """Inverted pendulum model for bipedal walking"""

    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)

    def compute_zmp(self, com_pos: np.ndarray, com_vel: np.ndarray, com_acc: np.ndarray) -> np.ndarray:
        """
        Compute Zero Moment Point from CoM state
        ZMP = [com_x - g/omega^2 * com_acc_x, com_y - g/omega^2 * com_acc_y]
        """
        zmp = np.array([
            com_pos[0] - self.com_height / self.gravity * com_acc[0],
            com_pos[1] - self.com_height / self.gravity * com_acc[1]
        ])
        return zmp

    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """
        Compute Capture Point: where to step to come to stop
        Capture Point = CoM position + CoM velocity / omega
        """
        cp = com_pos + com_vel / self.omega
        return cp

class LinearInvertedPendulumModel(InvertedPendulumModel):
    """Linear Inverted Pendulum Model with constant CoM height"""

    def __init__(self, com_height: float = 0.8, gravity: float = 9.81):
        super().__init__(com_height, gravity)

    def generate_footsteps_preview_control(self, initial_com: np.ndarray,
                                         final_com: np.ndarray,
                                         step_time: float,
                                         dt: float = 0.01) -> List[FootStep]:
        """
        Generate footsteps using preview control for LIPM
        """
        footsteps = []

        # Calculate number of steps needed
        distance = np.linalg.norm(final_com[:2] - initial_com[:2])
        num_steps = int(distance / 0.3) + 1  # Assuming 0.3m per step

        # Generate alternating footsteps
        for i in range(num_steps):
            # Calculate target position along the path
            t = i / num_steps
            target_pos = initial_com[:2] + t * (final_com[:2] - initial_com[:2])

            # Alternate between left and right foot
            if i % 2 == 0:
                # Right foot step
                foot_pos = (target_pos[0], target_pos[1] - 0.1, 0)  # Slightly to the side
                support_leg = "left"
            else:
                # Left foot step
                foot_pos = (target_pos[0], target_pos[1] + 0.1, 0)  # Slightly to the side
                support_leg = "right"

            # Create footstep
            footstep = FootStep(
                x=foot_pos[0],
                y=foot_pos[1],
                theta=foot_pos[2],
                time=i * step_time,
                support_leg=support_leg
            )

            footsteps.append(footstep)

        return footsteps

    def compute_com_trajectory(self, zmp_trajectory: np.ndarray,
                             initial_com: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Compute CoM trajectory from ZMP reference using LIPM dynamics
        CoM'' = omega^2 * (CoM - ZMP)
        """
        n_points = zmp_trajectory.shape[0]
        com_trajectory = np.zeros((n_points, 2))
        com_trajectory[0] = initial_com[:2]  # Initial CoM position

        # Use simple Euler integration
        com_vel = np.zeros(2)  # Initial velocity

        for i in range(1, n_points):
            # CoM acceleration: com_acc = omega^2 * (com - zmp)
            com_acc = self.omega**2 * (com_trajectory[i-1] - zmp_trajectory[i-1])

            # Update velocity and position
            com_vel += com_acc * dt
            com_trajectory[i] = com_trajectory[i-1] + com_vel * dt

        return com_trajectory

class WalkingPatternGenerator:
    """Generates walking patterns for humanoid robots"""

    def __init__(self, gait_params: GaitParameters):
        self.gait_params = gait_params

    def generate_walk_pattern(self, num_steps: int, start_pos: Tuple[float, float] = (0, 0)) -> List[Dict[str, Any]]:
        """
        Generate a walking pattern with alternating footsteps
        """
        pattern = []

        current_x, current_y = start_pos
        current_theta = 0.0

        for i in range(num_steps):
            # Determine support leg (alternating)
            support_leg = "left" if i % 2 == 0 else "right"
            swing_leg = "right" if i % 2 == 0 else "left"

            # Calculate step position
            if support_leg == "left":
                # Right foot moves forward
                step_x = current_x + self.gait_params.step_length
                step_y = current_y - self.gait_params.step_width  # Move to right side
            else:
                # Left foot moves forward
                step_x = current_x + self.gait_params.step_length
                step_y = current_y + self.gait_params.step_width  # Move to left side

            # Create step entry
            step = {
                "step_number": i + 1,
                "swing_leg": swing_leg,
                "support_leg": support_leg,
                "step_position": (step_x, step_y, current_theta),
                "time": i * self.gait_params.step_time,
                "com_reference": (step_x - self.gait_params.step_length/2, 0, self.gait_params.z_com)
            }

            pattern.append(step)

            # Update current position for next step
            current_x = step_x
            current_y = step_y

        return pattern

class BalanceController:
    """Balance controller for humanoid robots"""

    def __init__(self, kp: float = 10.0, ki: float = 0.1, kd: float = 1.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)

    def compute_balance_correction(self, current_zmp: np.ndarray,
                                 desired_zmp: np.ndarray,
                                 dt: float = 0.01) -> np.ndarray:
        """
        Compute balance correction using PID control
        """
        error = desired_zmp - current_zmp

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error

        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else np.zeros(2)
        d_term = self.kd * derivative

        # Store for next iteration
        self.previous_error = error

        correction = p_term + i_term + d_term
        return correction

class GaitAnalyzer:
    """Analyzes gait parameters and stability"""

    @staticmethod
    def calculate_gait_parameters(footsteps: List[FootStep]) -> Dict[str, float]:
        """
        Calculate gait parameters from footsteps
        """
        if len(footsteps) < 2:
            return {}

        # Calculate step lengths
        step_lengths = []
        step_times = []
        step_widths = []

        for i in range(1, len(footsteps)):
            # Calculate step length (distance between consecutive footsteps)
            dx = footsteps[i].x - footsteps[i-1].x
            dy = footsteps[i].y - footsteps[i-1].y
            step_length = sqrt(dx**2 + dy**2)
            step_lengths.append(step_length)

            # Calculate step time
            dt = footsteps[i].time - footsteps[i-1].time
            step_times.append(dt)

            # Calculate step width (lateral distance)
            step_width = abs(dy)
            step_widths.append(step_width)

        # Calculate averages
        avg_step_length = sum(step_lengths) / len(step_lengths) if step_lengths else 0
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        avg_step_width = sum(step_widths) / len(step_widths) if step_widths else 0

        # Calculate walking speed
        avg_speed = avg_step_length / avg_step_time if avg_step_time > 0 else 0

        return {
            "avg_step_length": avg_step_length,
            "avg_step_time": avg_step_time,
            "avg_step_width": avg_step_width,
            "avg_speed": avg_speed,
            "step_length_variance": np.var(step_lengths) if step_lengths else 0,
            "step_width_variance": np.var(step_widths) if step_widths else 0
        }

    @staticmethod
    def calculate_stability_margin(current_zmp: np.ndarray,
                                 support_polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate stability margin as distance from ZMP to support polygon boundary
        """
        # For simplicity, assume rectangular support polygon
        # Calculate min distance from ZMP to polygon edges
        if not support_polygon:
            return 0.0

        # Find bounding box of support polygon
        xs = [p[0] for p in support_polygon]
        ys = [p[1] for p in support_polygon]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate distance to each edge
        dist_left = current_zmp[0] - min_x
        dist_right = max_x - current_zmp[0]
        dist_front = current_zmp[1] - min_y
        dist_back = max_y - current_zmp[1]

        # Return minimum distance (stability margin)
        stability_margin = min(dist_left, dist_right, dist_front, dist_back)
        return stability_margin

class CentralPatternGenerator:
    """Central Pattern Generator for rhythmic walking patterns"""

    def __init__(self, frequency: float = 1.0, amplitude: float = 0.1):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase_offset = pi  # 180 degree phase difference for alternating legs

    def generate_oscillator_signal(self, time: float, phase_offset: float = 0.0) -> float:
        """
        Generate oscillator signal for CPG
        """
        return self.amplitude * sin(2 * pi * self.frequency * time + phase_offset)

    def generate_leg_trajectories(self, times: List[float]) -> Tuple[List[float], List[float]]:
        """
        Generate trajectories for left and right legs
        """
        left_leg = [self.generate_oscillator_signal(t) for t in times]
        right_leg = [self.generate_oscillator_signal(t, self.phase_offset) for t in times]
        return left_leg, right_leg

def simulate_walking_step(com_initial: np.ndarray, zmp_reference: np.ndarray,
                         dt: float = 0.01, duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a walking step using LIPM
    """
    # Create time vector
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Initialize state
    com_state = np.zeros((n_steps, 3))  # x, y, z
    com_state[0] = com_initial

    # Initialize velocity
    com_vel = np.zeros(3)

    # LIPM parameters
    g = 9.81
    h = com_initial[2]  # height
    omega = sqrt(g / h)

    # Simulation loop
    for i in range(1, n_steps):
        # Calculate acceleration based on LIPM: com_ddot = omega^2 * (com - zmp)
        if i < len(zmp_reference):
            zmp = zmp_reference[i]
        else:
            zmp = zmp_reference[-1]  # Use last value if reference is shorter

        com_acc = np.array([
            omega**2 * (com_state[i-1, 0] - zmp[0]),
            omega**2 * (com_state[i-1, 1] - zmp[1]),
            0  # z acceleration is 0 (constant height)
        ])

        # Update velocity and position
        com_vel += com_acc * dt
        com_state[i] = com_state[i-1] + com_vel * dt

    return t, com_state