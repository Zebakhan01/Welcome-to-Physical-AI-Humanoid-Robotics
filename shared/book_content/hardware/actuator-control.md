---
sidebar_position: 3
---

# Actuator Control

## Introduction to Actuator Control Systems

Actuator control is fundamental to humanoid robotics, enabling precise movement, manipulation, and interaction with the physical environment. This section covers the comprehensive control of various actuator types including servo motors, hydraulic systems, pneumatic actuators, and specialized robotic actuators. The focus is on achieving stable, accurate, and responsive control while maintaining safety and efficiency.

## Actuator Types and Characteristics

### Servo Motor Control

#### Basic Servo Control Architecture

```python
# servo_control.py
import numpy as np
import time
from abc import ABC, abstractmethod
import threading
import queue

class ActuatorInterface(ABC):
    """
    Abstract interface for actuator control
    """

    def __init__(self, name, actuator_type, control_mode='position'):
        self.name = name
        self.actuator_type = actuator_type
        self.control_mode = control_mode
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_effort = 0.0
        self.target_position = 0.0
        self.target_velocity = 0.0
        self.target_effort = 0.0
        self.is_enabled = False
        self.is_calibrated = False
        self.last_update_time = time.time()

    @abstractmethod
    def set_position(self, position):
        """Set target position for actuator"""
        pass

    @abstractmethod
    def set_velocity(self, velocity):
        """Set target velocity for actuator"""
        pass

    @abstractmethod
    def set_effort(self, effort):
        """Set target effort/torque for actuator"""
        pass

    @abstractmethod
    def get_position(self):
        """Get current position of actuator"""
        pass

    @abstractmethod
    def get_velocity(self):
        """Get current velocity of actuator"""
        pass

    @abstractmethod
    def get_effort(self):
        """Get current effort/torque of actuator"""
        pass

    @abstractmethod
    def enable(self):
        """Enable the actuator"""
        pass

    @abstractmethod
    def disable(self):
        """Disable the actuator"""
        pass

    @abstractmethod
    def calibrate(self):
        """Calibrate the actuator"""
        pass


class ServoMotor(ActuatorInterface):
    """
    Servo motor control implementation
    """

    def __init__(self, name, gear_ratio=100, max_torque=10.0, max_speed=5.0):
        super().__init__(name, 'servo', 'position')
        self.gear_ratio = gear_ratio
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.position_limits = [-np.pi, np.pi]  # radians
        self.velocity_limits = [-max_speed, max_speed]
        self.effort_limits = [-max_torque, max_torque]

        # Control parameters
        self.kp = 100.0  # Proportional gain
        self.ki = 10.0   # Integral gain
        self.kd = 5.0    # Derivative gain

        # PID controller state
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.error_derivative = 0.0

        # Safety parameters
        self.temperature = 25.0  # Celsius
        self.max_temperature = 80.0
        self.current_limit = max_torque * 1.2

    def set_position(self, position):
        """Set target position with safety checks"""
        # Clamp to limits
        clamped_position = np.clip(position, self.position_limits[0], self.position_limits[1])

        # Check safety conditions
        if self.temperature > self.max_temperature:
            raise RuntimeError(f"Actuator {self.name} overheating: {self.temperature}°C")

        self.target_position = clamped_position

        # Send command to hardware (simplified)
        self._send_position_command(clamped_position)

    def _send_position_command(self, position):
        """
        Send position command to hardware
        """
        # In real implementation, this would send command via CAN, serial, etc.
        # For simulation, update internal state
        self._simulate_position_update(position)

    def _simulate_position_update(self, target_position):
        """
        Simulate position update (for simulation purposes)
        """
        # Simple first-order dynamics simulation
        error = target_position - self.current_position
        dt = time.time() - self.last_update_time
        self.last_update_time = time.time()

        # Update position based on error and dynamics
        velocity = min(self.max_speed, max(-self.max_speed, error * 0.1))
        self.current_position += velocity * dt

    def set_velocity(self, velocity):
        """Set target velocity with safety checks"""
        # Clamp to velocity limits
        clamped_velocity = np.clip(velocity, self.velocity_limits[0], self.velocity_limits[1])

        self.target_velocity = clamped_velocity

        # Send velocity command to hardware
        self._send_velocity_command(clamped_velocity)

    def set_effort(self, effort):
        """Set target effort/torque with safety checks"""
        # Clamp to effort limits
        clamped_effort = np.clip(effort, self.effort_limits[0], self.effort_limits[1])

        self.target_effort = clamped_effort

        # Send effort command to hardware
        self._send_effort_command(clamped_effort)

    def get_position(self):
        """Get current position from hardware"""
        # In real implementation, read from encoder
        # For simulation, return current position
        return self.current_position

    def get_velocity(self):
        """Get current velocity from hardware"""
        # In real implementation, calculate from encoder data
        # For simulation, return current velocity
        return self.current_velocity

    def get_effort(self):
        """Get current effort/torque from hardware"""
        # In real implementation, read from force/torque sensor or current monitoring
        # For simulation, return current effort
        return self.current_effort

    def update_control(self):
        """
        Update control loop for PID control
        """
        if not self.is_enabled:
            return

        # Calculate error
        position_error = self.target_position - self.current_position

        # Update integral term (with anti-windup)
        self.error_integral += position_error * (time.time() - self.last_update_time)
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)

        # Calculate derivative term
        dt = time.time() - self.last_update_time
        if dt > 0:
            self.error_derivative = (position_error - self.previous_error) / dt
        else:
            self.error_derivative = 0.0

        # Calculate PID output
        effort_output = (
            self.kp * position_error +
            self.ki * self.error_integral +
            self.kd * self.error_derivative
        )

        # Clamp to effort limits
        effort_output = np.clip(effort_output, self.effort_limits[0], self.effort_limits[1])

        # Apply effort command
        self.set_effort(effort_output)

        # Update previous error
        self.previous_error = position_error
        self.last_update_time = time.time()

    def enable(self):
        """Enable the servo motor"""
        self.is_enabled = True
        self._send_enable_command()

    def disable(self):
        """Disable the servo motor"""
        self.is_enabled = False
        self._send_disable_command()

    def calibrate(self):
        """Calibrate the servo motor"""
        # Move to calibration position
        self.set_position(0.0)  # Move to zero position
        time.sleep(1.0)  # Wait for movement to complete

        # Set current position as zero reference
        self.current_position = 0.0
        self.is_calibrated = True

        return True

    def _send_enable_command(self):
        """Send enable command to hardware"""
        # Implementation depends on communication protocol
        pass

    def _send_disable_command(self):
        """Send disable command to hardware"""
        # Implementation depends on communication protocol
        pass

    def _send_velocity_command(self, velocity):
        """Send velocity command to hardware"""
        # Implementation depends on communication protocol
        pass

    def _send_effort_command(self, effort):
        """Send effort command to hardware"""
        # Implementation depends on communication protocol
        pass

    def get_status(self):
        """Get actuator status information"""
        return {
            'name': self.name,
            'type': self.actuator_type,
            'enabled': self.is_enabled,
            'calibrated': self.is_calibrated,
            'current_position': self.current_position,
            'current_velocity': self.current_velocity,
            'current_effort': self.current_effort,
            'target_position': self.target_position,
            'target_velocity': self.target_velocity,
            'target_effort': self.target_effort,
            'temperature': self.temperature,
            'error': self.target_position - self.current_position
        }
```

### Advanced Servo Control

```python
class AdvancedServoController:
    """
    Advanced servo controller with feedforward and adaptive control
    """

    def __init__(self, servo_motor, control_frequency=1000):
        self.servo = servo_motor
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # PID gains
        self.kp = 100.0
        self.ki = 10.0
        self.kd = 5.0

        # Feedforward gains
        self.kff_pos = 0.0  # Position feedforward
        self.kff_vel = 1.0  # Velocity feedforward
        self.kff_acc = 0.1  # Acceleration feedforward

        # Adaptive control parameters
        self.adaptive_learning_rate = 0.01
        self.model_uncertainty = np.zeros(3)  # [pos_err, vel_err, effort_err]

        # Trajectory tracking
        self.trajectory_generator = TrajectoryGenerator()
        self.current_trajectory_point = None

        # Safety limits
        self.max_position_error = 0.1  # radians
        self.max_velocity_error = 1.0  # rad/s
        self.max_effort = servo_motor.max_torque

        # Control history for adaptive learning
        self.error_history = []
        self.control_history = []
        self.max_history_length = 1000

    def update_control(self, desired_state, current_state):
        """
        Advanced control update with feedforward and adaptation
        """
        # Calculate tracking errors
        position_error = desired_state['position'] - current_state['position']
        velocity_error = desired_state['velocity'] - current_state['velocity']

        # Store errors for adaptation
        self._store_error_history(position_error, velocity_error, current_state['effort'])

        # Calculate feedforward terms
        feedforward_effort = self._calculate_feedforward_effort(desired_state)

        # Calculate feedback terms using PID
        feedback_effort = self._calculate_feedback_effort(position_error, velocity_error)

        # Calculate adaptive compensation
        adaptive_compensation = self._calculate_adaptive_compensation(position_error, velocity_error)

        # Combine all terms
        total_effort = feedforward_effort + feedback_effort + adaptive_compensation

        # Apply safety limits
        total_effort = np.clip(total_effort, -self.max_effort, self.max_effort)

        # Apply control command
        self.servo.set_effort(total_effort)

        # Update adaptive parameters
        self._update_adaptive_parameters(position_error, velocity_error, total_effort)

        return {
            'feedforward_effort': feedforward_effort,
            'feedback_effort': feedback_effort,
            'adaptive_compensation': adaptive_compensation,
            'total_effort': total_effort,
            'position_error': position_error,
            'velocity_error': velocity_error
        }

    def _calculate_feedforward_effort(self, desired_state):
        """
        Calculate feedforward effort based on desired trajectory
        """
        # Position feedforward (bias term)
        pos_ff = self.kff_pos * desired_state['position']

        # Velocity feedforward (damping compensation)
        vel_ff = self.kff_vel * desired_state['velocity']

        # Acceleration feedforward (inertia compensation)
        acc_ff = self.kff_acc * desired_state['acceleration']

        return pos_ff + vel_ff + acc_ff

    def _calculate_feedback_effort(self, position_error, velocity_error):
        """
        Calculate feedback effort using PID
        """
        # Update integral term with anti-windup
        self.error_integral += position_error * self.dt
        # Anti-windup: limit integral term
        max_integral = self.max_effort / self.ki if self.ki != 0 else 1000.0
        self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)

        # Calculate derivative (use velocity error directly if available)
        derivative_term = self.kd * velocity_error

        # PID calculation
        feedback_effort = (
            self.kp * position_error +
            self.ki * self.error_integral +
            derivative_term
        )

        return feedback_effort

    def _calculate_adaptive_compensation(self, position_error, velocity_error):
        """
        Calculate adaptive compensation based on model uncertainty
        """
        # Simple adaptive compensation based on error patterns
        adaptive_effort = 0.0

        # Learn from position error
        if abs(position_error) > 0.01:  # Threshold for learning
            adaptive_effort += -self.adaptive_learning_rate * position_error

        # Learn from velocity error
        if abs(velocity_error) > 0.1:  # Threshold for learning
            adaptive_effort += -self.adaptive_learning_rate * 0.1 * velocity_error

        return adaptive_effort

    def _update_adaptive_parameters(self, position_error, velocity_error, effort):
        """
        Update adaptive parameters based on recent errors
        """
        # Update uncertainty model based on recent performance
        recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history

        if recent_errors:
            avg_pos_error = np.mean([e['position'] for e in recent_errors])
            avg_vel_error = np.mean([e['velocity'] for e in recent_errors])

            # Adjust feedforward parameters based on systematic errors
            self.kff_pos += self.adaptive_learning_rate * avg_pos_error
            self.kff_vel += self.adaptive_learning_rate * avg_vel_error * 0.1

            # Keep parameters bounded
            self.kff_pos = np.clip(self.kff_pos, -1.0, 1.0)
            self.kff_vel = np.clip(self.kff_vel, 0.8, 1.2)

    def _store_error_history(self, pos_error, vel_error, effort):
        """
        Store error history for adaptive learning
        """
        error_data = {
            'position': pos_error,
            'velocity': vel_error,
            'effort': effort,
            'timestamp': time.time()
        }

        self.error_history.append(error_data)
        if len(self.error_history) > self.max_history_length:
            self.error_history.pop(0)

    def set_trajectory(self, trajectory):
        """
        Set trajectory for tracking
        """
        self.trajectory_generator.set_trajectory(trajectory)

    def get_trajectory_state(self):
        """
        Get current trajectory state
        """
        if self.trajectory_generator:
            return self.trajectory_generator.get_current_state(time.time())
        return None


class TrajectoryGenerator:
    """
    Trajectory generation for smooth motion control
    """

    def __init__(self):
        self.trajectory_segments = []
        self.current_segment_idx = 0
        self.segment_start_time = 0
        self.is_active = False

    def generate_min_jerk_trajectory(self, start_pos, end_pos, duration, dt=0.001):
        """
        Generate minimum jerk trajectory
        """
        t = np.arange(0, duration, dt)
        n_points = len(t)

        # Minimum jerk polynomial coefficients
        a0 = start_pos
        a1 = 0  # Initial velocity = 0
        a2 = 0  # Initial acceleration = 0
        a3 = (20 * (end_pos - start_pos)) / (2 * duration**3)
        a4 = (-30 * (end_pos - start_pos)) / (2 * duration**4)
        a5 = (12 * (end_pos - start_pos)) / (2 * duration**5)

        # Calculate trajectory points
        trajectory = []
        for i, time_point in enumerate(t):
            s = time_point / duration if duration > 0 else 0

            pos = (a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5)
            vel = (a1 + 2*a2*s + 3*a3*s**2 + 4*a4*s**3 + 5*a5*s**4) / duration
            acc = (2*a2 + 6*a3*s + 12*a4*s**2 + 20*a5*s**3) / (duration**2)

            trajectory.append({
                'time': time_point,
                'position': pos,
                'velocity': vel,
                'acceleration': acc
            })

        return trajectory

    def generate_spline_trajectory(self, waypoints, duration_per_segment=None):
        """
        Generate spline trajectory through waypoints
        """
        if len(waypoints) < 2:
            return []

        if duration_per_segment is None:
            duration_per_segment = [1.0] * (len(waypoints) - 1)

        trajectory = []
        cumulative_time = 0

        for i in range(len(waypoints) - 1):
            segment_duration = duration_per_segment[i]

            segment_traj = self.generate_min_jerk_trajectory(
                waypoints[i], waypoints[i+1], segment_duration
            )

            # Adjust times for cumulative trajectory
            for point in segment_traj:
                point['time'] += cumulative_time

            trajectory.extend(segment_traj[:-1])  # Exclude last point to avoid duplicates
            cumulative_time += segment_duration

        # Add final point
        if segment_traj:
            trajectory.append(segment_traj[-1])

        return trajectory

    def set_trajectory(self, trajectory_points):
        """
        Set trajectory from precomputed points
        """
        self.trajectory_segments = trajectory_points
        self.current_segment_idx = 0
        self.segment_start_time = time.time()
        self.is_active = True

    def get_current_state(self, current_time):
        """
        Get current state from trajectory
        """
        if not self.is_active or not self.trajectory_segments:
            return None

        # Find current trajectory point
        elapsed_time = current_time - self.segment_start_time

        # Simple linear interpolation between trajectory points
        for i, point in enumerate(self.trajectory_segments):
            if i < len(self.trajectory_segments) - 1:
                next_point = self.trajectory_segments[i + 1]
                if point['time'] <= elapsed_time < next_point['time']:
                    # Interpolate between points
                    t_ratio = (elapsed_time - point['time']) / (next_point['time'] - point['time'])

                    interpolated_state = {
                        'position': point['position'] + t_ratio * (next_point['position'] - point['position']),
                        'velocity': point['velocity'] + t_ratio * (next_point['velocity'] - point['velocity']),
                        'acceleration': point['acceleration'] + t_ratio * (next_point['acceleration'] - point['acceleration']),
                        'time': elapsed_time
                    }
                    return interpolated_state

        # If past the end of trajectory, return last point
        if self.trajectory_segments:
            last_point = self.trajectory_segments[-1].copy()
            last_point['time'] = elapsed_time  # Update time
            return last_point

        return None
```

## Joint Control Systems

### Multi-Joint Coordination

```python
class JointController:
    """
    Multi-joint controller for coordinated movement
    """

    def __init__(self, joint_names, actuator_interfaces):
        self.joint_names = joint_names
        self.actuators = {name: actuator for name, actuator in zip(joint_names, actuator_interfaces)}
        self.joint_states = {name: {'position': 0.0, 'velocity': 0.0, 'effort': 0.0} for name in joint_names}
        self.desired_states = {name: {'position': 0.0, 'velocity': 0.0, 'effort': 0.0} for name in joint_names}

        # Joint limits
        self.position_limits = {}
        self.velocity_limits = {}
        self.effort_limits = {}

        # Control parameters
        self.control_mode = 'position'  # position, velocity, or effort
        self.control_frequency = 1000  # Hz
        self.dt = 1.0 / self.control_frequency

        # Initialize joint limits
        self._initialize_joint_limits()

    def _initialize_joint_limits(self):
        """
        Initialize joint limits based on actuator capabilities
        """
        for joint_name, actuator in self.actuators.items():
            # Set limits based on actuator specifications
            self.position_limits[joint_name] = actuator.position_limits
            self.velocity_limits[joint_name] = actuator.velocity_limits
            self.effort_limits[joint_name] = actuator.effort_limits

    def update_joint_states(self):
        """
        Update current joint states from actuators
        """
        for joint_name, actuator in self.actuators.items():
            self.joint_states[joint_name] = {
                'position': actuator.get_position(),
                'velocity': actuator.get_velocity(),
                'effort': actuator.get_effort()
            }

    def set_joint_positions(self, positions_dict):
        """
        Set desired joint positions for all joints
        """
        for joint_name, position in positions_dict.items():
            if joint_name in self.actuators:
                # Clamp to limits
                clamped_position = np.clip(
                    position,
                    self.position_limits[joint_name][0],
                    self.position_limits[joint_name][1]
                )
                self.desired_states[joint_name]['position'] = clamped_position
                self.actuators[joint_name].set_position(clamped_position)

    def set_joint_velocities(self, velocities_dict):
        """
        Set desired joint velocities for all joints
        """
        for joint_name, velocity in velocities_dict.items():
            if joint_name in self.actuators:
                # Clamp to limits
                clamped_velocity = np.clip(
                    velocity,
                    self.velocity_limits[joint_name][0],
                    self.velocity_limits[joint_name][1]
                )
                self.desired_states[joint_name]['velocity'] = clamped_velocity
                self.actuators[joint_name].set_velocity(clamped_velocity)

    def set_joint_efforts(self, efforts_dict):
        """
        Set desired joint efforts for all joints
        """
        for joint_name, effort in efforts_dict.items():
            if joint_name in self.actuators:
                # Clamp to limits
                clamped_effort = np.clip(
                    effort,
                    self.effort_limits[joint_name][0],
                    self.effort_limits[joint_name][1]
                )
                self.desired_states[joint_name]['effort'] = clamped_effort
                self.actuators[joint_name].set_effort(clamped_effort)

    def move_to_configuration(self, joint_config, duration=5.0, control_rate=100):
        """
        Move joints to specified configuration over time
        """
        start_config = {name: state['position'] for name, state in self.joint_states.items()}
        end_config = joint_config

        # Generate trajectory
        trajectory = self._generate_joint_trajectory(start_config, end_config, duration)

        # Execute trajectory
        start_time = time.time()
        total_steps = int(duration * control_rate)

        for step in range(total_steps):
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            # Get current trajectory state
            progress = min(1.0, elapsed_time / duration)
            current_config = self._interpolate_configurations(start_config, end_config, progress)

            # Set joint positions
            self.set_joint_positions(current_config)

            # Wait for next control cycle
            time.sleep(1.0 / control_rate)

    def _generate_joint_trajectory(self, start_config, end_config, duration):
        """
        Generate joint trajectory between configurations
        """
        trajectory = []
        num_points = int(duration * self.control_frequency)

        for i in range(num_points + 1):
            progress = i / num_points
            current_config = self._interpolate_configurations(start_config, end_config, progress)

            trajectory.append({
                'time': i * self.dt,
                'configuration': current_config
            })

        return trajectory

    def _interpolate_configurations(self, start_config, end_config, progress):
        """
        Interpolate between two joint configurations
        """
        interpolated_config = {}

        for joint_name in self.joint_names:
            start_pos = start_config.get(joint_name, 0.0)
            end_pos = end_config.get(joint_name, 0.0)

            # Use cubic interpolation for smooth motion
            cubic_progress = 3 * progress**2 - 2 * progress**3  # Smooth interpolation
            interpolated_config[joint_name] = start_pos + cubic_progress * (end_pos - start_pos)

        return interpolated_config

    def get_joint_states(self):
        """
        Get current joint states
        """
        return self.joint_states.copy()

    def get_desired_states(self):
        """
        Get desired joint states
        """
        return self.desired_states.copy()

    def compute_inverse_kinematics(self, end_effector_pose, joint_seed=None):
        """
        Compute inverse kinematics to achieve desired end-effector pose
        """
        # This would interface with an IK solver
        # For now, return a placeholder implementation
        if joint_seed is None:
            joint_seed = {name: state['position'] for name, state in self.joint_states.items()}

        # Placeholder IK solution
        ik_solution = joint_seed.copy()

        # In real implementation, this would use an IK solver like:
        # - Analytical IK for simple robots
        # - Numerical IK (Jacobian-based)
        # - Optimization-based IK
        # - Deep learning-based IK

        return ik_solution

    def enforce_joint_limits(self, joint_config):
        """
        Enforce joint limits on configuration
        """
        limited_config = joint_config.copy()

        for joint_name, position in joint_config.items():
            if joint_name in self.position_limits:
                limits = self.position_limits[joint_name]
                limited_config[joint_name] = np.clip(position, limits[0], limits[1])

        return limited_config

    def check_collisions(self, joint_config):
        """
        Check for self-collisions at given configuration
        """
        # This would interface with collision detection system
        # For now, return False (no collisions)
        return False

    def get_control_error(self):
        """
        Get current control error for all joints
        """
        errors = {}
        for joint_name in self.joint_names:
            current_pos = self.joint_states[joint_name]['position']
            desired_pos = self.desired_states[joint_name]['position']
            errors[joint_name] = abs(current_pos - desired_pos)
        return errors
```

## Force Control and Impedance Control

### Force Control Implementation

```python
class ForceController:
    """
    Force/torque control for compliant manipulation
    """

    def __init__(self, actuator_interface, force_sensor_interface):
        self.actuator = actuator_interface
        self.force_sensor = force_sensor_interface
        self.is_force_control_active = False

        # Force control parameters
        self.desired_force = np.zeros(6)  # [Fx, Fy, Fz, Mx, My, Mz]
        self.current_force = np.zeros(6)
        self.force_error_integral = np.zeros(6)
        self.previous_force_error = np.zeros(6)

        # Force control PID gains
        self.kp_force = 1.0
        self.ki_force = 0.1
        self.kd_force = 0.05

        # Force limits
        self.max_force = 100.0  # Newtons
        self.max_torque = 10.0  # Nm

        # Safety parameters
        self.force_deadband = 0.5  # Ignore small force errors
        self.max_force_change_rate = 50.0  # Limit rate of force change

    def start_force_control(self, desired_force):
        """
        Start force control mode
        """
        self.desired_force = np.array(desired_force)
        self.is_force_control_active = True

        # Switch actuator to effort control mode
        self.actuator.set_control_mode('effort')

    def stop_force_control(self):
        """
        Stop force control mode
        """
        self.is_force_control_active = False
        self.actuator.set_control_mode('position')  # Return to position control

    def update_force_control(self):
        """
        Update force control loop
        """
        if not self.is_force_control_active:
            return

        # Get current force measurement
        self.current_force = self.force_sensor.get_force_torque()

        # Calculate force error
        force_error = self.desired_force - self.current_force

        # Apply deadband to small errors
        force_error = self._apply_force_deadband(force_error)

        # Update integral term (with anti-windup)
        self.force_error_integral += force_error * self.dt
        integral_limit = self.max_force / self.ki_force if self.ki_force != 0 else 1000.0
        self.force_error_integral = np.clip(self.force_error_integral, -integral_limit, integral_limit)

        # Calculate derivative term
        force_derivative = (force_error - self.previous_force_error) / self.dt if self.dt > 0 else np.zeros(6)
        self.previous_force_error = force_error

        # Calculate force control output
        force_control_output = (
            self.kp_force * force_error +
            self.ki_force * self.force_error_integral +
            self.kd_force * force_derivative
        )

        # Limit force control output
        force_control_output = np.clip(force_control_output, -self.max_force_change_rate, self.max_force_change_rate)

        # Get current position-based control
        position_based_effort = self._get_position_control_effort()

        # Combine position and force control
        combined_effort = position_based_effort + force_control_output

        # Apply final effort limits
        combined_effort = np.clip(combined_effort, -self.actuator.max_effort, self.actuator.max_effort)

        # Apply to actuator
        self.actuator.set_effort(combined_effort)

    def _apply_force_deadband(self, force_error):
        """
        Apply deadband to force error
        """
        masked_error = np.copy(force_error)
        for i in range(len(force_error)):
            if abs(force_error[i]) < self.force_deadband:
                masked_error[i] = 0.0
        return masked_error

    def _get_position_control_effort(self):
        """
        Get position control effort for hybrid control
        """
        # This would typically be from the position controller
        # For now, return zero (pure force control)
        return np.zeros(6)


class ImpedanceController:
    """
    Impedance control for compliant behavior
    """

    def __init__(self, actuator_interface, stiffness_matrix=None, damping_matrix=None):
        self.actuator = actuator_interface
        self.is_impedance_active = False

        # Default impedance parameters (for 6-DOF)
        if stiffness_matrix is None:
            self.stiffness = np.diag([1000, 1000, 1000, 100, 100, 100])  # [N/m, Nm/rad]
        else:
            self.stiffness = np.array(stiffness_matrix)

        if damping_matrix is None:
            self.damping = np.diag([50, 50, 50, 10, 10, 10])  # Damping coefficients
        else:
            self.damping = np.array(damping_matrix)

        # Desired equilibrium position/pose
        self.equilibrium_position = np.zeros(3)
        self.equilibrium_orientation = np.array([0, 0, 0, 1])  # Quaternion

        # Current state
        self.current_position = np.zeros(3)
        self.current_orientation = np.array([0, 0, 0, 1])
        self.current_velocity = np.zeros(6)  # [linear_vel, angular_vel]

        # Force/torque limits
        self.max_force = 200.0
        self.max_torque = 20.0

    def set_impedance_parameters(self, stiffness_matrix, damping_matrix):
        """
        Set impedance parameters (stiffness and damping matrices)
        """
        self.stiffness = np.array(stiffness_matrix)
        self.damping = np.array(damping_matrix)

    def set_equilibrium_pose(self, position, orientation=None):
        """
        Set desired equilibrium pose
        """
        self.equilibrium_position = np.array(position)
        if orientation is not None:
            self.equilibrium_orientation = np.array(orientation)

    def update_impedance_control(self, current_pose, current_velocity):
        """
        Update impedance control
        """
        if not self.is_impedance_active:
            return np.zeros(6)  # No force output

        # Calculate pose error
        position_error = self.equilibrium_position - current_pose[:3]

        # For orientation, we need to calculate orientation error
        # This is a simplified version - in practice, quaternion error calculation is more complex
        orientation_error = np.zeros(3)  # Simplified for now
        if len(current_pose) >= 7:  # Has orientation quaternion
            current_quat = current_pose[3:7]
            # Calculate quaternion error (simplified)
            quat_error = self._quaternion_error(self.equilibrium_orientation, current_quat)
            orientation_error = self._quaternion_to_euler_error(quat_error)

        # Combine position and orientation errors
        pose_error = np.concatenate([position_error, orientation_error])

        # Calculate impedance forces
        spring_force = self.stiffness @ pose_error
        damping_force = self.damping @ current_velocity

        # Total impedance force
        impedance_force = spring_force + damping_force

        # Limit forces
        impedance_force[:3] = np.clip(impedance_force[:3], -self.max_force, self.max_force)
        impedance_force[3:] = np.clip(impedance_force[3:], -self.max_torque, self.max_torque)

        # Apply to actuator
        self.actuator.set_effort(impedance_force)

        return impedance_force

    def _quaternion_error(self, quat1, quat2):
        """
        Calculate quaternion error between two quaternions
        """
        # q_error = quat2 * inverse(quat1)
        # This is a simplified calculation
        q1_inv = np.array([quat1[0], -quat1[1], -quat1[2], -quat1[3]])  # conjugate
        q_error = self._quaternion_multiply(quat2, q1_inv)
        return q_error

    def _quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def _quaternion_to_euler_error(self, quaternion):
        """
        Convert quaternion to Euler angle error (simplified)
        """
        # Extract rotation vector from quaternion
        # This is a simplified approximation
        if quaternion[0] < 0:  # Choose shortest rotation path
            quaternion = -quaternion

        # For small angles, rotation vector is approximately 2*[x,y,z]/w
        if abs(quaternion[0]) > 0.001:
            rotation_vector = 2 * quaternion[1:4] / quaternion[0]
        else:
            rotation_vector = np.zeros(3)

        return rotation_vector

    def start_impedance_control(self):
        """
        Start impedance control mode
        """
        self.is_impedance_active = True
        self.actuator.set_control_mode('effort')

    def stop_impedance_control(self):
        """
        Stop impedance control mode
        """
        self.is_impedance_active = False
        self.actuator.set_control_mode('position')


class HybridPositionForceController:
    """
    Hybrid position/force controller for constrained tasks
    """

    def __init__(self, actuator_interface):
        self.actuator = actuator_interface
        self.is_active = False

        # Task frame (defines which directions are position-controlled vs force-controlled)
        self.task_frame = np.eye(6)  # Initially identity (all DOF same frame)

        # Control modes for each DOF [pos, pos, pos, force, force, force] for x,y,z,pos,rot,rot
        self.control_modes = np.array([1, 1, 1, 0, 0, 0])  # 1=position, 0=force

        # Desired values
        self.desired_position = np.zeros(3)
        self.desired_force = np.zeros(3)

        # Current values
        self.current_position = np.zeros(3)
        self.current_force = np.zeros(3)

        # Gains
        self.position_gains = {'kp': 100, 'ki': 10, 'kd': 5}
        self.force_gains = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}

    def set_task_frame(self, rotation_matrix, translation_vector=None):
        """
        Set task frame for hybrid control
        """
        if translation_vector is not None:
            # Create 4x4 transformation matrix
            self.task_frame = np.eye(4)
            self.task_frame[:3, :3] = rotation_matrix
            self.task_frame[:3, 3] = translation_vector
        else:
            # Just rotation
            self.task_frame[:3, :3] = rotation_matrix

    def set_control_mode(self, dof_index, mode):
        """
        Set control mode for specific DOF (0=force, 1=position)
        """
        if 0 <= dof_index < 6:
            self.control_modes[dof_index] = mode

    def update_hybrid_control(self, current_pose, current_wrench):
        """
        Update hybrid position/force control
        """
        if not self.is_active:
            return

        # Transform current wrench to task frame
        task_wrench = self._transform_wrench_to_task_frame(current_wrench)

        # Calculate control commands for each DOF
        control_commands = np.zeros(6)

        for i in range(6):
            if self.control_modes[i] == 1:  # Position control
                pos_error = self.desired_position[i] - current_pose[i]
                control_commands[i] = self._calculate_position_control(pos_error, i)
            else:  # Force control
                force_error = self.desired_force[i] - task_wrench[i]
                control_commands[i] = self._calculate_force_control(force_error, i)

        # Apply combined control command
        self.actuator.set_effort(control_commands)

    def _transform_wrench_to_task_frame(self, wrench):
        """
        Transform wrench to task frame
        """
        # Extract rotation matrix from task frame
        if self.task_frame.shape == (4, 4):
            rotation = self.task_frame[:3, :3]
        else:
            rotation = self.task_frame

        # Transform force part
        force_transformed = rotation @ wrench[:3]
        # Transform torque part
        torque_transformed = rotation @ wrench[3:]

        return np.concatenate([force_transformed, torque_transformed])

    def _calculate_position_control(self, error, dof_index):
        """
        Calculate position control command
        """
        # Simple PD control for position
        control_output = (
            self.position_gains['kp'] * error +
            self.position_gains['kd'] * 0  # Velocity term would come from actual velocity
        )
        return np.clip(control_output, -100, 100)  # Limit output

    def _calculate_force_control(self, error, dof_index):
        """
        Calculate force control command
        """
        # Simple PD control for force
        control_output = (
            self.force_gains['kp'] * error +
            self.force_gains['kd'] * 0  # Derivative term would come from actual force derivative
        )
        return np.clip(control_output, -50, 50)  # Limit output

    def start_hybrid_control(self):
        """
        Start hybrid control mode
        """
        self.is_active = True
        self.actuator.set_control_mode('effort')

    def stop_hybrid_control(self):
        """
        Stop hybrid control mode
        """
        self.is_active = False
        self.actuator.set_control_mode('position')
```

## Safety and Limit Management

### Safety Systems for Actuator Control

```python
class ActuatorSafetyManager:
    """
    Safety manager for actuator control systems
    """

    def __init__(self):
        self.actuators = {}
        self.safety_limits = {}
        self.emergency_stop_active = False
        self.safety_violations = []
        self.safety_log = []

    def register_actuator(self, actuator_name, actuator_interface):
        """
        Register an actuator with the safety manager
        """
        self.actuators[actuator_name] = actuator_interface
        self.safety_limits[actuator_name] = self._get_default_safety_limits(actuator_interface)

    def _get_default_safety_limits(self, actuator):
        """
        Get default safety limits for an actuator
        """
        return {
            'position': actuator.position_limits,
            'velocity': actuator.velocity_limits,
            'effort': actuator.effort_limits,
            'temperature': [0, 80],  # Celsius
            'current': [0, actuator.max_current if hasattr(actuator, 'max_current') else 10],  # Amperes
            'collision_force': 50.0,  # Newtons
            'power': [0, 1000]  # Watts
        }

    def set_custom_limits(self, actuator_name, limits):
        """
        Set custom safety limits for specific actuator
        """
        if actuator_name in self.safety_limits:
            self.safety_limits[actuator_name].update(limits)

    def check_safety(self):
        """
        Check all actuators for safety violations
        """
        violations = []

        for actuator_name, actuator in self.actuators.items():
            limits = self.safety_limits[actuator_name]

            # Check position limits
            current_pos = actuator.get_position()
            if (current_pos < limits['position'][0] or current_pos > limits['position'][1]):
                violations.append({
                    'actuator': actuator_name,
                    'type': 'position_limit',
                    'current': current_pos,
                    'limit': limits['position'],
                    'severity': 'critical'
                })

            # Check velocity limits
            current_vel = actuator.get_velocity()
            vel_limit = limits['velocity']
            if (abs(current_vel) > max(abs(vel_limit[0]), abs(vel_limit[1]))):
                violations.append({
                    'actuator': actuator_name,
                    'type': 'velocity_limit',
                    'current': current_vel,
                    'limit': vel_limit,
                    'severity': 'warning'
                })

            # Check effort limits
            current_effort = actuator.get_effort()
            effort_limit = limits['effort']
            if (abs(current_effort) > max(abs(effort_limit[0]), abs(effort_limit[1]))):
                violations.append({
                    'actuator': actuator_name,
                    'type': 'effort_limit',
                    'current': current_effort,
                    'limit': effort_limit,
                    'severity': 'critical'
                })

            # Check temperature (if available)
            if hasattr(actuator, 'get_temperature'):
                temp = actuator.get_temperature()
                temp_limit = limits['temperature']
                if temp < temp_limit[0] or temp > temp_limit[1]:
                    violations.append({
                        'actuator': actuator_name,
                        'type': 'temperature_limit',
                        'current': temp,
                        'limit': temp_limit,
                        'severity': 'critical'
                    })

        # Log violations
        if violations:
            self.safety_violations.extend(violations)
            self._log_safety_violations(violations)

            # Trigger safety responses based on severity
            critical_violations = [v for v in violations if v['severity'] == 'critical']
            if critical_violations:
                self._trigger_critical_safety_response(critical_violations)

        return len(violations) == 0  # Return True if no violations

    def _log_safety_violations(self, violations):
        """
        Log safety violations
        """
        for violation in violations:
            log_entry = {
                'timestamp': time.time(),
                'violation': violation,
                'system_state': self._get_system_state()
            }
            self.safety_log.append(log_entry)

    def _get_system_state(self):
        """
        Get current system state for logging
        """
        return {
            'emergency_stop': self.emergency_stop_active,
            'actuator_states': {
                name: act.get_status() for name, act in self.actuators.items()
            }
        }

    def _trigger_critical_safety_response(self, violations):
        """
        Trigger response for critical safety violations
        """
        print(f"CRITICAL SAFETY VIOLATIONS: {violations}")

        # Emergency stop all actuators
        self.trigger_emergency_stop()

        # Log emergency event
        self._log_emergency_event(violations)

    def trigger_emergency_stop(self):
        """
        Trigger emergency stop for all actuators
        """
        self.emergency_stop_active = True

        for actuator in self.actuators.values():
            try:
                actuator.emergency_stop()
            except AttributeError:
                # If actuator doesn't have emergency_stop method, disable it
                actuator.disable()

        print("EMERGENCY STOP ACTIVATED - All actuators disabled")

    def clear_emergency_stop(self):
        """
        Clear emergency stop condition
        """
        self.emergency_stop_active = False

        # Optionally re-enable actuators after safety check
        # This should only be done after confirming safety
        for actuator in self.actuators.values():
            try:
                actuator.clear_emergency_stop()
            except AttributeError:
                pass  # Some actuators might not need clearing

        print("Emergency stop cleared")

    def get_safety_status(self):
        """
        Get current safety status
        """
        return {
            'emergency_stop_active': self.emergency_stop_active,
            'total_violations': len(self.safety_violations),
            'recent_violations': self.safety_violations[-10:],  # Last 10 violations
            'actuator_safety_status': {
                name: self._check_single_actuator_safety(act)
                for name, act in self.actuators.items()
            }
        }

    def _check_single_actuator_safety(self, actuator):
        """
        Check safety status for single actuator
        """
        status = {
            'position_safe': True,
            'velocity_safe': True,
            'effort_safe': True,
            'temperature_safe': True,
            'overall_safe': True
        }

        # This would check individual actuator parameters
        # Implementation depends on specific actuator interface
        return status

    def validate_trajectory_safety(self, trajectory):
        """
        Validate that a trajectory is safe to execute
        """
        for point in trajectory:
            # Check each trajectory point against limits
            for joint_name, joint_pos in point['configuration'].items():
                if joint_name in self.safety_limits:
                    pos_limits = self.safety_limits[joint_name]['position']
                    if joint_pos < pos_limits[0] or joint_pos > pos_limits[1]:
                        return False, f"Unsafe trajectory: {joint_name} exceeds position limits at point {point['time']}"

        return True, "Trajectory is safe"


class SoftMotionController:
    """
    Soft motion controller for safe and compliant movement
    """

    def __init__(self, actuator_interface):
        self.actuator = actuator_interface
        self.is_active = False

        # Soft motion parameters
        self.max_velocity = 0.5  # rad/s
        self.max_acceleration = 1.0  # rad/s^2
        self.smoothness_factor = 0.1  # Controls motion smoothness

        # Trajectory planning
        self.trajectory_planner = SoftTrajectoryPlanner()

    def move_to_position_smooth(self, target_position, duration=2.0):
        """
        Move to position with smooth, soft motion
        """
        current_position = self.actuator.get_position()

        # Plan smooth trajectory
        trajectory = self.trajectory_planner.plan_smooth_trajectory(
            current_position, target_position, duration
        )

        # Execute trajectory with safety checks
        start_time = time.time()
        for point in trajectory:
            if self.emergency_stop_active:
                break

            # Check safety before each command
            if not self._is_motion_safe(point['position']):
                print(f"Motion interrupted - safety limit exceeded at {point['position']}")
                break

            self.actuator.set_position(point['position'])

            # Sleep to follow trajectory timing
            sleep_time = point['time'] - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _is_motion_safe(self, position):
        """
        Check if motion to position is safe
        """
        # Check against position limits
        limits = self.safety_limits.get(self.actuator.name, {}).get('position', [-np.inf, np.inf])
        return limits[0] <= position <= limits[1]

    def set_softness_parameters(self, max_vel, max_acc, smoothness):
        """
        Set soft motion parameters
        """
        self.max_velocity = max_vel
        self.max_acceleration = max_acc
        self.smoothness_factor = smoothness


class SoftTrajectoryPlanner:
    """
    Trajectory planner for smooth, soft motions
    """

    def __init__(self):
        self.velocity_limit = 0.5
        self.acceleration_limit = 1.0

    def plan_smooth_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """
        Plan smooth trajectory with velocity and acceleration limits
        """
        trajectory = []
        num_points = int(duration / dt)

        # Calculate coefficients for quintic polynomial (smooth start/end)
        a0 = start_pos
        a1 = 0  # Start velocity = 0
        a2 = 0  # Start acceleration = 0
        a3 = (20 * (end_pos - start_pos)) / (2 * duration**3)
        a4 = (-30 * (end_pos - start_pos)) / (2 * duration**4)
        a5 = (12 * (end_pos - start_pos)) / (2 * duration**5)

        for i in range(num_points + 1):
            t = i * dt
            if t > duration:
                t = duration

            s = t / duration if duration > 0 else 0  # Normalized time

            pos = (a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5)
            vel = (a1 + 2*a2*s + 3*a3*s**2 + 4*a4*s**3 + 5*a5*s**4) / duration
            acc = (2*a2 + 6*a3*s + 12*a4*s**2 + 20*a5*s**3) / (duration**2)

            # Limit velocity and acceleration
            vel = np.clip(vel, -self.velocity_limit, self.velocity_limit)
            acc = np.clip(acc, -self.acceleration_limit, self.acceleration_limit)

            trajectory.append({
                'time': t,
                'position': pos,
                'velocity': vel,
                'acceleration': acc
            })

        return trajectory

    def plan_minimum_jerk_trajectory(self, start_pos, end_pos, duration):
        """
        Plan minimum jerk trajectory
        """
        # This is essentially the same as the quintic polynomial above
        # which naturally minimizes jerk
        return self.plan_smooth_trajectory(start_pos, end_pos, duration)
```

## Real-time Control Considerations

### Real-time Performance Optimization

```python
import threading
import time
from collections import deque
import numpy as np

class RealTimeActuatorController:
    """
    Real-time actuator controller with deterministic timing
    """

    def __init__(self, control_frequency=1000):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.is_running = False
        self.control_thread = None

        # Actuator interfaces
        self.actuators = {}
        self.control_modes = {}  # position, velocity, effort

        # Control data
        self.desired_states = {}
        self.current_states = {}
        self.control_outputs = {}

        # Timing and performance
        self.timing_stats = {
            'period': [],
            'jitter': [],
            'missed_periods': 0
        }

        # Control loop timing
        self.loop_start_time = 0
        self.expected_period = 1.0 / control_frequency
        self.max_jitter = 0.001  # 1ms maximum jitter allowed

    def add_actuator(self, name, actuator_interface, control_mode='position'):
        """
        Add actuator to real-time controller
        """
        self.actuators[name] = actuator_interface
        self.control_modes[name] = control_mode
        self.desired_states[name] = {'position': 0.0, 'velocity': 0.0, 'effort': 0.0}
        self.current_states[name] = {'position': 0.0, 'velocity': 0.0, 'effort': 0.0}
        self.control_outputs[name] = 0.0

    def start_control_loop(self):
        """
        Start real-time control loop in separate thread
        """
        if self.control_thread is not None and self.control_thread.is_alive():
            return

        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """
        Stop real-time control loop
        """
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)  # Wait up to 2 seconds

    def _control_loop(self):
        """
        Real-time control loop with deterministic timing
        """
        while self.is_running:
            loop_start = time.time()

            try:
                # Update current states from actuators
                self._update_current_states()

                # Calculate control outputs
                self._calculate_control_outputs()

                # Apply control commands to actuators
                self._apply_control_commands()

                # Monitor timing performance
                self._monitor_timing(loop_start)

                # Maintain timing
                loop_duration = time.time() - loop_start
                sleep_time = self.expected_period - loop_duration

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Control loop took too long - missed deadline
                    self.timing_stats['missed_periods'] += 1

            except Exception as e:
                print(f"Control loop error: {e}")
                time.sleep(0.001)  # Brief sleep to prevent busy waiting

    def _update_current_states(self):
        """
        Update current states from all actuators
        """
        for name, actuator in self.actuators.items():
            try:
                self.current_states[name]['position'] = actuator.get_position()
                self.current_states[name]['velocity'] = actuator.get_velocity()
                self.current_states[name]['effort'] = actuator.get_effort()
            except Exception as e:
                print(f"Error getting state for {name}: {e}")

    def _calculate_control_outputs(self):
        """
        Calculate control outputs for all actuators
        """
        for name, actuator in self.actuators.items():
            mode = self.control_modes[name]

            if mode == 'position':
                error = self.desired_states[name]['position'] - self.current_states[name]['position']
                # Simple proportional control for example
                output = 100 * error  # This would be a full PID controller in practice
            elif mode == 'velocity':
                error = self.desired_states[name]['velocity'] - self.current_states[name]['velocity']
                output = 50 * error
            elif mode == 'effort':
                output = self.desired_states[name]['effort']
            else:
                output = 0.0

            # Apply effort limits
            max_effort = actuator.max_effort if hasattr(actuator, 'max_effort') else 100.0
            self.control_outputs[name] = np.clip(output, -max_effort, max_effort)

    def _apply_control_commands(self):
        """
        Apply calculated control outputs to actuators
        """
        for name, output in self.control_outputs.items():
            if name in self.actuators:
                try:
                    mode = self.control_modes[name]
                    if mode == 'position':
                        self.actuators[name].set_position(output)
                    elif mode == 'velocity':
                        self.actuators[name].set_velocity(output)
                    elif mode == 'effort':
                        self.actuators[name].set_effort(output)
                except Exception as e:
                    print(f"Error setting command for {name}: {e}")

    def _monitor_timing(self, loop_start_time):
        """
        Monitor and record timing statistics
        """
        actual_period = time.time() - loop_start_time
        expected = self.expected_period
        jitter = abs(actual_period - expected)

        self.timing_stats['period'].append(actual_period)
        self.timing_stats['jitter'].append(jitter)

        # Keep statistics bounded
        if len(self.timing_stats['period']) > 1000:
            self.timing_stats['period'].pop(0)
            self.timing_stats['jitter'].pop(0)

    def set_desired_state(self, actuator_name, position=None, velocity=None, effort=None):
        """
        Set desired state for specific actuator
        """
        if actuator_name in self.desired_states:
            if position is not None:
                self.desired_states[actuator_name]['position'] = position
            if velocity is not None:
                self.desired_states[actuator_name]['velocity'] = velocity
            if effort is not None:
                self.desired_states[actuator_name]['effort'] = effort

    def get_timing_performance(self):
        """
        Get timing performance statistics
        """
        if not self.timing_stats['period']:
            return {'average_period': 0, 'max_jitter': 0, 'missed_periods': 0}

        avg_period = np.mean(self.timing_stats['period'])
        max_jitter = np.max(self.timing_stats['jitter']) if self.timing_stats['jitter'] else 0
        missed = self.timing_stats['missed_periods']

        return {
            'average_period': avg_period,
            'expected_period': self.expected_period,
            'max_jitter': max_jitter,
            'missed_periods': missed,
            'timing_jitter_percent': (max_jitter / self.expected_period) * 100 if self.expected_period > 0 else 0
        }

    def get_control_performance(self):
        """
        Get control performance metrics
        """
        errors = {}
        for name in self.actuators.keys():
            error = abs(self.desired_states[name]['position'] - self.current_states[name]['position'])
            errors[name] = error

        return {
            'tracking_errors': errors,
            'average_error': np.mean(list(errors.values())) if errors else 0,
            'max_error': max(errors.values()) if errors else 0
        }


class ActuatorControlOptimizer:
    """
    Optimizer for actuator control parameters
    """

    def __init__(self, controller):
        self.controller = controller
        self.parameter_ranges = {
            'kp': (10, 1000),
            'ki': (0.1, 50),
            'kd': (0.1, 50),
            'max_velocity': (0.1, 10.0),
            'max_acceleration': (0.5, 20.0)
        }

    def optimize_control_parameters(self, target_trajectory, performance_metric='minimize_error'):
        """
        Optimize control parameters for best performance
        """
        from scipy.optimize import minimize

        def objective_function(params):
            # Set controller parameters
            self._set_controller_params(params)

            # Execute trajectory and measure performance
            performance = self._evaluate_trajectory_performance(target_trajectory)

            # Return value to minimize
            if performance_metric == 'minimize_error':
                return performance['average_error']
            elif performance_metric == 'minimize_overshoot':
                return performance['max_overshoot']
            elif performance_metric == 'minimize_settling_time':
                return performance['settling_time']
            else:
                return performance['average_error']  # Default

        # Initial parameters
        initial_params = self._get_current_params()

        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            method='powell',  # Good for this type of problem
            bounds=self._get_parameter_bounds()
        )

        if result.success:
            self._set_controller_params(result.x)
            return result.x, result.fun
        else:
            return initial_params, float('inf')

    def _set_controller_params(self, params):
        """
        Set controller parameters from optimization vector
        """
        # This would set the actual controller parameters
        # Implementation depends on specific controller architecture
        pass

    def _get_current_params(self):
        """
        Get current controller parameters as optimization vector
        """
        # This would return current parameters
        # Implementation depends on specific controller architecture
        return [100, 10, 5, 1.0, 2.0]  # Example parameters

    def _get_parameter_bounds(self):
        """
        Get parameter bounds for optimization
        """
        return [
            self.parameter_ranges['kp'],
            self.parameter_ranges['ki'],
            self.parameter_ranges['kd'],
            self.parameter_ranges['max_velocity'],
            self.parameter_ranges['max_acceleration']
        ]

    def _evaluate_trajectory_performance(self, trajectory):
        """
        Evaluate trajectory performance
        """
        # Execute trajectory and measure various performance metrics
        errors = []
        overshoots = []
        settling_times = []

        # This would involve executing the trajectory and collecting data
        # For now, return example values
        return {
            'average_error': 0.01,
            'max_overshoot': 0.05,
            'settling_time': 0.5
        }
```

## Week Summary

This section covered comprehensive actuator control techniques for Physical AI and humanoid robotics systems. We explored servo motor control with PID and advanced control algorithms, multi-joint coordination systems, force and impedance control for compliant manipulation, safety management systems, and real-time performance optimization. These control strategies are essential for achieving precise, safe, and responsive actuator behavior in robotic systems, enabling complex manipulation and locomotion tasks while maintaining system safety and stability.