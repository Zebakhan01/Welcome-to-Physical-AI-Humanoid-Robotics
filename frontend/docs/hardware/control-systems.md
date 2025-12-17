---
sidebar_position: 4
---

# Control Systems

## Introduction to Robotic Control Systems

Robotic control systems form the nervous system of humanoid robots, coordinating sensor inputs, processing information, and generating appropriate actuator commands to achieve desired behaviors. These systems must handle real-time constraints, safety requirements, and complex multi-degree-of-freedom coordination while maintaining stability and performance. This section covers the architecture, design principles, and implementation strategies for humanoid robot control systems.

## Control System Architecture

### Hierarchical Control Structure

Humanoid robot control systems typically employ a hierarchical structure that separates different control functions by time scale and complexity:

```
┌─────────────────────────────────────────┐
│           Task Level                    │ ← High-level planning
├─────────────────────────────────────────┤
│           Motion Level                  │ ← Trajectory generation
├─────────────────────────────────────────┤
│           Balance Level                 │ ← Posture and balance
├─────────────────────────────────────────┤
│           Joint Level                   │ ← Individual joint control
└─────────────────────────────────────────┘
```

#### High-Level Control (Task Planning)

**Function:**
- Long-term goal planning
- Task decomposition
- Path planning and navigation
- Human-robot interaction management

**Characteristics:**
- Update rates: 1-10 Hz
- Decision-making focus
- Environmental modeling
- Multi-modal integration

#### Mid-Level Control (Motion Generation)

**Function:**
- Trajectory planning and execution
- Whole-body motion coordination
- Inverse kinematics solutions
- Dynamic balance maintenance

**Characteristics:**
- Update rates: 10-100 Hz
- Motion planning focus
- Kinematic constraints
- Dynamic modeling

#### Low-Level Control (Joint Servos)

**Function:**
- Individual joint position/velocity/torque control
- Feedback control implementation
- Safety monitoring
- Hardware interface management

**Characteristics:**
- Update rates: 100-1000 Hz
- Real-time performance
- Hardware-level control
- Safety-critical operations

### Real-Time Operating Systems (RTOS)

#### Requirements for Robotic Control

**Deterministic Timing:**
- Guaranteed response times
- Predictable interrupt handling
- Priority-based task scheduling
- Minimal jitter requirements

**Resource Management:**
- Memory allocation strategies
- CPU time allocation
- I/O management
- Power consumption optimization

**Safety Features:**
- Watchdog timers
- Memory protection
- Task isolation
- Fault detection and recovery

#### Popular RTOS Options

**PREEMPT_RT Linux:**
- Real-time Linux kernel patches
- POSIX compliance
- Extensive hardware support
- Large development community

**VxWorks:**
- Commercial real-time OS
- Deterministic performance
- Safety certification options
- Industrial automation focus

**QNX:**
- Microkernel architecture
- High reliability
- Automotive and industrial use
- Commercial licensing

### Control Hardware Platforms

#### Single Board Computers (SBCs)

**NVIDIA Jetson Series:**
- GPU-accelerated processing
- AI and computer vision capabilities
- Real-time performance
- Power-efficient design

**Raspberry Pi:**
- Cost-effective for educational use
- Extensive community support
- Limited computational power
- Good for prototyping

**Intel NUC:**
- High-performance computing
- Multiple interface options
- Industrial-grade reliability
- Higher power consumption

#### Microcontroller Units (MCUs)

**ARM Cortex-M Series:**
- Real-time performance
- Low power consumption
- Extensive peripheral support
- Cost-effective for simple control

**Arduino Family:**
- Educational and prototyping use
- Simple programming model
- Limited computational power
- Extensive library support

## Joint-Level Control

### PID Control Implementation

#### Basic PID Controller

```python
import time
import numpy as np

class JointPIDController:
    """
    PID controller for robotic joint control
    """

    def __init__(self, kp=10.0, ki=1.0, kd=0.1,
                 max_output=100.0, max_integral=50.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.max_output = max_output
        self.max_integral = max_integral

        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
        self.output = 0.0

    def update(self, setpoint, actual_value):
        """
        Update PID controller and return control output
        """
        current_time = time.time()
        dt = current_time - self.previous_time

        if dt <= 0:
            dt = 1e-6  # Prevent division by zero

        # Calculate error
        error = setpoint - actual_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        # Anti-windup protection
        self.integral = max(-self.max_integral,
                           min(self.max_integral, self.integral))
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Total output
        self.output = p_term + i_term + d_term

        # Output limiting
        self.output = max(-self.max_output,
                         min(self.max_output, self.output))

        # Update previous values
        self.previous_error = error
        self.previous_time = current_time

        return self.output

    def reset(self):
        """Reset controller internal state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
        self.output = 0.0
```

#### Advanced PID Features

```python
class AdvancedJointController(JointPIDController):
    """
    Advanced joint controller with additional features
    """

    def __init__(self, kp=10.0, ki=1.0, kd=0.1,
                 max_output=100.0, max_integral=50.0,
                 feedforward_gain=1.0):
        super().__init__(kp, ki, kd, max_output, max_integral)
        self.feedforward_gain = feedforward_gain
        self.velocity_feedforward = 0.0
        self.acceleration_feedforward = 0.0

    def update_advanced(self, setpoint, actual_value,
                       setpoint_velocity=0.0, setpoint_acceleration=0.0):
        """
        Update with feedforward terms
        """
        # Basic PID control
        pid_output = super().update(setpoint, actual_value)

        # Feedforward terms
        velocity_ff = self.feedforward_gain * setpoint_velocity
        acceleration_ff = self.feedforward_gain * 0.1 * setpoint_acceleration

        # Total output
        total_output = pid_output + velocity_ff + acceleration_ff

        # Apply limits
        total_output = max(-self.max_output,
                          min(self.max_output, total_output))

        self.output = total_output
        return total_output

    def set_gains(self, kp, ki, kd):
        """Dynamically adjust PID gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
```

### Trajectory Generation

#### Minimum Jerk Trajectory

```python
class MinimumJerkTrajectory:
    """
    Generate minimum jerk trajectories for smooth motion
    """

    def __init__(self, start_pos, end_pos, duration, dt=0.01):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.dt = dt
        self.coeffs = self._calculate_coefficients()

    def _calculate_coefficients(self):
        """
        Calculate coefficients for minimum jerk polynomial
        """
        a0 = self.start_pos
        a1 = 0  # Start velocity = 0
        a2 = 0  # Start acceleration = 0
        a3 = (20 * (self.end_pos - self.start_pos)) / (2 * self.duration**3)
        a4 = (-30 * (self.end_pos - self.start_pos)) / (2 * self.duration**4)
        a5 = (12 * (self.end_pos - self.start_pos)) / (2 * self.duration**5)

        return [a0, a1, a2, a3, a4, a5]

    def get_position(self, t):
        """Get position at time t"""
        if t >= self.duration:
            return self.end_pos
        if t <= 0:
            return self.start_pos

        t_rel = t
        coeffs = self.coeffs
        return (coeffs[0] +
                coeffs[1] * t_rel +
                coeffs[2] * t_rel**2 +
                coeffs[3] * t_rel**3 +
                coeffs[4] * t_rel**4 +
                coeffs[5] * t_rel**5)

    def get_velocity(self, t):
        """Get velocity at time t"""
        if t >= self.duration or t <= 0:
            return 0.0

        t_rel = t
        coeffs = self.coeffs
        return (coeffs[1] +
                2 * coeffs[2] * t_rel +
                3 * coeffs[3] * t_rel**2 +
                4 * coeffs[4] * t_rel**3 +
                5 * coeffs[5] * t_rel**4)

    def get_acceleration(self, t):
        """Get acceleration at time t"""
        if t >= self.duration or t <= 0:
            return 0.0

        t_rel = t
        coeffs = self.coeffs
        return (2 * coeffs[2] +
                6 * coeffs[3] * t_rel +
                12 * coeffs[4] * t_rel**2 +
                20 * coeffs[5] * t_rel**3)

    def get_trajectory(self):
        """Generate complete trajectory"""
        times = np.arange(0, self.duration, self.dt)
        positions = [self.get_position(t) for t in times]
        velocities = [self.get_velocity(t) for t in times]
        accelerations = [self.get_acceleration(t) for t in times]

        return {
            'time': times,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations
        }
```

### Safety Systems

#### Joint Limit Protection

```python
class JointLimitProtection:
    """
    Joint limit protection system
    """

    def __init__(self, joint_limits, soft_limit_margin=0.1):
        self.joint_limits = joint_limits  # [min, max] for each joint
        self.soft_limit_margin = soft_limit_margin
        self.emergency_stop_active = False

    def check_limits(self, joint_positions, joint_velocities):
        """
        Check if joints are within safe limits
        Returns: (safe_to_move, recommended_velocities)
        """
        safe = True
        recommended_velocities = joint_velocities.copy()

        for i, (pos, vel) in enumerate(zip(joint_positions, joint_velocities)):
            limits = self.joint_limits[i]
            min_pos, max_pos = limits

            # Check hard limits
            if pos <= min_pos or pos >= max_pos:
                safe = False
                recommended_velocities[i] = 0  # Stop immediately
                continue

            # Check soft limits
            soft_min = min_pos + self.soft_limit_margin
            soft_max = max_pos - self.soft_limit_margin

            if pos <= soft_min and vel < 0:
                # Moving toward limit, slow down
                recommended_velocities[i] = max(0, vel)
            elif pos >= soft_max and vel > 0:
                # Moving toward limit, slow down
                recommended_velocities[i] = min(0, vel)

        return safe, recommended_velocities

    def emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        return [0.0] * len(self.joint_limits)  # Zero all joint velocities
```

## Whole-Body Control

### Operational Space Control

#### Cartesian Space Control

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class OperationalSpaceController:
    """
    Operational space controller for Cartesian motion control
    """

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.jacobian_cache = {}
        self.cartesian_gains = {
            'position': np.diag([1000, 1000, 1000]),  # High stiffness
            'orientation': np.diag([100, 100, 100])   # Lower for orientation
        }

    def compute_cartesian_control(self, target_pose, current_pose,
                                 target_twist=None, current_twist=None):
        """
        Compute Cartesian space control commands
        """
        # Position error
        pos_error = target_pose[:3] - current_pose[:3]

        # Orientation error (using rotation vector)
        current_rot = R.from_matrix(current_pose[3:].reshape(3,3))
        target_rot = R.from_matrix(target_pose[3:].reshape(3,3))
        relative_rot = target_rot * current_rot.inv()
        orientation_error = relative_rot.as_rotvec()

        # Combine position and orientation errors
        pose_error = np.concatenate([pos_error, orientation_error])

        # Desired Cartesian velocity
        if target_twist is not None:
            desired_twist = target_twist
        else:
            # Proportional control
            desired_twist = np.zeros(6)
            desired_twist[:3] = self.cartesian_gains['position'] @ pos_error[:3]
            desired_twist[3:] = self.cartesian_gains['orientation'] @ orientation_error

        # If we have current twist, add damping
        if current_twist is not None:
            damping = 0.1 * current_twist  # Simple damping
            desired_twist -= damping

        return desired_twist

    def cartesian_to_joint_velocity(self, cartesian_velocity, joint_angles):
        """
        Convert Cartesian velocity to joint velocity using Jacobian
        """
        jacobian = self.robot_model.get_jacobian(joint_angles)

        # Pseudo-inverse for redundant systems
        joint_velocity = np.linalg.pinv(jacobian) @ cartesian_velocity

        return joint_velocity

    def compute_operational_force(self, desired_force, joint_angles):
        """
        Compute joint torques from desired Cartesian force
        """
        jacobian = self.robot_model.get_jacobian(joint_angles)

        # Transpose for force transformation
        joint_torques = jacobian.T @ desired_force

        return joint_torques
```

### Balance Control

#### Center of Mass Control

```python
class BalanceController:
    """
    Balance controller for humanoid robots
    """

    def __init__(self, robot_mass, gravity=9.81):
        self.robot_mass = robot_mass
        self.gravity = gravity
        self.com_pid = JointPIDController(kp=100, ki=10, kd=5)
        self.zmp_pid = JointPIDController(kp=50, ki=5, kd=2)

    def compute_balance_control(self, desired_com, actual_com,
                               desired_zmp=None, actual_zmp=None):
        """
        Compute balance control commands
        """
        # Center of Mass control
        com_correction = self.com_pid.update(
            desired_com, actual_com
        )

        # Zero Moment Point control if available
        if desired_zmp is not None and actual_zmp is not None:
            zmp_correction = self.zmp_pid.update(
                desired_zmp, actual_zmp
            )
        else:
            zmp_correction = 0.0

        return com_correction + zmp_correction

    def compute_stabilizing_torques(self, balance_correction, joint_angles):
        """
        Compute joint torques for balance correction
        """
        # This would involve inverse dynamics
        # For simplicity, return proportional to correction
        num_joints = len(joint_angles)
        torques = np.zeros(num_joints)

        # Distribute correction across joints
        # In reality, this would use whole-body inverse dynamics
        for i in range(num_joints):
            torques[i] = balance_correction * (i + 1) / num_joints

        return torques
```

## Communication and Synchronization

### Real-Time Communication

#### CAN Bus Implementation

```python
import can

class CANControlInterface:
    """
    CAN bus interface for real-time control
    """

    def __init__(self, channel='can0', bitrate=1000000):
        self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
        self.node_ids = []  # List of connected node IDs
        self.message_buffer = []

    def send_joint_command(self, node_id, position, velocity=0.0, effort=0.0):
        """
        Send joint command via CAN bus
        """
        # Pack command data
        data = self._pack_joint_command(position, velocity, effort)

        # Create CAN message
        msg = can.Message(
            arbitration_id=node_id,
            data=data,
            is_extended_id=True
        )

        # Send message
        try:
            self.bus.send(msg)
            return True
        except can.CanError:
            return False

    def _pack_joint_command(self, position, velocity, effort):
        """
        Pack joint command into CAN message format
        """
        # Convert to integer representation
        pos_int = int(position * 1000)  # Scale for precision
        vel_int = int(velocity * 1000)
        eff_int = int(effort * 1000)

        # Pack into bytes (little endian)
        data = (
            pos_int.to_bytes(4, 'little', signed=True) +
            vel_int.to_bytes(2, 'little', signed=True) +
            eff_int.to_bytes(2, 'little', signed=True)
        )

        return data[:8]  # CAN message is max 8 bytes

    def receive_sensor_data(self):
        """
        Receive sensor data from CAN network
        """
        try:
            msg = self.bus.recv(timeout=0.001)  # 1ms timeout
            if msg:
                return self._unpack_sensor_data(msg)
        except can.CanError:
            pass

        return None

    def _unpack_sensor_data(self, msg):
        """
        Unpack sensor data from CAN message
        """
        node_id = msg.arbitration_id
        data = msg.data

        # Unpack position, velocity, effort
        position = int.from_bytes(data[:4], 'little', signed=True) / 1000.0
        velocity = int.from_bytes(data[4:6], 'little', signed=True) / 1000.0
        effort = int.from_bytes(data[6:8], 'little', signed=True) / 1000.0

        return {
            'node_id': node_id,
            'position': position,
            'velocity': velocity,
            'effort': effort,
            'timestamp': time.time()
        }
```

### Synchronization Strategies

#### Multi-Rate Control Synchronization

```python
import threading
import time
from collections import deque

class MultiRateSynchronizer:
    """
    Synchronize multi-rate control loops
    """

    def __init__(self):
        self.highest_rate = 1000  # Hz
        self.time_step = 1.0 / self.highest_rate
        self.global_time = 0.0
        self.sync_lock = threading.Lock()
        self.data_buffers = {}
        self.callbacks = []

    def register_callback(self, callback, rate):
        """
        Register a callback to run at specific rate
        """
        interval = int(self.highest_rate / rate)
        callback_info = {
            'callback': callback,
            'interval': interval,
            'counter': 0
        }
        self.callbacks.append(callback_info)

    def run_synchronized_loops(self):
        """
        Run synchronized control loops
        """
        while True:
            start_time = time.time()

            # Update global time
            with self.sync_lock:
                self.global_time += self.time_step

            # Execute callbacks at their rates
            for callback_info in self.callbacks:
                callback_info['counter'] += 1
                if callback_info['counter'] >= callback_info['interval']:
                    callback_info['callback'](self.global_time)
                    callback_info['counter'] = 0

            # Maintain timing
            elapsed = time.time() - start_time
            sleep_time = self.time_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
```

## Safety and Emergency Systems

### Safety Architecture

#### Multi-Level Safety System

```python
class SafetySystem:
    """
    Multi-level safety system for humanoid robots
    """

    def __init__(self):
        self.emergency_stop = False
        self.safety_levels = {
            'level_1': {'name': 'Operational', 'active': True},
            'level_2': {'name': 'Reduced Speed', 'active': False},
            'level_3': {'name': 'Safe Stop', 'active': False},
            'level_4': {'name': 'Emergency Stop', 'active': False}
        }
        self.safety_monitors = []
        self.last_safe_time = time.time()

    def add_monitor(self, monitor_func, level=1):
        """
        Add a safety monitor function
        """
        self.safety_monitors.append({
            'function': monitor_func,
            'level': level,
            'last_check': time.time()
        })

    def check_safety(self):
        """
        Check all safety conditions
        """
        current_time = time.time()

        for monitor in self.safety_monitors:
            try:
                safe, reason = monitor['function']()

                if not safe:
                    self.trigger_safety_level(monitor['level'], reason)
                    return False

            except Exception as e:
                # Safety system failure - emergency stop
                self.trigger_emergency_stop(f"Monitor error: {e}")
                return False

        # Update last safe time
        self.last_safe_time = current_time
        return True

    def trigger_safety_level(self, level, reason="Safety violation"):
        """
        Trigger appropriate safety level
        """
        if level >= 4:
            self.trigger_emergency_stop(reason)
        elif level >= 3:
            self.safe_stop(reason)
        elif level >= 2:
            self.reduce_speed(reason)

    def trigger_emergency_stop(self, reason="Emergency"):
        """
        Trigger emergency stop
        """
        self.emergency_stop = True
        self.safety_levels['level_4']['active'] = True

        # Stop all motion
        self._stop_all_actuators()

        print(f"EMERGENCY STOP: {reason}")

    def safe_stop(self, reason="Safe stop required"):
        """
        Perform safe stop procedure
        """
        self.safety_levels['level_3']['active'] = True

        # Execute safe stop sequence
        self._execute_safe_stop_sequence()

        print(f"SAFE STOP: {reason}")

    def reduce_speed(self, reason="Speed reduction required"):
        """
        Reduce operational speed
        """
        self.safety_levels['level_2']['active'] = True

        # Reduce speed commands
        self._reduce_speed_commands()

        print(f"SPEED REDUCTION: {reason}")

    def _stop_all_actuators(self):
        """
        Send stop commands to all actuators
        """
        # Implementation depends on actuator interface
        pass

    def _execute_safe_stop_sequence(self):
        """
        Execute safe stop sequence
        """
        # Move to safe configuration
        # Reduce power gradually
        # Monitor for safe state
        pass

    def _reduce_speed_commands(self):
        """
        Reduce all speed commands
        """
        # Implementation depends on control system
        pass
```

## Control System Design Patterns

### State Machine Controller

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    INITIALIZING = 2
    READY = 3
    EXECUTING = 4
    PAUSED = 5
    ERROR = 6
    SHUTDOWN = 7

class StateMachineController:
    """
    State machine based controller
    """

    def __init__(self):
        self.state = RobotState.IDLE
        self.previous_state = None
        self.state_entry_time = time.time()

    def update(self, sensor_data, user_commands):
        """
        Update state machine
        """
        current_state = self.state
        new_state = self._determine_next_state(sensor_data, user_commands)

        if new_state != current_state:
            self._exit_state(current_state)
            self._transition(current_state, new_state)
            self._enter_state(new_state)
            self.state = new_state
            self.state_entry_time = time.time()

        return self._execute_current_state(sensor_data, user_commands)

    def _determine_next_state(self, sensor_data, user_commands):
        """
        Determine next state based on current state and inputs
        """
        current = self.state

        if user_commands.get('emergency_stop', False):
            return RobotState.ERROR

        if current == RobotState.IDLE:
            if user_commands.get('initialize', False):
                return RobotState.INITIALIZING
        elif current == RobotState.INITIALIZING:
            if self._initialization_complete(sensor_data):
                return RobotState.READY
        elif current == RobotState.READY:
            if user_commands.get('execute', False):
                return RobotState.EXECUTING
        elif current == RobotState.EXECUTING:
            if user_commands.get('pause', False):
                return RobotState.PAUSED
            elif self._task_complete(sensor_data):
                return RobotState.READY
        elif current == RobotState.PAUSED:
            if user_commands.get('resume', False):
                return RobotState.EXECUTING

        return current

    def _enter_state(self, state):
        """
        Actions to perform when entering a state
        """
        if state == RobotState.INITIALIZING:
            self._initialize_system()
        elif state == RobotState.READY:
            self._ready_system()

    def _exit_state(self, state):
        """
        Actions to perform when exiting a state
        """
        pass

    def _transition(self, from_state, to_state):
        """
        Actions to perform during state transition
        """
        print(f"Transitioning from {from_state.name} to {to_state.name}")

    def _execute_current_state(self, sensor_data, user_commands):
        """
        Execute actions for current state
        """
        if self.state == RobotState.IDLE:
            return self._execute_idle(sensor_data, user_commands)
        elif self.state == RobotState.INITIALIZING:
            return self._execute_initializing(sensor_data, user_commands)
        elif self.state == RobotState.READY:
            return self._execute_ready(sensor_data, user_commands)
        elif self.state == RobotState.EXECUTING:
            return self._execute_executing(sensor_data, user_commands)
        elif self.state == RobotState.PAUSED:
            return self._execute_paused(sensor_data, user_commands)
        elif self.state == RobotState.ERROR:
            return self._execute_error(sensor_data, user_commands)

    def _initialize_system(self):
        """
        Initialize the robot system
        """
        pass

    def _ready_system(self):
        """
        Prepare system for operation
        """
        pass

    def _execute_idle(self, sensor_data, user_commands):
        """
        Execute idle state
        """
        return {'status': 'idle', 'actions': []}

    def _execute_initializing(self, sensor_data, user_commands):
        """
        Execute initialization state
        """
        return {'status': 'initializing', 'actions': []}

    def _execute_ready(self, sensor_data, user_commands):
        """
        Execute ready state
        """
        return {'status': 'ready', 'actions': []}

    def _execute_executing(self, sensor_data, user_commands):
        """
        Execute main operation state
        """
        return {'status': 'executing', 'actions': []}

    def _execute_paused(self, sensor_data, user_commands):
        """
        Execute paused state
        """
        return {'status': 'paused', 'actions': []}

    def _execute_error(self, sensor_data, user_commands):
        """
        Execute error state
        """
        return {'status': 'error', 'actions': []}
```

## Week Summary

This section covered the comprehensive control systems required for humanoid robots, including hierarchical control architecture, real-time operating systems, joint-level control with PID controllers, whole-body control approaches, communication systems, and safety mechanisms. The control system is the brain of the humanoid robot, coordinating all components to achieve safe, stable, and effective operation.