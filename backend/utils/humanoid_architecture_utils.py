import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import uuid
from datetime import datetime


class HumanoidPlatform(Enum):
    """Types of humanoid platforms"""
    ASIMO = "asimo"
    ATLAS = "atlas"
    NAO = "nao"
    PEPPER = "pepper"
    HSR = "hsr"
    BAXTER = "baxter"
    SAWYER = "sawyer"
    CUSTOM = "custom"


class ActuatorType(Enum):
    """Types of actuators for humanoid robots"""
    SERVO = "servo"
    SERIES_ELASTIC = "series_elastic"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    SMA = "shape_memory_alloy"


class ControlLevel(Enum):
    """Levels of control hierarchy"""
    HIGH_LEVEL = "high_level"  # Task planning, path planning
    MID_LEVEL = "mid_level"    # Whole-body motion, balance
    LOW_LEVEL = "low_level"    # Joint servo, sensor feedback


@dataclass
class JointConfiguration:
    """Configuration for a single joint"""
    name: str
    position: float  # radians
    velocity: float  # rad/s
    effort: float    # Nm
    limits: Tuple[float, float]  # min, max position
    stiffness: float = 1.0
    damping: float = 0.1


@dataclass
class LinkProperties:
    """Properties of a robot link"""
    name: str
    mass: float  # kg
    com: Tuple[float, float, float]  # center of mass (x, y, z)
    inertia: Tuple[float, float, float, float, float, float]  # Ixx, Iyy, Izz, Ixy, Ixz, Iyz


@dataclass
class HumanoidBodyPlan:
    """Body plan for a humanoid robot"""
    platform: HumanoidPlatform
    total_dof: int
    joint_configurations: List[JointConfiguration]
    link_properties: List[LinkProperties]
    actuator_types: Dict[str, ActuatorType]  # joint_name -> actuator_type
    sensor_configurations: Dict[str, List[str]]  # body_part -> sensor_types


@dataclass
class BalanceState:
    """Current balance state of the humanoid"""
    com_position: Tuple[float, float, float]  # Center of Mass (x, y, z)
    com_velocity: Tuple[float, float, float]  # CoM velocity
    support_polygon: List[Tuple[float, float]]  # Convex hull of support points
    zmp_position: Tuple[float, float]  # Zero Moment Point
    stability_margin: float  # Distance from CoM projection to support polygon edge
    balance_mode: str  # "static", "dynamic", "recovery"
    double_support: bool  # Whether in double support phase


@dataclass
class HumanoidState:
    """Complete state of the humanoid robot"""
    body_plan: HumanoidBodyPlan
    joint_states: Dict[str, JointConfiguration]
    balance_state: BalanceState
    sensor_data: Dict[str, Any]
    timestamp: float
    is_safe: bool
    control_mode: str  # "idle", "walking", "standing", "manipulation"


class BalanceController:
    """Controller for humanoid balance and stability"""

    def __init__(self):
        self.com_filter = np.zeros(3)  # Low-pass filter for CoM
        self.zmp_filter = np.zeros(2)  # Low-pass filter for ZMP
        self.stability_threshold = 0.05  # Minimum stability margin (m)
        self.recovery_active = False

    def calculate_support_polygon(self, foot_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Calculate support polygon from contact points"""
        if not foot_positions:
            return []

        # For simplicity, return convex hull of foot positions
        # In practice, this would be more complex with hands, etc.
        return self._convex_hull(foot_positions)

    def _convex_hull(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Calculate 2D convex hull of points (simplified implementation)"""
        if len(points) <= 1:
            return points

        # Graham scan algorithm (simplified)
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove last point of each half because it's repeated
        return lower[:-1] + upper[:-1]

    def _cross(self, o, a, b):
        """Cross product of 2D vectors"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def calculate_zmp(self, com_state: Tuple[float, float, float, float, float, float],
                     cop_state: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """Calculate Zero Moment Point from CoM state"""
        com_x, com_y, com_z, com_dx, com_dy, com_dz = com_state
        gravity = 9.81

        # Simplified ZMP calculation: ZMP_x = CoM_x - (CoM_z / g) * CoM_ddx
        # For now, we'll return CoM projection (more complex physics needed for real ZMP)
        zmp_x = com_x
        zmp_y = com_y

        return (zmp_x, zmp_y)

    def calculate_stability_margin(self, com_proj: Tuple[float, float],
                                 support_polygon: List[Tuple[float, float]]) -> float:
        """Calculate stability margin (distance from CoM projection to support polygon edge)"""
        if not support_polygon:
            return -1.0  # No support

        # Calculate minimum distance from CoM projection to polygon edges
        min_dist = float('inf')
        com_x, com_y = com_proj

        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]

            # Distance from point to line segment
            dist = self._point_to_line_distance(com_x, com_y, p1[0], p1[1], p2[0], p2[1])
            min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 0.0

    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        # Vector from line start to point
        dx = px - x1
        dy = py - y1

        # Vector of line
        ldx = x2 - x1
        ldy = y2 - y1

        # Line length squared
        length_sq = ldx * ldx + ldy * ldy
        if length_sq == 0:
            # Line is actually a point
            return np.sqrt(dx * dx + dy * dy)

        # Project point onto line
        t = max(0, min(1, (dx * ldx + dy * ldy) / length_sq))

        # Closest point on line segment
        closest_x = x1 + t * ldx
        closest_y = y1 + t * ldy

        # Distance to closest point
        return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def update_balance_state(self, current_state: HumanoidState) -> BalanceState:
        """Update balance state based on current robot state"""
        # Extract CoM from joint states (simplified calculation)
        com_pos = self._calculate_com_position(current_state.joint_states)
        com_vel = self._calculate_com_velocity(current_state.joint_states)

        # Get foot positions for support polygon (simplified)
        foot_positions = self._get_foot_positions(current_state.joint_states)
        support_polygon = self.calculate_support_polygon(foot_positions)

        # Calculate ZMP (simplified)
        com_state = (*com_pos, *com_vel)  # Combine position and velocity
        zmp_pos = self.calculate_zmp(com_state[:6])

        # Calculate stability margin
        stability_margin = self.calculate_stability_margin(
            (com_pos[0], com_pos[1]), support_polygon
        )

        # Determine balance mode
        if stability_margin < 0.02:  # Very unstable
            balance_mode = "recovery"
            self.recovery_active = True
        elif stability_margin < self.stability_threshold:
            balance_mode = "dynamic"
        else:
            balance_mode = "stable"

        # Determine support phase
        double_support = len(foot_positions) >= 2

        return BalanceState(
            com_position=com_pos,
            com_velocity=com_vel,
            support_polygon=support_polygon,
            zmp_position=zmp_pos,
            stability_margin=stability_margin,
            balance_mode=balance_mode,
            double_support=double_support
        )

    def _calculate_com_position(self, joint_states: Dict[str, JointConfiguration]) -> Tuple[float, float, float]:
        """Calculate center of mass position (simplified)"""
        # For simplicity, return a fixed position
        # In reality, this would require full kinematic and mass calculation
        return (0.0, 0.0, 0.8)  # Typical humanoid CoM height

    def _calculate_com_velocity(self, joint_states: Dict[str, JointConfiguration]) -> Tuple[float, float, float]:
        """Calculate center of mass velocity (simplified)"""
        # For simplicity, return zero velocity
        # In reality, this would require derivative of CoM position
        return (0.0, 0.0, 0.0)

    def _get_foot_positions(self, joint_states: Dict[str, JointConfiguration]) -> List[Tuple[float, float]]:
        """Get foot contact positions (simplified)"""
        # For simplicity, return fixed positions for left and right foot
        # In reality, this would come from force/torque sensors or kinematics
        return [(-0.1, -0.1), (-0.1, 0.1)]  # Approximate foot positions


class WholeBodyController:
    """Controller for whole-body motion planning and coordination"""

    def __init__(self):
        self.task_queue = []
        self.active_constraints = []
        self.optimization_solver = None  # Would be a QP solver in real implementation

    def add_task(self, task_type: str, priority: int, constraints: Dict[str, Any]):
        """Add a task to the whole-body controller"""
        task = {
            "type": task_type,
            "priority": priority,
            "constraints": constraints,
            "timestamp": time.time()
        }
        self.task_queue.append(task)
        # Sort by priority (higher number = higher priority)
        self.task_queue.sort(key=lambda x: x["priority"], reverse=True)

    def solve_motion(self, current_state: HumanoidState) -> Dict[str, Any]:
        """Solve whole-body motion for current state and tasks"""
        # This would implement optimization-based control in a real system
        # For simulation, we'll return a simple solution

        solution = {
            "joint_commands": {},
            "task_execution": [],
            "constraints_satisfied": True,
            "com_trajectory": [],
            "contact_forces": {}
        }

        # Generate simple joint commands based on current tasks
        for joint_name in current_state.joint_states.keys():
            solution["joint_commands"][joint_name] = {
                "position": current_state.joint_states[joint_name].position,
                "velocity": 0.0,
                "effort": 0.0
            }

        return solution

    def add_constraint(self, constraint_type: str, parameters: Dict[str, Any]):
        """Add a constraint to the optimization problem"""
        constraint = {
            "type": constraint_type,
            "parameters": parameters,
            "active": True
        }
        self.active_constraints.append(constraint)

    def remove_constraint(self, constraint_type: str):
        """Remove a constraint from the optimization problem"""
        self.active_constraints = [
            c for c in self.active_constraints
            if c["type"] != constraint_type
        ]


class HumanoidController:
    """Main controller for humanoid robot with hierarchical architecture"""

    def __init__(self, body_plan: HumanoidBodyPlan):
        self.body_plan = body_plan
        self.balance_controller = BalanceController()
        self.whole_body_controller = WholeBodyController()
        self.current_state = self._initialize_state()
        self.control_level = ControlLevel.MID_LEVEL
        self.control_mode = "idle"
        self.safety_system = None

    def _initialize_state(self) -> HumanoidState:
        """Initialize the humanoid state"""
        # Initialize joint states to neutral positions
        joint_states = {}
        for joint_config in self.body_plan.joint_configurations:
            joint_states[joint_config.name] = JointConfiguration(
                name=joint_config.name,
                position=0.0,  # Neutral position
                velocity=0.0,
                effort=0.0,
                limits=joint_config.limits,
                stiffness=joint_config.stiffness,
                damping=joint_config.damping
            )

        # Initialize balance state
        balance_state = BalanceState(
            com_position=(0.0, 0.0, 0.8),
            com_velocity=(0.0, 0.0, 0.0),
            support_polygon=[(-0.1, -0.1), (-0.1, 0.1)],
            zmp_position=(0.0, 0.0),
            stability_margin=0.1,
            balance_mode="stable",
            double_support=True
        )

        return HumanoidState(
            body_plan=self.body_plan,
            joint_states=joint_states,
            balance_state=balance_state,
            sensor_data={},
            timestamp=time.time(),
            is_safe=True,
            control_mode="idle"
        )

    def update_state(self, sensor_data: Dict[str, Any]) -> HumanoidState:
        """Update the humanoid state with new sensor data"""
        # Update sensor data
        self.current_state.sensor_data.update(sensor_data)
        self.current_state.timestamp = time.time()

        # Update balance state
        self.current_state.balance_state = self.balance_controller.update_balance_state(
            self.current_state
        )

        # Check safety
        self.current_state.is_safe = self._check_safety()

        return self.current_state

    def _check_safety(self) -> bool:
        """Check if current state is safe for operation"""
        stability_margin = self.current_state.balance_state.stability_margin
        balance_mode = self.current_state.balance_state.balance_mode

        # Check if stability margin is sufficient
        if stability_margin < 0.01:  # Dangerously unstable
            return False

        # Check joint limits
        for joint_name, joint_state in self.current_state.joint_states.items():
            min_pos, max_pos = joint_state.limits
            if joint_state.position < min_pos or joint_state.position > max_pos:
                return False

        return True

    def compute_control_commands(self) -> Dict[str, Any]:
        """Compute control commands based on current state and tasks"""
        if not self.current_state.is_safe:
            # Emergency stop if not safe
            return self._emergency_stop_commands()

        # Solve whole-body motion
        motion_solution = self.whole_body_controller.solve_motion(self.current_state)

        # Apply balance corrections if needed
        if self.current_state.balance_state.balance_mode == "recovery":
            motion_solution = self._apply_balance_recovery(motion_solution)

        return motion_solution

    def _emergency_stop_commands(self) -> Dict[str, Any]:
        """Generate emergency stop commands"""
        commands = {
            "joint_commands": {},
            "task_execution": [],
            "constraints_satisfied": True,
            "com_trajectory": [],
            "contact_forces": {}
        }

        # Set all joint velocities to zero
        for joint_name in self.current_state.joint_states.keys():
            commands["joint_commands"][joint_name] = {
                "position": self.current_state.joint_states[joint_name].position,
                "velocity": 0.0,
                "effort": 0.0
            }

        return commands

    def _apply_balance_recovery(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply balance recovery adjustments to motion solution"""
        # This would implement balance recovery strategies
        # For simulation, we'll just return the solution as-is
        return solution

    def set_control_mode(self, mode: str):
        """Set the control mode (walking, standing, manipulation, etc.)"""
        self.control_mode = mode
        if mode == "walking":
            # Add walking-specific constraints
            self.whole_body_controller.add_constraint("foot_placement", {})
            self.whole_body_controller.add_constraint("com_tracking", {})
        elif mode == "standing":
            # Add standing-specific constraints
            self.whole_body_controller.add_constraint("balance", {})
        elif mode == "manipulation":
            # Add manipulation-specific constraints
            self.whole_body_controller.add_constraint("end_effector", {})


class HumanoidPlatformManager:
    """Manager for different humanoid platforms"""

    def __init__(self):
        self.platforms = {}
        self.active_robots = {}

    def create_platform(self, platform_type: HumanoidPlatform, name: str) -> HumanoidController:
        """Create a humanoid platform with appropriate body plan"""
        body_plan = self._create_body_plan(platform_type)
        controller = HumanoidController(body_plan)

        self.platforms[name] = {
            "type": platform_type,
            "body_plan": body_plan,
            "controller": controller
        }

        self.active_robots[name] = controller
        return controller

    def _create_body_plan(self, platform_type: HumanoidPlatform) -> HumanoidBodyPlan:
        """Create body plan for specific platform type"""
        if platform_type == HumanoidPlatform.NAO:
            # NAO has 25 DOF
            joint_configs = []
            # Add joints for NAO (simplified)
            for i in range(25):
                joint_configs.append(JointConfiguration(
                    name=f"joint_{i}",
                    position=0.0,
                    velocity=0.0,
                    effort=0.0,
                    limits=(-2.0, 2.0)
                ))

            link_props = []
            for i in range(25):
                link_props.append(LinkProperties(
                    name=f"link_{i}",
                    mass=0.1,
                    com=(0.0, 0.0, 0.0),
                    inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0)
                ))

            actuator_types = {f"joint_{i}": ActuatorType.SERVO for i in range(25)}

        elif platform_type == HumanoidPlatform.ATLAS:
            # Atlas has ~28 DOF with hydraulic actuators
            joint_configs = []
            for i in range(28):
                joint_configs.append(JointConfiguration(
                    name=f"joint_{i}",
                    position=0.0,
                    velocity=0.0,
                    effort=0.0,
                    limits=(-3.0, 3.0)
                ))

            link_props = []
            for i in range(28):
                link_props.append(LinkProperties(
                    name=f"link_{i}",
                    mass=0.5,
                    com=(0.0, 0.0, 0.0),
                    inertia=(0.05, 0.05, 0.05, 0.0, 0.0, 0.0)
                ))

            actuator_types = {f"joint_{i}": ActuatorType.HYDRAULIC for i in range(28)}

        else:
            # Default: 20 DOF for custom platform
            joint_configs = []
            for i in range(20):
                joint_configs.append(JointConfiguration(
                    name=f"joint_{i}",
                    position=0.0,
                    velocity=0.0,
                    effort=0.0,
                    limits=(-2.0, 2.0)
                ))

            link_props = []
            for i in range(20):
                link_props.append(LinkProperties(
                    name=f"link_{i}",
                    mass=0.2,
                    com=(0.0, 0.0, 0.0),
                    inertia=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0)
                ))

            actuator_types = {f"joint_{i}": ActuatorType.SERVO for i in range(20)}

        return HumanoidBodyPlan(
            platform=platform_type,
            total_dof=len(joint_configs),
            joint_configurations=joint_configs,
            link_properties=link_props,
            actuator_types=actuator_types,
            sensor_configurations={
                "head": ["camera", "imu"],
                "torso": ["imu", "force_torque"],
                "feet": ["force_torque", "contact"],
                "hands": ["tactile", "force_torque"]
            }
        )

    def get_robot(self, name: str) -> Optional[HumanoidController]:
        """Get a robot controller by name"""
        return self.active_robots.get(name)

    def list_platforms(self) -> List[str]:
        """List all available platforms"""
        return list(self.platforms.keys())

    def evaluate_performance(self, robot_name: str, task: str) -> Dict[str, Any]:
        """Evaluate robot performance on a specific task"""
        robot = self.get_robot(robot_name)
        if not robot:
            return {"success": False, "error": f"Robot {robot_name} not found"}

        # Performance metrics based on task
        if task == "walking":
            stability = robot.current_state.balance_state.stability_margin
            return {
                "success": True,
                "metrics": {
                    "stability_margin": stability,
                    "balance_mode": robot.current_state.balance_state.balance_mode,
                    "step_efficiency": 0.85,  # Simulated efficiency
                    "energy_consumption": 50.0  # Simulated energy (W)
                }
            }
        elif task == "manipulation":
            return {
                "success": True,
                "metrics": {
                    "precision": 0.95,  # Simulated precision
                    "success_rate": 0.90,  # Task success rate
                    "dexterity_score": 0.88,
                    "grasp_stability": 0.92
                }
            }
        elif task == "balance":
            stability = robot.current_state.balance_state.stability_margin
            recovery_capability = robot.current_state.balance_state.balance_mode != "recovery"
            return {
                "success": True,
                "metrics": {
                    "stability_margin": stability,
                    "recovery_capability": recovery_capability,
                    "disturbance_tolerance": 0.15,  # meters
                    "balance_time": 10.0  # seconds
                }
            }
        else:
            return {
                "success": False,
                "error": f"Task {task} not supported for evaluation"
            }


# Global humanoid platform manager instance
humanoid_manager = HumanoidPlatformManager()