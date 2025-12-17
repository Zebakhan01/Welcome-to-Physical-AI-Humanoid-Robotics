import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import uuid
from datetime import datetime


class SimulationPlatform(Enum):
    """Types of simulation platforms"""
    GAZEBO = "gazebo"
    UNITY = "unity"
    ISAAC_SIM = "isaac_sim"
    WEBOTS = "webots"
    CUSTOM = "custom"


class PhysicsEngine(Enum):
    """Physics engines for simulation"""
    ODE = "ode"
    BULLET = "bullet"
    DART = "dart"
    PHYSX = "physx"
    CUSTOM = "custom"


@dataclass
class SimulationWorld:
    """Represents a simulation world/environment"""
    id: str
    name: str
    description: str
    physics_engine: PhysicsEngine
    gravity: Tuple[float, float, float]
    time_step: float
    max_update_rate: float
    models: List[Dict[str, Any]]
    lights: List[Dict[str, Any]]
    cameras: List[Dict[str, Any]]
    sensors: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class RobotModel:
    """Represents a robot model in simulation"""
    id: str
    name: str
    urdf_path: str
    sdf_path: Optional[str]
    joints: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    sensors: List[Dict[str, Any]]
    initial_pose: Tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw
    mass: float
    inertia: Tuple[float, float, float, float, float, float]


@dataclass
class SimulationSensor:
    """Represents a sensor in simulation"""
    id: str
    name: str
    sensor_type: str  # camera, lidar, imu, gps, etc.
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    parameters: Dict[str, Any]
    topic: str
    noise_model: Optional[Dict[str, Any]] = None


@dataclass
class SimulationState:
    """Current state of the simulation"""
    world_id: str
    time: float
    paused: bool
    running: bool
    robot_states: Dict[str, Dict[str, Any]]  # robot_id -> state
    sensor_data: Dict[str, Any]  # sensor_id -> data
    physics_stats: Dict[str, float]
    last_updated: datetime


@dataclass
class SimulationMetrics:
    """Performance metrics for simulation"""
    real_time_factor: float
    update_rate: float
    cpu_usage: float
    memory_usage: float
    physics_step_time: float
    rendering_time: float
    simulation_time: float
    wall_time: float


class PhysicsSimulator:
    """Simulates physics in the virtual environment"""

    def __init__(self, engine: PhysicsEngine = PhysicsEngine.BULLET):
        self.engine = engine
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.time_step = 0.001  # 1ms default
        self.objects = {}
        self.constraints = {}

    def set_gravity(self, gravity: Tuple[float, float, float]):
        """Set gravity vector"""
        self.gravity = np.array(gravity)

    def set_time_step(self, time_step: float):
        """Set physics simulation time step"""
        self.time_step = time_step

    def add_rigid_body(self, obj_id: str, mass: float, position: np.ndarray,
                      orientation: np.ndarray, shape: str = "box"):
        """Add a rigid body to the simulation"""
        # Ensure orientation is a 4-element quaternion
        if len(orientation) == 3:  # If it's Euler angles, convert to quaternion
            # For simplicity, just create a default quaternion if 3D vector provided
            orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
        elif len(orientation) == 4:  # If already a quaternion
            orientation_quat = orientation.copy()
        else:
            # Default quaternion
            orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.objects[obj_id] = {
            "mass": mass,
            "position": position,
            "orientation": orientation_quat,
            "velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "shape": shape,
            "inertia": np.eye(3) * mass * 0.4  # Approximate inertia for sphere
        }

    def add_constraint(self, constraint_id: str, body1_id: str, body2_id: str,
                      constraint_type: str, parameters: Dict[str, Any]):
        """Add a constraint between bodies"""
        self.constraints[constraint_id] = {
            "body1": body1_id,
            "body2": body2_id,
            "type": constraint_type,
            "parameters": parameters
        }

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Step the physics simulation forward in time"""
        step_dt = dt or self.time_step

        # Update positions and velocities based on forces
        for obj_id, obj in self.objects.items():
            # Apply gravity
            force = obj["mass"] * self.gravity
            acceleration = force / obj["mass"]

            # Update velocity
            obj["velocity"] += acceleration * step_dt

            # Update position
            obj["position"] += obj["velocity"] * step_dt

            # Update orientation - for simplicity, we'll just keep the quaternion as is
            # In a real implementation, we'd properly integrate angular velocity
            # to update the quaternion orientation
            # For now, we'll just normalize the quaternion to maintain unit length
            q = obj["orientation"]
            norm = np.linalg.norm(q)
            if norm > 0:
                obj["orientation"] = q / norm

        # Return physics statistics
        return {
            "objects_count": len(self.objects),
            "step_time": step_dt,
            "gravity": self.gravity.tolist()
        }


class SensorSimulator:
    """Simulates various sensors in the virtual environment"""

    def __init__(self):
        self.sensors = {}

    def add_camera(self, sensor_id: str, position: Tuple[float, float, float],
                   orientation: Tuple[float, float, float, float],
                   params: Dict[str, Any]):
        """Add a camera sensor"""
        self.sensors[sensor_id] = {
            "type": "camera",
            "position": position,
            "orientation": orientation,
            "parameters": params,
            "last_data": None
        }

    def add_lidar(self, sensor_id: str, position: Tuple[float, float, float],
                  orientation: Tuple[float, float, float, float],
                  params: Dict[str, Any]):
        """Add a LIDAR sensor"""
        self.sensors[sensor_id] = {
            "type": "lidar",
            "position": position,
            "orientation": orientation,
            "parameters": params,
            "last_data": None
        }

    def add_imu(self, sensor_id: str, position: Tuple[float, float, float],
                orientation: Tuple[float, float, float, float],
                params: Dict[str, Any]):
        """Add an IMU sensor"""
        self.sensors[sensor_id] = {
            "type": "imu",
            "position": position,
            "orientation": orientation,
            "parameters": params,
            "last_data": None
        }

    def generate_sensor_data(self, sensor_id: str, physics_state: Dict[str, Any]) -> Any:
        """Generate simulated sensor data based on physics state"""
        if sensor_id not in self.sensors:
            return None

        sensor = self.sensors[sensor_id]

        if sensor["type"] == "camera":
            # Simulate camera data (simplified)
            width = sensor["parameters"].get("width", 640)
            height = sensor["parameters"].get("height", 480)
            # Generate random image data for simulation
            image_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            sensor["last_data"] = {
                "image": image_data.tolist(),  # In real implementation, this would be encoded
                "timestamp": time.time(),
                "width": width,
                "height": height
            }
        elif sensor["type"] == "lidar":
            # Simulate LIDAR data (simplified)
            num_points = sensor["parameters"].get("num_points", 360)
            # Generate random distance measurements
            distances = np.random.uniform(0.1, 10.0, num_points).tolist()
            sensor["last_data"] = {
                "ranges": distances,
                "timestamp": time.time(),
                "angle_min": -np.pi,
                "angle_max": np.pi,
                "angle_increment": 2 * np.pi / num_points
            }
        elif sensor["type"] == "imu":
            # Simulate IMU data
            sensor["last_data"] = {
                "linear_acceleration": [0.0, 0.0, 9.81],  # Gravity vector
                "angular_velocity": [0.0, 0.0, 0.0],  # No rotation initially
                "orientation": [0.0, 0.0, 0.0, 1.0],  # No rotation quaternion
                "timestamp": time.time()
            }

        return sensor["last_data"]


class SimulationEnvironment:
    """Main simulation environment manager"""

    def __init__(self, platform: SimulationPlatform = SimulationPlatform.GAZEBO):
        self.platform = platform
        self.worlds = {}
        self.robots = {}
        self.physics_simulator = PhysicsSimulator()
        self.sensor_simulator = SensorSimulator()
        self.current_state = None
        self.is_running = False
        self.is_paused = False
        self.simulation_time = 0.0

    def create_world(self, name: str, description: str = "",
                     physics_engine: PhysicsEngine = PhysicsEngine.BULLET) -> SimulationWorld:
        """Create a new simulation world"""
        world_id = str(uuid.uuid4())
        world = SimulationWorld(
            id=world_id,
            name=name,
            description=description,
            physics_engine=physics_engine,
            gravity=(0.0, 0.0, -9.81),
            time_step=0.001,
            max_update_rate=1000.0,
            models=[],
            lights=[],
            cameras=[],
            sensors=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.worlds[world_id] = world
        return world

    def load_robot_model(self, world_id: str, urdf_path: str, name: str,
                        initial_pose: Tuple[float, float, float, float, float, float] = (0,0,0,0,0,0)) -> RobotModel:
        """Load a robot model into the simulation"""
        if world_id not in self.worlds:
            raise ValueError(f"World {world_id} does not exist")

        robot_id = str(uuid.uuid4())

        # In a real implementation, this would parse the URDF file
        # For simulation, we'll create a basic robot model
        robot = RobotModel(
            id=robot_id,
            name=name,
            urdf_path=urdf_path,
            sdf_path=None,
            joints=[{"name": "joint1", "type": "revolute", "limits": {"lower": -1.57, "upper": 1.57}}],
            links=[{"name": "base_link", "mass": 1.0}],
            sensors=[],
            initial_pose=initial_pose,
            mass=10.0,
            inertia=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        )

        self.robots[robot_id] = robot

        # Add robot to world models
        world = self.worlds[world_id]
        world.models.append({
            "id": robot_id,
            "name": name,
            "type": "robot",
            "pose": initial_pose
        })
        world.updated_at = datetime.now()

        return robot

    def add_sensor_to_robot(self, robot_id: str, sensor_config: Dict[str, Any]) -> SimulationSensor:
        """Add a sensor to a robot in the simulation"""
        if robot_id not in self.robots:
            raise ValueError(f"Robot {robot_id} does not exist")

        sensor_id = str(uuid.uuid4())

        sensor = SimulationSensor(
            id=sensor_id,
            name=sensor_config.get("name", f"sensor_{sensor_id[:8]}"),
            sensor_type=sensor_config["type"],
            position=sensor_config.get("position", (0.0, 0.0, 0.0)),
            orientation=sensor_config.get("orientation", (0.0, 0.0, 0.0, 1.0)),
            parameters=sensor_config.get("parameters", {}),
            topic=sensor_config.get("topic", f"/sensor/{sensor_id}")
        )

        # Add to sensor simulator
        if sensor.sensor_type == "camera":
            self.sensor_simulator.add_camera(
                sensor_id, sensor.position, sensor.orientation, sensor.parameters
            )
        elif sensor.sensor_type == "lidar":
            self.sensor_simulator.add_lidar(
                sensor_id, sensor.position, sensor.orientation, sensor.parameters
            )
        elif sensor.sensor_type == "imu":
            self.sensor_simulator.add_imu(
                sensor_id, sensor.position, sensor.orientation, sensor.parameters
            )

        # Add to robot
        robot = self.robots[robot_id]
        robot.sensors.append({
            "id": sensor_id,
            "name": sensor.name,
            "type": sensor.sensor_type,
            "position": sensor.position
        })

        return sensor

    def start_simulation(self):
        """Start the simulation"""
        self.is_running = True
        self.is_paused = False
        self._update_simulation_state()

    def pause_simulation(self):
        """Pause the simulation"""
        self.is_paused = True

    def resume_simulation(self):
        """Resume the simulation"""
        self.is_paused = False

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.is_paused = False

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.simulation_time = 0.0
        self.current_state = None

    def step_simulation(self, dt: Optional[float] = None):
        """Step the simulation forward by dt seconds"""
        if not self.is_running or self.is_paused:
            return

        step_dt = dt or 0.01  # Default to 10ms if not specified

        # Update physics
        physics_stats = self.physics_simulator.step(step_dt)

        # Update simulation time
        self.simulation_time += step_dt

        # Update simulation state
        self._update_simulation_state()

    def _update_simulation_state(self):
        """Update the current simulation state"""
        robot_states = {}
        for robot_id, robot in self.robots.items():
            robot_states[robot_id] = {
                "position": robot.initial_pose[:3],
                "orientation": robot.initial_pose[3:],
                "joints": [{"position": 0.0, "velocity": 0.0} for _ in robot.joints]
            }

        sensor_data = {}
        for sensor_id in self.sensor_simulator.sensors.keys():
            sensor_data[sensor_id] = self.sensor_simulator.generate_sensor_data(
                sensor_id, self.physics_simulator.objects
            )

        self.current_state = SimulationState(
            world_id=list(self.worlds.keys())[0] if self.worlds else "",
            time=self.simulation_time,
            paused=self.is_paused,
            running=self.is_running,
            robot_states=robot_states,
            sensor_data=sensor_data,
            physics_stats={"objects_count": len(self.physics_simulator.objects)},
            last_updated=datetime.now()
        )

    def get_simulation_metrics(self) -> SimulationMetrics:
        """Get current simulation performance metrics"""
        # In a real implementation, these would be measured
        return SimulationMetrics(
            real_time_factor=1.0,
            update_rate=100.0,
            cpu_usage=25.0,
            memory_usage=512.0,
            physics_step_time=0.0001,
            rendering_time=0.001,
            simulation_time=self.simulation_time,
            wall_time=time.time()
        )

    def get_current_state(self) -> Optional[SimulationState]:
        """Get the current simulation state"""
        return self.current_state


class DomainRandomization:
    """Implements domain randomization for sim-to-real transfer"""

    def __init__(self):
        self.randomization_params = {}

    def add_randomization_parameter(self, name: str, param_type: str,
                                  min_val: float, max_val: float):
        """Add a parameter to be randomized"""
        self.randomization_params[name] = {
            "type": param_type,
            "min": min_val,
            "max": max_val,
            "current_value": np.random.uniform(min_val, max_val)
        }

    def randomize_parameter(self, name: str) -> float:
        """Randomize a specific parameter"""
        if name not in self.randomization_params:
            raise ValueError(f"Parameter {name} not found")

        param = self.randomization_params[name]
        new_val = np.random.uniform(param["min"], param["max"])
        param["current_value"] = new_val
        return new_val

    def randomize_all_parameters(self) -> Dict[str, float]:
        """Randomize all registered parameters"""
        results = {}
        for name in self.randomization_params:
            results[name] = self.randomize_parameter(name)
        return results

    def get_current_randomization_values(self) -> Dict[str, float]:
        """Get current values of all randomized parameters"""
        return {name: param["current_value"]
                for name, param in self.randomization_params.items()}


class SimulationManager:
    """Main manager for all simulation activities"""

    def __init__(self):
        self.environments = {}
        self.domain_randomizer = DomainRandomization()

    def create_environment(self, name: str, platform: SimulationPlatform = SimulationPlatform.GAZEBO) -> SimulationEnvironment:
        """Create a new simulation environment"""
        env_id = name.replace(" ", "_").lower()
        env = SimulationEnvironment(platform)
        self.environments[env_id] = env
        return env

    def get_environment(self, name: str) -> Optional[SimulationEnvironment]:
        """Get a simulation environment by name"""
        env_id = name.replace(" ", "_").lower()
        return self.environments.get(env_id)

    def list_environments(self) -> List[str]:
        """List all simulation environments"""
        return list(self.environments.keys())


# Global simulation manager instance
simulation_manager = SimulationManager()