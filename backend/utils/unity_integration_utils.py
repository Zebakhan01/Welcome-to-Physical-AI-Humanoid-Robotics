import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime
import threading
from pathlib import Path
import numpy as np


class UnityPhysicsEngine(Enum):
    """Physics engines available in Unity"""
    PHYSX = "physx"
    BULLET = "bullet"  # Through Unity integration
    CUSTOM = "custom"


class UnityObjectType(Enum):
    """Types of objects in Unity scenes"""
    GAME_OBJECT = "game_object"
    MODEL = "model"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    LIGHT = "light"
    CAMERA = "camera"
    TERRAIN = "terrain"


class UnityComponentType(Enum):
    """Component types in Unity"""
    RIGIDBODY = "rigidbody"
    COLLIDER = "collider"
    MESH_RENDERER = "mesh_renderer"
    TRANSFORM = "transform"
    SCRIPT = "script"


@dataclass
class UnityVector3:
    """3D vector for Unity coordinates"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: List[float]) -> 'UnityVector3':
        if len(values) != 3:
            raise ValueError("Vector3 requires 3 values")
        return cls(values[0], values[1], values[2])


@dataclass
class UnityQuaternion:
    """Quaternion for Unity rotations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_list(cls, values: List[float]) -> 'UnityQuaternion':
        if len(values) != 4:
            raise ValueError("Quaternion requires 4 values")
        return cls(values[0], values[1], values[2], values[3])


@dataclass
class UnityTransform:
    """Unity Transform component"""
    position: UnityVector3
    rotation: UnityQuaternion
    scale: UnityVector3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_list(),
            "rotation": self.rotation.to_list(),
            "scale": self.scale.to_list()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnityTransform':
        return cls(
            position=UnityVector3.from_list(data.get("position", [0, 0, 0])),
            rotation=UnityQuaternion.from_list(data.get("rotation", [0, 0, 0, 1])),
            scale=UnityVector3.from_list(data.get("scale", [1, 1, 1]))
        )


@dataclass
class UnityPhysicsProperties:
    """Physics properties for Unity objects"""
    mass: float = 1.0
    drag: float = 0.0
    angular_drag: float = 0.05
    use_gravity: bool = True
    is_kinematic: bool = False
    freeze_position: UnityVector3 = None
    freeze_rotation: UnityVector3 = None

    def __post_init__(self):
        if self.freeze_position is None:
            self.freeze_position = UnityVector3(0, 0, 0)
        if self.freeze_rotation is None:
            self.freeze_rotation = UnityVector3(0, 0, 0)


@dataclass
class UnityCollider:
    """Unity Collider component"""
    type: str  # box, sphere, capsule, mesh, terrain
    center: UnityVector3 = None
    size: UnityVector3 = None
    radius: float = 0.5
    height: float = 2.0
    is_trigger: bool = False

    def __post_init__(self):
        if self.center is None:
            self.center = UnityVector3(0, 0, 0)
        if self.size is None:
            self.size = UnityVector3(1, 1, 1)


@dataclass
class UnityRigidbody:
    """Unity Rigidbody component"""
    mass: float = 1.0
    drag: float = 0.0
    angular_drag: float = 0.05
    use_gravity: bool = True
    is_kinematic: bool = False
    interpolation: str = "none"  # none, interpolate, extrapolate
    collision_detection: str = "discrete"  # discrete, continuous, continuous_dynamic


@dataclass
class UnitySceneObject:
    """Represents an object in a Unity scene"""
    id: str
    name: str
    object_type: UnityObjectType
    transform: UnityTransform
    components: Dict[str, Any]
    tags: List[str]
    active: bool
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "object_type": self.object_type.value,
            "transform": self.transform.to_dict(),
            "components": self.components,
            "tags": self.tags,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class UnityScene:
    """Represents a Unity scene"""
    id: str
    name: str
    objects: Dict[str, UnitySceneObject]
    physics_engine: UnityPhysicsEngine
    gravity: UnityVector3
    time_scale: float
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "physics_engine": self.physics_engine.value,
            "gravity": self.gravity.to_list(),
            "time_scale": self.time_scale,
            "object_count": len(self.objects),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class UnitySceneManager:
    """Manager for Unity scenes and objects"""

    def __init__(self):
        self.scenes: Dict[str, UnityScene] = {}
        self.active_scene: Optional[str] = None
        self.physics_simulator = UnityPhysicsSimulator()
        self.asset_manager = UnityAssetManager()
        self.simulation_active = False
        self.simulation_thread = None

    def create_scene(self, name: str) -> str:
        """Create a new Unity scene"""
        scene_id = f"scene_{name}_{uuid.uuid4().hex[:8]}"

        scene = UnityScene(
            id=scene_id,
            name=name,
            objects={},
            physics_engine=UnityPhysicsEngine.PHYSX,
            gravity=UnityVector3(0, -9.81, 0),
            time_scale=1.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.scenes[scene_id] = scene
        return scene_id

    def get_scene(self, scene_id: str) -> Optional[UnityScene]:
        """Get a scene by ID"""
        return self.scenes.get(scene_id)

    def load_scene(self, scene_id: str) -> bool:
        """Load a scene as the active scene"""
        if scene_id in self.scenes:
            self.active_scene = scene_id
            return True
        return False

    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene"""
        if scene_id in self.scenes:
            del self.scenes[scene_id]
            if self.active_scene == scene_id:
                self.active_scene = None
            return True
        return False

    def list_scenes(self) -> List[Dict[str, Any]]:
        """List all scenes"""
        return [scene.to_dict() for scene in self.scenes.values()]

    def create_object(self, scene_id: str, name: str, object_type: UnityObjectType,
                     position: Tuple[float, float, float] = (0, 0, 0),
                     rotation: Tuple[float, float, float, float] = (0, 0, 0, 1),
                     tags: List[str] = None) -> Optional[str]:
        """Create an object in a scene"""
        scene = self.get_scene(scene_id)
        if not scene:
            return None

        object_id = f"obj_{name}_{uuid.uuid4().hex[:8]}"

        unity_obj = UnitySceneObject(
            id=object_id,
            name=name,
            object_type=object_type,
            transform=UnityTransform(
                position=UnityVector3(*position),
                rotation=UnityQuaternion(*rotation),
                scale=UnityVector3(1, 1, 1)
            ),
            components={},
            tags=tags or [],
            active=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        scene.objects[object_id] = unity_obj
        scene.updated_at = datetime.now()

        # Initialize physics state for the object if it has physics components
        # For now, we'll initialize all objects to ensure they can receive forces
        self.physics_simulator.initialize_object_state(object_id)

        return object_id

    def get_object(self, scene_id: str, object_id: str) -> Optional[UnitySceneObject]:
        """Get an object from a scene"""
        scene = self.get_scene(scene_id)
        if scene:
            return scene.objects.get(object_id)
        return None

    def delete_object(self, scene_id: str, object_id: str) -> bool:
        """Delete an object from a scene"""
        scene = self.get_scene(scene_id)
        if scene and object_id in scene.objects:
            del scene.objects[object_id]
            scene.updated_at = datetime.now()
            return True
        return False

    def list_objects(self, scene_id: str) -> List[Dict[str, Any]]:
        """List all objects in a scene"""
        scene = self.get_scene(scene_id)
        if scene:
            return [obj.to_dict() for obj in scene.objects.values()]
        return []

    def add_component(self, scene_id: str, object_id: str, component_type: UnityComponentType,
                     component_data: Dict[str, Any]) -> bool:
        """Add a component to an object"""
        obj = self.get_object(scene_id, object_id)
        if obj:
            obj.components[component_type.value] = component_data
            obj.updated_at = datetime.now()

            # If we're adding a physics component, initialize it in the physics simulator
            if component_type in [UnityComponentType.RIGIDBODY, UnityComponentType.COLLIDER]:
                # Initialize physics state for this object
                self.physics_simulator.initialize_object_state(object_id)

            return True
        return False

    def get_component(self, scene_id: str, object_id: str, component_type: UnityComponentType) -> Optional[Dict[str, Any]]:
        """Get a component from an object"""
        obj = self.get_object(scene_id, object_id)
        if obj:
            return obj.components.get(component_type.value)
        return None

    def set_transform(self, scene_id: str, object_id: str,
                     position: Optional[Tuple[float, float, float]] = None,
                     rotation: Optional[Tuple[float, float, float, float]] = None,
                     scale: Optional[Tuple[float, float, float]] = None) -> bool:
        """Set the transform of an object"""
        obj = self.get_object(scene_id, object_id)
        if obj:
            if position:
                obj.transform.position = UnityVector3(*position)
            if rotation:
                obj.transform.rotation = UnityQuaternion(*rotation)
            if scale:
                obj.transform.scale = UnityVector3(*scale)

            obj.updated_at = datetime.now()
            return True
        return False

    def start_simulation(self):
        """Start the Unity simulation"""
        if not self.simulation_active:
            self.simulation_active = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()

    def stop_simulation(self):
        """Stop the Unity simulation"""
        self.simulation_active = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)

    def _simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_active:
            if self.active_scene:
                # Update physics
                self.physics_simulator.update_scene(self.active_scene, self.scenes[self.active_scene])

                # Update object states
                self._update_objects()

            time.sleep(0.016)  # ~60 FPS

    def _update_objects(self):
        """Update all objects in the active scene"""
        if not self.active_scene:
            return

        scene = self.scenes[self.active_scene]
        for obj_id, obj in scene.objects.items():
            # Apply physics if object has rigidbody
            if "rigidbody" in obj.components:
                self.physics_simulator.update_object_physics(obj_id, obj)


class UnityPhysicsSimulator:
    """Simulates Unity physics engine behavior"""

    def __init__(self):
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.time_step = 0.016  # 60 FPS
        self.objects_state: Dict[str, Dict[str, Any]] = {}

    def update_scene(self, scene_id: str, scene: UnityScene):
        """Update physics for entire scene"""
        # Update gravity based on scene settings
        self.gravity = np.array(scene.gravity.to_list())

        # Process all objects with physics components
        for obj_id, obj in scene.objects.items():
            self.update_object_physics(obj_id, obj)

    def update_object_physics(self, obj_id: str, obj: UnitySceneObject):
        """Update physics for a single object"""
        if "rigidbody" not in obj.components:
            return

        rigidbody = obj.components["rigidbody"]

        # Initialize object state if not present
        if obj_id not in self.objects_state:
            self.objects_state[obj_id] = {
                "velocity": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "last_update": time.time()
            }

        state = self.objects_state[obj_id]
        current_time = time.time()
        dt = current_time - state["last_update"]
        state["last_update"] = current_time

        if dt > 0.1:  # Prevent large time jumps
            dt = 0.016

        # Apply gravity if enabled
        if rigidbody.get("use_gravity", True):
            gravity_force = self.gravity * rigidbody.get("mass", 1.0)
            state["velocity"] += gravity_force * dt

        # Apply drag
        drag = rigidbody.get("drag", 0.0)
        if drag > 0:
            state["velocity"] *= (1.0 - drag * dt)

        # Update position based on velocity
        pos = np.array(obj.transform.position.to_list())
        pos += state["velocity"] * dt

        # Update the object's transform
        obj.transform.position = UnityVector3(pos[0], pos[1], pos[2])
        obj.updated_at = datetime.now()

    def apply_force(self, obj_id: str, force: Tuple[float, float, float]) -> bool:
        """Apply a force to an object"""
        if obj_id in self.objects_state:
            force_vec = np.array(force)
            self.objects_state[obj_id]["velocity"] += force_vec * self.time_step
            return True
        return False

    def initialize_object_state(self, obj_id: str):
        """Initialize physics state for an object"""
        if obj_id not in self.objects_state:
            self.objects_state[obj_id] = {
                "velocity": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "last_update": time.time()
            }

    def apply_torque(self, obj_id: str, torque: Tuple[float, float, float]):
        """Apply a torque to an object"""
        # Initialize state if not present
        if obj_id not in self.objects_state:
            self.initialize_object_state(obj_id)

        torque_vec = np.array(torque)
        self.objects_state[obj_id]["angular_velocity"] += torque_vec * self.time_step

    def set_velocity(self, obj_id: str, velocity: Tuple[float, float, float]):
        """Set the velocity of an object"""
        if obj_id not in self.objects_state:
            self.objects_state[obj_id] = {
                "velocity": np.array([0.0, 0.0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "last_update": time.time()
            }

        self.objects_state[obj_id]["velocity"] = np.array(velocity)

    def get_object_state(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Get the physics state of an object"""
        if obj_id in self.objects_state:
            state = self.objects_state[obj_id]
            return {
                "velocity": state["velocity"].tolist(),
                "angular_velocity": state["angular_velocity"].tolist(),
                "position": self.objects_state.get("position", [0, 0, 0]),
                "mass": 1.0  # Default mass
            }
        return None


class UnityAssetManager:
    """Manager for Unity assets (models, materials, textures, etc.)"""

    def __init__(self):
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.asset_paths: Dict[str, str] = {}  # asset_id -> file_path

    def register_asset(self, asset_id: str, asset_type: str, file_path: str,
                      metadata: Dict[str, Any] = None) -> bool:
        """Register an asset in the system"""
        if metadata is None:
            metadata = {}

        self.assets[asset_id] = {
            "id": asset_id,
            "type": asset_type,
            "file_path": file_path,
            "metadata": metadata,
            "registered_at": datetime.now()
        }
        self.asset_paths[asset_id] = file_path
        return True

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset information"""
        return self.assets.get(asset_id)

    def list_assets(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all assets or assets of a specific type"""
        if asset_type:
            return [asset for asset in self.assets.values() if asset["type"] == asset_type]
        return list(self.assets.values())

    def load_model(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Load a model asset"""
        asset = self.get_asset(asset_id)
        if asset and asset["type"] == "model":
            # Simulate model loading
            return {
                "id": asset_id,
                "type": "loaded_model",
                "mesh_count": 1,
                "material_count": 1,
                "has_colliders": True,
                "bounds": {"center": [0, 0, 0], "size": [1, 1, 1]}
            }
        return None

    def create_material(self, name: str, color: Tuple[float, float, float, float] = (1, 1, 1, 1),
                       shader: str = "standard") -> Dict[str, Any]:
        """Create a material"""
        material_id = f"mat_{name}_{uuid.uuid4().hex[:8]}"

        material = {
            "id": material_id,
            "name": name,
            "type": "material",
            "properties": {
                "color": color,
                "shader": shader,
                "metallic": 0.0,
                "smoothness": 0.5
            },
            "created_at": datetime.now()
        }

        self.assets[material_id] = material
        return material


class UnityRobotController:
    """Controller for robot objects in Unity simulation"""

    def __init__(self, unity_manager: UnitySceneManager):
        self.unity_manager = unity_manager
        self.robot_objects: Dict[str, str] = {}  # robot_id -> object_id

    def create_robot(self, scene_id: str, robot_name: str,
                    robot_type: str = "mobile_base") -> Optional[str]:
        """Create a robot in the scene"""
        # Create the main robot object
        robot_obj_id = self.unity_manager.create_object(
            scene_id, robot_name, UnityObjectType.MODEL,
            position=(0, 0, 0), tags=["robot", robot_type]
        )

        if robot_obj_id:
            # Add rigidbody component
            self.unity_manager.add_component(
                scene_id, robot_obj_id, UnityComponentType.RIGIDBODY,
                {
                    "mass": 10.0,
                    "drag": 0.1,
                    "angular_drag": 0.05,
                    "use_gravity": True,
                    "is_kinematic": False
                }
            )

            # Add collider component
            self.unity_manager.add_component(
                scene_id, robot_obj_id, UnityComponentType.COLLIDER,
                {
                    "type": "capsule",
                    "center": [0, 0.5, 0],
                    "radius": 0.3,
                    "height": 1.0,
                    "is_trigger": False
                }
            )

            self.robot_objects[f"robot_{robot_name}_{uuid.uuid4().hex[:8]}"] = robot_obj_id
            return robot_obj_id

        return None

    def move_robot(self, scene_id: str, robot_object_id: str,
                   target_position: Tuple[float, float, float]) -> bool:
        """Move a robot to a target position"""
        # In a real implementation, this would involve more complex pathfinding and physics
        # For simulation, we'll just set the position
        return self.unity_manager.set_transform(
            scene_id, robot_object_id, position=target_position
        )

    def rotate_robot(self, scene_id: str, robot_object_id: str,
                     target_rotation: Tuple[float, float, float, float]) -> bool:
        """Rotate a robot to a target orientation"""
        return self.unity_manager.set_transform(
            scene_id, robot_object_id, rotation=target_rotation
        )

    def apply_force_to_robot(self, robot_object_id: str, force: Tuple[float, float, float]) -> bool:
        """Apply a force to a robot"""
        # Apply force through the physics simulator
        success = self.unity_manager.physics_simulator.apply_force(robot_object_id, force)
        return success

    def get_robot_state(self, scene_id: str, robot_object_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a robot"""
        robot_obj = self.unity_manager.get_object(scene_id, robot_object_id)
        if not robot_obj:
            return None

        # Get physics state
        physics_state = self.unity_manager.physics_simulator.get_object_state(robot_object_id)

        return {
            "object_id": robot_object_id,
            "name": robot_obj.name,
            "position": robot_obj.transform.position.to_list(),
            "rotation": robot_obj.transform.rotation.to_list(),
            "physics_state": physics_state,
            "components": list(robot_obj.components.keys()),
            "tags": robot_obj.tags
        }


class UnitySensorSystem:
    """System for managing sensors in Unity simulation"""

    def __init__(self, unity_manager: UnitySceneManager):
        self.unity_manager = unity_manager
        self.sensors: Dict[str, Dict[str, Any]] = {}  # sensor_id -> sensor_config

    def add_camera_sensor(self, scene_id: str, object_id: str, sensor_name: str,
                         fov: float = 60.0, resolution: Tuple[int, int] = (640, 480)) -> Optional[str]:
        """Add a camera sensor to an object"""
        sensor_id = f"cam_{sensor_name}_{uuid.uuid4().hex[:8]}"

        sensor_config = {
            "id": sensor_id,
            "type": "camera",
            "fov": fov,
            "resolution": resolution,
            "attached_to": object_id,
            "enabled": True
        }

        # Add sensor component to the object
        success = self.unity_manager.add_component(
            scene_id, object_id, UnityComponentType.SCRIPT,  # Using script as sensor placeholder
            {"sensor_type": "camera", "config": sensor_config}
        )

        if success:
            self.sensors[sensor_id] = sensor_config
            return sensor_id

        return None

    def add_lidar_sensor(self, scene_id: str, object_id: str, sensor_name: str,
                        range_m: float = 10.0, rays: int = 360) -> Optional[str]:
        """Add a LIDAR sensor to an object"""
        sensor_id = f"lidar_{sensor_name}_{uuid.uuid4().hex[:8]}"

        sensor_config = {
            "id": sensor_id,
            "type": "lidar",
            "range": range_m,
            "rays": rays,
            "attached_to": object_id,
            "enabled": True
        }

        # Add sensor component to the object
        success = self.unity_manager.add_component(
            scene_id, object_id, UnityComponentType.SCRIPT,  # Using script as sensor placeholder
            {"sensor_type": "lidar", "config": sensor_config}
        )

        if success:
            self.sensors[sensor_id] = sensor_config
            return sensor_id

        return None

    def add_imu_sensor(self, scene_id: str, object_id: str, sensor_name: str) -> Optional[str]:
        """Add an IMU sensor to an object"""
        sensor_id = f"imu_{sensor_name}_{uuid.uuid4().hex[:8]}"

        sensor_config = {
            "id": sensor_id,
            "type": "imu",
            "attached_to": object_id,
            "enabled": True
        }

        # Add sensor component to the object
        success = self.unity_manager.add_component(
            scene_id, object_id, UnityComponentType.SCRIPT,  # Using script as sensor placeholder
            {"sensor_type": "imu", "config": sensor_config}
        )

        if success:
            self.sensors[sensor_id] = sensor_config
            return sensor_id

        return None

    def get_sensor_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get data from a sensor (simulated)"""
        if sensor_id not in self.sensors:
            return None

        sensor = self.sensors[sensor_id]
        sensor_type = sensor["type"]

        if sensor_type == "camera":
            # Simulate camera data
            width, height = sensor["resolution"]
            # Generate a simple simulated image (in practice, this would come from Unity's rendering)
            image_data = {
                "width": width,
                "height": height,
                "channels": 3,  # RGB
                "data": [[128, 128, 128] for _ in range(width * height)]  # Gray image for simulation
            }
            return {
                "sensor_id": sensor_id,
                "type": "camera",
                "timestamp": time.time(),
                "image": image_data
            }

        elif sensor_type == "lidar":
            # Simulate LIDAR data
            rays = sensor["rays"]
            # Generate simulated distance readings
            distances = [np.random.uniform(0.1, sensor["range"]) for _ in range(rays)]
            return {
                "sensor_id": sensor_id,
                "type": "lidar",
                "timestamp": time.time(),
                "ranges": distances,
                "min_range": 0.1,
                "max_range": sensor["range"],
                "angle_min": -np.pi,
                "angle_max": np.pi,
                "angle_increment": (2 * np.pi) / rays
            }

        elif sensor_type == "imu":
            # Simulate IMU data
            return {
                "sensor_id": sensor_id,
                "type": "imu",
                "timestamp": time.time(),
                "linear_acceleration": [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(9.81, 0.1)],
                "angular_velocity": [np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)],
                "orientation": [0, 0, 0, 1]  # Quaternion (identity)
            }

        return None

    def list_sensors(self) -> List[Dict[str, Any]]:
        """List all sensors"""
        return list(self.sensors.values())


class UnityIntegrationManager:
    """Main manager for Unity integration functionality"""

    def __init__(self):
        self.scene_manager = UnitySceneManager()
        self.asset_manager = UnityAssetManager()
        self.robot_controller = UnityRobotController(self.scene_manager)
        self.sensor_system = UnitySensorSystem(self.scene_manager)

    def create_unity_environment(self, name: str) -> Dict[str, Any]:
        """Create a Unity simulation environment"""
        scene_id = self.scene_manager.create_scene(name)

        return {
            "scene_id": scene_id,
            "name": name,
            "success": True,
            "message": f"Unity environment '{name}' created successfully"
        }

    def get_scene_info(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a scene"""
        scene = self.scene_manager.get_scene(scene_id)
        if scene:
            return {
                "scene": scene.to_dict(),
                "objects": self.scene_manager.list_objects(scene_id),
                "success": True
            }
        return None

    def spawn_robot_in_environment(self, scene_id: str, robot_name: str,
                                  robot_type: str = "mobile_base") -> Optional[Dict[str, Any]]:
        """Spawn a robot in a Unity environment"""
        robot_obj_id = self.robot_controller.create_robot(scene_id, robot_name, robot_type)
        if robot_obj_id:
            robot_state = self.robot_controller.get_robot_state(scene_id, robot_obj_id)
            return {
                "robot_object_id": robot_obj_id,
                "robot_state": robot_state,
                "success": True,
                "message": f"Robot '{robot_name}' spawned successfully"
            }
        return None

    def get_robot_state(self, scene_id: str, robot_object_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of a robot in the Unity environment"""
        return self.robot_controller.get_robot_state(scene_id, robot_object_id)

    def move_robot(self, scene_id: str, robot_object_id: str,
                   target_position: Tuple[float, float, float]) -> bool:
        """Move a robot in the Unity environment"""
        return self.robot_controller.move_robot(scene_id, robot_object_id, target_position)

    def add_sensor_to_robot(self, scene_id: str, robot_object_id: str,
                           sensor_type: str, sensor_name: str) -> Optional[str]:
        """Add a sensor to a robot in the Unity environment"""
        if sensor_type.lower() == "camera":
            return self.sensor_system.add_camera_sensor(
                scene_id, robot_object_id, sensor_name
            )
        elif sensor_type.lower() == "lidar":
            return self.sensor_system.add_lidar_sensor(
                scene_id, robot_object_id, sensor_name
            )
        elif sensor_type.lower() == "imu":
            return self.sensor_system.add_imu_sensor(
                scene_id, robot_object_id, sensor_name
            )
        return None

    def get_sensor_data(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get data from a sensor in the Unity environment"""
        return self.sensor_system.get_sensor_data(sensor_id)

    def list_scenes(self) -> List[Dict[str, Any]]:
        """List all Unity scenes"""
        return self.scene_manager.list_scenes()

    def list_robots(self, scene_id: str) -> List[Dict[str, Any]]:
        """List all robots in a scene"""
        objects = self.scene_manager.list_objects(scene_id)
        robots = [obj for obj in objects if "robot" in obj["tags"]]
        return robots

    def list_sensors(self) -> List[Dict[str, Any]]:
        """List all sensors in the system"""
        return self.sensor_system.list_sensors()

    def start_simulation(self):
        """Start the Unity simulation"""
        self.scene_manager.start_simulation()

    def stop_simulation(self):
        """Stop the Unity simulation"""
        self.scene_manager.stop_simulation()


# Global Unity integration manager instance
unity_integration_manager = UnityIntegrationManager()