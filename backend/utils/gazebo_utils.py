import asyncio
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime
import threading
from pathlib import Path


class PhysicsEngine(Enum):
    """Supported physics engines in Gazebo"""
    ODE = "ode"
    BULLET = "bullet"
    DART = "dart"
    SIMBODY = "simbody"


class ModelType(Enum):
    """Types of models in Gazebo"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ARTICULATED = "articulated"


class SensorType(Enum):
    """Types of sensors supported in Gazebo"""
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    CONTACT = "contact"
    FORCE_TORQUE = "force_torque"


@dataclass
class GazeboPose:
    """Pose representation (position and orientation)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    @classmethod
    def from_list(cls, pose_list: List[float]) -> 'GazeboPose':
        if len(pose_list) != 6:
            raise ValueError("Pose list must have 6 elements [x, y, z, roll, pitch, yaw]")
        return cls(*pose_list)


@dataclass
class SDFModel:
    """SDF Model representation"""
    name: str
    pose: GazeboPose
    model_type: ModelType
    links: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]
    sensors: List[Dict[str, Any]]
    static: bool = False
    sdf_xml: str = ""


@dataclass
class SDFWorld:
    """SDF World representation"""
    name: str
    physics_engine: PhysicsEngine
    gravity: Tuple[float, float, float]
    models: List[SDFModel]
    lights: List[Dict[str, Any]]
    plugins: List[Dict[str, Any]]
    scene_properties: Dict[str, Any]
    sdf_xml: str = ""


class GazeboModel:
    """Simulated Gazebo model with state tracking"""

    def __init__(self, model: SDFModel):
        self.model = model
        self.id = f"{model.name}_{uuid.uuid4().hex[:8]}"
        self.pose = model.pose
        self.velocity = GazeboPose()
        self.acceleration = GazeboPose()
        self.links = model.links
        self.joints = model.joints
        self.sensors = model.sensors
        self.state = "active"  # active, paused, destroyed
        self.mass = self._calculate_mass()
        self.last_update = time.time()

    def _calculate_mass(self) -> float:
        """Calculate total mass of the model from its links"""
        total_mass = 0.0
        for link in self.links:
            inertial = link.get("inertial", {})
            mass = inertial.get("mass", 0.0)
            total_mass += mass
        return max(total_mass, 0.01)  # Minimum mass to prevent division by zero

    def update_pose(self, new_pose: GazeboPose, dt: float = 0.001):
        """Update model pose based on velocity and acceleration"""
        if self.state != "active":
            return

        # Update velocity based on acceleration
        self.velocity.x += self.acceleration.x * dt
        self.velocity.y += self.acceleration.y * dt
        self.velocity.z += self.acceleration.z * dt
        self.velocity.roll += self.acceleration.roll * dt
        self.velocity.pitch += self.acceleration.pitch * dt
        self.velocity.yaw += self.acceleration.yaw * dt

        # Update pose based on velocity
        self.pose.x += self.velocity.x * dt
        self.pose.y += self.velocity.y * dt
        self.pose.z += self.velocity.z * dt
        self.pose.roll += self.velocity.roll * dt
        self.pose.pitch += self.velocity.pitch * dt
        self.pose.yaw += self.velocity.yaw * dt

        self.last_update = time.time()

    def apply_force(self, force: GazeboPose):
        """Apply force to the model"""
        # F = ma, so a = F/m
        self.acceleration.x = force.x / self.mass
        self.acceleration.y = force.y / self.mass
        self.acceleration.z = force.z / self.mass
        self.acceleration.roll = force.roll / self.mass
        self.acceleration.pitch = force.pitch / self.mass
        self.acceleration.yaw = force.yaw / self.mass

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the model"""
        return {
            "id": self.id,
            "name": self.model.name,
            "pose": self.pose.to_list(),
            "velocity": self.velocity.to_list(),
            "acceleration": self.acceleration.to_list(),
            "state": self.state,
            "mass": self.mass,
            "last_update": self.last_update
        }


class GazeboWorld:
    """Simulated Gazebo world with physics and models"""

    def __init__(self, world: SDFWorld):
        self.world = world
        self.id = f"{world.name}_{uuid.uuid4().hex[:8]}"
        self.physics_engine = world.physics_engine
        self.gravity = world.gravity
        self.scene_properties = world.scene_properties
        self.lights = world.lights
        self.plugins = world.plugins
        self.models: Dict[str, GazeboModel] = {}
        self.time_step = 0.001  # 1ms default time step
        self.sim_time = 0.0
        self.real_time_factor = 1.0
        self.paused = False
        self.update_thread = None
        self.update_active = False
        self.lock = threading.Lock()

        # Initialize models
        for sdf_model in world.models:
            gazebo_model = GazeboModel(sdf_model)
            self.models[gazebo_model.id] = gazebo_model

    def add_model(self, sdf_model: SDFModel) -> str:
        """Add a model to the world"""
        with self.lock:
            gazebo_model = GazeboModel(sdf_model)
            self.models[gazebo_model.id] = gazebo_model
            return gazebo_model.id

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the world"""
        with self.lock:
            if model_id in self.models:
                model = self.models[model_id]
                model.state = "destroyed"
                del self.models[model_id]
                return True
            return False

    def get_model(self, model_id: str) -> Optional[GazeboModel]:
        """Get a model by ID"""
        return self.models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the world"""
        with self.lock:
            return [model.get_state() for model in self.models.values()]

    def apply_gravity(self):
        """Apply gravity to all dynamic models"""
        with self.lock:
            for model in self.models.values():
                if not model.model.static:  # Only apply to non-static models
                    gravity_force = GazeboPose(
                        x=0.0,
                        y=0.0,
                        z=-model.mass * self.gravity[2]  # Assuming gravity is in -Z direction
                    )
                    model.apply_force(gravity_force)

    def step_simulation(self, dt: Optional[float] = None):
        """Step the simulation forward in time"""
        step_dt = dt or self.time_step

        if not self.paused:
            with self.lock:
                # Apply gravity
                self.apply_gravity()

                # Update all model poses
                for model in self.models.values():
                    model.update_pose(model.pose, step_dt)

                # Update simulation time
                self.sim_time += step_dt

    def start_simulation(self):
        """Start the simulation update loop"""
        if not self.update_active:
            self.update_active = True
            self.paused = False
            self.update_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.update_thread.start()

    def stop_simulation(self):
        """Stop the simulation update loop"""
        self.update_active = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True

    def unpause_simulation(self):
        """Unpause the simulation"""
        self.paused = False

    def _simulation_loop(self):
        """Background simulation update loop"""
        while self.update_active:
            if not self.paused:
                self.step_simulation()
            # Sleep based on real-time factor
            sleep_time = self.time_step / self.real_time_factor
            time.sleep(max(0.001, sleep_time))  # Minimum sleep time

    def get_world_state(self) -> Dict[str, Any]:
        """Get current state of the world"""
        with self.lock:
            return {
                "id": self.id,
                "name": self.world.name,
                "physics_engine": self.physics_engine.value,
                "gravity": self.gravity,
                "sim_time": self.sim_time,
                "real_time_factor": self.real_time_factor,
                "paused": self.paused,
                "model_count": len(self.models),
                "models": [model.get_state() for model in self.models.values()],
                "lights": self.lights,
                "plugins": self.plugins
            }


class GazeboPlugin:
    """Base class for Gazebo plugins"""

    def __init__(self, name: str, plugin_type: str):
        self.name = name
        self.plugin_type = plugin_type
        self.loaded = False
        self.parameters: Dict[str, Any] = {}

    def load(self, world: 'GazeboWorld', params: Dict[str, Any]) -> bool:
        """Load the plugin with given parameters"""
        self.parameters = params
        self.loaded = True
        return True

    def unload(self):
        """Unload the plugin"""
        self.loaded = False

    def update(self, world: 'GazeboWorld'):
        """Update plugin state"""
        pass


class ROSBridgePlugin(GazeboPlugin):
    """Plugin to simulate ROS bridge functionality"""

    def __init__(self):
        super().__init__("ros_bridge", "ros_bridge")
        self.published_topics: Dict[str, List[Any]] = {}
        self.subscribed_topics: Dict[str, List[callable]] = {}
        self.ros_node_name = "gazebo_ros"

    def load(self, world: GazeboWorld, params: Dict[str, Any]) -> bool:
        """Load ROS bridge plugin"""
        super().load(world, params)

        # Initialize ROS communication channels
        self.published_topics = {
            "/clock": [],
            "/gazebo/model_states": [],
            "/gazebo/link_states": []
        }

        return True

    def update(self, world: GazeboWorld):
        """Publish simulation state to ROS topics"""
        if not self.loaded:
            return

        # Publish clock
        clock_msg = {"clock": world.sim_time}
        self.published_topics["/clock"].append(clock_msg)

        # Publish model states
        model_states = {
            "name": [model["name"] for model in world.list_models()],
            "pose": [model["pose"] for model in world.list_models()],
            "twist": [model["velocity"] for model in world.list_models()]
        }
        self.published_topics["/gazebo/model_states"].append(model_states)


class GazeboSceneManager:
    """Manager for Gazebo worlds and models"""

    def __init__(self):
        self.worlds: Dict[str, GazeboWorld] = {}
        self.models: Dict[str, GazeboModel] = {}
        self.plugins: Dict[str, GazeboPlugin] = {}
        self.active_plugins: Dict[str, List[str]] = {}  # world_id -> plugin_ids

    def create_world(self, world: SDFWorld) -> str:
        """Create a new simulation world"""
        gazebo_world = GazeboWorld(world)
        world_id = gazebo_world.id
        self.worlds[world_id] = gazebo_world
        return world_id

    def get_world(self, world_id: str) -> Optional[GazeboWorld]:
        """Get a world by ID"""
        return self.worlds.get(world_id)

    def list_worlds(self) -> List[Dict[str, Any]]:
        """List all worlds"""
        return [
            {
                "id": world_id,
                "name": world.world.name,
                "model_count": len(world.models),
                "paused": world.paused
            }
            for world_id, world in self.worlds.items()
        ]

    def delete_world(self, world_id: str) -> bool:
        """Delete a world"""
        if world_id in self.worlds:
            world = self.worlds[world_id]
            world.stop_simulation()
            del self.worlds[world_id]
            return True
        return False

    def spawn_model(self, world_id: str, sdf_model: SDFModel) -> Optional[str]:
        """Spawn a model in a world"""
        world = self.get_world(world_id)
        if world:
            return world.add_model(sdf_model)
        return None

    def delete_model(self, world_id: str, model_id: str) -> bool:
        """Delete a model from a world"""
        world = self.get_world(world_id)
        if world:
            return world.remove_model(model_id)
        return False

    def register_plugin(self, plugin: GazeboPlugin) -> str:
        """Register a plugin"""
        plugin_id = f"{plugin.name}_{uuid.uuid4().hex[:8]}"
        self.plugins[plugin_id] = plugin
        return plugin_id

    def load_plugin(self, plugin_id: str, world_id: str, params: Dict[str, Any] = None) -> bool:
        """Load a plugin into a world"""
        if params is None:
            params = {}

        plugin = self.plugins.get(plugin_id)
        world = self.get_world(world_id)

        if plugin and world:
            success = plugin.load(world, params)
            if success:
                if world_id not in self.active_plugins:
                    self.active_plugins[world_id] = []
                self.active_plugins[world_id].append(plugin_id)
            return success

        return False

    def get_world_state(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of a world"""
        world = self.get_world(world_id)
        if world:
            return world.get_world_state()
        return None

    def get_model_state(self, world_id: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of a model in a world"""
        world = self.get_world(world_id)
        if world:
            model = world.get_model(model_id)
            if model:
                return model.get_state()
        return None

    def apply_force_to_model(self, world_id: str, model_id: str, force: GazeboPose) -> bool:
        """Apply a force to a model"""
        world = self.get_world(world_id)
        if world:
            model = world.get_model(model_id)
            if model:
                model.apply_force(force)
                return True
        return False

    def set_model_pose(self, world_id: str, model_id: str, pose: GazeboPose) -> bool:
        """Set the pose of a model"""
        world = self.get_world(world_id)
        if world:
            model = world.get_model(model_id)
            if model:
                model.pose = pose
                return True
        return False


class SDFParser:
    """Parser for SDF (Simulation Description Format) files"""

    @staticmethod
    def parse_world_from_xml(xml_content: str) -> Optional[SDFWorld]:
        """Parse SDF world from XML content"""
        try:
            root = ET.fromstring(xml_content)

            # Find world element
            world_elem = root.find('world')
            if world_elem is None:
                return None

            name = world_elem.get('name', 'default_world')

            # Parse physics
            physics_elem = world_elem.find('physics')
            physics_engine = PhysicsEngine.ODE
            if physics_elem is not None:
                physics_type = physics_elem.get('type', 'ode')
                try:
                    physics_engine = PhysicsEngine(physics_type)
                except ValueError:
                    physics_engine = PhysicsEngine.ODE

            # Parse gravity
            gravity_elem = world_elem.find('physics/gravity')
            gravity = (0, 0, -9.8)  # Default gravity
            if gravity_elem is not None:
                gravity_text = gravity_elem.text
                if gravity_text:
                    gravity_vals = [float(x) for x in gravity_text.split()]
                    if len(gravity_vals) == 3:
                        gravity = tuple(gravity_vals)

            # Parse models
            models = []
            for model_elem in world_elem.findall('model'):
                sdf_model = SDFParser._parse_model_element(model_elem)
                if sdf_model:
                    models.append(sdf_model)

            # Parse lights
            lights = []
            for light_elem in world_elem.findall('light'):
                light = SDFParser._parse_light_element(light_elem)
                lights.append(light)

            # Parse plugins
            plugins = []
            for plugin_elem in world_elem.findall('plugin'):
                plugin = SDFParser._parse_plugin_element(plugin_elem)
                plugins.append(plugin)

            # Parse scene properties
            scene_elem = world_elem.find('scene')
            scene_properties = {}
            if scene_elem is not None:
                for child in scene_elem:
                    scene_properties[child.tag] = child.text or ""

            world = SDFWorld(
                name=name,
                physics_engine=physics_engine,
                gravity=gravity,
                models=models,
                lights=lights,
                plugins=plugins,
                scene_properties=scene_properties,
                sdf_xml=xml_content
            )

            return world

        except ET.ParseError as e:
            print(f"Error parsing SDF XML: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error parsing SDF: {e}")
            return None

    @staticmethod
    def parse_model_from_xml(xml_content: str) -> Optional[SDFModel]:
        """Parse SDF model from XML content"""
        try:
            root = ET.fromstring(xml_content)

            # Look for model element at root or inside sdf
            model_elem = root.find('model')
            if model_elem is None and root.tag == 'model':
                model_elem = root

            if model_elem is None:
                return None

            name = model_elem.get('name', 'unnamed_model')

            # Parse pose
            pose_elem = model_elem.find('pose')
            pose = GazeboPose()
            if pose_elem is not None:
                pose_text = pose_elem.text
                if pose_text:
                    pose_vals = [float(x) for x in pose_text.split()]
                    if len(pose_vals) >= 6:
                        pose = GazeboPose(*pose_vals[:6])

            # Parse static property
            static_elem = model_elem.find('static')
            static = False
            if static_elem is not None:
                static = static_elem.text.lower() == 'true' if static_elem.text else False

            # Parse links
            links = []
            for link_elem in model_elem.findall('link'):
                link = SDFParser._parse_link_element(link_elem)
                links.append(link)

            # Parse joints
            joints = []
            for joint_elem in model_elem.findall('joint'):
                joint = SDFParser._parse_joint_element(joint_elem)
                joints.append(joint)

            # Parse sensors
            sensors = []
            for sensor_elem in model_elem.findall('.//sensor'):
                sensor = SDFParser._parse_sensor_element(sensor_elem)
                sensors.append(sensor)

            model_type = ModelType.DYNAMIC
            if static:
                model_type = ModelType.STATIC

            sdf_model = SDFModel(
                name=name,
                pose=pose,
                model_type=model_type,
                links=links,
                joints=joints,
                sensors=sensors,
                static=static,
                sdf_xml=xml_content
            )

            return sdf_model

        except ET.ParseError as e:
            print(f"Error parsing SDF model XML: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error parsing SDF model: {e}")
            return None

    @staticmethod
    def _parse_model_element(model_elem) -> Optional[SDFModel]:
        """Parse a model element from SDF XML"""
        name = model_elem.get('name', 'unnamed_model')

        # Parse pose
        pose_elem = model_elem.find('pose')
        pose = GazeboPose()
        if pose_elem is not None:
            pose_text = pose_elem.text
            if pose_text:
                pose_vals = [float(x) for x in pose_text.split()]
                if len(pose_vals) >= 6:
                    pose = GazeboPose(*pose_vals[:6])

        # Parse static property
        static_elem = model_elem.find('static')
        static = False
        if static_elem is not None:
            static = static_elem.text.lower() == 'true' if static_elem.text else False

        # Parse links
        links = []
        for link_elem in model_elem.findall('link'):
            link = SDFParser._parse_link_element(link_elem)
            links.append(link)

        # Parse joints
        joints = []
        for joint_elem in model_elem.findall('joint'):
            joint = SDFParser._parse_joint_element(joint_elem)
            joints.append(joint)

        # Parse sensors
        sensors = []
        for sensor_elem in model_elem.findall('.//sensor'):
            sensor = SDFParser._parse_sensor_element(sensor_elem)
            sensors.append(sensor)

        model_type = ModelType.DYNAMIC
        if static:
            model_type = ModelType.STATIC

        return SDFModel(
            name=name,
            pose=pose,
            model_type=model_type,
            links=links,
            joints=joints,
            sensors=sensors,
            static=static
        )

    @staticmethod
    def _parse_link_element(link_elem) -> Dict[str, Any]:
        """Parse a link element from SDF XML"""
        name = link_elem.get('name', 'unnamed_link')

        # Parse inertial
        inertial_elem = link_elem.find('inertial')
        inertial = {}
        if inertial_elem is not None:
            mass_elem = inertial_elem.find('mass')
            mass = float(mass_elem.text) if mass_elem is not None and mass_elem.text else 1.0

            inertia_elem = inertial_elem.find('inertia')
            inertia = {
                "ixx": 1.0, "ixy": 0.0, "ixz": 0.0,
                "iyy": 1.0, "iyz": 0.0, "izz": 1.0
            }
            if inertia_elem is not None:
                for tag in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                    elem = inertia_elem.find(tag)
                    if elem is not None and elem.text:
                        inertia[tag] = float(elem.text)

            inertial = {
                "mass": mass,
                "inertia": inertia
            }

        # Parse visual
        visuals = []
        for visual_elem in link_elem.findall('visual'):
            visual = {
                "name": visual_elem.get('name', ''),
                "geometry": SDFParser._parse_geometry_element(visual_elem.find('geometry'))
            }
            visuals.append(visual)

        # Parse collision
        collisions = []
        for collision_elem in link_elem.findall('collision'):
            collision = {
                "name": collision_elem.get('name', ''),
                "geometry": SDFParser._parse_geometry_element(collision_elem.find('geometry'))
            }
            collisions.append(collision)

        return {
            "name": name,
            "inertial": inertial,
            "visuals": visuals,
            "collisions": collisions
        }

    @staticmethod
    def _parse_joint_element(joint_elem) -> Dict[str, Any]:
        """Parse a joint element from SDF XML"""
        name = joint_elem.get('name', 'unnamed_joint')
        joint_type = joint_elem.get('type', 'revolute')

        parent_elem = joint_elem.find('parent')
        parent = parent_elem.text if parent_elem is not None and parent_elem.text else ''

        child_elem = joint_elem.find('child')
        child = child_elem.text if child_elem is not None and child_elem.text else ''

        axis_elem = joint_elem.find('axis')
        axis = {"xyz": [0, 0, 1], "limit": {"lower": -1, "upper": 1}}
        if axis_elem is not None:
            xyz_elem = axis_elem.find('xyz')
            if xyz_elem is not None and xyz_elem.text:
                xyz_vals = [float(x) for x in xyz_elem.text.split()]
                axis["xyz"] = xyz_vals[:3] if len(xyz_vals) >= 3 else [0, 0, 1]

            limit_elem = axis_elem.find('limit')
            if limit_elem is not None:
                lower_elem = limit_elem.find('lower')
                upper_elem = limit_elem.find('upper')

                if lower_elem is not None and lower_elem.text:
                    axis["limit"]["lower"] = float(lower_elem.text)
                if upper_elem is not None and upper_elem.text:
                    axis["limit"]["upper"] = float(upper_elem.text)

        return {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "axis": axis
        }

    @staticmethod
    def _parse_sensor_element(sensor_elem) -> Dict[str, Any]:
        """Parse a sensor element from SDF XML"""
        name = sensor_elem.get('name', 'unnamed_sensor')
        sensor_type = sensor_elem.get('type', 'camera')

        return {
            "name": name,
            "type": sensor_type,
            "always_on": True,
            "update_rate": 30
        }

    @staticmethod
    def _parse_geometry_element(geometry_elem) -> Dict[str, Any]:
        """Parse a geometry element from SDF XML"""
        if geometry_elem is None:
            return {"type": "box", "box": {"size": [1, 1, 1]}}

        # Check for different geometry types
        box_elem = geometry_elem.find('box')
        if box_elem is not None:
            size_elem = box_elem.find('size')
            if size_elem is not None and size_elem.text:
                size_vals = [float(x) for x in size_elem.text.split()]
                return {"type": "box", "box": {"size": size_vals[:3]}}

        cylinder_elem = geometry_elem.find('cylinder')
        if cylinder_elem is not None:
            radius_elem = cylinder_elem.find('radius')
            length_elem = cylinder_elem.find('length')
            radius = float(radius_elem.text) if radius_elem is not None and radius_elem.text else 1.0
            length = float(length_elem.text) if length_elem is not None and length_elem.text else 1.0
            return {"type": "cylinder", "cylinder": {"radius": radius, "length": length}}

        sphere_elem = geometry_elem.find('sphere')
        if sphere_elem is not None:
            radius_elem = sphere_elem.find('radius')
            radius = float(radius_elem.text) if radius_elem is not None and radius_elem.text else 1.0
            return {"type": "sphere", "sphere": {"radius": radius}}

        # Default to box
        return {"type": "box", "box": {"size": [1, 1, 1]}}

    @staticmethod
    def _parse_light_element(light_elem) -> Dict[str, Any]:
        """Parse a light element from SDF XML"""
        name = light_elem.get('name', 'unnamed_light')
        light_type = light_elem.get('type', 'point')

        pose_elem = light_elem.find('pose')
        pose = [0, 0, 1, 0, 0, 0]
        if pose_elem is not None and pose_elem.text:
            pose_vals = [float(x) for x in pose_elem.text.split()]
            pose = pose_vals[:6] if len(pose_vals) >= 6 else pose

        diffuse_elem = light_elem.find('diffuse')
        diffuse = [1, 1, 1, 1]
        if diffuse_elem is not None and diffuse_elem.text:
            diffuse_vals = [float(x) for x in diffuse_elem.text.split()]
            diffuse = diffuse_vals[:4] if len(diffuse_vals) >= 4 else diffuse

        return {
            "name": name,
            "type": light_type,
            "pose": pose,
            "diffuse": diffuse
        }

    @staticmethod
    def _parse_plugin_element(plugin_elem) -> Dict[str, Any]:
        """Parse a plugin element from SDF XML"""
        name = plugin_elem.get('name', 'unnamed_plugin')
        filename = plugin_elem.get('filename', '')

        # Extract parameters from child elements
        params = {}
        for child in plugin_elem:
            if child.text:
                params[child.tag] = child.text

        return {
            "name": name,
            "filename": filename,
            "parameters": params
        }


# Global Gazebo scene manager instance
gazebo_scene_manager = GazeboSceneManager()