"""
Data models for NVIDIA Isaac API module
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import numpy as np


# Configuration to disable protected namespace warning
class IsaacBaseModel(BaseModel):
    class Config:
        protected_namespaces = ()


# Simulation Models
class IsaacEnvironmentConfig(IsaacBaseModel):
    """Configuration for Isaac simulation environment"""
    name: str
    scene_path: Optional[str] = None
    gravity: List[float] = [0.0, 0.0, -9.81]
    physics_dt: float = 1.0/60.0
    substeps: int = 1
    enable_collisions: bool = True
    lighting_config: Optional[Dict[str, Any]] = None


class IsaacRobotConfig(IsaacBaseModel):
    """Configuration for robot in Isaac simulation"""
    name: str
    urdf_path: str
    position: List[float] = [0.0, 0.0, 0.0]
    orientation: List[float] = [0.0, 0.0, 0.0, 1.0]  # quaternion
    joint_limits: Optional[Dict[str, List[float]]] = None
    enable_observations: bool = True
    enable_actuators: bool = True


class IsaacSensorConfig(IsaacBaseModel):
    """Configuration for sensor in Isaac simulation"""
    sensor_type: str  # 'camera', 'lidar', 'imu', 'contact_sensor'
    name: str
    position: List[float] = [0.0, 0.0, 0.0]
    orientation: List[float] = [0.0, 0.0, 0.0, 1.0]
    parameters: Optional[Dict[str, Any]] = None  # sensor-specific parameters


class IsaacPhysicsConfig(IsaacBaseModel):
    """Physics configuration for Isaac simulation"""
    fixed_timestep: float = 1.0/60.0
    substeps: int = 1
    solver_position_iteration_count: int = 4
    solver_velocity_iteration_count: int = 1
    enable_gravity: bool = True
    gravity: List[float] = [0.0, 0.0, -9.81]


# ROS Bridge Models
class IsaacROSMessage(IsaacBaseModel):
    """Base model for Isaac-ROS message conversion"""
    isaac_message_type: str
    ros_message_type: str
    data: Dict[str, Any]
    frame_id: Optional[str] = None


class IsaacROSPublisherConfig(IsaacBaseModel):
    """Configuration for ROS publisher bridge"""
    topic_name: str
    ros_message_type: str
    isaac_source: str  # Isaac data source
    publish_rate: float = 30.0


class IsaacROSSubscriberConfig(IsaacBaseModel):
    """Configuration for ROS subscriber bridge"""
    topic_name: str
    ros_message_type: str
    isaac_target: str  # Isaac target for data
    queue_size: int = 10


# AI Models
class IsaacDetectionRequest(IsaacBaseModel):
    """Request model for Isaac DetectNet"""
    image_data: str  # base64 encoded image or URL
    model_pathname: Optional[str] = None  # Changed from model_path to avoid Pydantic warning
    confidence_threshold: float = 0.5
    classes: Optional[List[str]] = None


class IsaacDetectionResponse(IsaacBaseModel):
    """Response model for Isaac DetectNet"""
    detections: List[Dict[str, Any]]  # List of detections with bbox, confidence, class
    processing_time: float
    image_dimensions: List[int]  # [width, height]


class IsaacSegmentationRequest(IsaacBaseModel):
    """Request model for Isaac SegNet"""
    image_data: str  # base64 encoded image or URL
    model_pathname: Optional[str] = None  # Changed from model_path to avoid Pydantic warning
    num_classes: int = 28


class IsaacSegmentationResponse(IsaacBaseModel):
    """Response model for Isaac SegNet"""
    segmentation_map: str  # base64 encoded segmentation image
    colorized_map: Optional[str] = None  # base64 encoded colorized segmentation
    class_statistics: Dict[str, Dict[str, float]]  # class: {count, percentage}
    processing_time: float


class IsaacNavigationRequest(IsaacBaseModel):
    """Request model for Isaac Navigation"""
    start_pose: List[float]  # [x, y, theta]
    goal_pose: List[float]   # [x, y, theta]
    map_data: Optional[Dict[str, Any]] = None
    planner_type: str = "astar"  # astar, dwa, teb, etc.


class IsaacNavigationResponse(IsaacBaseModel):
    """Response model for Isaac Navigation"""
    path: List[List[float]]  # List of [x, y] coordinates
    velocity_commands: Optional[Dict[str, float]] = None  # linear, angular velocities
    status: str  # success, failed, executing
    processing_time: float


class IsaacManipulationRequest(IsaacBaseModel):
    """Request model for Isaac Manipulation"""
    task_type: str  # pick, place, move_to_pose, follow_trajectory
    target_pose: Optional[Dict[str, List[float]]] = None  # position and orientation
    object_info: Optional[Dict[str, Any]] = None
    arm_name: str = "arm"


class IsaacManipulationResponse(IsaacBaseModel):
    """Response model for Isaac Manipulation"""
    success: bool
    trajectory: Optional[List[List[float]]] = None  # joint angles over time
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


# App Framework Models
class IsaacAppConfig(IsaacBaseModel):
    """Configuration for Isaac application"""
    app_name: str
    config_path: Optional[str] = None
    robot_config: Optional[IsaacRobotConfig] = None
    sensor_configs: List[IsaacSensorConfig] = []
    simulation_config: Optional[IsaacEnvironmentConfig] = None
    enable_gui: bool = True
    headless: bool = False


class IsaacAppStatus(IsaacBaseModel):
    """Status of Isaac application"""
    app_name: str
    is_running: bool
    pid: Optional[int] = None
    status_message: str
    start_time: Optional[float] = None