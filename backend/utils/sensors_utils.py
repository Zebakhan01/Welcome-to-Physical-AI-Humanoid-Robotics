import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
import cv2
from math import sqrt, atan2, cos, sin
import math

@dataclass
class SensorReading:
    """Represents a sensor reading with timestamp and confidence"""
    value: float
    timestamp: float
    confidence: float = 1.0
    sensor_type: str = "generic"

@dataclass
class IMUReading:
    """Represents an IMU reading with accelerometer, gyroscope, and magnetometer data"""
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    timestamp: float

@dataclass
class RangeReading:
    """Represents a range sensor reading"""
    distance: float
    angle: float  # in radians
    confidence: float
    sensor_type: str  # "ultrasonic", "lidar", "infrared"

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion coefficient
    k2: float = 0.0  # Radial distortion coefficient
    p1: float = 0.0  # Tangential distortion coefficient
    p2: float = 0.0  # Tangential distortion coefficient

class SensorNoiseSimulator:
    """Simulates various types of sensor noise"""

    @staticmethod
    def add_gaussian_noise(value: float, std_dev: float) -> float:
        """Add Gaussian noise to a sensor reading"""
        noise = np.random.normal(0, std_dev)
        return value + noise

    @staticmethod
    def add_systematic_error(value: float, bias: float) -> float:
        """Add systematic bias to a sensor reading"""
        return value + bias

    @staticmethod
    def generate_outlier(value: float, probability: float = 0.01, max_error: float = 1.0) -> float:
        """Generate occasional outliers in sensor readings"""
        if np.random.random() < probability:
            outlier = value + np.random.uniform(-max_error, max_error)
            return outlier
        return value

class ImageProcessor:
    """Handles basic computer vision operations"""

    @staticmethod
    def detect_edges(image: np.ndarray, method: str = "canny") -> np.ndarray:
        """Detect edges in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if method == "canny":
            edges = cv2.Canny(gray, 50, 150)
        elif method == "sobel":
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(grad_x**2 + grad_y**2)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        return edges

    @staticmethod
    def detect_corners(image: np.ndarray, method: str = "shi_tomasi", max_corners: int = 100) -> List[Tuple[int, int]]:
        """Detect corners in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if method == "shi_tomasi":
            corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
        elif method == "harris":
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            # Find centroids of corner regions
            corners = cv2.goodFeaturesToTrack(corners, max_corners, 0.01, 10)
        else:
            raise ValueError(f"Unknown corner detection method: {method}")

        if corners is not None:
            corners = np.int0(corners)
            return [(corner[0][0], corner[0][1]) for corner in corners]
        else:
            return []

    @staticmethod
    def compute_features(image: np.ndarray, method: str = "orb") -> Tuple[List[Tuple[int, int]], List[float]]:
        """Compute features in an image (simplified implementation)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if method == "orb":
            # For a simplified implementation, we'll use corner detection
            corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
            if corners is not None:
                corners = np.int0(corners)
                points = [(corner[0][0], corner[0][1]) for corner in corners]
                # Simplified descriptor as distance from center
                center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
                descriptors = [sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points]
                return points, descriptors
            else:
                return [], []
        else:
            raise ValueError(f"Unknown feature method: {method}")

    @staticmethod
    def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Undistort an image using camera intrinsics"""
        h, w = image.shape[:2]

        # Create camera matrix
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ])

        # Create distortion coefficients
        dist_coeffs = np.array([
            intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, 0
        ])

        # Undistort image
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        return undistorted

class SensorFusion:
    """Implementation of sensor fusion algorithms"""

    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Initialize state vector (position, velocity, etc.)
        self.x = np.zeros((state_dim, 1))

        # Initialize covariance matrix
        self.P = np.eye(state_dim) * 1000

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0

        # State transition matrix (simplified - identity for now)
        self.F = np.eye(state_dim)

        # Measurement matrix
        self.H = np.eye(measurement_dim, state_dim) if measurement_dim <= state_dim else np.zeros((measurement_dim, state_dim))
        if measurement_dim <= state_dim:
            self.H = np.zeros((measurement_dim, state_dim))
            for i in range(measurement_dim):
                self.H[i, i] = 1

    def predict(self):
        """Prediction step of the Kalman filter"""
        # State prediction: x = F * x
        self.x = self.F @ self.x

        # Covariance prediction: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """Update step of the Kalman filter"""
        # Innovation: y = z - H * x
        y = z.reshape(-1, 1) - self.H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Get the current state estimate"""
        return self.x.flatten()

    @staticmethod
    def fuse_sensors_simple(readings: List[SensorReading]) -> Tuple[float, float]:
        """
        Simple sensor fusion using weighted average based on confidence
        Returns (fused_value, combined_confidence)
        """
        if not readings:
            return 0.0, 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for reading in readings:
            weight = reading.confidence
            weighted_sum += reading.value * weight
            total_weight += weight

        if total_weight == 0:
            return readings[0].value, readings[0].confidence

        fused_value = weighted_sum / total_weight
        combined_confidence = total_weight / len(readings)

        return fused_value, combined_confidence

class RangeSensorSimulator:
    """Simulates range sensor readings"""

    @staticmethod
    def simulate_lidar_2d(angles: List[float], robot_pose: Tuple[float, float, float],
                         obstacles: List[Tuple[float, float, float]],
                         noise_std: float = 0.01) -> List[RangeReading]:
        """
        Simulate 2D LIDAR readings
        robot_pose: (x, y, theta) in world coordinates
        obstacles: List of (x, y, radius) for circular obstacles
        """
        readings = []

        robot_x, robot_y, robot_theta = robot_pose

        for angle in angles:
            # Transform angle to world frame
            world_angle = robot_theta + angle

            # Initialize minimum distance to max range (e.g., 10m)
            min_distance = 10.0

            # Check for intersections with each obstacle
            for obs_x, obs_y, obs_radius in obstacles:
                # Vector from robot to obstacle
                dx = obs_x - robot_x
                dy = obs_y - robot_y

                # Distance to obstacle center
                dist_to_center = sqrt(dx*dx + dy*dy)

                # If robot is inside obstacle, distance is negative
                if dist_to_center < obs_radius:
                    distance = 0.0
                else:
                    # Calculate distance to obstacle surface
                    distance = dist_to_center - obs_radius

                # Check if ray intersects with obstacle
                # Calculate angle from robot to obstacle
                obs_angle = atan2(dy, dx)
                angle_diff = abs(world_angle - obs_angle)

                # Simplified check: if angle is close enough, consider it an intersection
                if abs(angle_diff) < 0.1 or abs(angle_diff) > 2*math.pi - 0.1:
                    if distance < min_distance:
                        min_distance = distance

            # Add noise
            noisy_distance = min_distance + np.random.normal(0, noise_std)
            if noisy_distance < 0:
                noisy_distance = 0.0

            readings.append(RangeReading(
                distance=noisy_distance,
                angle=angle,
                confidence=0.95,
                sensor_type="lidar"
            ))

        return readings

class IMUSimulator:
    """Simulates IMU sensor readings"""

    @staticmethod
    def simulate_imu(accel_true: Tuple[float, float, float],
                     gyro_true: Tuple[float, float, float],
                     noise_std: float = 0.01) -> IMUReading:
        """
        Simulate IMU reading with added noise
        """
        timestamp = 0.0  # In a real implementation, this would be current time

        # Add noise to true values
        accel_x = accel_true[0] + np.random.normal(0, noise_std)
        accel_y = accel_true[1] + np.random.normal(0, noise_std)
        accel_z = accel_true[2] + np.random.normal(0, noise_std)

        gyro_x = gyro_true[0] + np.random.normal(0, noise_std)
        gyro_y = gyro_true[1] + np.random.normal(0, noise_std)
        gyro_z = gyro_true[2] + np.random.normal(0, noise_std)

        # Magnetometer - simplified as constant magnetic field
        mag_x = 0.2 + np.random.normal(0, noise_std*0.1)
        mag_y = 0.0 + np.random.normal(0, noise_std*0.1)
        mag_z = 0.4 + np.random.normal(0, noise_std*0.1)

        return IMUReading(
            accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z,
            timestamp=timestamp
        )

def process_lidar_scan(ranges: List[float], angles: List[float]) -> Dict[str, Any]:
    """
    Process LIDAR scan data to extract features
    """
    # Convert to Cartesian coordinates
    points = []
    for r, theta in zip(ranges, angles):
        if r > 0 and r < 10:  # Valid range
            x = r * cos(theta)
            y = r * sin(theta)
            points.append((x, y))

    # Calculate simple statistics
    if points:
        xs, ys = zip(*points)
        stats = {
            "min_x": min(xs) if xs else 0,
            "max_x": max(xs) if xs else 0,
            "min_y": min(ys) if ys else 0,
            "max_y": max(ys) if ys else 0,
            "center_x": sum(xs) / len(xs) if xs else 0,
            "center_y": sum(ys) / len(ys) if ys else 0,
            "num_points": len(points),
            "avg_distance": sum(ranges) / len(ranges) if ranges else 0
        }
    else:
        stats = {
            "min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0,
            "center_x": 0, "center_y": 0, "num_points": 0,
            "avg_distance": 0
        }

    return stats