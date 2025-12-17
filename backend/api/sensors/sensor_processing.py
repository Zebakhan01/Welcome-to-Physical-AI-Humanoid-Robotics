from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.sensors_utils import (
    SensorReading, IMUReading, RangeReading, CameraIntrinsics,
    SensorNoiseSimulator, ImageProcessor, SensorFusion,
    RangeSensorSimulator, IMUSimulator, process_lidar_scan
)
from backend.utils.logger import logger

router = APIRouter()

class SensorReadingRequest(BaseModel):
    value: float
    timestamp: float
    confidence: float = 1.0
    sensor_type: str = "generic"

class SensorReadingResponse(BaseModel):
    corrected_value: float
    confidence: float
    processed_at: float

class IMUReadingRequest(BaseModel):
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

class IMUReadingResponse(BaseModel):
    euler_angles: Dict[str, float]  # roll, pitch, yaw
    linear_acceleration: Dict[str, float]  # x, y, z
    angular_velocity: Dict[str, float]  # x, y, z
    processed_at: float

class RangeReadingRequest(BaseModel):
    distance: float
    angle: float
    confidence: float
    sensor_type: str

class RangeReadingResponse(BaseModel):
    corrected_distance: float
    obstacle_detected: bool
    confidence: float

class SensorFusionRequest(BaseModel):
    sensor_readings: List[SensorReadingRequest]
    fusion_method: str = "kalman"  # "kalman", "weighted_average", "particle"

class SensorFusionResponse(BaseModel):
    fused_value: float
    confidence: float
    method_used: str

class ImageProcessingRequest(BaseModel):
    image_data: str  # Base64 encoded image
    operations: List[str]  # List of operations to perform
    camera_intrinsics: Optional[CameraIntrinsics] = None

class ImageProcessingResponse(BaseModel):
    processed_data: Dict[str, Any]
    success: bool
    message: str

class LIDARSimulationRequest(BaseModel):
    angles: List[float]  # Angles in radians
    robot_pose: List[float]  # [x, y, theta]
    obstacles: List[List[float]]  # List of [x, y, radius]
    noise_std: float = 0.01

class LIDARSimulationResponse(BaseModel):
    ranges: List[float]
    angles: List[float]
    point_cloud: List[Dict[str, float]]
    processed_stats: Dict[str, Any]

class SensorNoiseSimulationRequest(BaseModel):
    value: float
    noise_type: str  # "gaussian", "systematic", "outlier"
    noise_params: Dict[str, float]  # Parameters for noise model

class SensorNoiseSimulationResponse(BaseModel):
    noisy_value: float
    original_value: float

@router.post("/process-sensor-reading", response_model=SensorReadingResponse)
async def process_sensor_reading(request: SensorReadingRequest):
    """
    Process a single sensor reading with noise correction
    """
    try:
        # In a real implementation, this would apply calibration and noise correction
        corrected_value = request.value  # Placeholder - in reality, apply calibration

        response = SensorReadingResponse(
            corrected_value=corrected_value,
            confidence=request.confidence,
            processed_at=request.timestamp
        )

        logger.info(f"Processed sensor reading of type {request.sensor_type}")

        return response

    except Exception as e:
        logger.error(f"Error processing sensor reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing sensor reading: {str(e)}")

@router.post("/process-imu", response_model=IMUReadingResponse)
async def process_imu_reading(request: IMUReadingRequest):
    """
    Process IMU reading to extract meaningful information
    """
    try:
        # Calculate Euler angles from IMU data (simplified)
        # Roll (rotation around X axis)
        roll = np.arctan2(request.accel_y, request.accel_z)

        # Pitch (rotation around Y axis)
        pitch = np.arctan2(-request.accel_x, np.sqrt(request.accel_y**2 + request.accel_z**2))

        # Yaw calculation would require magnetometer data and more complex computation
        # For now, we'll use a simplified approach
        yaw = np.arctan2(request.mag_y, request.mag_x)

        response = IMUReadingResponse(
            euler_angles={"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
            linear_acceleration={"x": request.accel_x, "y": request.accel_y, "z": request.accel_z},
            angular_velocity={"x": request.gyro_x, "y": request.gyro_y, "z": request.gyro_z},
            processed_at=request.timestamp
        )

        logger.info(f"Processed IMU reading at timestamp {request.timestamp}")

        return response

    except Exception as e:
        logger.error(f"Error processing IMU reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing IMU reading: {str(e)}")

@router.post("/process-range", response_model=RangeReadingResponse)
async def process_range_reading(request: RangeReadingRequest):
    """
    Process range sensor reading
    """
    try:
        # Determine if obstacle is detected based on distance threshold
        obstacle_threshold = 1.0  # meters
        obstacle_detected = request.distance < obstacle_threshold and request.distance > 0

        # Apply corrections if needed
        corrected_distance = request.distance  # In real implementation, apply calibration

        response = RangeReadingResponse(
            corrected_distance=corrected_distance,
            obstacle_detected=obstacle_detected,
            confidence=request.confidence
        )

        logger.info(f"Processed range reading: {request.distance}m, obstacle detected: {obstacle_detected}")

        return response

    except Exception as e:
        logger.error(f"Error processing range reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing range reading: {str(e)}")

@router.post("/fuse-sensors", response_model=SensorFusionResponse)
async def fuse_sensors(request: SensorFusionRequest):
    """
    Fuse multiple sensor readings using specified method
    """
    try:
        if not request.sensor_readings:
            raise HTTPException(status_code=400, detail="No sensor readings provided")

        if request.fusion_method == "weighted_average":
            # Convert to internal SensorReading objects
            readings = [
                SensorReading(
                    value=r.value,
                    timestamp=r.timestamp,
                    confidence=r.confidence,
                    sensor_type=r.sensor_type
                )
                for r in request.sensor_readings
            ]

            fused_value, confidence = SensorFusion.fuse_sensors_simple(readings)

            response = SensorFusionResponse(
                fused_value=fused_value,
                confidence=confidence,
                method_used=request.fusion_method
            )

        elif request.fusion_method == "kalman":
            # For Kalman filter, we'll create a simple implementation
            # In practice, the state dimension would depend on what we're estimating
            fusion_filter = SensorFusion(state_dim=2, measurement_dim=len(request.sensor_readings))

            # Prepare measurements from sensor readings
            measurements = np.array([r.value for r in request.sensor_readings])

            # Perform prediction and update steps
            fusion_filter.predict()
            fusion_filter.update(measurements)

            state = fusion_filter.get_state()
            fused_value = float(state[0]) if len(state) > 0 else measurements[0]
            confidence = 0.9  # Placeholder confidence

            response = SensorFusionResponse(
                fused_value=fused_value,
                confidence=confidence,
                method_used=request.fusion_method
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown fusion method: {request.fusion_method}")

        logger.info(f"Fused {len(request.sensor_readings)} sensor readings using {request.fusion_method}")

        return response

    except Exception as e:
        logger.error(f"Error fusing sensors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fusing sensors: {str(e)}")

@router.post("/simulate-lidar", response_model=LIDARSimulationResponse)
async def simulate_lidar(request: LIDARSimulationRequest):
    """
    Simulate LIDAR sensor readings
    """
    try:
        if len(request.robot_pose) != 3:
            raise HTTPException(status_code=400, detail="Robot pose must have 3 elements [x, y, theta]")

        # Convert obstacles to tuples
        obstacles = [(obs[0], obs[1], obs[2]) for obs in request.obstacles]

        # Create temporary range readings
        temp_readings = RangeSensorSimulator.simulate_lidar_2d(
            request.angles,
            (request.robot_pose[0], request.robot_pose[1], request.robot_pose[2]),
            obstacles,
            request.noise_std
        )

        # Extract ranges and create point cloud
        ranges = [reading.distance for reading in temp_readings]

        point_cloud = []
        for i, (angle, distance) in enumerate(zip(request.angles, ranges)):
            if distance > 0 and distance < 10:  # Valid range
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                point_cloud.append({"x": float(x), "y": float(y), "z": 0.0, "angle": angle, "distance": distance})

        # Process statistics
        stats = process_lidar_scan(ranges, request.angles)

        response = LIDARSimulationResponse(
            ranges=ranges,
            angles=request.angles,
            point_cloud=point_cloud,
            processed_stats=stats
        )

        logger.info(f"Simulated LIDAR with {len(request.angles)} beams")

        return response

    except Exception as e:
        logger.error(f"Error simulating LIDAR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error simulating LIDAR: {str(e)}")

@router.post("/simulate-sensor-noise", response_model=SensorNoiseSimulationResponse)
async def simulate_sensor_noise(request: SensorNoiseSimulationRequest):
    """
    Simulate various types of sensor noise
    """
    try:
        if request.noise_type == "gaussian":
            std_dev = request.noise_params.get("std_dev", 0.1)
            noisy_value = SensorNoiseSimulator.add_gaussian_noise(request.value, std_dev)
        elif request.noise_type == "systematic":
            bias = request.noise_params.get("bias", 0.0)
            noisy_value = SensorNoiseSimulator.add_systematic_error(request.value, bias)
        elif request.noise_type == "outlier":
            probability = request.noise_params.get("probability", 0.01)
            max_error = request.noise_params.get("max_error", 1.0)
            noisy_value = SensorNoiseSimulator.generate_outlier(request.value, probability, max_error)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown noise type: {request.noise_type}")

        response = SensorNoiseSimulationResponse(
            noisy_value=noisy_value,
            original_value=request.value
        )

        logger.info(f"Applied {request.noise_type} noise to value {request.value}")

        return response

    except Exception as e:
        logger.error(f"Error simulating sensor noise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error simulating sensor noise: {str(e)}")

class IMUSimulationRequest(BaseModel):
    accel_true: List[float]  # [x, y, z]
    gyro_true: List[float]  # [x, y, z]
    noise_std: float = 0.01

@router.post("/simulate-imu", response_model=IMUReadingResponse)
async def simulate_imu(request: IMUSimulationRequest):
    """
    Simulate IMU sensor readings
    """
    try:
        if len(request.accel_true) != 3 or len(request.gyro_true) != 3:
            raise HTTPException(status_code=400, detail="Acceleration and gyro must have 3 elements each")

        # Convert to tuples
        accel_tuple = (request.accel_true[0], request.accel_true[1], request.accel_true[2])
        gyro_tuple = (request.gyro_true[0], request.gyro_true[1], request.gyro_true[2])

        # Simulate IMU reading
        imu_reading = IMUSimulator.simulate_imu(accel_tuple, gyro_tuple, request.noise_std)

        # Calculate Euler angles from the simulated reading
        roll = np.arctan2(imu_reading.accel_y, imu_reading.accel_z)
        pitch = np.arctan2(-imu_reading.accel_x, np.sqrt(imu_reading.accel_y**2 + imu_reading.accel_z**2))
        yaw = np.arctan2(imu_reading.mag_y, imu_reading.mag_x)

        response = IMUReadingResponse(
            euler_angles={"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
            linear_acceleration={"x": imu_reading.accel_x, "y": imu_reading.accel_y, "z": imu_reading.accel_z},
            angular_velocity={"x": imu_reading.gyro_x, "y": imu_reading.gyro_y, "z": imu_reading.gyro_z},
            processed_at=imu_reading.timestamp
        )

        logger.info("Simulated IMU reading with noise")

        return response

    except Exception as e:
        logger.error(f"Error simulating IMU: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error simulating IMU: {str(e)}")