---
sidebar_position: 2
---

# Sensors Integration

## Introduction to Sensor Integration

Sensor integration is a critical component of Physical AI and humanoid robotics systems, providing the perception capabilities necessary for intelligent interaction with the physical world. This section covers the integration of various sensor types, their calibration, data fusion techniques, and the implementation of sensor-based control systems that enable robots to perceive and respond to their environment effectively.

## Sensor Types and Classification

### Proprioceptive Sensors

Proprioceptive sensors provide information about the robot's own state and configuration.

#### Joint Position Sensors

```python
# joint_position_sensor.py
import numpy as np
from abc import ABC, abstractmethod
import time

class JointPositionSensor(ABC):
    """
    Abstract base class for joint position sensors
    """

    def __init__(self, joint_name, sensor_type, resolution=16):
        self.joint_name = joint_name
        self.sensor_type = sensor_type
        self.resolution = resolution
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0
        self.timestamp = time.time()

    @abstractmethod
    def read_position(self):
        """
        Read the current joint position
        """
        pass

    @abstractmethod
    def get_accuracy(self):
        """
        Get the accuracy of the sensor
        """
        pass

    def update_state(self):
        """
        Update the sensor state including position, velocity, and effort
        """
        self.position = self.read_position()
        self.velocity = self.calculate_velocity()
        self.effort = self.calculate_effort()
        self.timestamp = time.time()

    def calculate_velocity(self):
        """
        Calculate velocity based on position changes
        """
        # This would typically use a differentiation filter
        # For now, return a simple approximation
        return 0.0

    def calculate_effort(self):
        """
        Calculate effort based on motor current or other measurements
        """
        # This would typically be based on motor current or force sensors
        return 0.0


class EncoderSensor(JointPositionSensor):
    """
    Encoder-based joint position sensor
    """

    def __init__(self, joint_name, encoder_resolution=4096, gear_ratio=1.0):
        super().__init__(joint_name, 'encoder', encoder_resolution)
        self.encoder_resolution = encoder_resolution
        self.gear_ratio = gear_ratio
        self.raw_count = 0
        self.multi_turn_count = 0
        self.last_raw_count = 0

    def read_position(self):
        """
        Read position from encoder with multi-turn capability
        """
        raw_count = self.get_raw_encoder_count()

        # Handle encoder overflow/underflow
        if raw_count < self.last_raw_count:
            # Encoder wrapped around backwards
            if self.last_raw_count - raw_count > self.encoder_resolution / 2:
                self.multi_turn_count += 1
        elif raw_count > self.last_raw_count:
            # Encoder wrapped around forwards
            if raw_count - self.last_raw_count > self.encoder_resolution / 2:
                self.multi_turn_count -= 1

        self.raw_count = raw_count
        self.last_raw_count = raw_count

        # Calculate absolute position in radians
        encoder_position = (raw_count + self.multi_turn_count * self.encoder_resolution) / self.encoder_resolution
        joint_position = encoder_position * 2 * np.pi / self.gear_ratio

        return joint_position

    def get_raw_encoder_count(self):
        """
        Get raw encoder count (this would interface with actual hardware)
        """
        # In real implementation, this would read from hardware
        # For simulation, return a value based on expected position
        return int(np.random.uniform(0, self.encoder_resolution))

    def get_accuracy(self):
        """
        Get encoder accuracy based on resolution
        """
        return (2 * np.pi) / (self.encoder_resolution * self.gear_ratio)


class PotentiometerSensor(JointPositionSensor):
    """
    Potentiometer-based joint position sensor
    """

    def __init__(self, joint_name, min_voltage=0.0, max_voltage=3.3, gear_ratio=1.0):
        super().__init__(joint_name, 'potentiometer')
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.gear_ratio = gear_ratio
        self.voltage_range = max_voltage - min_voltage

    def read_position(self):
        """
        Read position from potentiometer voltage
        """
        voltage = self.get_voltage_reading()
        normalized = (voltage - self.min_voltage) / self.voltage_range
        position = normalized * 2 * np.pi / self.gear_ratio  # Convert to radians
        return position

    def get_voltage_reading(self):
        """
        Get voltage reading from potentiometer (simulated)
        """
        # In real implementation, this would read from ADC
        return np.random.uniform(self.min_voltage, self.max_voltage)

    def get_accuracy(self):
        """
        Get potentiometer accuracy
        """
        # Limited by ADC resolution and potentiometer linearity
        return 0.017  # Approximately 1 degree


class AbsoluteEncoderSensor(JointPositionSensor):
    """
    Absolute encoder sensor for position sensing
    """

    def __init__(self, joint_name, bits=12, gear_ratio=1.0):
        super().__init__(joint_name, 'absolute_encoder', bits)
        self.bits = bits
        self.max_count = 2**bits - 1
        self.gear_ratio = gear_ratio

    def read_position(self):
        """
        Read absolute position from encoder
        """
        raw_position = self.get_absolute_position()
        position = (raw_position / self.max_count) * 2 * np.pi / self.gear_ratio
        return position

    def get_absolute_position(self):
        """
        Get absolute position from encoder (0 to max_count)
        """
        # In real implementation, this would read from encoder hardware
        return np.random.randint(0, self.max_count + 1)

    def get_accuracy(self):
        """
        Get absolute encoder accuracy
        """
        return (2 * np.pi) / (self.max_count * self.gear_ratio)
```

### Inertial Measurement Units (IMUs)

```python
# imu_sensor.py
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class IMUSensor:
    """
    Inertial Measurement Unit sensor class
    """

    def __init__(self, name, update_rate=100, noise_density=None):
        self.name = name
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        self.noise_density = noise_density or {
            'accelerometer': 0.01,  # m/s^2/sqrt(Hz)
            'gyroscope': 0.001,     # rad/s/sqrt(Hz)
            'magnetometer': 0.01    # Tesla/sqrt(Hz)
        }

        # Bias and drift parameters
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.mag_bias = np.zeros(3)

        # Scale factor errors
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        self.mag_scale = np.ones(3)

        # Current measurements
        self.orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self.angular_velocity = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        self.magnetic_field = np.array([0.25, 0, 0.45])  # Earth's magnetic field

        # Timestamp
        self.timestamp = time.time()

    def update_measurements(self, true_state):
        """
        Update IMU measurements based on true robot state
        """
        current_time = time.time()
        dt = current_time - self.timestamp

        # Get true state from robot
        true_orientation = true_state.get('orientation', [0, 0, 0, 1])
        true_angular_velocity = true_state.get('angular_velocity', [0, 0, 0])
        true_linear_acceleration = true_state.get('linear_acceleration', [0, 0, -9.81])

        # Apply sensor model
        self.orientation = self._model_orientation(true_orientation)
        self.angular_velocity = self._model_angular_velocity(true_angular_velocity)
        self.linear_acceleration = self._model_linear_acceleration(true_linear_acceleration, true_orientation)
        self.magnetic_field = self._model_magnetic_field()

        self.timestamp = current_time

    def _model_orientation(self, true_orientation):
        """
        Model orientation measurements with drift and noise
        """
        # In practice, IMUs don't directly measure orientation
        # This would integrate gyroscope measurements
        # For now, return a slightly perturbed version
        noise = np.random.normal(0, 0.001, 4)  # Small orientation noise
        noisy_orientation = np.array(true_orientation) + noise
        # Normalize quaternion
        noisy_orientation = noisy_orientation / np.linalg.norm(noisy_orientation)
        return noisy_orientation

    def _model_angular_velocity(self, true_angular_velocity):
        """
        Model gyroscope measurements with bias, scale factor, and noise
        """
        true_av = np.array(true_angular_velocity)

        # Apply scale factor
        scaled_av = true_av * self.gyro_scale

        # Apply bias
        biased_av = scaled_av + self.gyro_bias

        # Add noise (considering bandwidth)
        noise_std = self.noise_density['gyroscope'] * np.sqrt(self.update_rate / 2)
        noise = np.random.normal(0, noise_std, 3)

        measured_av = biased_av + noise

        return measured_av

    def _model_linear_acceleration(self, true_linear_acceleration, true_orientation):
        """
        Model accelerometer measurements with bias, scale factor, and noise
        """
        true_la = np.array(true_linear_acceleration)

        # Apply scale factor
        scaled_la = true_la * self.accel_scale

        # Apply bias
        biased_la = scaled_la + self.accel_bias

        # Add noise
        noise_std = self.noise_density['accelerometer'] * np.sqrt(self.update_rate / 2)
        noise = np.random.normal(0, noise_std, 3)

        measured_la = biased_la + noise

        return measured_la

    def _model_magnetic_field(self):
        """
        Model magnetometer measurements
        """
        # Earth's magnetic field with local variations
        base_field = np.array([0.25, 0, 0.45])  # Approximate Earth's field in local area

        # Add local magnetic disturbances
        disturbance = np.random.normal(0, 0.01, 3)

        # Apply scale factor and bias
        scaled_field = (base_field + disturbance) * self.mag_scale + self.mag_bias

        # Add noise
        noise_std = self.noise_density['magnetometer'] * np.sqrt(self.update_rate / 2)
        noise = np.random.normal(0, noise_std, 3)

        measured_field = scaled_field + noise

        return measured_field

    def get_measurements(self):
        """
        Get current IMU measurements
        """
        return {
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'linear_acceleration': self.linear_acceleration.copy(),
            'magnetic_field': self.magnetic_field.copy(),
            'timestamp': self.timestamp
        }

    def calibrate(self, calibration_data):
        """
        Calibrate IMU using calibration data
        """
        # Calculate biases
        self.accel_bias = np.mean(calibration_data['accelerometer_static'], axis=0) - [0, 0, 9.81]
        self.gyro_bias = np.mean(calibration_data['gyroscope_static'], axis=0)
        self.mag_bias = np.mean(calibration_data['magnetometer_static'], axis=0)

        # Calculate scale factors (simplified calibration)
        # This would be more complex in practice
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        self.mag_scale = np.ones(3)

    def get_calibration_status(self):
        """
        Get current calibration status
        """
        return {
            'accelerometer_bias': self.accel_bias,
            'gyroscope_bias': self.gyro_bias,
            'magnetometer_bias': self.mag_bias,
            'accelerometer_scale': self.accel_scale,
            'gyroscope_scale': self.gyro_scale,
            'magnetometer_scale': self.mag_scale
        }


class AHRS(IMUSensor):
    """
    Attitude and Heading Reference System using IMU data
    """

    def __init__(self, name, update_rate=100):
        super().__init__(name, update_rate)
        self.orientation_filter = MadgwickAHRS(update_rate=update_rate)

    def update_orientation(self, accel, gyro, mag=None):
        """
        Update orientation estimate using IMU data
        """
        # Update AHRS filter
        self.orientation_filter.update_imu(gyro, accel)

        # If magnetometer data available, use it for heading correction
        if mag is not None:
            self.orientation_filter.update(gyro, accel, mag)

        # Get estimated orientation
        self.orientation = self.orientation_filter.get_quaternion()


class MadgwickAHRS:
    """
    Madgwick AHRS algorithm implementation
    """

    def __init__(self, update_rate=100, beta=0.1):
        self.update_rate = update_rate
        self.beta = beta  # Algorithm gain
        self.q = np.array([1, 0, 0, 0], dtype=float)  # Quaternion [w, x, y, z]
        self.inv_dt = update_rate

    def update_imu(self, gyro, accel):
        """
        Update orientation using gyroscope and accelerometer
        """
        # Convert measurements to numpy arrays
        gyroscope = np.array(gyro, dtype=float)
        accelerometer = np.array(accel, dtype=float)

        # Normalize accelerometer measurement
        if np.linalg.norm(accelerometer) != 0:
            accelerometer = accelerometer / np.linalg.norm(accelerometer)

        # Rate of change of quaternion from gyroscope
        qDot = self._quat_multiply(self.q, np.concatenate(([0], gyroscope))) * 0.5

        # Auxiliary variables to avoid repeated calculations
        _2q0 = 2 * self.q[0]
        _2q1 = 2 * self.q[1]
        _2q2 = 2 * self.q[2]
        _2q3 = 2 * self.q[3]
        _4q0 = 4 * self.q[0]
        _4q1 = 4 * self.q[1]
        _4q2 = 4 * self.q[2]
        _8q1 = 8 * self.q[1]
        _8q2 = 8 * self.q[2]
        q0q0 = self.q[0] * self.q[0]
        q1q1 = self.q[1] * self.q[1]
        q2q2 = self.q[2] * self.q[2]
        q3q3 = self.q[3] * self.q[3]

        # Gradient decent algorithm corrective step
        s = np.zeros(4)

        # Compute feedback only if accelerometer measurement valid (avoids NaN in gradient)
        if np.linalg.norm(accelerometer) != 0:
            # Auxiliary variables to avoid repeated arithmetic
            _2q0x = _2q0 * accelerometer[0]
            _2q0y = _2q0 * accelerometer[1]
            _2q0z = _2q0 * accelerometer[2]
            _2q1x = _2q1 * accelerometer[0]
            _2q1y = _2q1 * accelerometer[1]
            _2q1z = _2q1 * accelerometer[2]
            _2q2x = _2q2 * accelerometer[0]
            _2q2y = _2q2 * accelerometer[1]
            _2q2z = _2q2 * accelerometer[2]
            _2q3x = _2q3 * accelerometer[0]
            _2q3y = _2q3 * accelerometer[1]
            _2q3z = _2q3 * accelerometer[2]

            # Gradient decent algorithm corrective step
            s[0] = _4q0 * q2q2 + _2q2 * accelerometer[1] + _4q0 * q1q1 - _2q1 * accelerometer[0]
            s[1] = _2q1 * accelerometer[2] - _4q2 * accelerometer[1] + _4q1 * q3q3 - _2q3 * accelerometer[2]
            s[2] = _2q0 * accelerometer[2] - _4q3 * accelerometer[0] + _4q2 * q3q3 - _2q2 * accelerometer[2]
            s[3] = _4q3 * accelerometer[1] - _2q0 * accelerometer[2] + _4q3 * q1q1 - _2q1 * accelerometer[1]

            # Normalize step magnitude
            s_norm = np.linalg.norm(s)
            if s_norm != 0:
                s = s / s_norm

        # Compute rate of change of quaternion
        qDot = qDot - self.beta * s

        # Integrate to yield quaternion
        self.q = self.q + qDot / self.inv_dt

        # Normalize quaternion
        self.q = self.q / np.linalg.norm(self.q)

    def update(self, gyro, accel, mag):
        """
        Update orientation using gyroscope, accelerometer, and magnetometer
        """
        # This would include magnetometer correction
        # Simplified implementation uses only IMU
        self.update_imu(gyro, accel)

    def get_quaternion(self):
        """
        Get current quaternion estimate
        """
        return self.q.copy()

    def _quat_multiply(self, q1, q2):
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])
```

### Exteroceptive Sensors

#### Vision Systems

```python
# vision_sensor.py
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraSensor:
    """
    Camera sensor class for vision-based perception
    """

    def __init__(self, name, width=640, height=480, fov=60, fps=30):
        self.name = name
        self.width = width
        self.height = height
        self.fov = fov  # Field of view in degrees
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Camera intrinsic parameters
        self.fx = (width / 2) / np.tan(np.radians(fov / 2))
        self.fy = self.fx  # Assume square pixels
        self.cx = width / 2
        self.cy = height / 2

        # Distortion parameters (for now, assume no distortion)
        self.distortion_coeffs = np.zeros(5)

        # Sensor noise and parameters
        self.noise_std = 10.0  # Pixel noise standard deviation
        self.exposure_time = 1.0 / 60  # seconds
        self.gain = 1.0

        # Current image and timestamp
        self.current_image = None
        self.timestamp = time.time()

        # Initialize CV bridge for ROS integration
        self.cv_bridge = CvBridge()

    def capture_image(self, scene_data=None):
        """
        Capture image from camera
        """
        if scene_data is not None:
            # Generate image from scene data (for simulation)
            image = self._generate_image_from_scene(scene_data)
        else:
            # In real implementation, this would capture from actual camera
            image = self._simulate_image_capture()

        # Apply camera effects
        image = self._apply_camera_effects(image)

        # Update timestamp
        self.timestamp = time.time()
        self.current_image = image

        return image

    def _generate_image_from_scene(self, scene_data):
        """
        Generate image from 3D scene data (for simulation)
        """
        # This would typically use a 3D renderer
        # For now, create a placeholder image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add some simulated objects
        for obj in scene_data.get('objects', []):
            if obj.get('type') == 'object':
                # Draw bounding box
                bbox = obj.get('bbox', [100, 100, 200, 200])
                color = obj.get('color', (255, 0, 0))
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return image

    def _simulate_image_capture(self):
        """
        Simulate image capture with realistic effects
        """
        # Create a test image with some patterns
        image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        # Add some structured elements
        cv2.circle(image, (320, 240), 50, (0, 255, 0), -1)
        cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), 2)

        return image

    def _apply_camera_effects(self, image):
        """
        Apply realistic camera effects to image
        """
        # Add noise
        noise = np.random.normal(0, self.noise_std, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply gamma correction
        gamma = 1.0 + np.random.uniform(-0.1, 0.1)  # Small gamma variation
        gamma_corrected = np.power(noisy_image / 255.0, gamma) * 255.0
        corrected_image = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

        return corrected_image

    def get_intrinsic_matrix(self):
        """
        Get camera intrinsic matrix
        """
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        return K

    def undistort_image(self, image):
        """
        Undistort image using distortion coefficients
        """
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.get_intrinsic_matrix(),
            self.distortion_coeffs,
            (w, h),
            1,
            (w, h)
        )

        undistorted = cv2.undistort(
            image,
            self.get_intrinsic_matrix(),
            self.distortion_coeffs,
            None,
            newcameramtx
        )

        return undistorted

    def project_3d_to_2d(self, points_3d):
        """
        Project 3D points to 2D image coordinates
        """
        K = self.get_intrinsic_matrix()
        points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_2d_homo = K @ points_3d_homo.T
        points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
        return points_2d.T

    def get_ros_image_message(self):
        """
        Get ROS Image message from current image
        """
        if self.current_image is not None:
            return self.cv_bridge.cv2_to_imgmsg(self.current_image, encoding="bgr8")
        return None


class StereoCamera:
    """
    Stereo camera system for depth perception
    """

    def __init__(self, left_camera_config, right_camera_config, baseline=0.1):
        self.left_camera = CameraSensor(**left_camera_config)
        self.right_camera = CameraSensor(**right_camera_config)
        self.baseline = baseline  # Distance between cameras in meters

        # Epipolar geometry
        self.focal_length = self.left_camera.fx
        self.principal_point = (self.left_camera.cx, self.left_camera.cy)

    def capture_stereo_pair(self, scene_data=None):
        """
        Capture synchronized stereo image pair
        """
        left_image = self.left_camera.capture_image(scene_data)
        right_image = self.right_camera.capture_image(scene_data)

        return {
            'left': left_image,
            'right': right_image,
            'timestamp': time.time()
        }

    def compute_disparity_map(self, stereo_pair):
        """
        Compute disparity map from stereo images
        """
        left_gray = cv2.cvtColor(stereo_pair['left'], cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(stereo_pair['right'], cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        return disparity

    def compute_depth_map(self, stereo_pair):
        """
        Compute depth map from stereo images
        """
        disparity = self.compute_disparity_map(stereo_pair)

        # Convert disparity to depth: depth = (baseline * focal_length) / disparity
        depth_map = np.zeros_like(disparity)
        valid_disparity = disparity > 0
        depth_map[valid_disparity] = (self.baseline * self.focal_length) / disparity[valid_disparity]

        return depth_map

    def get_point_cloud(self, stereo_pair):
        """
        Generate 3D point cloud from stereo images
        """
        depth_map = self.compute_depth_map(stereo_pair)
        disparity_map = self.compute_disparity_map(stereo_pair)

        # Generate point cloud
        points = []
        colors = []

        for v in range(depth_map.shape[0]):
            for u in range(depth_map.shape[1]):
                depth = depth_map[v, u]
                if depth > 0:  # Valid depth
                    # Convert pixel coordinates to 3D
                    x = (u - self.principal_point[0]) * depth / self.focal_length
                    y = (v - self.principal_point[1]) * depth / self.focal_length
                    z = depth

                    points.append([x, y, z])
                    colors.append(stereo_pair['left'][v, u])

        return np.array(points), np.array(colors)


class DepthCamera:
    """
    Depth camera sensor (RGB-D)
    """

    def __init__(self, name, width=640, height=480, fov=60, fps=30, min_depth=0.1, max_depth=10.0):
        self.rgb_camera = CameraSensor(name + "_rgb", width, height, fov, fps)
        self.depth_camera = CameraSensor(name + "_depth", width, height, fov, fps)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_noise_std = 0.01  # meters

    def capture_rgbd(self, scene_data=None):
        """
        Capture RGB-D image pair
        """
        rgb_image = self.rgb_camera.capture_image(scene_data)
        depth_map = self._generate_depth_map(scene_data)

        return {
            'rgb': rgb_image,
            'depth': depth_map,
            'timestamp': time.time()
        }

    def _generate_depth_map(self, scene_data):
        """
        Generate depth map from scene data
        """
        if scene_data is not None:
            # Generate depth from 3D scene
            depth_map = np.full((self.rgb_camera.height, self.rgb_camera.width), self.max_depth, dtype=np.float32)

            for obj in scene_data.get('objects', []):
                if 'position' in obj and 'size' in obj:
                    # Calculate depth based on object position
                    obj_pos = obj['position']
                    obj_size = obj['size']

                    # For now, just set depth to object distance
                    depth = np.sqrt(obj_pos[0]**2 + obj_pos[1]**2 + obj_pos[2]**2)

                    # Add some noise
                    depth_with_noise = depth + np.random.normal(0, self.depth_noise_std)

                    # Update depth map in the area of the object
                    # This is a simplified implementation
                    center_u = int(self.rgb_camera.cx)
                    center_v = int(self.rgb_camera.cy)

                    # Set depth in a region around the center
                    for dv in range(-20, 21):
                        for du in range(-20, 21):
                            v, u = center_v + dv, center_u + du
                            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                                depth_map[v, u] = min(depth_with_noise, depth_map[v, u])

            return depth_map

        # Simulated depth map
        depth_map = np.random.uniform(self.min_depth, self.max_depth,
                                     (self.rgb_camera.height, self.rgb_camera.width)).astype(np.float32)
        return depth_map

    def convert_depth_to_point_cloud(self, depth_map):
        """
        Convert depth map to 3D point cloud
        """
        K = self.rgb_camera.get_intrinsic_matrix()

        # Generate coordinate grids
        v, u = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]

        # Convert to homogeneous coordinates
        uv_homogeneous = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)

        # Invert intrinsic matrix
        K_inv = np.linalg.inv(K)

        # Get normalized coordinates
        normalized_coords = (K_inv @ uv_homogeneous.T).T

        # Get depths
        depths = depth_map.flatten()

        # Compute 3D points
        points_3d = normalized_coords * depths[:, np.newaxis]

        # Reshape back
        points_3d = points_3d.reshape(depth_map.shape[0], depth_map.shape[1], 3)

        return points_3d

    def get_field_of_view(self):
        """
        Get camera field of view in radians
        """
        return np.radians(self.rgb_camera.fov)
```

#### Range Sensors

```python
# range_sensors.py
import numpy as np
import time

class LIDARSensor:
    """
    LIDAR sensor class for 3D range sensing
    """

    def __init__(self, name, horizontal_fov=360, vertical_fov=30,
                 horizontal_resolution=0.1, vertical_resolution=2.0,
                 min_range=0.1, max_range=25.0, update_rate=10):
        self.name = name
        self.horizontal_fov = horizontal_fov  # degrees
        self.vertical_fov = vertical_fov      # degrees
        self.horizontal_resolution = horizontal_resolution  # degrees
        self.vertical_resolution = vertical_resolution      # degrees
        self.min_range = min_range
        self.max_range = max_range
        self.update_rate = update_rate

        # Calculate number of beams
        self.horizontal_beams = int(horizontal_fov / horizontal_resolution)
        self.vertical_beams = int(vertical_fov / vertical_resolution)

        # Sensor parameters
        self.noise_std = 0.02  # meters
        self.bias_error = 0.01  # meters

        # Current scan data
        self.ranges = np.full((self.vertical_beams, self.horizontal_beams), np.inf)
        self.intensities = np.zeros((self.vertical_beams, self.horizontal_beams))
        self.timestamp = time.time()

        # Angular arrays
        self.angles_horizontal = np.linspace(
            -horizontal_fov/2, horizontal_fov/2, self.horizontal_beams
        )
        self.angles_vertical = np.linspace(
            -vertical_fov/2, vertical_fov/2, self.vertical_beams
        )

    def scan_environment(self, environment_model=None):
        """
        Perform LIDAR scan of environment
        """
        if environment_model is not None:
            # Simulate scan based on environment model
            self.ranges, self.intensities = self._simulate_scan(environment_model)
        else:
            # Simulate random environment
            self.ranges = np.random.uniform(
                self.min_range, self.max_range,
                (self.vertical_beams, self.horizontal_beams)
            )
            self.intensities = np.random.uniform(0, 1000,
                                               (self.vertical_beams, self.horizontal_beams))

        # Add sensor noise and bias
        self._apply_sensor_errors()

        self.timestamp = time.time()

        return self._get_scan_data()

    def _simulate_scan(self, environment_model):
        """
        Simulate LIDAR scan based on environment model
        """
        ranges = np.full((self.vertical_beams, self.horizontal_beams), self.max_range)
        intensities = np.zeros((self.vertical_beams, self.horizontal_beams))

        # For each beam, cast ray and find intersections
        for v_idx, v_angle in enumerate(self.angles_vertical):
            for h_idx, h_angle in enumerate(self.angles_horizontal):
                # Convert angles to 3D direction vector
                h_rad = np.radians(h_angle)
                v_rad = np.radians(v_angle)

                # Calculate direction vector
                direction = np.array([
                    np.cos(v_rad) * np.cos(h_rad),
                    np.cos(v_rad) * np.sin(h_rad),
                    np.sin(v_rad)
                ])

                # Cast ray and find nearest intersection
                distance, intensity = self._ray_intersect(environment_model, direction)

                ranges[v_idx, h_idx] = distance
                intensities[v_idx, h_idx] = intensity

        return ranges, intensities

    def _ray_intersect(self, environment_model, direction):
        """
        Perform ray intersection with environment model
        """
        # Simplified ray intersection - in real implementation,
        # this would use proper 3D collision detection
        ray_origin = np.array([0, 0, 0])  # LIDAR is at origin
        ray_direction = direction / np.linalg.norm(direction)

        # For simulation, return distance based on some simple geometry
        # This would be replaced with actual environment intersection logic
        distance = self.max_range * 0.5  # Default distance
        intensity = 500  # Default intensity

        # Add some variation based on direction
        distance += np.random.normal(0, 0.1)
        intensity += np.random.normal(0, 50)

        # Clamp to sensor range
        distance = np.clip(distance, self.min_range, self.max_range)

        return distance, intensity

    def _apply_sensor_errors(self):
        """
        Apply sensor noise and bias errors to measurements
        """
        # Add random noise
        noise = np.random.normal(0, self.noise_std, self.ranges.shape)

        # Add bias error
        bias = self.bias_error

        # Apply errors
        self.ranges = np.clip(self.ranges + noise + bias,
                             self.min_range, self.max_range)

    def _get_scan_data(self):
        """
        Get current scan data
        """
        return {
            'ranges': self.ranges.copy(),
            'intensities': self.intensities.copy(),
            'timestamp': self.timestamp,
            'horizontal_angles': self.angles_horizontal.copy(),
            'vertical_angles': self.angles_vertical.copy()
        }

    def get_point_cloud(self):
        """
        Convert LIDAR scan to 3D point cloud
        """
        points = []
        intensities = []

        for v_idx, v_angle in enumerate(self.angles_vertical):
            for h_idx, h_angle in enumerate(self.angles_horizontal):
                range_val = self.ranges[v_idx, h_idx]

                if range_val < self.max_range:  # Valid measurement
                    # Convert polar to Cartesian coordinates
                    h_rad = np.radians(h_angle)
                    v_rad = np.radians(v_angle)

                    x = range_val * np.cos(v_rad) * np.cos(h_rad)
                    y = range_val * np.cos(v_rad) * np.sin(h_rad)
                    z = range_val * np.sin(v_rad)

                    points.append([x, y, z])
                    intensities.append(self.intensities[v_idx, h_idx])

        return np.array(points), np.array(intensities)

    def get_laser_scan_msg(self):
        """
        Get ROS LaserScan message format
        """
        # This would convert to ROS message format
        # For now, return a dictionary with similar structure
        return {
            'angle_min': np.radians(-self.horizontal_fov/2),
            'angle_max': np.radians(self.horizontal_fov/2),
            'angle_increment': np.radians(self.horizontal_resolution),
            'time_increment': 1.0/(self.horizontal_beams * self.update_rate),
            'scan_time': 1.0/self.update_rate,
            'range_min': self.min_range,
            'range_max': self.max_range,
            'ranges': self.ranges[self.vertical_beams//2, :].tolist(),  # Horizontal slice
            'intensities': self.intensities[self.vertical_beams//2, :].tolist()
        }


class UltrasonicSensor:
    """
    Ultrasonic sensor for proximity detection
    """

    def __init__(self, name, max_range=4.0, update_rate=10, beam_angle=30):
        self.name = name
        self.max_range = max_range
        self.update_rate = update_rate
        self.beam_angle = beam_angle  # Full cone angle in degrees
        self.noise_std = 0.05  # meters

        # Current measurement
        self.distance = max_range
        self.timestamp = time.time()

    def measure_distance(self, environment_model=None):
        """
        Measure distance to nearest obstacle
        """
        if environment_model is not None:
            # Simulate measurement based on environment
            self.distance = self._simulate_measurement(environment_model)
        else:
            # Simulate random distance
            self.distance = np.random.uniform(0.1, self.max_range)

        # Add noise
        noise = np.random.normal(0, self.noise_std)
        self.distance = np.clip(self.distance + noise, 0.01, self.max_range)

        self.timestamp = time.time()

        return self.distance

    def _simulate_measurement(self, environment_model):
        """
        Simulate ultrasonic measurement
        """
        # Simplified simulation - in real implementation,
        # this would check for obstacles in the sensor's cone
        distance = self.max_range * 0.3  # Default distance

        # Add some variation
        distance += np.random.normal(0, 0.05)

        return np.clip(distance, 0.01, self.max_range)

    def is_object_detected(self, threshold=1.0):
        """
        Check if object is detected within threshold
        """
        return self.distance < threshold


class InfraredSensor:
    """
    Infrared sensor for proximity detection
    """

    def __init__(self, name, max_range=1.0, update_rate=20, response_curve=None):
        self.name = name
        self.max_range = max_range
        self.update_rate = update_rate

        # Response curve (distance vs. signal strength)
        if response_curve is None:
            # Default inverse square response
            self.response_curve = lambda d: 1.0 / (d**2 + 0.1) if d > 0 else float('inf')
        else:
            self.response_curve = response_curve

        # Current measurement
        self.signal_strength = 0.0
        self.distance = max_range
        self.timestamp = time.time()

    def measure_proximity(self, environment_model=None):
        """
        Measure proximity using infrared sensor
        """
        if environment_model is not None:
            # Simulate measurement based on environment
            distance = self._simulate_ir_measurement(environment_model)
        else:
            # Simulate random distance
            distance = np.random.uniform(0.05, self.max_range)

        # Calculate signal strength based on distance
        signal_strength = self.response_curve(distance)

        # Add noise
        noise = np.random.normal(0, 0.05)
        signal_strength = max(0, signal_strength + noise)

        # Convert back to distance estimate
        estimated_distance = self._signal_to_distance(signal_strength)

        self.signal_strength = signal_strength
        self.distance = estimated_distance
        self.timestamp = time.time()

        return self.distance, self.signal_strength

    def _simulate_ir_measurement(self, environment_model):
        """
        Simulate infrared measurement
        """
        # In real implementation, this would calculate based on
        # infrared reflection properties
        distance = self.max_range * 0.2  # Default distance
        distance += np.random.normal(0, 0.02)
        return np.clip(distance, 0.01, self.max_range)

    def _signal_to_distance(self, signal_strength):
        """
        Convert signal strength to distance estimate
        """
        # Simplified conversion - real implementation would be more complex
        # based on the sensor's specific characteristics
        if signal_strength > 0:
            # Inverse of response curve
            return np.sqrt(max(0, 1.0 / signal_strength - 0.1))
        else:
            return self.max_range


class TactileSensor:
    """
    Tactile sensor for contact detection and force measurement
    """

    def __init__(self, name, sensor_array_size=(8, 8), max_force=50.0, update_rate=100):
        self.name = name
        self.sensor_array_size = sensor_array_size
        self.max_force = max_force
        self.update_rate = update_rate

        # Initialize sensor array
        self.force_array = np.zeros(sensor_array_size)
        self.contact_array = np.zeros(sensor_array_size, dtype=bool)
        self.temperature_array = np.zeros(sensor_array_size) + 25.0  # Room temperature

        # Noise parameters
        self.force_noise_std = 0.1  # Newtons
        self.temperature_noise_std = 0.5  # Celsius

        # Timestamp
        self.timestamp = time.time()

    def sense_contact(self, contact_data=None):
        """
        Sense contact and forces
        """
        if contact_data is not None:
            # Update based on contact simulation
            self._update_from_contact_data(contact_data)
        else:
            # Simulate random contact
            self.force_array = np.random.uniform(0, 1, self.sensor_array_size) * self.max_force
            self.contact_array = self.force_array > 0.1
            self.temperature_array = 25.0 + np.random.normal(0, self.temperature_noise_std, self.sensor_array_size)

        # Add noise
        noise = np.random.normal(0, self.force_noise_std, self.force_array.shape)
        self.force_array = np.clip(self.force_array + noise, 0, self.max_force)

        self.timestamp = time.time()

        return self._get_tactile_data()

    def _update_from_contact_data(self, contact_data):
        """
        Update tactile sensor from contact simulation data
        """
        # This would integrate contact forces from physics simulation
        # For now, use simplified model
        contact_positions = contact_data.get('contact_positions', [])
        contact_forces = contact_data.get('contact_forces', [])

        # Reset arrays
        self.force_array = np.zeros(self.sensor_array_size)
        self.contact_array = np.zeros(self.sensor_array_size, dtype=bool)

        # Distribute contact forces to sensor array
        for pos, force in zip(contact_positions, contact_forces):
            # Map contact position to sensor array coordinates
            sensor_x = int(pos[0] * self.sensor_array_size[0])
            sensor_y = int(pos[1] * self.sensor_array_size[1])

            if (0 <= sensor_x < self.sensor_array_size[0] and
                0 <= sensor_y < self.sensor_array_size[1]):
                self.force_array[sensor_x, sensor_y] = min(force, self.max_force)
                self.contact_array[sensor_x, sensor_y] = True

    def _get_tactile_data(self):
        """
        Get current tactile sensor data
        """
        return {
            'force_array': self.force_array.copy(),
            'contact_array': self.contact_array.copy(),
            'temperature_array': self.temperature_array.copy(),
            'timestamp': self.timestamp,
            'total_force': np.sum(self.force_array),
            'contact_points': np.sum(self.contact_array)
        }

    def get_contact_locations(self):
        """
        Get locations of contact points
        """
        contact_y, contact_x = np.where(self.contact_array)
        return list(zip(contact_x, contact_y))

    def get_max_force_location(self):
        """
        Get location of maximum force
        """
        max_idx = np.unravel_index(np.argmax(self.force_array), self.force_array.shape)
        return max_idx[1], max_idx[0]  # Return as (x, y)
```

## Sensor Fusion Techniques

### Kalman Filter Implementation

```python
# sensor_fusion.py
import numpy as np
from scipy.linalg import block_diag

class KalmanFilter:
    """
    Kalman Filter for sensor fusion
    """

    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        # For position, velocity, and acceleration
        self.x = np.zeros(state_dim)  # State vector
        self.P = np.eye(state_dim) * 1000  # Covariance matrix
        self.Q = np.eye(state_dim) * 0.1  # Process noise
        self.R = np.eye(measurement_dim) * 1.0  # Measurement noise

        # Control input matrix (if applicable)
        self.B = np.zeros((state_dim, 1)) if state_dim > 0 else np.array([])

    def predict(self, u=None, F=None, Q=None):
        """
        Prediction step of Kalman Filter
        """
        if F is None:
            # Default state transition matrix (constant velocity model)
            dt = 0.01  # Time step - in real implementation, this would come from timing
            F = self._get_state_transition_matrix(dt)

        if Q is None:
            Q = self.Q

        # Predict state
        if u is not None and self.B.size > 0:
            self.x = F @ self.x + self.B @ u
        else:
            self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H=None, R=None):
        """
        Update step of Kalman Filter
        """
        if H is None:
            # Default measurement matrix
            H = self._get_measurement_matrix()

        if R is None:
            R = self.R

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    def _get_state_transition_matrix(self, dt):
        """
        Get state transition matrix for constant velocity model
        """
        F = np.eye(self.state_dim)

        if self.state_dim >= 6:  # At least position and velocity
            # Position-velocity relationships
            F[0, 3] = dt  # x += vx*dt
            F[1, 4] = dt  # y += vy*dt
            F[2, 5] = dt  # z += vz*dt

            if self.state_dim >= 9:  # Include acceleration
                # Velocity-acceleration relationships
                F[3, 6] = dt  # vx += ax*dt
                F[4, 7] = dt  # vy += ay*dt
                F[5, 8] = dt  # vz += az*dt

        return F

    def _get_measurement_matrix(self):
        """
        Get measurement matrix
        """
        # For now, assume we measure position directly
        H = np.zeros((self.measurement_dim, self.state_dim))
        for i in range(min(self.measurement_dim, self.state_dim)):
            H[i, i] = 1.0  # Direct measurement of first measurement_dim states

        return H

    def get_state(self):
        """
        Get current state estimate
        """
        return self.x.copy()

    def get_covariance(self):
        """
        Get current covariance estimate
        """
        return self.P.copy()


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems
    """

    def __init__(self, state_dim, measurement_dim):
        super().__init__(state_dim, measurement_dim)
        self.process_function = None
        self.measurement_function = None

    def set_process_model(self, f, F_jacobian):
        """
        Set nonlinear process model and its Jacobian
        """
        self.process_function = f
        self.F_jacobian = F_jacobian

    def set_measurement_model(self, h, H_jacobian):
        """
        Set nonlinear measurement model and its Jacobian
        """
        self.measurement_function = h
        self.H_jacobian = H_jacobian

    def predict(self, u=None, dt=0.01):
        """
        Prediction step for EKF
        """
        if self.process_function is None:
            raise ValueError("Process function not set")

        # Predict state using nonlinear model
        self.x = self.process_function(self.x, u, dt)

        # Compute Jacobian
        F = self.F_jacobian(self.x, u, dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update step for EKF
        """
        if self.measurement_function is None:
            raise ValueError("Measurement function not set")

        # Compute measurement prediction
        h_x = self.measurement_function(self.x)

        # Compute Jacobian
        H = self.H_jacobian(self.x)

        # Innovation
        y = z - h_x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P


class MultiSensorFusion:
    """
    Multi-sensor fusion system
    """

    def __init__(self):
        self.sensors = {}
        self.kalman_filter = None
        self.fusion_data = {}
        self.is_initialized = False

    def add_sensor(self, sensor_name, sensor_type, measurement_indices):
        """
        Add a sensor to the fusion system
        """
        self.sensors[sensor_name] = {
            'type': sensor_type,
            'measurement_indices': measurement_indices,
            'last_measurement': None,
            'measurement_time': 0
        }

    def initialize_filter(self, state_dim):
        """
        Initialize the Kalman filter for fusion
        """
        # For position, velocity, and orientation
        measurement_dim = sum(
            len(sensor['measurement_indices']) for sensor in self.sensors.values()
        )

        self.kalman_filter = KalmanFilter(state_dim, measurement_dim)
        self.is_initialized = True

    def fuse_sensor_data(self, sensor_measurements):
        """
        Fuse data from multiple sensors
        """
        if not self.is_initialized:
            raise ValueError("Filter not initialized")

        # Prepare combined measurement vector
        z_combined = []
        R_combined = []

        for sensor_name, measurement in sensor_measurements.items():
            if sensor_name in self.sensors:
                sensor_info = self.sensors[sensor_name]

                # Add measurements to combined vector
                if isinstance(measurement, (list, tuple, np.ndarray)):
                    z_combined.extend(measurement)
                else:
                    z_combined.append(measurement)

                # Add noise covariances
                if hasattr(measurement, 'shape'):
                    R_combined.extend([sensor_info.get('noise_covariance', 1.0)] * len(measurement))
                else:
                    R_combined.append(sensor_info.get('noise_covariance', 1.0))

                # Update sensor timestamp
                sensor_info['last_measurement'] = measurement
                sensor_info['measurement_time'] = time.time()

        if z_combined:
            z = np.array(z_combined)
            R = np.diag(R_combined)

            # Update Kalman filter
            self.kalman_filter.R = R
            self.kalman_filter.update(z)

        return self.kalman_filter.get_state()

    def predict_state(self, dt=0.01):
        """
        Predict state using filter
        """
        if self.kalman_filter:
            self.kalman_filter.predict()
            return self.kalman_filter.get_state()
        return None

    def get_fusion_status(self):
        """
        Get status of sensor fusion system
        """
        return {
            'sensors': list(self.sensors.keys()),
            'initialized': self.is_initialized,
            'last_state': self.kalman_filter.get_state() if self.kalman_filter else None
        }


class ParticleFilter:
    """
    Particle Filter for nonlinear/non-Gaussian systems
    """

    def __init__(self, state_dim, num_particles=1000):
        self.state_dim = state_dim
        self.num_particles = num_particles

        # Initialize particles
        self.particles = np.random.randn(num_particles, state_dim) * 10
        self.weights = np.ones(num_particles) / num_particles

        # Process noise
        self.process_noise = np.eye(state_dim) * 0.1

    def predict(self, control_input=None, dt=0.01):
        """
        Predict particle states
        """
        for i in range(self.num_particles):
            # Apply motion model with noise
            motion = self._motion_model(self.particles[i], control_input, dt)
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise)
            self.particles[i] = motion + noise

    def update(self, measurement, measurement_function, measurement_noise):
        """
        Update particle weights based on measurement
        """
        for i in range(self.num_particles):
            # Predict measurement for this particle
            predicted_measurement = measurement_function(self.particles[i])

            # Calculate likelihood of actual measurement given particle
            diff = measurement - predicted_measurement
            likelihood = self._gaussian_likelihood(diff, measurement_noise)

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid numerical issues
        self.weights /= np.sum(self.weights)

        # Resample if effective sample size is too low
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        if effective_sample_size < self.num_particles / 2.0:
            self._resample()

    def _motion_model(self, state, control, dt):
        """
        Simple motion model (constant velocity)
        """
        new_state = state.copy()
        if len(state) >= 6:  # Has position and velocity
            new_state[0:3] += state[3:6] * dt  # Update position
            # Velocity remains the same (or apply control)
            if control is not None and len(control) >= 3:
                new_state[3:6] += control * dt
        return new_state

    def _gaussian_likelihood(self, diff, covariance):
        """
        Calculate Gaussian likelihood
        """
        inv_cov = np.linalg.inv(covariance)
        exponent = -0.5 * diff.T @ inv_cov @ diff
        norm = 1.0 / np.sqrt((2 * np.pi)**len(diff) * np.linalg.det(covariance))
        return norm * np.exp(exponent)

    def _resample(self):
        """
        Resample particles based on weights
        """
        # Systematic resampling
        indices = []
        cumulative_sum = np.cumsum(self.weights)
        start = np.random.uniform(0, 1.0 / self.num_particles)
        i, j = 0, 0
        while i < self.num_particles:
            if start + i * (1.0 / self.num_particles) <= cumulative_sum[j]:
                indices.append(j)
                i += 1
            else:
                j += 1

        # Resample particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_state(self):
        """
        Estimate state as weighted average of particles
        """
        return np.average(self.particles, axis=0, weights=self.weights)

    def get_particles(self):
        """
        Get current particles
        """
        return self.particles.copy()

    def get_weights(self):
        """
        Get current weights
        """
        return self.weights.copy()
```

## Sensor Calibration

### Calibration Procedures

```python
# sensor_calibration.py
import numpy as np
from scipy.optimize import minimize
import cv2

class SensorCalibrator:
    """
    Generic sensor calibrator
    """

    def __init__(self):
        self.calibration_data = []
        self.calibration_parameters = {}
        self.is_calibrated = False

    def collect_calibration_data(self, measurements, ground_truth):
        """
        Collect calibration data pairs
        """
        self.calibration_data.append({
            'measurements': measurements,
            'ground_truth': ground_truth
        })

    def calibrate(self):
        """
        Perform calibration optimization
        """
        if len(self.calibration_data) < 10:
            raise ValueError("Need at least 10 calibration data points")

        # Define objective function to minimize
        def objective(params):
            total_error = 0
            for data_point in self.calibration_data:
                corrected_measurement = self._apply_correction(
                    data_point['measurements'], params
                )
                error = np.sum((corrected_measurement - data_point['ground_truth'])**2)
                total_error += error
            return total_error

        # Initial parameter guess
        initial_params = self._get_initial_parameters()

        # Optimize parameters
        result = minimize(objective, initial_params, method='BFGS')

        if result.success:
            self.calibration_parameters = result.x
            self.is_calibrated = True
            return True
        else:
            return False

    def _apply_correction(self, measurements, params):
        """
        Apply correction based on parameters
        """
        # This would be specific to the sensor type
        # For now, return identity transformation
        return measurements

    def _get_initial_parameters(self):
        """
        Get initial parameter guess
        """
        # This would be specific to the sensor type
        return np.zeros(6)  # Placeholder

    def correct_measurement(self, measurement):
        """
        Correct a measurement using calibration parameters
        """
        if not self.is_calibrated:
            raise ValueError("Sensor not calibrated")

        return self._apply_correction(measurement, self.calibration_parameters)


class CameraCalibrator(SensorCalibrator):
    """
    Camera calibration using chessboard patterns
    """

    def __init__(self, pattern_size=(9, 6), square_size=0.025):
        super().__init__()
        self.pattern_size = pattern_size  # Number of inner corners
        self.square_size = square_size   # Size of chessboard squares in meters
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane

    def detect_chessboard(self, image):
        """
        Detect chessboard pattern in image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            return ret, corners_refined
        else:
            return ret, None

    def add_calibration_image(self, image):
        """
        Add calibration image to dataset
        """
        ret, corners = self.detect_chessboard(image)

        if ret:
            # Prepare object points (3D points of chessboard corners)
            objp = np.zeros((np.prod(self.pattern_size), 3), dtype=np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size

            self.object_points.append(objp)
            self.image_points.append(corners)

            return True
        else:
            return False

    def calibrate_camera(self):
        """
        Perform camera calibration
        """
        if len(self.object_points) < 10:
            raise ValueError("Need at least 10 calibration images")

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points,
            (640, 480), None, None  # Image size - adjust as needed
        )

        if ret:
            self.calibration_parameters = {
                'camera_matrix': mtx,
                'distortion_coefficients': dist,
                'rotation_vectors': rvecs,
                'translation_vectors': tvecs
            }
            self.is_calibrated = True

            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.object_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error

            avg_error = total_error / len(self.object_points)
            print(f"Average reprojection error: {avg_error}")

            return True, avg_error
        else:
            return False, 0.0

    def undistort_image(self, image):
        """
        Undistort image using calibration parameters
        """
        if not self.is_calibrated:
            raise ValueError("Camera not calibrated")

        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.calibration_parameters['camera_matrix'],
            self.calibration_parameters['distortion_coefficients'],
            (w, h), 1, (w, h)
        )

        # Undistort
        dst = cv2.undistort(
            image,
            self.calibration_parameters['camera_matrix'],
            self.calibration_parameters['distortion_coefficients'],
            None,
            newcameramtx
        )

        # Crop the image
        x, y, w, h = roi
        if roi:
            dst = dst[y:y+h, x:x+w]

        return dst

    def get_camera_intrinsics(self):
        """
        Get camera intrinsic parameters
        """
        if self.is_calibrated:
            return self.calibration_parameters['camera_matrix']
        else:
            return None

    def get_distortion_coefficients(self):
        """
        Get camera distortion coefficients
        """
        if self.is_calibrated:
            return self.calibration_parameters['distortion_coefficients']
        else:
            return None


class IMUCalibrator(SensorCalibrator):
    """
    IMU calibrator for bias and scale factor correction
    """

    def __init__(self):
        super().__init__()
        self.static_measurements = []
        self.rotating_measurements = []

    def collect_static_data(self, accel_data, gyro_data, mag_data=None):
        """
        Collect static calibration data (IMU at rest)
        """
        self.static_measurements.append({
            'accel': accel_data,
            'gyro': gyro_data,
            'mag': mag_data
        })

    def collect_rotating_data(self, measurements_sequence):
        """
        Collect rotating calibration data (IMU rotated through various orientations)
        """
        self.rotating_measurements.append(measurements_sequence)

    def calibrate_imu(self):
        """
        Calibrate IMU biases and scale factors
        """
        if len(self.static_measurements) < 10:
            raise ValueError("Need sufficient static measurements for calibration")

        # Calculate accelerometer bias (should measure gravity when static)
        avg_accel = np.mean([m['accel'] for m in self.static_measurements], axis=0)
        gravity_magnitude = 9.81
        measured_gravity_mag = np.linalg.norm(avg_accel)

        # Calculate scale factor
        scale_factor = gravity_magnitude / measured_gravity_mag if measured_gravity_mag != 0 else 1.0

        # Accelerometer bias (assuming z-axis should measure gravity)
        bias = avg_accel - np.array([0, 0, gravity_magnitude])

        # Gyroscope bias (should be zero when static)
        avg_gyro = np.mean([m['gyro'] for m in self.static_measurements], axis=0)

        # Magnetometer calibration (hard iron and soft iron)
        if self.static_measurements[0]['mag'] is not None:
            hard_iron_bias, soft_iron_matrix = self._calibrate_magnetometer()

        self.calibration_parameters = {
            'accel_bias': bias,
            'accel_scale': scale_factor,
            'gyro_bias': avg_gyro,
            'gyro_scale': 1.0,  # Typically close to 1.0
            'mag_hard_iron_bias': hard_iron_bias if 'hard_iron_bias' in locals() else np.zeros(3),
            'mag_soft_iron_matrix': soft_iron_matrix if 'soft_iron_matrix' in locals() else np.eye(3)
        }

        self.is_calibrated = True
        return True

    def _calibrate_magnetometer(self):
        """
        Calibrate magnetometer for hard and soft iron effects
        """
        # Collect all magnetometer measurements
        all_mag_data = []
        for seq in self.rotating_measurements:
            for meas in seq:
                if meas['mag'] is not None:
                    all_mag_data.append(meas['mag'])

        if not all_mag_data:
            return np.zeros(3), np.eye(3)

        mag_data = np.array(all_mag_data)

        # Simple sphere fitting for hard iron bias
        # More sophisticated methods exist for soft iron correction
        center = np.mean(mag_data, axis=0)
        radius = np.mean(np.linalg.norm(mag_data - center, axis=1))

        # Hard iron bias is the offset
        hard_iron_bias = center

        # Soft iron correction matrix (simplified)
        # In practice, this would use more sophisticated ellipsoid fitting
        soft_iron_matrix = np.eye(3)

        return hard_iron_bias, soft_iron_matrix

    def correct_imu_data(self, accel, gyro, mag=None):
        """
        Correct IMU measurements using calibration parameters
        """
        if not self.is_calibrated:
            raise ValueError("IMU not calibrated")

        # Correct accelerometer
        corrected_accel = (np.array(accel) - self.calibration_parameters['accel_bias']) * self.calibration_parameters['accel_scale']

        # Correct gyroscope
        corrected_gyro = np.array(gyro) - self.calibration_parameters['gyro_bias']

        # Correct magnetometer if available
        corrected_mag = None
        if mag is not None:
            raw_mag = np.array(mag) - self.calibration_parameters['mag_hard_iron_bias']
            corrected_mag = self.calibration_parameters['mag_soft_iron_matrix'] @ raw_mag

        return corrected_accel, corrected_gyro, corrected_mag


class LIDARCalibrator(SensorCalibrator):
    """
    LIDAR calibrator for extrinsic parameter calibration
    """

    def __init__(self):
        super().__init__()
        self.lidar_poses = []  # LIDAR poses in robot coordinate frame
        self.correspondences = []  # Point correspondences

    def collect_calibration_scan(self, lidar_scan, robot_pose, static_features):
        """
        Collect calibration data with known static features
        """
        self.lidar_poses.append(robot_pose)
        self.correspondences.append({
            'lidar_scan': lidar_scan,
            'static_features': static_features,
            'robot_pose': robot_pose
        })

    def calibrate_extrinsics(self):
        """
        Calibrate LIDAR extrinsic parameters (position and orientation relative to robot)
        """
        if len(self.correspondences) < 3:
            raise ValueError("Need at least 3 calibration scans")

        # Use point-to-plane ICP or similar method
        # This is a simplified approach - real implementation would be more sophisticated
        lidar_to_robot_translation = np.zeros(3)
        lidar_to_robot_rotation = np.eye(3)

        # Calculate average offset
        for corr in self.correspondences:
            # Match LIDAR points to known features
            # Calculate transformation that best aligns them
            pass

        self.calibration_parameters = {
            'translation': lidar_to_robot_translation,
            'rotation_matrix': lidar_to_robot_rotation,
            'rotation_quaternion': self._matrix_to_quaternion(lidar_to_robot_rotation)
        }

        self.is_calibrated = True
        return True

    def _matrix_to_quaternion(self, rotation_matrix):
        """
        Convert rotation matrix to quaternion
        """
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def transform_lidar_scan(self, scan_data, robot_pose):
        """
        Transform LIDAR scan to robot coordinate frame using calibration
        """
        if not self.is_calibrated:
            raise ValueError("LIDAR not calibrated")

        # Apply extrinsic transformation
        translation = self.calibration_parameters['translation']
        rotation = self.calibration_parameters['rotation_matrix']

        # Transform each point in the scan
        transformed_scan = []
        for point in scan_data:
            # Apply rotation and translation
            world_point = rotation @ point + translation
            # Also account for robot pose
            robot_world_point = self._transform_to_world_frame(world_point, robot_pose)
            transformed_scan.append(robot_world_point)

        return transformed_scan

    def _transform_to_world_frame(self, local_point, robot_pose):
        """
        Transform point from robot frame to world frame
        """
        # Apply robot pose transformation
        # This would use the robot's current position and orientation
        return local_point  # Placeholder
```

## Week Summary

This section covered comprehensive sensor integration techniques for Physical AI and humanoid robotics systems. We explored various sensor types including proprioceptive (encoders, IMUs) and exteroceptive (cameras, LIDAR, tactile) sensors, their modeling, and integration approaches. The content included advanced topics like sensor fusion using Kalman filters and particle filters, calibration procedures for different sensor types, and optimization techniques for efficient sensor data processing. These techniques are essential for creating robust perception systems that enable robots to understand and interact with their physical environment effectively.