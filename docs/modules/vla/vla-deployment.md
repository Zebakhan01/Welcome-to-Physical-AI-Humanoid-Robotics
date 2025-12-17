---
sidebar_position: 5
---

# VLA Deployment

## Introduction to VLA Deployment

Deploying Vision-Language-Action (VLA) systems in real-world robotic applications presents unique challenges that go beyond training. The deployment process must consider computational constraints, safety requirements, real-time performance, and integration with existing robotic systems. This section covers the practical aspects of deploying VLA systems, from model optimization to integration with robotic platforms.

## Deployment Architecture

### Edge Deployment Considerations

Deploying VLA systems on edge devices requires careful optimization of computational resources:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import time

class VLADeploymentManager:
    """Manager for deploying VLA systems on edge devices"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Performance monitoring
        self.inference_times = []
        self.memory_usage = []

    def load_optimized_model(self):
        """Load and optimize model for deployment"""
        # Load model
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()

        # Apply optimizations
        if self.config.get('use_tensorrt', False):
            self.model = self._optimize_with_tensorrt()
        elif self.config.get('use_openvino', False):
            self.model = self._optimize_with_openvino()
        else:
            # Apply PyTorch optimizations
            self.model = self._optimize_pytorch()

        print(f"Model loaded and optimized for {self.device}")

    def _optimize_pytorch(self):
        """Apply PyTorch optimizations"""
        # Convert to TorchScript if specified
        if self.config.get('use_torchscript', False):
            self.model = torch.jit.script(self.model)

        # Apply quantization if specified
        if self.config.get('quantize', False):
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        return self.model

    def _optimize_with_tensorrt(self):
        """Optimize model with TensorRT (if available)"""
        try:
            import tensorrt as trt
            from torch2trt import torch2trt

            # Example optimization (simplified)
            dummy_input = {
                'image': torch.randn(1, 3, 224, 224).to(self.device),
                'instruction': ["dummy instruction"]
            }

            optimized_model = torch2trt(
                self.model,
                [dummy_input['image']],
                fp16_mode=self.config.get('fp16', False)
            )
            return optimized_model
        except ImportError:
            print("TensorRT not available, using PyTorch optimizations")
            return self._optimize_pytorch()

    def _optimize_with_openvino(self):
        """Optimize model with OpenVINO (if available)"""
        try:
            from openvino.runtime import Core

            # Convert PyTorch model to OpenVINO IR format
            # This is a simplified example
            return self.model
        except ImportError:
            print("OpenVINO not available, using PyTorch optimizations")
            return self._optimize_pytorch()

    def run_inference(self,
                     image: torch.Tensor,
                     instruction: str,
                     robot_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run inference with performance monitoring"""
        start_time = time.time()

        # Prepare inputs
        image = image.to(self.device).unsqueeze(0)  # Add batch dimension
        instructions = [instruction]

        # Run inference
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            outputs = self.model(image, instructions)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        inference_time = time.time() - start_time

        # Store performance metrics
        self.inference_times.append(inference_time)
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.memory_allocated())

        return {
            'actions': outputs['actions'].cpu(),
            'inference_time': inference_time,
            'success': True
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0
        }
```

### Real-time Inference Pipeline

```python
import threading
import queue
from collections import deque

class VLARealTimePipeline:
    """Real-time inference pipeline for VLA systems"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Input queues for different modalities
        self.image_queue = queue.Queue(maxsize=config.get('image_queue_size', 5))
        self.instruction_queue = queue.Queue(maxsize=config.get('instruction_queue_size', 10))
        self.state_queue = queue.Queue(maxsize=config.get('state_queue_size', 10))

        # Output queue for actions
        self.action_queue = queue.Queue(maxsize=config.get('action_queue_size', 5))

        # Threading for parallel processing
        self.processing_thread = threading.Thread(target=self._process_pipeline, daemon=True)
        self.running = False

        # Performance buffers
        self.fps_buffer = deque(maxlen=100)
        self.latency_buffer = deque(maxlen=100)

    def start_pipeline(self):
        """Start the real-time pipeline"""
        self.running = True
        self.processing_thread.start()

    def stop_pipeline(self):
        """Stop the real-time pipeline"""
        self.running = False
        self.processing_thread.join()

    def _process_pipeline(self):
        """Main processing pipeline"""
        while self.running:
            try:
                # Get latest inputs (non-blocking)
                image = self._get_latest_image()
                instruction = self._get_latest_instruction()
                state = self._get_latest_state()

                if image is not None and instruction is not None:
                    # Run inference
                    start_time = time.time()
                    action = self._run_inference(image, instruction, state)
                    latency = time.time() - start_time

                    # Put action in output queue
                    try:
                        self.action_queue.put_nowait(action)
                    except queue.Full:
                        # Drop oldest action if queue is full
                        try:
                            self.action_queue.get_nowait()
                            self.action_queue.put_nowait(action)
                        except queue.Empty:
                            pass

                    # Update performance metrics
                    self.latency_buffer.append(latency)
                    fps = 1.0 / latency if latency > 0 else 0
                    self.fps_buffer.append(fps)

            except Exception as e:
                print(f"Pipeline processing error: {e}")
                time.sleep(0.001)  # Small delay to prevent busy waiting

    def _get_latest_image(self):
        """Get the latest image from queue"""
        latest_image = None
        try:
            while not self.image_queue.empty():
                latest_image = self.image_queue.get_nowait()
        except queue.Empty:
            pass
        return latest_image

    def _get_latest_instruction(self):
        """Get the latest instruction from queue"""
        latest_instruction = None
        try:
            while not self.instruction_queue.empty():
                latest_instruction = self.instruction_queue.get_nowait()
        except queue.Empty:
            pass
        return latest_instruction

    def _get_latest_state(self):
        """Get the latest state from queue"""
        latest_state = None
        try:
            while not self.state_queue.empty():
                latest_state = self.state_queue.get_nowait()
        except queue.Empty:
            pass
        return latest_state

    def _run_inference(self, image, instruction, state):
        """Run VLA inference"""
        with torch.no_grad():
            image_tensor = image.to(self.device).unsqueeze(0)
            outputs = self.model(image_tensor, [instruction])

        return outputs['actions'].cpu()[0]

    def get_action(self, timeout=0.1) -> Dict[str, Any]:
        """Get the next action from the pipeline"""
        try:
            action = self.action_queue.get(timeout=timeout)
            return {
                'action': action,
                'timestamp': time.time(),
                'success': True
            }
        except queue.Empty:
            return {
                'action': None,
                'timestamp': time.time(),
                'success': False,
                'message': 'No action available'
            }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get real-time performance metrics"""
        if not self.fps_buffer or not self.latency_buffer:
            return {'fps': 0, 'avg_latency': 0, 'min_latency': 0, 'max_latency': 0}

        return {
            'fps': float(np.mean(self.fps_buffer)),
            'avg_latency': float(np.mean(self.latency_buffer)),
            'min_latency': float(np.min(self.latency_buffer)),
            'max_latency': float(np.max(self.latency_buffer)),
            'latency_std': float(np.std(self.latency_buffer))
        }
```

## Safety and Reliability Systems

### Safety Layer Implementation

```python
import numpy as np
from typing import Dict, Any, List

class VLASafetyLayer:
    """Safety layer for VLA systems"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_thresholds = config.get('safety_thresholds', {})
        self.emergency_stop = False
        self.safety_history = []

    def check_action_safety(self,
                           action: torch.Tensor,
                           robot_state: Dict[str, Any],
                           environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action is safe to execute"""
        safety_checks = {
            'collision_check': self._check_collision(action, environment_state),
            'joint_limit_check': self._check_joint_limits(action, robot_state),
            'velocity_check': self._check_velocity(action, robot_state),
            'force_check': self._check_force(action, robot_state),
            'workspace_check': self._check_workspace(action, robot_state)
        }

        # Overall safety decision
        is_safe = all(check['safe'] for check in safety_checks.values())

        safety_result = {
            'is_safe': is_safe,
            'checks': safety_checks,
            'modified_action': action if is_safe else self._safe_fallback_action(action),
            'emergency_stop': self._should_emergency_stop(safety_checks)
        }

        # Log safety decision
        self.safety_history.append(safety_result)

        return safety_result

    def _check_collision(self, action: torch.Tensor, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential collisions"""
        # Predict next state based on action
        predicted_state = self._predict_state(action, env_state['robot_state'])

        # Check collision with environment
        collision_objects = env_state.get('collision_objects', [])
        collision_distance = self._compute_collision_distance(predicted_state, collision_objects)

        is_safe = collision_distance > self.safety_thresholds.get('collision_distance', 0.1)

        return {
            'safe': is_safe,
            'collision_distance': collision_distance,
            'message': 'Collision detected' if not is_safe else 'No collision'
        }

    def _check_joint_limits(self, action: torch.Tensor, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check joint limits"""
        current_joints = robot_state.get('joint_positions', [])
        target_joints = current_joints + action[:len(current_joints)]  # Assuming action is delta

        joint_limits = robot_state.get('joint_limits', {})
        min_limits = joint_limits.get('min', [])
        max_limits = joint_limits.get('max', [])

        is_safe = True
        violations = []

        for i, (pos, min_lim, max_lim) in enumerate(zip(target_joints, min_limits, max_limits)):
            if pos < min_lim or pos > max_lim:
                is_safe = False
                violations.append(f'Joint {i}: {pos} not in [{min_lim}, {max_lim}]')

        return {
            'safe': is_safe,
            'violations': violations,
            'message': f'Joint limit violations: {violations}' if not is_safe else 'Joint limits OK'
        }

    def _check_velocity(self, action: torch.Tensor, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check velocity constraints"""
        max_velocity = self.safety_thresholds.get('max_velocity', 1.0)

        # Calculate velocity from action (simplified)
        velocity = torch.norm(action) / 0.1  # Assuming 0.1s control cycle

        is_safe = velocity <= max_velocity

        return {
            'safe': is_safe,
            'velocity': velocity.item(),
            'max_velocity': max_velocity,
            'message': f'Velocity {velocity:.2f} > max {max_velocity}' if not is_safe else 'Velocity OK'
        }

    def _safe_fallback_action(self, action: torch.Tensor) -> torch.Tensor:
        """Generate safe fallback action"""
        # For now, return zero action (stop)
        # In practice, this could be a more sophisticated fallback
        return torch.zeros_like(action)

    def _should_emergency_stop(self, safety_checks: Dict[str, Dict]) -> bool:
        """Determine if emergency stop is needed"""
        critical_violations = [
            'collision_check',  # Collision is critical
            'joint_limit_check'  # Joint limit violation is critical
        ]

        for check_name in critical_violations:
            if not safety_checks[check_name]['safe']:
                return True

        return False

    def _predict_state(self, action: torch.Tensor, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict robot state after action"""
        # Simplified state prediction
        # In practice, this would use robot kinematics/dynamics
        predicted_state = current_state.copy()

        # Update positions based on action
        if 'joint_positions' in predicted_state:
            predicted_state['joint_positions'] = (
                np.array(predicted_state['joint_positions']) +
                action[:len(predicted_state['joint_positions'])].numpy()
            ).tolist()

        return predicted_state

    def _compute_collision_distance(self, state: Dict[str, Any], objects: List[Dict[str, Any]]) -> float:
        """Compute minimum distance to collision objects"""
        robot_pos = state.get('position', [0, 0, 0])
        min_distance = float('inf')

        for obj in objects:
            obj_pos = obj.get('position', [0, 0, 0])
            distance = np.linalg.norm(np.array(robot_pos) - np.array(obj_pos))
            min_distance = min(min_distance, distance)

        return min_distance
```

## Integration with Robotic Platforms

### ROS Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

class VLAROSBridge(Node):
    """ROS bridge for VLA system integration"""

    def __init__(self, vla_model, config):
        super().__init__('vla_ros_bridge')

        self.vla_model = vla_model
        self.config = config
        self.bridge = CvBridge()

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.instruction_sub = self.create_subscription(
            String, '/vla_instruction', self.instruction_callback, 10)

        # Internal state
        self.current_image = None
        self.current_instruction = None
        self.inference_timer = self.create_timer(0.1, self.inference_callback)

        self.get_logger().info('VLA ROS Bridge initialized')

    def image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def instruction_callback(self, msg: String):
        """Handle incoming instructions"""
        self.current_instruction = msg.data
        self.get_logger().info(f'New instruction received: {msg.data}')

    def inference_callback(self):
        """Run VLA inference at regular intervals"""
        if self.current_image is not None and self.current_instruction is not None:
            try:
                # Convert image to tensor
                image_tensor = self._preprocess_image(self.current_image)

                # Run VLA inference
                with torch.no_grad():
                    outputs = self.vla_model(
                        image_tensor.unsqueeze(0),
                        [self.current_instruction]
                    )

                # Convert to ROS message
                cmd_msg = self._convert_to_twist(outputs['actions'][0])
                self.cmd_pub.publish(cmd_msg)

                # Publish status
                status_msg = String()
                status_msg.data = 'Inference completed successfully'
                self.status_pub.publish(status_msg)

            except Exception as e:
                self.get_logger().error(f'Inference error: {e}')
                status_msg = String()
                status_msg.data = f'Inference error: {e}'
                self.status_pub.publish(status_msg)

    def _preprocess_image(self, cv_image):
        """Preprocess image for VLA model"""
        import cv2
        # Resize and normalize image
        resized = cv2.resize(cv_image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(normalized).permute(2, 0, 1)
        return tensor_image

    def _convert_to_twist(self, action_tensor):
        """Convert action tensor to ROS Twist message"""
        from geometry_msgs.msg import Twist

        cmd = Twist()
        cmd.linear.x = float(action_tensor[0])  # Forward/backward
        cmd.linear.y = float(action_tensor[1])  # Left/right
        cmd.linear.z = float(action_tensor[2])  # Up/down
        cmd.angular.x = float(action_tensor[3])  # Roll
        cmd.angular.y = float(action_tensor[4])  # Pitch
        cmd.angular.z = float(action_tensor[5])  # Yaw

        return cmd

def main(args=None):
    rclpy.init(args=args)

    # Load VLA model (simplified)
    vla_model = torch.load('path/to/vla_model.pth')
    config = {}  # Load from config file

    vla_bridge = VLAROSBridge(vla_model, config)

    try:
        rclpy.spin(vla_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        vla_bridge.destroy_node()
        rclpy.shutdown()
```

### Hardware Integration

```python
class VLAHardwareInterface:
    """Hardware interface for VLA system deployment"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hardware_interfaces = {}
        self.safety_layer = VLASafetyLayer(config)

        # Initialize hardware components
        self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize hardware components"""
        # Initialize camera
        if self.config.get('camera_enabled', True):
            self.hardware_interfaces['camera'] = self._init_camera()

        # Initialize robot controller
        if self.config.get('robot_enabled', True):
            self.hardware_interfaces['robot'] = self._init_robot_controller()

        # Initialize sensors
        if self.config.get('sensors_enabled', True):
            self.hardware_interfaces['sensors'] = self._init_sensors()

    def _init_camera(self):
        """Initialize camera interface"""
        camera_type = self.config.get('camera_type', 'usb')

        if camera_type == 'usb':
            import cv2
            camera = cv2.VideoCapture(self.config.get('camera_id', 0))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return camera
        elif camera_type == 'realsense':
            try:
                import pyrealsense2 as rs
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(config)
                return pipeline
            except ImportError:
                raise Exception("RealSense camera requested but pyrealsense2 not installed")

        return None

    def _init_robot_controller(self):
        """Initialize robot controller interface"""
        robot_type = self.config.get('robot_type', 'simulated')

        if robot_type == 'franka':
            # Initialize Franka robot controller
            try:
                import franka_interface
                return franka_interface.RobotInterface()
            except ImportError:
                raise Exception("Franka robot requested but franka_interface not installed")
        elif robot_type == 'ur5':
            # Initialize Universal Robots controller
            try:
                import urx
                return urx.Robot(self.config.get('robot_ip', '192.168.1.100'))
            except ImportError:
                raise Exception("UR5 robot requested but urx not installed")
        else:
            # Simulated robot
            return SimulatedRobotController()

    def get_sensor_data(self) -> Dict[str, Any]:
        """Get data from all sensors"""
        sensor_data = {}

        # Get camera image
        if 'camera' in self.hardware_interfaces:
            sensor_data['image'] = self._get_camera_image()

        # Get robot state
        if 'robot' in self.hardware_interfaces:
            sensor_data['robot_state'] = self.hardware_interfaces['robot'].get_state()

        # Get additional sensors
        if 'sensors' in self.hardware_interfaces:
            sensor_data.update(self.hardware_interfaces['sensors'].get_all_data())

        return sensor_data

    def _get_camera_image(self):
        """Get image from camera"""
        if 'camera' not in self.hardware_interfaces:
            return None

        camera = self.hardware_interfaces['camera']

        if hasattr(camera, 'read'):  # USB camera
            ret, frame = camera.read()
            if ret:
                return frame
        elif hasattr(camera, 'wait_for_frames'):  # RealSense
            frames = camera.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                import numpy as np
                return np.asanyarray(color_frame.get_data())

        return None

    def execute_action(self, action: torch.Tensor, instruction: str = "") -> Dict[str, Any]:
        """Execute action on hardware with safety checks"""
        # Get current robot state
        robot_state = self.hardware_interfaces['robot'].get_state()

        # Get environment state
        env_state = self.get_environment_state()

        # Check action safety
        safety_result = self.safety_layer.check_action_safety(
            action, robot_state, env_state
        )

        if safety_result['is_safe']:
            # Execute safe action
            execution_result = self.hardware_interfaces['robot'].execute_action(
                safety_result['modified_action']
            )

            return {
                'success': True,
                'execution_result': execution_result,
                'safety_result': safety_result
            }
        else:
            # Action is unsafe, trigger emergency procedures
            if safety_result['emergency_stop']:
                self._emergency_stop()

            return {
                'success': False,
                'safety_result': safety_result,
                'message': 'Action blocked by safety system'
            }

    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        env_state = {
            'robot_state': self.hardware_interfaces['robot'].get_state(),
            'collision_objects': self._detect_collision_objects(),
            'workspace_bounds': self.config.get('workspace_bounds', {})
        }
        return env_state

    def _detect_collision_objects(self) -> List[Dict[str, Any]]:
        """Detect potential collision objects"""
        # This would use perception system to detect objects
        # For now, return empty list
        return []

    def _emergency_stop(self):
        """Execute emergency stop procedure"""
        if 'robot' in self.hardware_interfaces:
            self.hardware_interfaces['robot'].emergency_stop()

class SimulatedRobotController:
    """Simulated robot controller for testing"""

    def __init__(self):
        self.state = {
            'joint_positions': [0.0] * 7,
            'joint_velocities': [0.0] * 7,
            'end_effector_pose': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return self.state

    def execute_action(self, action: torch.Tensor) -> Dict[str, Any]:
        """Execute action in simulation"""
        # Update state based on action (simplified)
        for i in range(min(len(self.state['joint_positions']), len(action))):
            self.state['joint_positions'][i] += float(action[i]) * 0.01  # Small step

        return {'success': True, 'new_state': self.state}

    def emergency_stop(self):
        """Emergency stop in simulation"""
        print("Emergency stop executed in simulation")
```

## Monitoring and Maintenance

### System Monitoring

```python
import psutil
import GPUtil
from datetime import datetime
import json

class VLAMonitoringSystem:
    """Monitoring system for deployed VLA systems"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []
        self.alerts = []

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

        # GPU metrics if available
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assuming single GPU
            metrics.update({
                'gpu_usage': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature
            })

        # Add custom metrics from VLA pipeline
        if hasattr(self, 'vla_pipeline'):
            pipeline_metrics = self.vla_pipeline.get_performance_metrics()
            metrics.update(pipeline_metrics)

        self.metrics_history.append(metrics)

        # Check for alerts
        self._check_alerts(metrics)

        return metrics

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for system alerts based on metrics"""
        thresholds = self.config.get('monitoring_thresholds', {})

        if metrics.get('cpu_usage', 0) > thresholds.get('cpu_threshold', 90):
            self._add_alert('CPU usage high', 'cpu_high', metrics['cpu_usage'])

        if metrics.get('memory_usage', 0) > thresholds.get('memory_threshold', 90):
            self._add_alert('Memory usage high', 'memory_high', metrics['memory_usage'])

        if metrics.get('gpu_usage', 0) > thresholds.get('gpu_threshold', 95):
            self._add_alert('GPU usage high', 'gpu_high', metrics['gpu_usage'])

        if metrics.get('avg_latency', 0) > thresholds.get('latency_threshold', 0.1):
            self._add_alert('High inference latency', 'latency_high', metrics['avg_latency'])

    def _add_alert(self, message: str, alert_type: str, value: float):
        """Add alert to monitoring system"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'value': value
        }
        self.alerts.append(alert)

        # Log alert
        print(f"ALERT: {message} (value: {value})")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []

        if not recent_metrics:
            return {'status': 'unknown', 'metrics': {}}

        avg_metrics = {}
        for key in recent_metrics[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in recent_metrics if key in m]
                if values:
                    avg_metrics[key] = sum(values) / len(values)

        # Determine overall health
        health_score = self._calculate_health_score(avg_metrics)

        if health_score >= 0.8:
            status = 'healthy'
        elif health_score >= 0.6:
            status = 'warning'
        else:
            status = 'critical'

        return {
            'status': status,
            'health_score': health_score,
            'average_metrics': avg_metrics,
            'active_alerts': len([a for a in self.alerts if self._is_recent_alert(a)])
        }

    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate system health score from metrics"""
        score = 1.0

        # CPU health (lower is better)
        cpu_usage = metrics.get('cpu_usage', 0)
        if cpu_usage > 80:
            score *= (1 - (cpu_usage - 80) / 20)

        # Memory health
        memory_usage = metrics.get('memory_usage', 0)
        if memory_usage > 85:
            score *= (1 - (memory_usage - 85) / 15)

        # GPU health
        gpu_usage = metrics.get('gpu_usage', 0)
        if gpu_usage and gpu_usage > 90:
            score *= (1 - (gpu_usage - 90) / 10)

        # Latency health
        avg_latency = metrics.get('avg_latency', 0)
        if avg_latency > 0.2:  # 200ms threshold
            score *= (1 - min(0.5, (avg_latency - 0.2) / 0.3))

        return max(0.0, min(1.0, score))

    def _is_recent_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is recent (within last hour)"""
        from datetime import datetime, timedelta
        alert_time = datetime.fromisoformat(alert['timestamp'])
        return datetime.now() - alert_time < timedelta(hours=1)

    def save_metrics_report(self, filename: str):
        """Save metrics report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(self.metrics_history),
            'total_alerts': len(self.alerts),
            'system_health': self.get_system_health(),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'metrics_history': self.metrics_history[-100:]  # Last 100 metrics
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Metrics report saved to {filename}")
```

## Week Summary

This section covered VLA deployment considerations, including edge deployment optimizations, real-time inference pipelines, safety systems, hardware integration, and monitoring solutions. Deploying VLA systems requires careful attention to computational constraints, safety requirements, and integration with existing robotic platforms to ensure reliable and safe operation in real-world environments.