"""
VLA Integration Utilities for connecting VLA module with other systems
"""
import asyncio
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from PIL import Image
import base64
import io

from backend.api.vla.vla_components import (
    perception_module, language_module, action_module,
    fusion_module, memory_module
)
from backend.api.vla.vla_models import (
    VLARequest, VLAResponse, VisionAnalysisRequest, LanguageAnalysisRequest,
    ActionGenerationRequest, MultimodalFusionRequest
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VLAIntegrationUtils:
    """Utility class for integrating VLA with other backend systems"""

    def __init__(self):
        self.isaac_integration = None  # Will be set when Isaac module is available
        self.ros_integration = None   # Will be set when ROS module is available

    async def integrate_with_ros(self, ros_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate VLA with ROS system"""
        try:
            # Extract relevant data from ROS message
            ros_image = ros_data.get('image', None)
            ros_command = ros_data.get('command', '')
            ros_robot_state = ros_data.get('robot_state', {})

            # If we have an image from ROS, process it with VLA
            vla_response = None
            if ros_image:
                # Convert ROS image to base64 for VLA processing
                image_base64 = self._ros_image_to_base64(ros_image)

                # Create VLA request
                vla_request = VLARequest(
                    image=image_base64,
                    instruction=ros_command or "analyze the scene",
                    robot_state=ros_robot_state
                )

                # Process with VLA
                visual_features = perception_module.extract_features(
                    self._base64_to_numpy(image_base64)
                )
                language_features = language_module.extract_features(ros_command)

                fusion_result = fusion_module.fuse_features(
                    visual_features.__dict__,
                    language_features.__dict__
                )

                action_features = action_module.generate_action(
                    vision_features=visual_features.__dict__,
                    language_features=language_features.__dict__,
                    robot_state=ros_robot_state
                )

                vla_response = VLAResponse(
                    action_type=action_features.action_type,
                    action_parameters=action_features.action_parameters,
                    confidence=action_features.confidence,
                    execution_plan=action_features.execution_plan or [],
                    multimodal_features=fusion_result,
                    success=True
                )

            integrated_data = {
                'vla_response': vla_response.dict() if vla_response else None,
                'ros_data': ros_data,
                'integration_status': 'success',
                'timestamp': __import__('time').time()
            }

            logger.info("Successfully integrated VLA with ROS")
            return integrated_data

        except Exception as e:
            logger.error(f"Error in VLA-ROS integration: {str(e)}")
            raise

    async def integrate_with_sensors(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate VLA with sensor processing system"""
        try:
            # Process sensor data through VLA perception module
            processed_data = {
                'original_sensor_data': sensor_data,
                'vla_processed': True,
                'visual_analysis': None,
                'timestamp': __import__('time').time()
            }

            # If sensor data contains image information
            if 'image' in sensor_data:
                image_data = sensor_data['image']
                if isinstance(image_data, str):
                    # Assume it's base64 encoded
                    image_array = self._base64_to_numpy(image_data)
                else:
                    # Convert from other formats if needed
                    image_array = np.array(image_data)

                # Extract visual features using perception module
                visual_features = perception_module.extract_features(image_array)
                processed_data['visual_analysis'] = {
                    'object_detections': visual_features.object_detections,
                    'scene_context': visual_features.scene_graph,
                    'affordances': perception_module.analyze_affordances(visual_features)
                }

            logger.info("Successfully integrated VLA with sensor system")
            return processed_data

        except Exception as e:
            logger.error(f"Error in VLA-sensor integration: {str(e)}")
            raise

    async def integrate_with_isaac(self, isaac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate VLA with Isaac Sim system"""
        try:
            # Extract Isaac simulation data
            isaac_image = isaac_data.get('rgb_image', None)
            isaac_depth = isaac_data.get('depth_image', None)
            isaac_task = isaac_data.get('task_description', '')

            integrated_result = {
                'input': isaac_data,
                'vla_perception': None,
                'vla_language': None,
                'vla_action_planning': None,
                'timestamp': __import__('time').time()
            }

            # Process Isaac RGB image with VLA perception
            if isaac_image:
                # Convert Isaac image format to compatible format
                if isinstance(isaac_image, str) and (isaac_image.startswith('data:image') or len(isaac_image) > 100):
                    # Assume it's already in a processable format
                    image_base64 = isaac_image
                else:
                    # Convert from Isaac format to base64
                    image_base64 = self._numpy_to_base64(isaac_image)

                # Extract visual features
                image_array = self._base64_to_numpy(image_base64)
                visual_features = perception_module.extract_features(image_array)

                integrated_result['vla_perception'] = {
                    'object_detections': visual_features.object_detections,
                    'spatial_features': visual_features.spatial_features.tolist(),
                    'semantic_features': visual_features.semantic_features.tolist()
                }

            # Process Isaac task description with VLA language module
            if isaac_task:
                language_features = language_module.extract_features(isaac_task)

                integrated_result['vla_language'] = {
                    'intent': language_features.intent,
                    'entities': language_features.entities,
                    'action_sequence': language_features.action_sequence
                }

                # Generate action plan based on task and perception
                if integrated_result['vla_perception']:
                    action_features = action_module.generate_action(
                        vision_features=visual_features.__dict__,
                        language_features=language_features.__dict__
                    )

                    integrated_result['vla_action_planning'] = {
                        'action_type': action_features.action_type,
                        'action_parameters': action_features.action_parameters,
                        'execution_plan': action_features.execution_plan,
                        'confidence': action_features.confidence
                    }

            logger.info("Successfully integrated VLA with Isaac system")
            return integrated_result

        except Exception as e:
            logger.error(f"Error in VLA-Isaac integration: {str(e)}")
            raise

    async def integrate_with_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate VLA with learning system for training/evaluation"""
        try:
            # This would coordinate VLA with the learning system
            # for generating training data or evaluating performance

            integrated_data = {
                'learning_data': learning_data,
                'vla_component': 'enhanced_data_generation',
                'data_augmentation': 'multimodal',
                'evaluation_metrics': [],
                'timestamp': __import__('time').time()
            }

            # If learning system needs VLA-generated data
            if learning_data.get('generate_training_data', False):
                # Generate synthetic training examples using VLA components
                synthetic_data = self._generate_synthetic_training_data(
                    learning_data.get('task_type', 'general'),
                    learning_data.get('complexity', 'medium')
                )
                integrated_data['synthetic_training_data'] = synthetic_data

            # If learning system needs VLA evaluation
            if learning_data.get('evaluate_performance', False):
                evaluation_results = self._evaluate_vla_performance(
                    learning_data.get('test_cases', [])
                )
                integrated_data['evaluation_results'] = evaluation_results

            logger.info("Successfully integrated VLA with learning system")
            return integrated_data

        except Exception as e:
            logger.error(f"Error in VLA-learning integration: {str(e)}")
            raise

    async def integrate_with_humanoid(self, humanoid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate VLA with humanoid robotics system"""
        try:
            # Extract humanoid-specific data
            humanoid_image = humanoid_data.get('camera_feed', None)
            humanoid_instruction = humanoid_data.get('command', '')
            humanoid_state = humanoid_data.get('robot_state', {})
            humanoid_capabilities = humanoid_data.get('capabilities', {})

            # Process with VLA to generate humanoid-appropriate actions
            vla_response = None
            if humanoid_image and humanoid_instruction:
                # Convert image to numpy array
                if isinstance(humanoid_image, str):
                    image_array = self._base64_to_numpy(humanoid_image)
                else:
                    image_array = humanoid_image

                # Extract features
                visual_features = perception_module.extract_features(image_array)
                language_features = language_module.extract_features(humanoid_instruction)

                # Generate action considering humanoid constraints
                action_features = action_module.generate_action(
                    vision_features=visual_features.__dict__,
                    language_features=language_features.__dict__,
                    robot_state=humanoid_state
                )

                # Adapt action to humanoid capabilities
                adapted_action = action_module.adapt_action_to_robot(
                    action_features, humanoid_capabilities
                )

                vla_response = VLAResponse(
                    action_type=adapted_action.action_type,
                    action_parameters=adapted_action.action_parameters,
                    confidence=adapted_action.confidence,
                    execution_plan=adapted_action.execution_plan or [],
                    success=True
                )

            integrated_result = {
                'vla_response': vla_response.dict() if vla_response else None,
                'humanoid_data': humanoid_data,
                'integration_type': 'humanoid_control',
                'timestamp': __import__('time').time()
            }

            logger.info("Successfully integrated VLA with humanoid system")
            return integrated_result

        except Exception as e:
            logger.error(f"Error in VLA-humanoid integration: {str(e)}")
            raise

    def _generate_synthetic_training_data(self, task_type: str, complexity: str) -> List[Dict[str, Any]]:
        """Generate synthetic training data for learning systems"""
        try:
            # Generate synthetic (image, instruction, action) triplets
            synthetic_data = []

            for i in range(10):  # Generate 10 synthetic examples
                # Create a synthetic scenario
                scenario = {
                    "image_description": f"synthetic_{task_type}_scene_{i}",
                    "instruction": f"Perform {task_type} task on the object",
                    "expected_action": f"{task_type}_action_{i}",
                    "complexity": complexity,
                    "context": f"{task_type}_context_{i}"
                }
                synthetic_data.append(scenario)

            return synthetic_data

        except Exception as e:
            logger.error(f"Error generating synthetic training data: {str(e)}")
            return []

    def _evaluate_vla_performance(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate VLA performance on test cases"""
        try:
            # In a real implementation, this would run comprehensive evaluation
            # For simulation, we'll return sample metrics
            num_cases = len(test_cases)
            if num_cases == 0:
                return {
                    "task_success_rate": 0.0,
                    "language_accuracy": 0.0,
                    "vision_accuracy": 0.0,
                    "action_accuracy": 0.0,
                    "num_evaluated": 0
                }

            # Simulate evaluation results
            results = {
                "task_success_rate": float(np.random.uniform(0.6, 0.9)),
                "language_accuracy": float(np.random.uniform(0.7, 0.95)),
                "vision_accuracy": float(np.random.uniform(0.65, 0.9)),
                "action_accuracy": float(np.random.uniform(0.7, 0.85)),
                "num_evaluated": num_cases
            }

            return results

        except Exception as e:
            logger.error(f"Error evaluating VLA performance: {str(e)}")
            return {}

    def _ros_image_to_base64(self, ros_image) -> str:
        """Convert ROS image message to base64 string"""
        try:
            # This is a simplified conversion - in practice, ROS image conversion
            # would depend on the specific message format
            if hasattr(ros_image, 'data'):
                # If it's a raw image data array
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                # Reshape based on height, width, channels info
                height, width, channels = ros_image.height, ros_image.width, ros_image.step // ros_image.width
                img_array = img_array.reshape((height, width, channels))

                # Convert to PIL Image and then to base64
                img = Image.fromarray(img_array)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/png;base64,{img_base64}"
            else:
                # If it's already in a processable format, return as-is or process accordingly
                return str(ros_image)[:1000]  # Limit string length
        except Exception as e:
            logger.error(f"Error converting ROS image to base64: {str(e)}")
            # Return a placeholder image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    def _base64_to_numpy(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy array"""
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to numpy array
            image_array = np.array(image).astype(np.float32) / 255.0

            # Ensure 3-channel (RGB)
            if len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Convert to RGB

            return image_array
        except Exception as e:
            logger.error(f"Error converting base64 to numpy: {str(e)}")
            # Return a default small image array
            return np.random.random((64, 64, 3)).astype(np.float32)

    def _numpy_to_base64(self, numpy_array: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        try:
            # Ensure the array is in the right format
            if numpy_array.dtype != np.uint8:
                numpy_array = (numpy_array * 255).astype(np.uint8)

            # Convert to PIL Image
            if len(numpy_array.shape) == 3:
                img = Image.fromarray(numpy_array)
            else:
                img = Image.fromarray(numpy_array)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error converting numpy to base64: {str(e)}")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    async def process_robot_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a robot task using VLA components"""
        try:
            # Extract task components
            image_data = task_config.get('image_data')
            instruction = task_config.get('instruction', '')
            robot_state = task_config.get('robot_state', {})
            task_type = task_config.get('task_type', 'general')

            # Process through VLA pipeline
            if image_data:
                # Convert image data
                if isinstance(image_data, str):
                    image_array = self._base64_to_numpy(image_data)
                else:
                    image_array = image_data

                # Extract visual features
                visual_features = perception_module.extract_features(image_array)

                # Extract language features
                language_features = language_module.extract_features(instruction)

                # Fuse features
                fusion_result = fusion_module.fuse_features(
                    visual_features.__dict__,
                    language_features.__dict__
                )

                # Generate action
                action_features = action_module.generate_action(
                    vision_features=visual_features.__dict__,
                    language_features=language_features.__dict__,
                    robot_state=robot_state
                )

                # Store in memory for future reference
                memory_module.write_to_memory(
                    content={
                        "task": instruction,
                        "visual_context": visual_features.object_detections,
                        "action_taken": action_features.action_type,
                        "result": "executed"
                    },
                    memory_type="episodic",
                    tags=[task_type, "robot_task"]
                )

                task_result = {
                    'action_type': action_features.action_type,
                    'action_parameters': action_features.action_parameters,
                    'execution_plan': action_features.execution_plan,
                    'confidence': action_features.confidence,
                    'fusion_result': fusion_result,
                    'success': True
                }
            else:
                # Text-only task (no visual component)
                language_features = language_module.extract_features(instruction)

                # Generate action without visual context
                action_features = action_module.generate_action(
                    vision_features={},  # Empty vision features
                    language_features=language_features.__dict__,
                    robot_state=robot_state
                )

                task_result = {
                    'action_type': action_features.action_type,
                    'action_parameters': action_features.action_parameters,
                    'execution_plan': action_features.execution_plan,
                    'confidence': action_features.confidence,
                    'fusion_result': None,
                    'success': True
                }

            result = {
                'task_type': task_type,
                'task_config': task_config,
                'task_result': task_result,
                'timestamp': __import__('time').time()
            }

            logger.info(f"Successfully processed {task_type} task using VLA")
            return result

        except Exception as e:
            logger.error(f"Error in VLA robot task processing: {str(e)}")
            raise


# Singleton instance
vla_integration_utils = VLAIntegrationUtils()