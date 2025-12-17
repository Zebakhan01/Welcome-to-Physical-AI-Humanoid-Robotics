---
sidebar_position: 5
---

# Simulation Tips and Tricks

## Introduction to Simulation Optimization

Simulation in Physical AI and humanoid robotics serves as a crucial development tool, enabling rapid prototyping, algorithm testing, and safety validation before real-world deployment. This section provides advanced tips and tricks for optimizing simulation performance, improving realism, and maximizing the utility of simulation environments for robotics development.

## Performance Optimization Strategies

### Physics Engine Optimization

#### Optimizing Gazebo Performance

```python
# performance_optimization.py
import numpy as np
import time
from typing import Dict, List, Tuple
import threading

class SimulationPerformanceOptimizer:
    """
    Optimizes simulation performance for Physical AI systems
    """

    def __init__(self):
        self.performance_metrics = {
            'real_time_factor': [],
            'frame_rate': [],
            'physics_step_time': [],
            'render_time': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        self.optimization_settings = {
            'max_step_size': 0.001,
            'real_time_update_rate': 1000,
            'solver_iterations': 50,
            'contact_surface_layer': 0.001,
            'contact_max_correcting_vel': 100
        }
        self.is_optimizing = False
        self.optimization_thread = None

    def optimize_physics_settings(self, target_real_time_factor: float = 1.0):
        """
        Optimize physics settings for target real-time factor
        """
        if target_real_time_factor <= 0.5:
            # Prioritize speed over accuracy
            self.optimization_settings.update({
                'max_step_size': 0.01,  # Larger steps for speed
                'solver_iterations': 20,  # Fewer iterations
                'contact_surface_layer': 0.01,  # Larger surface layer
                'contact_max_correcting_vel': 50  # Lower correcting velocity
            })
        elif target_real_time_factor <= 1.0:
            # Balance speed and accuracy
            self.optimization_settings.update({
                'max_step_size': 0.005,
                'solver_iterations': 30,
                'contact_surface_layer': 0.005,
                'contact_max_correcting_vel': 75
            })
        else:
            # Prioritize accuracy over speed
            self.optimization_settings.update({
                'max_step_size': 0.001,
                'solver_iterations': 100,
                'contact_surface_layer': 0.001,
                'contact_max_correcting_vel': 100
            })

        print(f"Physics settings optimized for RTF: {target_real_time_factor}")
        return self.optimization_settings

    def optimize_collision_geometry(self, model_complexity: str = "medium"):
        """
        Optimize collision geometry based on complexity requirements
        """
        optimization_map = {
            'high_performance': {
                'use_convex_hulls': True,
                'simplify_complex_shapes': True,
                'reduce_triangle_count': 0.5,
                'use_primitive_shapes': True
            },
            'medium': {
                'use_convex_hulls': False,
                'simplify_complex_shapes': True,
                'reduce_triangle_count': 0.7,
                'use_primitive_shapes': False
            },
            'high_accuracy': {
                'use_convex_hulls': False,
                'simplify_complex_shapes': False,
                'reduce_triangle_count': 1.0,
                'use_primitive_shapes': False
            }
        }

        if model_complexity in optimization_map:
            settings = optimization_map[model_complexity]
            print(f"Collision geometry optimized for {model_complexity} performance")
            return settings
        else:
            print(f"Unknown complexity level: {model_complexity}")
            return optimization_map['medium']

    def optimize_rendering_settings(self, quality_level: str = "balanced"):
        """
        Optimize rendering settings for performance
        """
        quality_settings = {
            'high_performance': {
                'shadows': False,
                'reflections': False,
                'anti_aliasing': 1,
                'render_resolution': [640, 480],
                'texture_quality': 'low',
                'lighting_quality': 'fast'
            },
            'balanced': {
                'shadows': True,
                'reflections': False,
                'anti_aliasing': 2,
                'render_resolution': [1280, 720],
                'texture_quality': 'medium',
                'lighting_quality': 'balanced'
            },
            'high_quality': {
                'shadows': True,
                'reflections': True,
                'anti_aliasing': 4,
                'render_resolution': [1920, 1080],
                'texture_quality': 'high',
                'lighting_quality': 'accurate'
            }
        }

        if quality_level in quality_settings:
            settings = quality_settings[quality_level]
            print(f"Rendering settings optimized for {quality_level} quality")
            return settings
        else:
            print(f"Unknown quality level: {quality_level}")
            return quality_settings['balanced']

    def dynamic_performance_adjustment(self, current_metrics: Dict[str, float]):
        """
        Dynamically adjust performance settings based on current metrics
        """
        current_rtf = current_metrics.get('real_time_factor', 1.0)
        current_cpu = current_metrics.get('cpu_usage', 0.0)
        current_memory = current_metrics.get('memory_usage', 0.0)

        adjustments = {}

        # Adjust physics if RTF is too low
        if current_rtf < 0.5:
            adjustments['max_step_size'] = min(0.01, self.optimization_settings['max_step_size'] * 1.1)
            adjustments['solver_iterations'] = max(10, int(self.optimization_settings['solver_iterations'] * 0.9))

        # Adjust rendering if CPU usage is high
        if current_cpu > 80:
            adjustments['render_resolution'] = [640, 480]  # Lower resolution
            adjustments['shadows'] = False  # Disable shadows

        # Apply adjustments
        self.optimization_settings.update(adjustments)

        return self.optimization_settings

    def start_performance_monitoring(self):
        """
        Start performance monitoring in background thread
        """
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()

    def stop_performance_monitoring(self):
        """
        Stop performance monitoring
        """
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_performance(self):
        """
        Monitor performance in background thread
        """
        while self.is_monitoring:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()

                # Store metrics
                for key, value in metrics.items():
                    if key in self.performance_metrics:
                        self.performance_metrics[key].append(value)
                        # Keep only recent metrics (last 1000 samples)
                        if len(self.performance_metrics[key]) > 1000:
                            self.performance_metrics[key].pop(0)

                # Dynamic adjustment based on metrics
                self.dynamic_performance_adjustment(metrics)

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(1.0)

    def _collect_performance_metrics(self) -> Dict[str, float]:
        """
        Collect current performance metrics
        """
        import psutil
        import os

        # This would interface with simulation engine to get metrics
        # For now, return mock metrics
        return {
            'real_time_factor': np.random.uniform(0.8, 1.2),
            'frame_rate': np.random.uniform(25, 60),
            'physics_step_time': np.random.uniform(0.001, 0.005),
            'render_time': np.random.uniform(0.01, 0.03),
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent()
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary statistics
        """
        summary = {}
        for metric, values in self.performance_metrics.items():
            if values:
                summary[f"{metric}_avg"] = np.mean(values)
                summary[f"{metric}_min"] = min(values)
                summary[f"{metric}_max"] = max(values)
                summary[f"{metric}_std"] = np.std(values)
            else:
                summary[f"{metric}_avg"] = 0.0
                summary[f"{metric}_min"] = 0.0
                summary[f"{metric}_max"] = 0.0
                summary[f"{metric}_std"] = 0.0

        return summary


class ModelOptimization:
    """
    Model optimization for simulation performance
    """

    def __init__(self):
        self.model_complexity_settings = {}
        self.lod_system = LevelOfDetailSystem()

    def optimize_robot_model(self, robot_model_path: str, target_complexity: str = "medium"):
        """
        Optimize robot model for simulation performance
        """
        optimization_strategies = {
            'high_performance': {
                'reduce_mesh_resolution': 0.3,
                'simplify_collision_geometry': True,
                'remove_visual_details': True,
                'optimize_joints': True
            },
            'medium': {
                'reduce_mesh_resolution': 0.7,
                'simplify_collision_geometry': True,
                'remove_visual_details': False,
                'optimize_joints': True
            },
            'high_accuracy': {
                'reduce_mesh_resolution': 1.0,
                'simplify_collision_geometry': False,
                'remove_visual_details': False,
                'optimize_joints': False
            }
        }

        strategy = optimization_strategies.get(target_complexity, optimization_strategies['medium'])

        # Apply optimizations
        optimized_model = self._apply_model_optimizations(robot_model_path, strategy)

        return optimized_model

    def _apply_model_optimizations(self, model_path: str, strategy: Dict) -> str:
        """
        Apply model optimizations based on strategy
        """
        # This would use CAD/3D modeling tools to optimize the model
        # For now, return the original path with note
        print(f"Model optimizations applied with strategy: {strategy}")
        return model_path

    def create_level_of_detail_model(self, base_model_path: str, lod_levels: List[float]) -> List[str]:
        """
        Create multiple levels of detail for the same model
        """
        lod_models = []

        for i, lod_factor in enumerate(lod_levels):
            lod_model_path = self._create_lod_model(base_model_path, lod_factor, i)
            lod_models.append(lod_model_path)

        return lod_models

    def _create_lod_model(self, base_path: str, lod_factor: float, level: int) -> str:
        """
        Create a level of detail model at specified factor
        """
        import os
        from pathlib import Path

        base_name = Path(base_path).stem
        ext = Path(base_path).suffix
        lod_path = f"{base_path[:-len(ext)]}_lod{level}{ext}"

        # This would actually create the LOD model
        # For now, just return the path
        print(f"Created LOD model at {lod_path} with factor {lod_factor}")
        return lod_path


class LevelOfDetailSystem:
    """
    Level of Detail system for simulation optimization
    """

    def __init__(self):
        self.lod_models = {}
        self.visibility_distances = {}

    def add_lod_model(self, model_name: str, lod_models: List[str], distances: List[float]):
        """
        Add LOD models with corresponding visibility distances
        """
        if len(lod_models) != len(distances) + 1:
            raise ValueError("LOD models should be one more than distances")

        self.lod_models[model_name] = lod_models
        self.visibility_distances[model_name] = distances

    def get_appropriate_lod(self, model_name: str, distance: float) -> str:
        """
        Get appropriate LOD model based on distance
        """
        if model_name not in self.lod_models:
            return None

        distances = self.visibility_distances[model_name]
        models = self.lod_models[model_name]

        # Find appropriate LOD level
        for i, dist_threshold in enumerate(distances):
            if distance <= dist_threshold:
                return models[i]

        # Return lowest detail level if beyond all thresholds
        return models[-1]

    def optimize_scene_complexity(self, objects_in_scene: List[Dict]) -> List[Dict]:
        """
        Optimize scene complexity using LOD
        """
        optimized_scene = []

        for obj in objects_in_scene:
            # Calculate distance to camera/viewer
            distance = self._calculate_distance_to_viewer(obj['position'])

            # Get appropriate LOD model
            lod_model = self.get_appropriate_lod(obj['model_name'], distance)

            if lod_model:
                optimized_obj = obj.copy()
                optimized_obj['model_path'] = lod_model
                optimized_scene.append(optimized_obj)
            else:
                optimized_scene.append(obj)

        return optimized_scene

    def _calculate_distance_to_viewer(self, object_position: List[float]) -> float:
        """
        Calculate distance from object to viewer/camera
        """
        # This would calculate actual distance
        # For now, return mock distance
        return np.random.uniform(0, 10)
```

## Advanced Simulation Techniques

### Domain Randomization for Training

```python
# domain_randomization.py
import numpy as np
import random
from typing import Dict, Any, List

class DomainRandomization:
    """
    Domain randomization techniques for simulation-to-reality transfer
    """

    def __init__(self):
        self.randomization_ranges = {
            'lighting': {
                'intensity': [0.5, 2.0],
                'color_temperature': [3000, 8000],
                'direction': [[-1, -1, -1], [1, 1, 1]]
            },
            'materials': {
                'friction': [0.1, 1.0],
                'restitution': [0.0, 0.5],
                'color': [[0, 0, 0], [1, 1, 1]]
            },
            'objects': {
                'size': [0.5, 2.0],
                'position': [[-1, -1, 0], [1, 1, 2]],
                'rotation': [[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]]
            },
            'textures': {
                'roughness': [0.0, 1.0],
                'metallic': [0.0, 0.5],
                'normal_map_strength': [0.0, 1.0]
            }
        }

        self.randomization_weights = {
            'lighting': 0.2,
            'materials': 0.3,
            'objects': 0.3,
            'textures': 0.2
        }

    def randomize_environment(self, environment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize environment parameters
        """
        randomized_config = environment_config.copy()

        # Randomize lighting
        randomized_config['lighting'] = self._randomize_lighting(
            environment_config.get('lighting', {})
        )

        # Randomize materials
        randomized_config['materials'] = self._randomize_materials(
            environment_config.get('materials', {})
        )

        # Randomize objects
        if 'objects' in environment_config:
            randomized_config['objects'] = self._randomize_objects(
                environment_config['objects']
            )

        # Randomize textures
        randomized_config['textures'] = self._randomize_textures(
            environment_config.get('textures', {})
        )

        return randomized_config

    def _randomize_lighting(self, lighting_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize lighting parameters
        """
        randomized_lighting = lighting_config.copy()

        # Randomize intensity
        intensity_range = self.randomization_ranges['lighting']['intensity']
        randomized_lighting['intensity'] = np.random.uniform(
            intensity_range[0], intensity_range[1]
        )

        # Randomize color temperature
        temp_range = self.randomization_ranges['lighting']['color_temperature']
        randomized_lighting['color_temperature'] = np.random.uniform(
            temp_range[0], temp_range[1]
        )

        # Randomize direction (normalize the vector)
        direction_range = self.randomization_ranges['lighting']['direction']
        direction = np.random.uniform(
            direction_range[0], direction_range[1]
        )
        direction = direction / np.linalg.norm(direction)  # Normalize
        randomized_lighting['direction'] = direction.tolist()

        return randomized_lighting

    def _randomize_materials(self, materials_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize material properties
        """
        randomized_materials = materials_config.copy()

        # Randomize friction
        friction_range = self.randomization_ranges['materials']['friction']
        randomized_materials['friction'] = np.random.uniform(
            friction_range[0], friction_range[1]
        )

        # Randomize restitution
        restitution_range = self.randomization_ranges['materials']['restitution']
        randomized_materials['restitution'] = np.random.uniform(
            restitution_range[0], restitution_range[1]
        )

        # Randomize color
        color_range = self.randomization_ranges['materials']['color']
        randomized_materials['color'] = np.random.uniform(
            color_range[0], color_range[1]
        ).tolist()

        return randomized_materials

    def _randomize_objects(self, objects_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Randomize object properties in environment
        """
        randomized_objects = []

        for obj in objects_config:
            randomized_obj = obj.copy()

            # Randomize size
            size_range = self.randomization_ranges['objects']['size']
            size_factor = np.random.uniform(size_range[0], size_range[1])
            if 'size' in randomized_obj:
                if isinstance(randomized_obj['size'], list):
                    randomized_obj['size'] = [s * size_factor for s in randomized_obj['size']]
                else:
                    randomized_obj['size'] *= size_factor

            # Randomize position
            pos_range = self.randomization_ranges['objects']['position']
            pos_offset = np.random.uniform(pos_range[0], pos_range[1])
            if 'position' in randomized_obj:
                randomized_obj['position'] = [
                    pos + offset for pos, offset in zip(randomized_obj['position'], pos_offset)
                ]

            # Randomize rotation
            rot_range = self.randomization_ranges['objects']['rotation']
            rot_offset = np.random.uniform(rot_range[0], rot_range[1])
            if 'rotation' in randomized_obj:
                randomized_obj['rotation'] = [
                    rot + offset for rot, offset in zip(randomized_obj['rotation'], rot_offset)
                ]

            randomized_objects.append(randomized_obj)

        return randomized_objects

    def _randomize_textures(self, textures_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize texture properties
        """
        randomized_textures = textures_config.copy()

        # Randomize roughness
        roughness_range = self.randomization_ranges['textures']['roughness']
        randomized_textures['roughness'] = np.random.uniform(
            roughness_range[0], roughness_range[1]
        )

        # Randomize metallic
        metallic_range = self.randomization_ranges['textures']['metallic']
        randomized_textures['metallic'] = np.random.uniform(
            metallic_range[0], metallic_range[1]
        )

        # Randomize normal map strength
        normal_range = self.randomization_ranges['textures']['normal_map_strength']
        randomized_textures['normal_map_strength'] = np.random.uniform(
            normal_range[0], normal_range[1]
        )

        return randomized_textures

    def apply_progressive_randomization(self, training_step: int, max_steps: int = 10000) -> Dict[str, Any]:
        """
        Apply progressive domain randomization that increases over training
        """
        progress = min(1.0, training_step / max_steps)

        # Adjust randomization ranges based on progress
        adjusted_ranges = {}
        for category, ranges in self.randomization_ranges.items():
            adjusted_ranges[category] = {}
            for param, value_range in ranges.items():
                if isinstance(value_range[0], list):  # Vector ranges
                    center = [(a + b) / 2 for a, b in zip(value_range[0], value_range[1])]
                    span = [(b - a) * progress for a, b in zip(value_range[0], value_range[1])]
                    new_range = [
                        [c - s/2 for c, s in zip(center, span)],
                        [c + s/2 for c, s in zip(center, span)]
                    ]
                else:  # Scalar ranges
                    center = (value_range[0] + value_range[1]) / 2
                    span = (value_range[1] - value_range[0]) * progress
                    new_range = [center - span/2, center + span/2]

                adjusted_ranges[category][param] = new_range

        # Store original ranges temporarily
        original_ranges = self.randomization_ranges
        self.randomization_ranges = adjusted_ranges

        # Generate randomization
        randomization = self._generate_current_randomization()

        # Restore original ranges
        self.randomization_ranges = original_ranges

        return randomization

    def _generate_current_randomization(self) -> Dict[str, Any]:
        """
        Generate current randomization based on current ranges
        """
        return {
            'lighting': self._get_current_lighting_randomization(),
            'materials': self._get_current_material_randomization(),
            'objects': self._get_current_object_randomization(),
            'textures': self._get_current_texture_randomization()
        }

    def _get_current_lighting_randomization(self) -> Dict[str, Any]:
        """Get current lighting randomization"""
        intensity_range = self.randomization_ranges['lighting']['intensity']
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])

        temp_range = self.randomization_ranges['lighting']['color_temperature']
        color_temp = np.random.uniform(temp_range[0], temp_range[1])

        return {
            'intensity': intensity,
            'color_temperature': color_temp
        }

    def _get_current_material_randomization(self) -> Dict[str, Any]:
        """Get current material randomization"""
        friction_range = self.randomization_ranges['materials']['friction']
        friction = np.random.uniform(friction_range[0], friction_range[1])

        restitution_range = self.randomization_ranges['materials']['restitution']
        restitution = np.random.uniform(restitution_range[0], restitution_range[1])

        return {
            'friction': friction,
            'restitution': restitution
        }

    def _get_current_object_randomization(self) -> Dict[str, Any]:
        """Get current object randomization"""
        size_range = self.randomization_ranges['objects']['size']
        size_factor = np.random.uniform(size_range[0], size_range[1])

        pos_range = self.randomization_ranges['objects']['position']
        position_offset = np.random.uniform(pos_range[0], pos_range[1])

        return {
            'size_factor': size_factor,
            'position_offset': position_offset
        }

    def _get_current_texture_randomization(self) -> Dict[str, Any]:
        """Get current texture randomization"""
        roughness_range = self.randomization_ranges['textures']['roughness']
        roughness = np.random.uniform(roughness_range[0], roughness_range[1])

        metallic_range = self.randomization_ranges['textures']['metallic']
        metallic = np.random.uniform(metallic_range[0], metallic_range[1])

        return {
            'roughness': roughness,
            'metallic': metallic
        }

    def get_randomization_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about randomization application
        """
        return {
            'randomization_ranges': self.randomization_ranges,
            'randomization_weights': self.randomization_weights,
            'total_categories': len(self.randomization_ranges)
        }
```

### Sensor Simulation Optimization

#### High-Fidelity Sensor Simulation

```python
# sensor_simulation.py
import numpy as np
import cv2
from typing import Dict, Any, Tuple
import time

class HighFidelitySensorSimulation:
    """
    High-fidelity sensor simulation with realistic noise and distortions
    """

    def __init__(self):
        self.sensor_models = {}
        self.noise_parameters = {}
        self.distortion_models = {}

    def create_camera_model(self, name: str, config: Dict[str, Any]):
        """
        Create high-fidelity camera model with realistic characteristics
        """
        camera_model = {
            'name': name,
            'width': config['width'],
            'height': config['height'],
            'fov': config['fov'],
            'focal_length': config['width'] / (2 * np.tan(np.radians(config['fov'] / 2))),
            'principal_point': [config['width'] / 2, config['height'] / 2],

            # Noise parameters
            'noise_model': config.get('noise_model', 'gaussian'),
            'noise_std': config.get('noise_std', 10.0),
            'shot_noise_factor': config.get('shot_noise_factor', 0.01),
            'thermal_noise_std': config.get('thermal_noise_std', 5.0),

            # Distortion parameters
            'distortion_model': config.get('distortion_model', 'brown_conrady'),
            'k1': config.get('k1', 0),  # Radial distortion
            'k2': config.get('k2', 0),
            'k3': config.get('k3', 0),
            'p1': config.get('p1', 0),  # Tangential distortion
            'p2': config.get('p2', 0),

            # Dynamic range and gamma
            'dynamic_range': config.get('dynamic_range', [0.001, 1000]),  # lux
            'gamma': config.get('gamma', 1.0),
            'exposure_time': config.get('exposure_time', 1.0/30.0),  # seconds

            # Frame rate and timing
            'frame_rate': config.get('frame_rate', 30),
            'frame_delay': config.get('frame_delay', 0.001),  # seconds
        }

        self.sensor_models[name] = camera_model
        return camera_model

    def simulate_camera_capture(self, scene_data: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Simulate realistic camera capture with noise and distortions
        """
        if camera_name not in self.sensor_models:
            raise ValueError(f"Camera {camera_name} not found in sensor models")

        camera = self.sensor_models[camera_name]

        # Apply geometric distortion
        distorted_image = self._apply_geometric_distortion(scene_data, camera)

        # Apply noise
        noisy_image = self._apply_sensor_noise(distorted_image, camera)

        # Apply gamma correction
        gamma_corrected = self._apply_gamma_correction(noisy_image, camera)

        # Apply dynamic range limitations
        final_image = self._apply_dynamic_range(gamma_corrected, camera)

        return final_image

    def _apply_geometric_distortion(self, image: np.ndarray, camera: Dict[str, Any]) -> np.ndarray:
        """
        Apply geometric distortion to image
        """
        h, w = image.shape[:2]

        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        xv, yv = np.meshgrid(x, y)

        # Convert to normalized coordinates
        x_norm = (xv - camera['principal_point'][0]) / camera['focal_length']
        y_norm = (yv - camera['principal_point'][1]) / camera['focal_length']

        # Calculate distortion
        r_squared = x_norm**2 + y_norm**2
        radial_distortion = 1 + camera['k1'] * r_squared + camera['k2'] * r_squared**2 + camera['k3'] * r_squared**3
        tangential_distortion_x = 2 * camera['p1'] * x_norm * y_norm + camera['p2'] * (r_squared + 2 * x_norm**2)
        tangential_distortion_y = camera['p1'] * (r_squared + 2 * y_norm**2) + 2 * camera['p2'] * x_norm * y_norm

        # Apply distortion
        x_distorted = (x_norm * radial_distortion + tangential_distortion_x) * camera['focal_length'] + camera['principal_point'][0]
        y_distorted = (y_norm * radial_distortion + tangential_distortion_y) * camera['focal_length'] + camera['principal_point'][1]

        # Remap image
        distorted_image = cv2.remap(
            image.astype(np.float32),
            x_distorted.astype(np.float32),
            y_distorted.astype(np.float32),
            cv2.INTER_LINEAR
        )

        return distorted_image.astype(np.uint8)

    def _apply_sensor_noise(self, image: np.ndarray, camera: Dict[str, Any]) -> np.ndarray:
        """
        Apply realistic sensor noise to image
        """
        image_float = image.astype(np.float32)

        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, camera['noise_std'], image.shape)
        image_noisy = image_float + gaussian_noise

        # Add shot noise (photon noise) - proportional to signal
        shot_noise = np.random.poisson(image_float * camera['shot_noise_factor'])
        image_noisy += shot_noise

        # Add thermal noise
        thermal_noise = np.random.normal(0, camera['thermal_noise_std'], image.shape)
        image_noisy += thermal_noise

        # Add quantization noise (ADC discretization)
        quantization_step = 256.0 / (2**8)  # 8-bit quantization
        quantization_noise = np.random.uniform(-quantization_step/2, quantization_step/2, image.shape)
        image_noisy += quantization_noise

        # Clip to valid range
        image_noisy = np.clip(image_noisy, 0, 255)

        return image_noisy.astype(np.uint8)

    def _apply_gamma_correction(self, image: np.ndarray, camera: Dict[str, Any]) -> np.ndarray:
        """
        Apply gamma correction to image
        """
        image_normalized = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(image_normalized, 1.0 / camera['gamma'])
        return (gamma_corrected * 255.0).astype(np.uint8)

    def _apply_dynamic_range(self, image: np.ndarray, camera: Dict[str, Any]) -> np.ndarray:
        """
        Apply dynamic range limitations
        """
        # This would simulate the camera's response to different light levels
        # For now, just return the image
        return image

    def create_lidar_model(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create high-fidelity LIDAR model
        """
        lidar_model = {
            'name': name,
            'range_min': config.get('range_min', 0.1),
            'range_max': config.get('range_max', 25.0),
            'fov_horizontal': config.get('fov_horizontal', 360),  # degrees
            'fov_vertical': config.get('fov_vertical', 30),      # degrees
            'resolution_horizontal': config.get('resolution_horizontal', 0.1),  # degrees
            'resolution_vertical': config.get('resolution_vertical', 2.0),      # degrees
            'update_rate': config.get('update_rate', 10),  # Hz

            # Noise parameters
            'range_noise_std': config.get('range_noise_std', 0.02),  # meters
            'angular_noise_std': config.get('angular_noise_std', 0.01),  # degrees

            # Performance parameters
            'max_points': config.get('max_points', 360 * 15),  # typical for 360deg x 15deg
            'intensity_range': config.get('intensity_range', [0, 1000])
        }

        self.sensor_models[name] = lidar_model
        return lidar_model

    def simulate_lidar_scan(self, environment_data: Dict[str, Any], lidar_name: str) -> Dict[str, Any]:
        """
        Simulate realistic LIDAR scan
        """
        if lidar_name not in self.sensor_models:
            raise ValueError(f"LIDAR {lidar_name} not found in sensor models")

        lidar = self.sensor_models[lidar_name]

        # Calculate number of beams
        h_beams = int(lidar['fov_horizontal'] / lidar['resolution_horizontal'])
        v_beams = int(lidar['fov_vertical'] / lidar['resolution_vertical'])

        # Generate scan data
        ranges = np.full((v_beams, h_beams), lidar['range_max'])
        intensities = np.zeros((v_beams, h_beams))

        # Simulate ray casting for each beam
        for v_idx in range(v_beams):
            for h_idx in range(h_beams):
                # Calculate beam direction
                h_angle = (h_idx - h_beams/2) * np.radians(lidar['resolution_horizontal'])
                v_angle = (v_idx - v_beams/2) * np.radians(lidar['resolution_vertical'])

                # Calculate 3D direction vector
                direction = np.array([
                    np.cos(v_angle) * np.cos(h_angle),
                    np.cos(v_angle) * np.sin(h_angle),
                    np.sin(v_angle)
                ])

                # Find intersection with environment (simplified)
                range_reading = self._ray_intersect_environment(
                    environment_data, direction, lidar['range_max']
                )

                # Add noise
                noise = np.random.normal(0, lidar['range_noise_std'])
                ranges[v_idx, h_idx] = min(lidar['range_max'], max(lidar['range_min'], range_reading + noise))

                # Calculate intensity (simplified)
                intensities[v_idx, h_idx] = self._calculate_reflection_intensity(
                    range_reading, direction, environment_data
                )

        # Add angular noise
        angular_noise_h = np.random.normal(0, lidar['angular_noise_std'], ranges.shape)
        angular_noise_v = np.random.normal(0, lidar['angular_noise_std'], ranges.shape)

        return {
            'ranges': ranges,
            'intensities': intensities,
            'horizontal_angles': np.linspace(
                -np.radians(lidar['fov_horizontal']/2),
                np.radians(lidar['fov_horizontal']/2),
                h_beams
            ),
            'vertical_angles': np.linspace(
                -np.radians(lidar['fov_vertical']/2),
                np.radians(lidar['fov_vertical']/2),
                v_beams
            ),
            'timestamp': time.time()
        }

    def _ray_intersect_environment(self, env_data: Dict[str, Any], direction: np.ndarray, max_range: float) -> float:
        """
        Simulate ray-environment intersection
        """
        # Simplified ray intersection - in real implementation, this would be more sophisticated
        # This could involve collision detection with environment meshes
        return max_range * 0.7  # Return 70% of max range as default

    def _calculate_reflection_intensity(self, range_reading: float, direction: np.ndarray, env_data: Dict[str, Any]) -> float:
        """
        Calculate reflection intensity based on surface properties
        """
        # Simplified intensity calculation
        base_intensity = 500  # Base intensity
        distance_factor = max(0.1, 1.0 - (range_reading / self.lidar_model['range_max']))  # Closer = brighter
        surface_factor = 1.0  # Surface reflectivity factor

        intensity = base_intensity * distance_factor * surface_factor
        return min(self.lidar_model['intensity_range'][1], max(self.lidar_model['intensity_range'][0], intensity))

    def create_imu_model(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create high-fidelity IMU model
        """
        imu_model = {
            'name': name,
            'update_rate': config.get('update_rate', 100),  # Hz
            'accelerometer_noise_density': config.get('accelerometer_noise_density', 0.01),  # m/s^2/sqrt(Hz)
            'gyroscope_noise_density': config.get('gyroscope_noise_density', 0.001),      # rad/s/sqrt(Hz)
            'accelerometer_random_walk': config.get('accelerometer_random_walk', 0.001),   # m/s^2/sqrt(s)
            'gyroscope_random_walk': config.get('gyroscope_random_walk', 0.0001),         # rad/s/sqrt(s)
            'accelerometer_bias_instability': config.get('accelerometer_bias_instability', 0.01),  # m/s^2
            'gyroscope_bias_instability': config.get('gyroscope_bias_instability', 0.001),          # rad/s

            # Mounting parameters
            'mounting_position': config.get('mounting_position', [0, 0, 0]),  # meters
            'mounting_orientation': config.get('mounting_orientation', [0, 0, 0, 1]),  # quaternion [x,y,z,w]

            # Temperature coefficients
            'temp_coeff_accel': config.get('temp_coeff_accel', [0.0, 0.0, 0.0]),  # ppm/°C
            'temp_coeff_gyro': config.get('temp_coeff_gyro', [0.0, 0.0, 0.0]),   # ppm/°C
            'operating_temp_range': config.get('operating_temp_range', [-20, 85])  # °C
        }

        self.sensor_models[name] = imu_model
        return imu_model

    def simulate_imu_data(self, robot_state: Dict[str, Any], imu_name: str) -> Dict[str, Any]:
        """
        Simulate realistic IMU measurements
        """
        if imu_name not in self.sensor_models:
            raise ValueError(f"IMU {imu_name} not found in sensor models")

        imu = self.sensor_models[imu_name]

        # Get true values from robot state
        true_orientation = robot_state.get('orientation', [0, 0, 0, 1])
        true_angular_velocity = robot_state.get('angular_velocity', [0, 0, 0])
        true_linear_acceleration = robot_state.get('linear_acceleration', [0, 0, -9.81])

        # Apply mounting transformation
        mounted_orientation = self._transform_orientation_to_mount(
            true_orientation, imu['mounting_orientation']
        )
        mounted_angular_velocity = self._transform_vector_to_mount(
            true_angular_velocity, imu['mounting_orientation']
        )
        mounted_linear_acceleration = self._transform_vector_to_mount(
            true_linear_acceleration, imu['mounting_orientation']
        )

        # Add sensor noise and biases
        noisy_orientation = self._add_orientation_noise(mounted_orientation, imu)
        noisy_angular_velocity = self._add_gyro_noise(mounted_angular_velocity, imu)
        noisy_linear_acceleration = self._add_accel_noise(mounted_linear_acceleration, imu)

        return {
            'orientation': noisy_orientation,
            'angular_velocity': noisy_angular_velocity,
            'linear_acceleration': noisy_linear_acceleration,
            'timestamp': time.time(),
            'temperature': robot_state.get('temperature', 25.0)  # °C
        }

    def _transform_orientation_to_mount(self, orientation: List[float], mounting_offset: List[float]) -> List[float]:
        """
        Transform orientation to IMU mounting frame
        """
        # This would apply quaternion multiplication for mounting offset
        # For now, return the original orientation
        return orientation

    def _transform_vector_to_mount(self, vector: List[float], mounting_orientation: List[float]) -> List[float]:
        """
        Transform vector to IMU mounting frame
        """
        # Convert mounting orientation quaternion to rotation matrix
        q = mounting_orientation
        R = self._quaternion_to_rotation_matrix(q)

        # Transform vector
        transformed_vector = R @ np.array(vector)

        return transformed_vector.tolist()

    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        """
        Convert quaternion to rotation matrix
        """
        w, x, y, z = q

        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        return R

    def _add_orientation_noise(self, orientation: List[float], imu: Dict[str, Any]) -> List[float]:
        """
        Add noise to orientation measurements
        """
        # For now, just add small noise to quaternion
        noise = np.random.normal(0, 0.001, 4)
        noisy_orientation = np.array(orientation) + noise

        # Normalize quaternion
        noisy_orientation = noisy_orientation / np.linalg.norm(noisy_orientation)

        return noisy_orientation.tolist()

    def _add_gyro_noise(self, angular_velocity: List[float], imu: Dict[str, Any]) -> List[float]:
        """
        Add realistic gyroscope noise
        """
        # Add white noise
        white_noise_std = imu['gyroscope_noise_density'] * np.sqrt(imu['update_rate'] / 2)
        white_noise = np.random.normal(0, white_noise_std, 3)

        # Add bias instability (random walk)
        bias_drift = np.random.normal(0, imu['gyroscope_random_walk'] / np.sqrt(1/imu['update_rate']), 3)

        noisy_angular_velocity = np.array(angular_velocity) + white_noise + bias_drift

        return noisy_angular_velocity.tolist()

    def _add_accel_noise(self, linear_acceleration: List[float], imu: Dict[str, Any]) -> List[float]:
        """
        Add realistic accelerometer noise
        """
        # Add white noise
        white_noise_std = imu['accelerometer_noise_density'] * np.sqrt(imu['update_rate'] / 2)
        white_noise = np.random.normal(0, white_noise_std, 3)

        # Add bias instability
        bias_drift = np.random.normal(0, imu['accelerometer_random_walk'] / np.sqrt(1/imu['update_rate']), 3)

        noisy_linear_acceleration = np.array(linear_acceleration) + white_noise + bias_drift

        return noisy_linear_acceleration.tolist()
```

## Advanced Simulation Techniques

### Parallel Simulation and Batch Processing

```python
# parallel_simulation.py
import multiprocessing
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

class ParallelSimulationManager:
    """
    Manager for parallel simulation execution
    """

    def __init__(self, num_processes=None, num_threads=None):
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.num_threads = num_threads or min(32, multiprocessing.cpu_count() * 2)

        self.simulation_processes = []
        self.simulation_threads = []
        self.results_queue = queue.Queue()
        self.task_queue = queue.Queue()

    def run_parallel_simulations(self, simulation_configs: List[Dict[str, Any]],
                                num_parallel: int = None) -> List[Dict[str, Any]]:
        """
        Run multiple simulations in parallel
        """
        if num_parallel is None:
            num_parallel = min(len(simulation_configs), self.num_processes)

        results = []

        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            # Submit simulation tasks
            futures = []
            for config in simulation_configs:
                future = executor.submit(self._run_single_simulation, config)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'config': config})

        return results

    def _run_single_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single simulation with given configuration
        """
        import subprocess
        import tempfile
        import os

        # Create temporary simulation environment
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write simulation config to temporary file
            config_file = os.path.join(temp_dir, 'simulation_config.json')
            with open(config_file, 'w') as f:
                import json
                json.dump(config, f)

            # Run simulation process
            try:
                result = subprocess.run([
                    'gazebo', '--verbose', '--headless',
                    config.get('world_file', 'empty.world'),
                    '--ros-args', '--params-file', config_file
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

                return {
                    'config': config,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': result.returncode == 0
                }
            except subprocess.TimeoutExpired:
                return {
                    'config': config,
                    'error': 'Simulation timeout',
                    'success': False
                }
            except Exception as e:
                return {
                    'config': config,
                    'error': str(e),
                    'success': False
                }

    def run_batch_training_simulations(self, training_configs: List[Dict[str, Any]],
                                     domain_randomization: DomainRandomization = None) -> Dict[str, Any]:
        """
        Run batch of training simulations with domain randomization
        """
        start_time = time.time()

        # Apply domain randomization to configs
        randomized_configs = []
        for i, config in enumerate(training_configs):
            if domain_randomization:
                randomized_config = domain_randomization.randomize_environment(config)
            else:
                randomized_config = config.copy()

            # Add unique identifier
            randomized_config['simulation_id'] = f"train_{i:04d}"
            randomized_configs.append(randomized_config)

        # Run simulations in parallel
        results = self.run_parallel_simulations(randomized_configs)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze results
        successful_simulations = [r for r in results if r.get('success', False)]
        failed_simulations = [r for r in results if not r.get('success', True)]

        return {
            'total_simulations': len(training_configs),
            'successful_simulations': len(successful_simulations),
            'failed_simulations': len(failed_simulations),
            'success_rate': len(successful_simulations) / len(training_configs) if training_configs else 0,
            'total_time': total_time,
            'average_time_per_simulation': total_time / len(training_configs) if training_configs else 0,
            'results': results
        }

    def optimize_simulation_parameters(self, base_config: Dict[str, Any],
                                     parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize simulation parameters using parallel evaluation
        """
        from itertools import product

        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        combinations = list(product(*param_values))

        # Create configurations for each combination
        configs = []
        for combo in combinations:
            config = base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value
            configs.append(config)

        # Run simulations in parallel
        results = self.run_parallel_simulations(configs)

        # Find best configuration
        best_result = None
        best_score = float('-inf')

        for result in results:
            if result.get('success', False):
                score = self._calculate_simulation_score(result)
                if score > best_score:
                    best_score = score
                    best_result = result

        return {
            'best_configuration': best_result.get('config', {}) if best_result else {},
            'best_score': best_score,
            'all_results': results,
            'optimization_time': sum(r.get('simulation_time', 0) for r in results)
        }

    def _calculate_simulation_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate score for simulation result (higher is better)
        """
        # This would calculate a performance score based on simulation metrics
        # For now, return a simple score
        if result.get('success', False):
            return 1.0
        else:
            return 0.0

    def run_sensitivity_analysis(self, base_config: Dict[str, Any],
                                parameters_to_test: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Run sensitivity analysis for simulation parameters
        """
        analysis_results = {}

        for param_name, param_values in parameters_to_test.items():
            param_results = []

            for param_value in param_values:
                # Create config with modified parameter
                test_config = base_config.copy()
                test_config[param_name] = param_value

                # Run simulation
                result = self._run_single_simulation(test_config)
                param_results.append({
                    'parameter_value': param_value,
                    'result': result
                })

            analysis_results[param_name] = param_results

        return {
            'sensitivity_analysis': analysis_results,
            'base_config': base_config
        }

    def manage_simulation_resources(self) -> Dict[str, Any]:
        """
        Manage simulation resources and prevent overload
        """
        import psutil

        system_info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'total_processes': len(psutil.pids()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }

        # Determine if system can handle more simulations
        resource_limitations = {
            'cpu_limit': system_info['cpu_percent'] > 80,
            'memory_limit': system_info['memory_percent'] > 85,
            'disk_limit': system_info['disk_percent'] > 90,
            'safe_to_continue': all([
                system_info['cpu_percent'] < 80,
                system_info['memory_percent'] < 85,
                system_info['disk_percent'] < 90
            ])
        }

        return {
            'system_info': system_info,
            'resource_limitations': resource_limitations,
            'recommendation': 'continue' if resource_limitations['safe_to_continue'] else 'pause'
        }

    def schedule_simulation_jobs(self, job_queue: List[Dict[str, Any]]) -> List[str]:
        """
        Schedule simulation jobs based on resource availability
        """
        completed_jobs = []
        resource_manager = self.manage_simulation_resources()

        while job_queue and resource_manager['resource_limitations']['safe_to_continue']:
            job = job_queue.pop(0)

            # Run simulation
            result = self._run_single_simulation(job['config'])

            completed_jobs.append({
                'job_id': job['job_id'],
                'result': result,
                'completed_time': time.time()
            })

            # Check resources again
            resource_manager = self.manage_simulation_resources()

        return completed_jobs


class SimulationCacheManager:
    """
    Cache manager for simulation results and assets
    """

    def __init__(self, cache_directory: str = "./simulation_cache"):
        self.cache_directory = cache_directory
        self.cache_metadata = {}
        self.max_cache_size_gb = 10  # Maximum cache size

        import os
        os.makedirs(cache_directory, exist_ok=True)

    def cache_simulation_result(self, simulation_id: str, result_data: Dict[str, Any]):
        """
        Cache simulation result
        """
        import json
        import hashlib

        # Create cache file path
        cache_file = f"{self.cache_directory}/{simulation_id}.json"

        # Add metadata
        metadata = {
            'simulation_id': simulation_id,
            'timestamp': time.time(),
            'size': len(json.dumps(result_data).encode('utf-8')),
            'hash': hashlib.md5(json.dumps(result_data, sort_keys=True).encode('utf-8')).hexdigest()
        }

        # Save result
        with open(cache_file, 'w') as f:
            json.dump(result_data, f)

        # Save metadata
        self.cache_metadata[simulation_id] = metadata

        # Check cache size and clean if necessary
        self._manage_cache_size()

    def get_cached_result(self, simulation_id: str) -> Dict[str, Any]:
        """
        Retrieve cached simulation result
        """
        cache_file = f"{self.cache_directory}/{simulation_id}.json"

        if simulation_id in self.cache_metadata and os.path.exists(cache_file):
            import json
            with open(cache_file, 'r') as f:
                return json.load(f)

        return None

    def _manage_cache_size(self):
        """
        Manage cache size by removing oldest entries
        """
        import os
        from operator import itemgetter

        # Calculate total cache size
        total_size = sum(meta['size'] for meta in self.cache_metadata.values())
        total_size_gb = total_size / (1024**3)

        if total_size_gb > self.max_cache_size_gb:
            # Sort by timestamp (oldest first)
            sorted_meta = sorted(self.cache_metadata.items(), key=lambda x: x[1]['timestamp'])

            # Remove oldest entries until under limit
            removed_count = 0
            for sim_id, meta in sorted_meta:
                cache_file = f"{self.cache_directory}/{sim_id}.json"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    del self.cache_metadata[sim_id]
                    removed_count += 1

                # Recalculate size
                current_size = sum(m['size'] for m in self.cache_metadata.values()) / (1024**3)
                if current_size <= self.max_cache_size_gb * 0.8:  # 80% of limit
                    break

            print(f"Cache cleaned: {removed_count} entries removed")

    def invalidate_cache(self, simulation_ids: List[str] = None):
        """
        Invalidate specific cache entries or entire cache
        """
        import os

        if simulation_ids is None:
            # Clear entire cache
            import shutil
            shutil.rmtree(self.cache_directory)
            os.makedirs(self.cache_directory, exist_ok=True)
            self.cache_metadata = {}
        else:
            # Clear specific entries
            for sim_id in simulation_ids:
                cache_file = f"{self.cache_directory}/{sim_id}.json"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    if sim_id in self.cache_metadata:
                        del self.cache_metadata[sim_id]

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        total_size = sum(meta['size'] for meta in self.cache_metadata.values())
        total_size_gb = total_size / (1024**3)

        return {
            'total_entries': len(self.cache_metadata),
            'total_size_gb': total_size_gb,
            'hit_rate': 0,  # Would track hits vs misses
            'cache_directory': self.cache_directory,
            'max_size_gb': self.max_cache_size_gb
        }
```

## Simulation Validation and Verification

### Validation Techniques

```python
# simulation_validation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Any
import json

class SimulationValidator:
    """
    Validation system for simulation accuracy and realism
    """

    def __init__(self):
        self.validation_metrics = {}
        self.real_world_data = {}
        self.simulation_data = {}
        self.validation_results = {}

    def validate_physics_accuracy(self, real_data: Dict[str, Any], sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate physics simulation accuracy against real data
        """
        validation_results = {}

        # Validate position tracking
        if 'position' in real_data and 'position' in sim_data:
            pos_error = np.array(real_data['position']) - np.array(sim_data['position'])
            pos_rmse = np.sqrt(np.mean(pos_error**2))
            pos_mae = np.mean(np.abs(pos_error))

            validation_results['position'] = {
                'rmse': pos_rmse,
                'mae': pos_mae,
                'max_error': np.max(np.abs(pos_error)),
                'acceptable': pos_rmse < 0.05  # 5cm threshold
            }

        # Validate velocity tracking
        if 'velocity' in real_data and 'velocity' in sim_data:
            vel_error = np.array(real_data['velocity']) - np.array(sim_data['velocity'])
            vel_rmse = np.sqrt(np.mean(vel_error**2))
            vel_mae = np.mean(np.abs(vel_error))

            validation_results['velocity'] = {
                'rmse': vel_rmse,
                'mae': vel_mae,
                'max_error': np.max(np.abs(vel_error)),
                'acceptable': vel_rmse < 0.1  # 0.1 m/s threshold
            }

        # Validate acceleration tracking
        if 'acceleration' in real_data and 'acceleration' in sim_data:
            acc_error = np.array(real_data['acceleration']) - np.array(sim_data['acceleration'])
            acc_rmse = np.sqrt(np.mean(acc_error**2))
            acc_mae = np.mean(np.abs(acc_error))

            validation_results['acceleration'] = {
                'rmse': acc_rmse,
                'mae': acc_mae,
                'max_error': np.max(np.abs(acc_error)),
                'acceptable': acc_rmse < 1.0  # 1 m/s^2 threshold
            }

        overall_acceptable = all(result['acceptable'] for result in validation_results.values())

        return {
            'results': validation_results,
            'overall_acceptable': overall_acceptable,
            'physics_fidelity_score': self._calculate_physics_fidelity(validation_results)
        }

    def _calculate_physics_fidelity(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall physics fidelity score
        """
        if not validation_results:
            return 0.0

        # Calculate weighted score based on different metrics
        weights = {
            'position': 0.4,
            'velocity': 0.3,
            'acceleration': 0.3
        }

        score = 0.0
        total_weight = 0.0

        for metric, results in validation_results.items():
            if metric in weights:
                # Convert RMSE to fidelity score (lower RMSE = higher fidelity)
                rmse_score = max(0.0, 1.0 - results['rmse'])  # Simple normalization
                score += rmse_score * weights[metric]
                total_weight += weights[metric]

        return score / total_weight if total_weight > 0 else 0.0

    def validate_sensor_accuracy(self, real_sensor_data: Dict[str, Any],
                                sim_sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate sensor simulation accuracy
        """
        validation_results = {}

        # Validate camera images
        if 'camera' in real_sensor_data and 'camera' in sim_sensor_data:
            camera_validation = self._validate_camera_accuracy(
                real_sensor_data['camera'], sim_sensor_data['camera']
            )
            validation_results['camera'] = camera_validation

        # Validate LIDAR data
        if 'lidar' in real_sensor_data and 'lidar' in sim_sensor_data:
            lidar_validation = self._validate_lidar_accuracy(
                real_sensor_data['lidar'], sim_sensor_data['lidar']
            )
            validation_results['lidar'] = lidar_validation

        # Validate IMU data
        if 'imu' in real_sensor_data and 'imu' in sim_sensor_data:
            imu_validation = self._validate_imu_accuracy(
                real_sensor_data['imu'], sim_sensor_data['imu']
            )
            validation_results['imu'] = imu_validation

        overall_acceptable = all(result.get('acceptable', False) for result in validation_results.values())

        return {
            'results': validation_results,
            'overall_acceptable': overall_acceptable,
            'sensor_fidelity_score': self._calculate_sensor_fidelity(validation_results)
        }

    def _validate_camera_accuracy(self, real_camera: np.ndarray, sim_camera: np.ndarray) -> Dict[str, Any]:
        """
        Validate camera simulation accuracy
        """
        if real_camera.shape != sim_camera.shape:
            return {'acceptable': False, 'error': 'Shape mismatch'}

        # Calculate various image quality metrics
        mse = np.mean((real_camera - sim_camera) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

        # Structural Similarity Index (simplified)
        ssim = self._calculate_ssim(real_camera, sim_camera)

        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'acceptable': psnr > 30 and ssim > 0.8,  # PSNR > 30dB and SSIM > 0.8
            'image_quality_score': min(1.0, psnr / 50.0)  # Normalize PSNR to 0-1 scale
        }

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate simplified SSIM (Structural Similarity Index)
        """
        # Simplified SSIM calculation - in practice, use skimage.metrics.structural_similarity
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator if denominator != 0 else 0.0
        return ssim

    def _validate_lidar_accuracy(self, real_lidar: Dict[str, Any], sim_lidar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LIDAR simulation accuracy
        """
        real_ranges = real_lidar.get('ranges', [])
        sim_ranges = sim_lidar.get('ranges', [])

        if len(real_ranges) != len(sim_ranges):
            return {'acceptable': False, 'error': 'Range count mismatch'}

        # Calculate range errors
        range_errors = np.abs(np.array(real_ranges) - np.array(sim_ranges))
        rmse = np.sqrt(np.mean(range_errors**2))
        mae = np.mean(range_errors)

        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': np.max(range_errors),
            'acceptable': rmse < 0.05,  # 5cm threshold
            'accuracy_score': max(0.0, 1.0 - rmse)  # Simple accuracy score
        }

    def _validate_imu_accuracy(self, real_imu: Dict[str, Any], sim_imu: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate IMU simulation accuracy
        """
        validation_results = {}

        # Validate orientation
        if 'orientation' in real_imu and 'orientation' in sim_imu:
            real_quat = np.array(real_imu['orientation'])
            sim_quat = np.array(sim_imu['orientation'])

            # Calculate quaternion difference
            quat_diff = self._quaternion_difference(real_quat, sim_quat)
            angle_error_deg = np.degrees(2 * np.arccos(min(1.0, abs(np.dot(real_quat, sim_quat)))))

            validation_results['orientation'] = {
                'angle_error_deg': angle_error_deg,
                'acceptable': angle_error_deg < 5.0  # 5 degree threshold
            }

        # Validate angular velocity
        if 'angular_velocity' in real_imu and 'angular_velocity' in sim_imu:
            real_ang_vel = np.array(real_imu['angular_velocity'])
            sim_ang_vel = np.array(sim_imu['angular_velocity'])

            vel_error = np.linalg.norm(real_ang_vel - sim_ang_vel)
            validation_results['angular_velocity'] = {
                'error': vel_error,
                'acceptable': vel_error < 0.1  # 0.1 rad/s threshold
            }

        # Validate linear acceleration
        if 'linear_acceleration' in real_imu and 'linear_acceleration' in sim_imu:
            real_lin_acc = np.array(real_imu['linear_acceleration'])
            sim_lin_acc = np.array(sim_imu['linear_acceleration'])

            acc_error = np.linalg.norm(real_lin_acc - sim_lin_acc)
            validation_results['linear_acceleration'] = {
                'error': acc_error,
                'acceptable': acc_error < 0.5  # 0.5 m/s^2 threshold
            }

        overall_acceptable = all(result['acceptable'] for result in validation_results.values())

        return {
            'results': validation_results,
            'overall_acceptable': overall_acceptable
        }

    def _quaternion_difference(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Calculate difference between two quaternions
        """
        # Calculate the quaternion that rotates q1 to q2
        q_diff = self._quaternion_multiply(q2, self._quaternion_inverse(q1))
        return np.linalg.norm(q_diff[1:])  # Magnitude of imaginary part

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate quaternion inverse
        """
        return np.array([q[0], -q[1], -q[2], -q[3]]) / np.sum(q**2)

    def _calculate_sensor_fidelity(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall sensor fidelity score
        """
        if not validation_results:
            return 0.0

        weights = {
            'camera': 0.4,
            'lidar': 0.3,
            'imu': 0.3
        }

        score = 0.0
        total_weight = 0.0

        for sensor, results in validation_results.items():
            if sensor in weights and 'image_quality_score' in results.get('results', {}):
                sensor_score = results['results']['image_quality_score']
                score += sensor_score * weights[sensor]
                total_weight += weights[sensor]

        return score / total_weight if total_weight > 0 else 0.0

    def run_comprehensive_validation(self, real_world_data: Dict[str, Any],
                                   simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive validation of simulation
        """
        print("Running comprehensive simulation validation...")

        # Validate physics
        physics_validation = self.validate_physics_accuracy(
            real_world_data.get('physics', {}),
            simulation_data.get('physics', {})
        )

        # Validate sensors
        sensor_validation = self.validate_sensor_accuracy(
            real_world_data.get('sensors', {}),
            simulation_data.get('sensors', {})
        )

        # Calculate overall fidelity score
        overall_fidelity = (
            physics_validation.get('physics_fidelity_score', 0) * 0.6 +
            sensor_validation.get('sensor_fidelity_score', 0) * 0.4
        )

        # Determine validation status
        overall_pass = (
            physics_validation.get('overall_acceptable', False) and
            sensor_validation.get('overall_acceptable', False)
        )

        validation_report = {
            'timestamp': time.time(),
            'physics_validation': physics_validation,
            'sensor_validation': sensor_validation,
            'overall_fidelity_score': overall_fidelity,
            'overall_pass': overall_pass,
            'validation_status': 'PASS' if overall_pass else 'FAIL',
            'recommendations': self._generate_validation_recommendations(
                physics_validation, sensor_validation
            )
        }

        self.validation_results = validation_report
        return validation_report

    def _generate_validation_recommendations(self, physics_results: Dict[str, Any],
                                           sensor_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results
        """
        recommendations = []

        # Physics recommendations
        if not physics_results.get('overall_acceptable', True):
            recommendations.append("Improve physics model accuracy")
            if 'position' in physics_results.get('results', {}):
                pos_result = physics_results['results']['position']
                if not pos_result.get('acceptable', True):
                    recommendations.append(f"Reduce position error (current RMSE: {pos_result['rmse']:.3f})")

        # Sensor recommendations
        if not sensor_results.get('overall_acceptable', True):
            recommendations.append("Improve sensor simulation fidelity")
            if 'camera' in sensor_results.get('results', {}):
                cam_result = sensor_results['results']['camera']
                if not cam_result.get('acceptable', True):
                    recommendations.append(f"Improve camera simulation (PSNR: {cam_result['psnr']:.1f} dB)")

        if not recommendations:
            recommendations.append("Simulation validation passed - no improvements needed")

        return recommendations

    def generate_validation_report(self) -> str:
        """
        Generate detailed validation report
        """
        if not self.validation_results:
            return "No validation results available"

        report_lines = []
        report_lines.append("# Simulation Validation Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall results
        report_lines.append("## Overall Validation Results")
        overall_result = self.validation_results
        report_lines.append(f"- Status: {overall_result['validation_status']}")
        report_lines.append(f"- Overall Fidelity Score: {overall_result['overall_fidelity_score']:.3f}")
        report_lines.append("")

        # Physics validation
        report_lines.append("## Physics Validation")
        physics_results = overall_result['physics_validation']['results']
        for metric, result in physics_results.items():
            report_lines.append(f"- {metric.title()}:")
            report_lines.append(f"  - RMSE: {result['rmse']:.4f}")
            report_lines.append(f"  - MAE: {result['mae']:.4f}")
            report_lines.append(f"  - Acceptable: {result['acceptable']}")
        report_lines.append("")

        # Sensor validation
        report_lines.append("## Sensor Validation")
        sensor_results = overall_result['sensor_validation']['results']
        for sensor, result in sensor_results.items():
            report_lines.append(f"- {sensor.title()}:")
            if 'results' in result:
                for metric, value in result['results'].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  - {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"  - {metric}: {value}")
            report_lines.append(f"  - Acceptable: {result.get('overall_acceptable', False)}")
        report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        for rec in overall_result['recommendations']:
            report_lines.append(f"- {rec}")

        return "\n".join(report_lines)

    def save_validation_report(self, filename: str = "simulation_validation_report.txt"):
        """
        Save validation report to file
        """
        report = self.generate_validation_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Validation report saved to {filename}")
```

## Performance Benchmarking

### Simulation Performance Metrics

```python
# performance_benchmarking.py
import time
import psutil
import GPUtil
import threading
from collections import deque
import matplotlib.pyplot as plt

class SimulationPerformanceBenchmark:
    """
    Performance benchmarking for simulation systems
    """

    def __init__(self):
        self.metrics_history = {
            'real_time_factor': deque(maxlen=1000),
            'frame_rate': deque(maxlen=1000),
            'physics_time': deque(maxlen=1000),
            'render_time': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'network_usage': deque(maxlen=1000)
        }

        self.benchmark_results = {}
        self.is_monitoring = False
        self.monitoring_thread = None

    def start_monitoring(self):
        """
        Start performance monitoring
        """
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """
        Stop performance monitoring
        """
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_performance(self):
        """
        Monitor performance in background thread
        """
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                # Get GPU metrics if available
                gpu_percent = 0
                gpu_memory = 0
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUtil * 100

                # Get network usage
                net_io = psutil.net_io_counters()
                network_usage = net_io.bytes_sent + net_io.bytes_recv

                # Get simulation-specific metrics (would interface with simulation engine)
                rtf = self._get_real_time_factor()
                frame_rate = self._get_frame_rate()
                physics_time = self._get_physics_time()
                render_time = self._get_render_time()

                # Store metrics
                self.metrics_history['real_time_factor'].append(rtf)
                self.metrics_history['frame_rate'].append(frame_rate)
                self.metrics_history['physics_time'].append(physics_time)
                self.metrics_history['render_time'].append(render_time)
                self.metrics_history['cpu_usage'].append(cpu_percent)
                self.metrics_history['memory_usage'].append(memory_percent)
                self.metrics_history['gpu_usage'].append(gpu_percent)
                self.metrics_history['gpu_memory'].append(gpu_memory)
                self.metrics_history['network_usage'].append(network_usage)

                time.sleep(0.1)  # Monitor every 100ms

            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(1.0)

    def _get_real_time_factor(self) -> float:
        """
        Get current real-time factor from simulation
        """
        # This would interface with simulation engine
        # For now, return mock value
        return np.random.uniform(0.8, 1.2)

    def _get_frame_rate(self) -> float:
        """
        Get current frame rate
        """
        # This would interface with rendering system
        return np.random.uniform(30, 60)

    def _get_physics_time(self) -> float:
        """
        Get current physics computation time
        """
        # This would interface with physics engine
        return np.random.uniform(0.001, 0.005)

    def _get_render_time(self) -> float:
        """
        Get current rendering time
        """
        # This would interface with rendering system
        return np.random.uniform(0.005, 0.015)

    def run_performance_benchmark(self, test_duration: float = 60.0) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark
        """
        print(f"Running performance benchmark for {test_duration} seconds...")

        start_time = time.time()
        self.start_monitoring()

        # Run simulation workload
        self._run_workload_simulation(test_duration)

        self.stop_monitoring()
        end_time = time.time()

        # Calculate benchmark results
        benchmark_results = self._calculate_benchmark_results(test_duration)
        self.benchmark_results = benchmark_results

        return benchmark_results

    def _run_workload_simulation(self, duration: float):
        """
        Run simulation with typical workload
        """
        # This would run actual simulation with realistic robot movements
        # For now, just wait for the duration
        time.sleep(duration)

    def _calculate_benchmark_results(self, test_duration: float) -> Dict[str, Any]:
        """
        Calculate benchmark results from collected metrics
        """
        results = {}

        for metric_name, values in self.metrics_history.items():
            if values:
                results[metric_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'count': len(values)
                }
            else:
                results[metric_name] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }

        # Calculate performance scores
        rtf_score = min(1.0, results['real_time_factor']['mean'] / 1.0)
        frame_rate_score = min(1.0, results['frame_rate']['mean'] / 60.0)
        cpu_score = max(0.0, 1.0 - results['cpu_usage']['mean'] / 100.0)

        results['performance_score'] = {
            'overall': (rtf_score + frame_rate_score + cpu_score) / 3.0,
            'real_time_factor': rtf_score,
            'frame_rate': frame_rate_score,
            'cpu_efficiency': cpu_score
        }

        results['test_duration'] = test_duration
        results['benchmark_timestamp'] = time.time()

        return results

    def compare_benchmark_results(self, current_results: Dict[str, Any],
                                 baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current results to baseline
        """
        comparison = {
            'improvements': [],
            'regressions': [],
            'metrics_comparison': {}
        }

        for metric in current_results.keys():
            if metric in baseline_results:
                current_val = current_results[metric].get('mean', 0)
                baseline_val = baseline_results[metric].get('mean', 0)

                if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                    if current_val > baseline_val:
                        comparison['improvements'].append(metric)
                    elif current_val < baseline_val:
                        comparison['regressions'].append(metric)

                    comparison['metrics_comparison'][metric] = {
                        'current': current_val,
                        'baseline': baseline_val,
                        'difference': current_val - baseline_val,
                        'improvement': (current_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else 0
                    }

        return comparison

    def generate_performance_report(self) -> str:
        """
        Generate performance benchmark report
        """
        if not self.benchmark_results:
            return "No benchmark results available"

        report_lines = []
        report_lines.append("# Simulation Performance Benchmark Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        results = self.benchmark_results

        # Performance Summary
        report_lines.append("## Performance Summary")
        report_lines.append(f"- Test Duration: {results['test_duration']:.1f} seconds")
        report_lines.append(f"- Overall Performance Score: {results['performance_score']['overall']:.3f}")
        report_lines.append(f"- Average Real-Time Factor: {results['real_time_factor']['mean']:.3f}")
        report_lines.append(f"- Average Frame Rate: {results['frame_rate']['mean']:.1f} FPS")
        report_lines.append(f"- Average CPU Usage: {results['cpu_usage']['mean']:.1f}%")
        report_lines.append(f"- Average Memory Usage: {results['memory_usage']['mean']:.1f}%")
        report_lines.append("")

        # Detailed Metrics
        report_lines.append("## Detailed Metrics")
        metrics_to_report = [
            'real_time_factor', 'frame_rate', 'physics_time',
            'render_time', 'cpu_usage', 'memory_usage', 'gpu_usage'
        ]

        for metric in metrics_to_report:
            if metric in results:
                metric_data = results[metric]
                report_lines.append(f"- {metric.replace('_', ' ').title()}:")
                report_lines.append(f"  - Mean: {metric_data['mean']:.4f}")
                report_lines.append(f"  - Median: {metric_data['median']:.4f}")
                report_lines.append(f"  - Std Dev: {metric_data['std']:.4f}")
                report_lines.append(f"  - Min: {metric_data['min']:.4f}")
                report_lines.append(f"  - Max: {metric_data['max']:.4f}")
        report_lines.append("")

        # Performance Analysis
        report_lines.append("## Performance Analysis")
        if results['performance_score']['overall'] > 0.8:
            report_lines.append("- Performance: Excellent")
        elif results['performance_score']['overall'] > 0.6:
            report_lines.append("- Performance: Good")
        elif results['performance_score']['overall'] > 0.4:
            report_lines.append("- Performance: Fair")
        else:
            report_lines.append("- Performance: Poor")

        if results['real_time_factor']['mean'] < 0.9:
            report_lines.append("- Real-time performance: Below target (RTF < 1.0)")
        if results['cpu_usage']['mean'] > 80:
            report_lines.append("- CPU usage: High (may need optimization)")
        if results['memory_usage']['mean'] > 85:
            report_lines.append("- Memory usage: High (may need optimization)")

        return "\n".join(report_lines)

    def plot_performance_metrics(self, save_path: str = None):
        """
        Plot performance metrics over time
        """
        if not self.metrics_history:
            print("No metrics history available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Simulation Performance Metrics')

        # Real-time factor
        ax1 = axes[0, 0]
        if self.metrics_history['real_time_factor']:
            values = list(self.metrics_history['real_time_factor'])
            ax1.plot(values)
            ax1.set_title('Real-Time Factor')
            ax1.set_ylabel('RTF')
            ax1.axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
            ax1.legend()

        # Frame rate
        ax2 = axes[0, 1]
        if self.metrics_history['frame_rate']:
            values = list(self.metrics_history['frame_rate'])
            ax2.plot(values)
            ax2.set_title('Frame Rate')
            ax2.set_ylabel('FPS')

        # CPU usage
        ax3 = axes[0, 2]
        if self.metrics_history['cpu_usage']:
            values = list(self.metrics_history['cpu_usage'])
            ax3.plot(values)
            ax3.set_title('CPU Usage')
            ax3.set_ylabel('Percentage (%)')

        # Memory usage
        ax4 = axes[1, 0]
        if self.metrics_history['memory_usage']:
            values = list(self.metrics_history['memory_usage'])
            ax4.plot(values)
            ax4.set_title('Memory Usage')
            ax4.set_ylabel('Percentage (%)')

        # GPU usage
        ax5 = axes[1, 1]
        if self.metrics_history['gpu_usage']:
            values = list(self.metrics_history['gpu_usage'])
            ax5.plot(values)
            ax5.set_title('GPU Usage')
            ax5.set_ylabel('Percentage (%)')

        # Physics time
        ax6 = axes[1, 2]
        if self.metrics_history['physics_time']:
            values = list(self.metrics_history['physics_time'])
            ax6.plot(values)
            ax6.set_title('Physics Computation Time')
            ax6.set_ylabel('Seconds')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Performance plot saved to {save_path}")
        else:
            plt.show()

    def get_performance_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations
        """
        if not self.benchmark_results:
            return ["Run performance benchmark to get recommendations"]

        recommendations = []
        results = self.benchmark_results

        # RTF recommendations
        if results['real_time_factor']['mean'] < 0.9:
            recommendations.append("Consider reducing physics accuracy for better real-time performance")
            recommendations.append("Optimize collision geometry to simpler shapes")
            recommendations.append("Increase physics step size (but check accuracy impact)")

        # CPU recommendations
        if results['cpu_usage']['mean'] > 80:
            recommendations.append("Reduce simulation update rates")
            recommendations.append("Implement more efficient algorithms")
            recommendations.append("Consider parallel processing for non-critical tasks")

        # Memory recommendations
        if results['memory_usage']['mean'] > 85:
            recommendations.append("Optimize data structures and reduce memory allocations")
            recommendations.append("Implement object pooling for frequently created objects")
            recommendations.append("Reduce resolution of high-memory sensors")

        # GPU recommendations
        if results['gpu_usage']['mean'] > 90:
            recommendations.append("Reduce rendering quality or resolution")
            recommendations.append("Implement Level of Detail (LOD) systems")
            recommendations.append("Optimize shader complexity")

        return recommendations if recommendations else ["Performance looks good - no specific optimizations needed"]
```

## Integration Testing

### Comprehensive Integration Testing

```python
# integration_testing.py
import unittest
import time
import numpy as np
from typing import Dict, Any

class IntegrationTestSuite(unittest.TestCase):
    """
    Comprehensive integration test suite for Isaac system
    """

    def setUp(self):
        """
        Set up integration test environment
        """
        self.simulation_manager = SimulationPerformanceOptimizer()
        self.sensor_simulator = HighFidelitySensorSimulation()
        self.domain_randomizer = DomainRandomization()
        self.validator = SimulationValidator()
        self.performance_benchmarker = SimulationPerformanceBenchmark()

    def test_sensor_integration(self):
        """
        Test sensor system integration
        """
        # Create sensor models
        camera_config = {
            'width': 640,
            'height': 480,
            'fov': 60
        }
        camera_model = self.sensor_simulator.create_camera_model('test_camera', camera_config)

        lidar_config = {
            'range_min': 0.1,
            'range_max': 25.0,
            'fov_horizontal': 360,
            'resolution_horizontal': 0.5
        }
        lidar_model = self.sensor_simulator.create_lidar_model('test_lidar', lidar_config)

        # Test sensor data generation
        scene_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        camera_image = self.sensor_simulator.simulate_camera_capture(scene_data, 'test_camera')

        lidar_environment = {'objects': [], 'boundaries': []}
        lidar_data = self.sensor_simulator.simulate_lidar_scan(lidar_environment, 'test_lidar')

        # Verify data integrity
        self.assertIsNotNone(camera_image)
        self.assertEqual(camera_image.shape, (480, 640, 3))
        self.assertIsNotNone(lidar_data)
        self.assertIn('ranges', lidar_data)
        self.assertIn('intensities', lidar_data)

    def test_physics_integration(self):
        """
        Test physics system integration
        """
        # Test physics optimization
        optimized_settings = self.simulation_manager.optimize_physics_settings(target_real_time_factor=1.0)

        # Verify optimization results
        self.assertIn('max_step_size', optimized_settings)
        self.assertIn('solver_iterations', optimized_settings)
        self.assertGreater(optimized_settings['max_step_size'], 0)
        self.assertGreater(optimized_settings['solver_iterations'], 0)

        # Test collision optimization
        collision_settings = self.simulation_manager.optimize_collision_geometry('medium')
        self.assertIn('use_convex_hulls', collision_settings)
        self.assertIn('reduce_triangle_count', collision_settings)

    def test_domain_randomization_integration(self):
        """
        Test domain randomization integration
        """
        # Create test environment configuration
        env_config = {
            'lighting': {'intensity': 1.0, 'direction': [0, 0, -1]},
            'materials': {'friction': 0.5, 'restitution': 0.2},
            'objects': [
                {'position': [1, 1, 0], 'size': [0.1, 0.1, 0.1], 'type': 'cube'}
            ],
            'textures': {'roughness': 0.1, 'metallic': 0.0}
        }

        # Apply domain randomization
        randomized_config = self.domain_randomizer.randomize_environment(env_config)

        # Verify randomization occurred
        self.assertNotEqual(env_config['lighting']['intensity'], randomized_config['lighting']['intensity'])
        self.assertNotEqual(env_config['materials']['friction'], randomized_config['materials']['friction'])

    def test_parallel_simulation_integration(self):
        """
        Test parallel simulation integration
        """
        # Create test simulation configurations
        configs = [
            {'world_file': 'empty.world', 'robot_config': 'config1.yaml'},
            {'world_file': 'simple.world', 'robot_config': 'config2.yaml'},
            {'world_file': 'complex.world', 'robot_config': 'config3.yaml'}
        ]

        # Run parallel simulations
        parallel_manager = ParallelSimulationManager(num_processes=2)
        results = parallel_manager.run_parallel_simulations(configs, num_parallel=2)

        # Verify results
        self.assertEqual(len(results), len(configs))
        for result in results:
            self.assertIn('config', result)
            self.assertIn('success', result)

    def test_validation_integration(self):
        """
        Test validation system integration
        """
        # Create test data
        real_data = {
            'physics': {
                'position': [1.0, 1.0, 0.5],
                'velocity': [0.1, 0.0, 0.0],
                'acceleration': [0.0, 0.0, 0.0]
            },
            'sensors': {
                'camera': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'lidar': {'ranges': [1.0] * 360, 'intensities': [100] * 360},
                'imu': {
                    'orientation': [0, 0, 0, 1],
                    'angular_velocity': [0.0, 0.0, 0.0],
                    'linear_acceleration': [0.0, 0.0, -9.81]
                }
            }
        }

        sim_data = {
            'physics': {
                'position': [1.01, 1.01, 0.51],  # Slightly different
                'velocity': [0.11, 0.01, 0.01],
                'acceleration': [0.01, 0.01, 0.01]
            },
            'sensors': {
                'camera': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'lidar': {'ranges': [1.01] * 360, 'intensities': [101] * 360},
                'imu': {
                    'orientation': [0.01, 0.01, 0.01, 0.999],  # Slightly different
                    'angular_velocity': [0.01, 0.01, 0.01],
                    'linear_acceleration': [0.01, 0.01, -9.8]
                }
            }
        }

        # Run validation
        validation_result = self.validator.run_comprehensive_validation(real_data, sim_data)

        # Verify validation results
        self.assertIn('overall_pass', validation_result)
        self.assertIn('overall_fidelity_score', validation_result)
        self.assertIn('physics_validation', validation_result)
        self.assertIn('sensor_validation', validation_result)

    def test_performance_benchmark_integration(self):
        """
        Test performance benchmark integration
        """
        # Run short performance benchmark
        benchmark_results = self.performance_benchmarker.run_performance_benchmark(test_duration=5.0)

        # Verify benchmark results
        self.assertIn('real_time_factor', benchmark_results)
        self.assertIn('frame_rate', benchmark_results)
        self.assertIn('cpu_usage', benchmark_results)
        self.assertIn('performance_score', benchmark_results)

        # Verify metrics are reasonable
        rtf_data = benchmark_results['real_time_factor']
        self.assertGreater(rtf_data['mean'], 0.0)
        self.assertLess(rtf_data['mean'], 10.0)  # Reasonable RTF range

    def test_cache_management_integration(self):
        """
        Test cache management integration
        """
        cache_manager = SimulationCacheManager("./test_cache")

        # Create test simulation result
        test_result = {
            'position': [1.0, 2.0, 3.0],
            'velocity': [0.1, 0.2, 0.3],
            'timestamp': time.time(),
            'metrics': {'accuracy': 0.95, 'performance': 0.85}
        }

        # Cache result
        cache_manager.cache_simulation_result('test_simulation_001', test_result)

        # Retrieve result
        cached_result = cache_manager.get_cached_result('test_simulation_001')

        # Verify caching worked
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result['position'], test_result['position'])
        self.assertEqual(cached_result['velocity'], test_result['velocity'])

        # Clean up
        import shutil
        shutil.rmtree('./test_cache', ignore_errors=True)


class SystemIntegrationValidator:
    """
    System-level integration validator
    """

    def __init__(self):
        self.subsystem_validators = {}
        self.integration_score = 0.0
        self.integration_issues = []

    def add_subsystem_validator(self, name: str, validator_func: Callable):
        """
        Add a subsystem validator
        """
        self.subsystem_validators[name] = {
            'validator': validator_func,
            'last_result': None,
            'status': 'unknown'
        }

    def validate_full_integration(self) -> Dict[str, Any]:
        """
        Validate full system integration
        """
        validation_results = {}
        all_passed = True

        for subsystem_name, subsystem_info in self.subsystem_validators.items():
            try:
                result = subsystem_info['validator']()
                validation_results[subsystem_name] = result
                subsystem_info['last_result'] = result
                subsystem_info['status'] = 'pass' if result.get('pass', False) else 'fail'

                if not result.get('pass', False):
                    all_passed = False
                    self.integration_issues.append({
                        'subsystem': subsystem_name,
                        'issue': result.get('issues', ['unknown']),
                        'severity': result.get('severity', 'medium')
                    })

            except Exception as e:
                validation_results[subsystem_name] = {
                    'pass': False,
                    'error': str(e)
                }
                subsystem_info['status'] = 'error'
                all_passed = False
                self.integration_issues.append({
                    'subsystem': subsystem_name,
                    'issue': [f'Validator error: {e}'],
                    'severity': 'critical'
                })

        # Calculate overall integration score
        passed_subsystems = sum(1 for result in validation_results.values() if result.get('pass', False))
        total_subsystems = len(validation_results)
        self.integration_score = passed_subsystems / total_subsystems if total_subsystems > 0 else 0.0

        return {
            'overall_status': 'pass' if all_passed else 'fail',
            'integration_score': self.integration_score,
            'subsystem_results': validation_results,
            'issues': self.integration_issues,
            'total_subsystems': total_subsystems,
            'passed_subsystems': passed_subsystems
        }

    def generate_integration_report(self) -> str:
        """
        Generate system integration validation report
        """
        validation_result = self.validate_full_integration()

        report_lines = []
        report_lines.append("# System Integration Validation Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall status
        report_lines.append(f"## Overall Status: {validation_result['overall_status'].upper()}")
        report_lines.append(f"Integration Score: {validation_result['integration_score']:.2f}")
        report_lines.append(f"Subsystems: {validation_result['passed_subsystems']}/{validation_result['total_subsystems']} passed")
        report_lines.append("")

        # Subsystem details
        report_lines.append("## Subsystem Validation Results:")
        for subsystem, result in validation_result['subsystem_results'].items():
            status = "PASS" if result.get('pass', False) else "FAIL"
            report_lines.append(f"- {subsystem}: {status}")
            if 'issues' in result:
                for issue in result['issues']:
                    report_lines.append(f"  - {issue}")
        report_lines.append("")

        # Issues summary
        if validation_result['issues']:
            report_lines.append("## Integration Issues:")
            for issue in validation_result['issues']:
                report_lines.append(f"- {issue['subsystem']}: {', '.join(issue['issue'])} ({issue['severity']})")
        else:
            report_lines.append("## Integration Status: All subsystems validated successfully!")

        return "\n".join(report_lines)

    def get_integration_recommendations(self) -> List[str]:
        """
        Get integration recommendations based on validation results
        """
        recommendations = []

        if self.integration_score < 0.8:
            recommendations.append("Major integration issues detected - prioritize fixes")
        elif self.integration_score < 0.95:
            recommendations.append("Some integration issues - consider improvements")
        else:
            recommendations.append("Integration looks good - minor optimizations possible")

        # Add specific recommendations based on issues
        for issue in self.integration_issues:
            if issue['severity'] == 'critical':
                recommendations.append(f"Critical issue in {issue['subsystem']}: {', '.join(issue['issue'])}")
            elif issue['severity'] == 'high':
                recommendations.append(f"High priority fix needed for {issue['subsystem']}")

        return recommendations


def run_integration_tests():
    """
    Run the complete integration test suite
    """
    print("Running Isaac System Integration Tests...")

    # Create test suite
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTestSuite))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate integration validation report
    integrator = SystemIntegrationValidator()

    # Add validators for different subsystems
    def validate_sensors():
        return {'pass': True, 'issues': [], 'severity': 'low'}

    def validate_physics():
        return {'pass': True, 'issues': [], 'severity': 'low'}

    def validate_communication():
        return {'pass': True, 'issues': [], 'severity': 'low'}

    def validate_safety():
        return {'pass': True, 'issues': [], 'severity': 'low'}

    integrator.add_subsystem_validator('sensors', validate_sensors)
    integrator.add_subsystem_validator('physics', validate_physics)
    integrator.add_subsystem_validator('communication', validate_communication)
    integrator.add_subsystem_validator('safety', validate_safety)

    # Generate and print integration report
    integration_report = integrator.generate_integration_report()
    print("\n" + integration_report)

    # Print recommendations
    recommendations = integrator.get_integration_recommendations()
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")

    # Return success status
    return result.wasSuccessful() and integrator.integration_score >= 0.95


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
```

## Troubleshooting Integration Issues

### Common Integration Problems and Solutions

```python
# integration_troubleshooting.py
class IntegrationTroubleshooter:
    """
    Troubleshooting guide for common integration issues
    """

    def __init__(self):
        self.known_issues = {
            'communication_timeout': {
                'symptoms': ['ROS communication timeouts', 'No sensor data', 'Controller not responding'],
                'causes': ['Network issues', 'Firewall blocking', 'Wrong ROS master URI'],
                'solutions': [
                    'Check network connectivity',
                    'Verify ROS_IP and ROS_MASTER_URI settings',
                    'Disable firewall temporarily for testing',
                    'Use ROS network diagnostics tools'
                ]
            },
            'physics_instability': {
                'symptoms': ['Robot shaking', 'Joint oscillations', 'Unstable simulation'],
                'causes': ['High PID gains', 'Low physics update rate', 'Inappropriate time step'],
                'solutions': [
                    'Reduce PID gains',
                    'Increase physics update rate',
                    'Decrease time step size',
                    'Check joint limits and constraints'
                ]
            },
            'sensor_noise_excessive': {
                'symptoms': ['Noisy sensor readings', 'Inconsistent measurements', 'Poor perception'],
                'causes': ['High noise parameters', 'Insufficient filtering', 'Hardware issues'],
                'solutions': [
                    'Reduce noise parameters in simulation',
                    'Implement sensor filtering',
                    'Check sensor mounting and calibration',
                    'Verify sensor model parameters'
                ]
            },
            'performance_degradation': {
                'symptoms': ['Low frame rate', 'High CPU usage', 'Long simulation time'],
                'causes': ['Complex models', 'High resolution sensors', 'Inefficient algorithms'],
                'solutions': [
                    'Simplify collision geometry',
                    'Reduce sensor resolution',
                    'Optimize physics parameters',
                    'Use Level of Detail (LOD) systems'
                ]
            },
            'synchronization_issues': {
                'symptoms': ['Timing problems', 'Desynced components', 'Inconsistent states'],
                'causes': ['Different update rates', 'Clock synchronization', 'Threading issues'],
                'solutions': [
                    'Use common clock source',
                    'Synchronize update rates',
                    'Implement proper threading',
                    'Use message synchronization'
                ]
            }
        }

    def diagnose_issue(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        Diagnose issue based on symptoms
        """
        potential_issues = []

        for issue_name, issue_info in self.known_issues.items():
            symptom_matches = sum(1 for symptom in symptoms if symptom in issue_info['symptoms'])
            match_ratio = symptom_matches / len(issue_info['symptoms'])

            if match_ratio >= 0.5:  # At least 50% symptom match
                potential_issues.append({
                    'issue': issue_name,
                    'confidence': match_ratio,
                    'causes': issue_info['causes'],
                    'solutions': issue_info['solutions']
                })

        # Sort by confidence
        potential_issues.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'potential_issues': potential_issues,
            'recommended_solutions': self._get_recommended_solutions(potential_issues)
        }

    def _get_recommended_solutions(self, potential_issues: List[Dict[str, Any]]) -> List[str]:
        """
        Get recommended solutions based on potential issues
        """
        if not potential_issues:
            return ['No specific issues identified - run general diagnostics']

        # Get solutions for highest confidence issue
        primary_issue = potential_issues[0]
        return primary_issue['solutions']

    def run_diagnostic_procedures(self) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic procedures
        """
        diagnostics = {
            'network_status': self._check_network_connectivity(),
            'system_resources': self._check_system_resources(),
            'component_health': self._check_component_health(),
            'simulation_performance': self._check_simulation_performance(),
            'sensor_status': self._check_sensor_status(),
            'actuator_status': self._check_actuator_status()
        }

        # Identify issues based on diagnostic results
        issues = self._identify_issues_from_diagnostics(diagnostics)

        return {
            'diagnostics': diagnostics,
            'identified_issues': issues,
            'recommended_actions': self._generate_recommended_actions(issues)
        }

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """
        Check network connectivity for ROS communication
        """
        import socket
        import subprocess

        results = {
            'ros_master_reachable': False,
            'network_latency': 0.0,
            'bandwidth_available': 0.0
        }

        try:
            # Check if ROS master is reachable
            ros_master_uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
            # Parse ROS master URI to get host and port
            if 'localhost' in ros_master_uri or '127.0.0.1' in ros_master_uri:
                results['ros_master_reachable'] = True
            else:
                # Test actual connectivity
                host = ros_master_uri.split('//')[1].split(':')[0]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((host, 11311))
                results['ros_master_reachable'] = (result == 0)
                sock.close()
        except Exception as e:
            results['error'] = f"Network check failed: {e}"

        return results

    def _check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resource availability
        """
        import psutil
        import GPUtil

        resources = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'gpus_available': []
        }

        # Check GPU availability and usage
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                resources['gpus_available'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_util': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature
                })
        except ImportError:
            resources['gpus_available'] = 'GPUtil not installed'

        return resources

    def _check_component_health(self) -> Dict[str, Any]:
        """
        Check health of system components
        """
        # This would check ROS nodes, processes, etc.
        # For now, return mock data
        return {
            'ros_nodes_active': 15,
            'processes_running': 25,
            'memory_allocated': 2.5,  # GB
            'threads_active': 120
        }

    def _check_simulation_performance(self) -> Dict[str, Any]:
        """
        Check simulation performance metrics
        """
        # This would interface with simulation engine
        # For now, return mock data
        return {
            'real_time_factor': 0.95,
            'frame_rate': 45.0,
            'physics_time_ms': 2.5,
            'render_time_ms': 15.0
        }

    def _check_sensor_status(self) -> Dict[str, Any]:
        """
        Check sensor system status
        """
        # This would check sensor data availability and quality
        # For now, return mock data
        return {
            'sensors_active': 8,
            'data_rate_ok': True,
            'calibration_valid': True,
            'noise_levels_acceptable': True
        }

    def _check_actuator_status(self) -> Dict[str, Any]:
        """
        Check actuator system status
        """
        # This would check actuator health and control
        # For now, return mock data
        return {
            'actuators_enabled': 24,
            'control_loop_active': True,
            'safety_limits_active': True,
            'temperature_normal': True
        }

    def _identify_issues_from_diagnostics(self, diagnostics: Dict[str, Any]) -> List[str]:
        """
        Identify potential issues from diagnostic results
        """
        issues = []

        # Check network issues
        if not diagnostics['network_status'].get('ros_master_reachable', True):
            issues.append('communication_timeout')

        # Check resource issues
        resources = diagnostics['system_resources']
        if resources.get('cpu_usage', 0) > 90:
            issues.append('performance_degradation')
        if resources.get('memory_usage', 0) > 95:
            issues.append('performance_degradation')

        # Check simulation performance issues
        perf = diagnostics['simulation_performance']
        if perf.get('real_time_factor', 1.0) < 0.8:
            issues.append('performance_degradation')

        return issues

    def _generate_recommended_actions(self, issues: List[str]) -> List[str]:
        """
        Generate recommended actions based on identified issues
        """
        actions = []

        for issue in issues:
            if issue in self.known_issues:
                actions.extend(self.known_issues[issue]['solutions'])

        # Remove duplicates while preserving order
        unique_actions = []
        for action in actions:
            if action not in unique_actions:
                unique_actions.append(action)

        return unique_actions

    def get_troubleshooting_guide(self, issue_category: str) -> str:
        """
        Get detailed troubleshooting guide for specific issue category
        """
        if issue_category not in self.known_issues:
            return f"No troubleshooting guide available for: {issue_category}"

        issue_info = self.known_issues[issue_category]

        guide = f"""
# Troubleshooting Guide: {issue_category.replace('_', ' ').title()}

## Symptoms
"""
        for symptom in issue_info['symptoms']:
            guide += f"- {symptom}\n"

        guide += "\n## Likely Causes\n"
        for cause in issue_info['causes']:
            guide += f"- {cause}\n"

        guide += "\n## Recommended Solutions\n"
        for i, solution in enumerate(issue_info['solutions'], 1):
            guide += f"{i}. {solution}\n"

        return guide

    def run_system_health_check(self) -> str:
        """
        Run comprehensive system health check
        """
        diagnostics = self.run_diagnostic_procedures()

        health_report = []
        health_report.append("# System Health Check Report")
        health_report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        health_report.append("")

        # System resources
        resources = diagnostics['diagnostics']['system_resources']
        health_report.append("## System Resources")
        health_report.append(f"- CPU Usage: {resources['cpu_usage']:.1f}%")
        health_report.append(f"- Memory Usage: {resources['memory_usage']:.1f}%")
        health_report.append(f"- Disk Usage: {resources['disk_usage']:.1f}%")
        health_report.append(f"- Available Memory: {resources['available_memory_gb']:.1f} GB")

        if resources['cpu_usage'] > 80:
            health_report.append("⚠️ High CPU usage detected")
        if resources['memory_usage'] > 85:
            health_report.append("⚠️ High memory usage detected")
        health_report.append("")

        # Simulation performance
        perf = diagnostics['diagnostics']['simulation_performance']
        health_report.append("## Simulation Performance")
        health_report.append(f"- Real-time Factor: {perf['real_time_factor']:.2f}")
        health_report.append(f"- Frame Rate: {perf['frame_rate']:.1f} FPS")
        health_report.append(f"- Physics Time: {perf['physics_time_ms']:.1f} ms")
        health_report.append(f"- Render Time: {perf['render_time_ms']:.1f} ms")

        if perf['real_time_factor'] < 0.9:
            health_report.append("⚠️ Real-time performance below target")
        health_report.append("")

        # Identified issues
        issues = diagnostics['identified_issues']
        if issues:
            health_report.append("## Identified Issues")
            for issue in issues:
                guide = self.get_troubleshooting_guide(issue)
                health_report.append(f"- {issue}")
                health_report.append(f"  Recommended: {', '.join(self.known_issues[issue]['solutions'][:2])}")
            health_report.append("")
        else:
            health_report.append("## Status: No issues detected")
            health_report.append("")

        # Recommendations
        health_report.append("## Recommendations")
        for action in diagnostics['recommended_actions'][:5]:  # Show top 5
            health_report.append(f"- {action}")

        return "\n".join(health_report)
```

## Week Summary

This section covered comprehensive simulation tips and tricks for Physical AI and humanoid robotics systems. We explored performance optimization techniques, domain randomization for training, sensor simulation optimization, parallel simulation strategies, validation and verification procedures, and troubleshooting methodologies. These advanced simulation techniques are essential for creating realistic, efficient, and effective simulation environments that bridge the gap between virtual development and real-world deployment of humanoid robots.