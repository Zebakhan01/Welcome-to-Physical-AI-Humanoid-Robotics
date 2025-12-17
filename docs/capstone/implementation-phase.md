---
sidebar_position: 3
---

# Implementation Phase

## Introduction to Implementation

The implementation phase transforms the planned autonomous humanoid system into a functional reality. This phase involves developing, integrating, and testing all system components according to the architectural design and project plan. Success in this phase requires careful attention to detail, systematic development practices, and continuous validation.

## Phase 1: Component Development

### Perception System Implementation

#### Vision Processing Pipeline

```python
# Vision Processing Component Implementation
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisionProcessor:
    """
    Vision processing component for humanoid robot perception
    """

    def __init__(self, config: Dict):
        self.config = config
        self.bridge = CvBridge()

        # Initialize deep learning models
        self.object_detector = self._load_object_detector()
        self.pose_estimator = self._load_pose_estimator()
        self.depth_estimator = self._load_depth_estimator()

        # Initialize ROS subscribers
        self.image_sub = rospy.Subscriber(
            config['camera_topic'],
            Image,
            self._image_callback
        )

        # Processing parameters
        self.image_queue = []
        self.max_queue_size = config.get('max_queue_size', 5)
        self.processing_rate = config.get('processing_rate', 10)  # Hz

    def _load_object_detector(self):
        """Load pre-trained object detection model"""
        # Example using YOLO or similar
        import torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model

    def _load_pose_estimator(self):
        """Load human pose estimation model"""
        # Example using OpenPose or similar
        return None  # Placeholder

    def _load_depth_estimator(self):
        """Load depth estimation model"""
        # Example using MiDaS or similar
        return None  # Placeholder

    def _image_callback(self, msg: Image):
        """ROS callback for image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Add to processing queue
            if len(self.image_queue) < self.max_queue_size:
                self.image_queue.append({
                    'image': cv_image,
                    'timestamp': msg.header.stamp,
                    'frame_id': msg.header.frame_id
                })
            else:
                # Remove oldest image if queue is full
                self.image_queue.pop(0)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def process_frame(self) -> Dict:
        """Process the latest frame and return results"""
        if not self.image_queue:
            return {}

        # Get latest image
        frame_data = self.image_queue[-1]
        image = frame_data['image']

        # Run object detection
        object_results = self._detect_objects(image)

        # Run pose estimation
        pose_results = self._estimate_pose(image)

        # Run depth estimation
        depth_results = self._estimate_depth(image)

        return {
            'objects': object_results,
            'poses': pose_results,
            'depth': depth_results,
            'timestamp': frame_data['timestamp'],
            'frame_id': frame_data['frame_id']
        }

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in the image"""
        # Preprocess image
        results = self.object_detector(image)

        # Extract bounding boxes and labels
        detections = []
        for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
            x1, y1, x2, y2, conf, cls = detection
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.object_detector.names[int(cls)]
            })

        return detections

    def _estimate_pose(self, image: np.ndarray) -> List[Dict]:
        """Estimate human poses in the image"""
        # Placeholder implementation
        return []

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from monocular image"""
        # Placeholder implementation
        return np.zeros((image.shape[0], image.shape[1]))

class SensorFusion:
    """
    Multi-modal sensor fusion component
    """

    def __init__(self, config: Dict):
        self.config = config
        self.fusion_weights = config.get('fusion_weights', {})
        self.tracking_system = ObjectTracker(config)

    def fuse_sensors(self,
                    vision_data: Dict,
                    lidar_data: Dict,
                    other_sensors: Dict) -> Dict:
        """Fuse data from multiple sensors"""

        # Fuse object detections from vision and LIDAR
        fused_objects = self._fuse_object_detections(
            vision_data.get('objects', []),
            lidar_data.get('objects', [])
        )

        # Fuse pose estimates
        fused_poses = self._fuse_pose_estimates(
            vision_data.get('poses', []),
            other_sensors.get('poses', [])
        )

        # Update object tracking
        tracked_objects = self.tracking_system.update(
            fused_objects, vision_data['timestamp']
        )

        return {
            'fused_objects': fused_objects,
            'fused_poses': fused_poses,
            'tracked_objects': tracked_objects,
            'environment_map': self._build_environment_map(
                fused_objects, fused_poses
            )
        }

    def _fuse_object_detections(self, vision_objects: List, lidar_objects: List) -> List:
        """Fuse object detections from vision and LIDAR"""
        # Simple fusion based on spatial overlap and confidence
        fused_objects = []

        for vision_obj in vision_objects:
            best_match = None
            best_score = 0

            for lidar_obj in lidar_objects:
                # Calculate spatial overlap score
                overlap_score = self._calculate_overlap_score(vision_obj, lidar_obj)

                if overlap_score > best_score:
                    best_score = overlap_score
                    best_match = lidar_obj

            if best_match and best_score > self.config.get('fusion_threshold', 0.5):
                # Combine vision and LIDAR information
                fused_obj = self._combine_object_info(vision_obj, best_match)
                fused_objects.append(fused_obj)
            else:
                # Use vision-only detection
                fused_objects.append(vision_obj)

        # Add LIDAR-only detections that didn't match
        for lidar_obj in lidar_objects:
            matched = False
            for fused_obj in fused_objects:
                overlap_score = self._calculate_overlap_score(fused_obj, lidar_obj)
                if overlap_score > self.config.get('fusion_threshold', 0.5):
                    matched = True
                    break

            if not matched:
                fused_objects.append(lidar_obj)

        return fused_objects

    def _calculate_overlap_score(self, obj1: Dict, obj2: Dict) -> float:
        """Calculate overlap score between two objects"""
        # Simple 2D bounding box overlap
        x1_1, y1_1, x2_1, y2_1 = obj1['bbox']
        x1_2, y1_2, x2_2, y2_2 = obj2['bbox']

        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x1_int < x2_int and y1_int < y2_int:
            intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area

            return intersection_area / union_area if union_area > 0 else 0
        else:
            return 0

    def _combine_object_info(self, vision_obj: Dict, lidar_obj: Dict) -> Dict:
        """Combine information from vision and LIDAR objects"""
        combined = vision_obj.copy()

        # Average position information
        if 'position' in lidar_obj:
            if 'position' in combined:
                # Weighted average based on confidence
                vision_weight = combined['confidence']
                lidar_weight = lidar_obj.get('confidence', 0.8)

                total_weight = vision_weight + lidar_weight
                combined['position'] = [
                    (vision_weight * combined['position'][0] +
                     lidar_weight * lidar_obj['position'][0]) / total_weight,
                    (vision_weight * combined['position'][1] +
                     lidar_weight * lidar_obj['position'][1]) / total_weight,
                    (vision_weight * combined['position'][2] +
                     lidar_weight * lidar_obj['position'][2]) / total_weight
                ]
            else:
                combined['position'] = lidar_obj['position']

        # Update confidence as average of both
        combined['confidence'] = (combined['confidence'] +
                                lidar_obj.get('confidence', 0.8)) / 2

        return combined

class ObjectTracker:
    """
    Object tracking component for maintaining object states over time
    """

    def __init__(self, config: Dict):
        self.config = config
        self.tracks = {}  # {track_id: track_data}
        self.next_track_id = 0
        self.max_displacement = config.get('max_displacement', 1.0)  # meters
        self.max_time_gap = config.get('max_time_gap', 2.0)  # seconds

    def update(self, detections: List[Dict], timestamp: rospy.Time) -> List[Dict]:
        """Update object tracks with new detections"""
        # Associate detections with existing tracks
        associations = self._associate_detections(detections)

        # Update existing tracks
        for det_idx, track_id in associations.items():
            if track_id in self.tracks:
                self._update_track(self.tracks[track_id], detections[det_idx], timestamp)

        # Create new tracks for unassociated detections
        associated_dets = set(associations.keys())
        for i, detection in enumerate(detections):
            if i not in associated_dets:
                new_track_id = self._create_new_track(detection, timestamp)
                associations[i] = new_track_id

        # Remove old tracks
        self._cleanup_tracks(timestamp)

        # Return active tracks
        return [track for track in self.tracks.values() if track['active']]

    def _associate_detections(self, detections: List[Dict]) -> Dict[int, int]:
        """Associate detections with existing tracks"""
        associations = {}

        for det_idx, detection in enumerate(detections):
            best_track_id = None
            best_score = 0

            for track_id, track in self.tracks.items():
                if not track['active']:
                    continue

                score = self._calculate_association_score(detection, track)

                if score > best_score and score > self.config.get('association_threshold', 0.3):
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None:
                associations[det_idx] = best_track_id

        return associations

    def _calculate_association_score(self, detection: Dict, track: Dict) -> float:
        """Calculate association score between detection and track"""
        # Calculate spatial distance
        if 'position' in detection and 'position' in track:
            pos_det = np.array(detection['position'])
            pos_track = np.array(track['position'])
            distance = np.linalg.norm(pos_det - pos_track)

            # Convert to similarity score (higher is better)
            spatial_score = max(0, 1 - distance / self.max_displacement)
        else:
            spatial_score = 0.5  # Default if no position info

        # Calculate class similarity
        class_score = 1.0 if (detection.get('class_name') == track.get('class_name')) else 0.7

        # Combine scores
        return (spatial_score * 0.7 + class_score * 0.3)

    def _update_track(self, track: Dict, detection: Dict, timestamp: rospy.Time):
        """Update track with new detection"""
        # Update position with weighted average
        dt = (timestamp - track['last_update']).to_sec()

        if 'position' in detection:
            alpha = self.config.get('position_smoothing', 0.3)
            if 'position' in track:
                track['position'] = [
                    alpha * detection['position'][0] + (1 - alpha) * track['position'][0],
                    alpha * detection['position'][1] + (1 - alpha) * track['position'][1],
                    alpha * detection['position'][2] + (1 - alpha) * track['position'][2]
                ]
            else:
                track['position'] = detection['position']

        # Update other attributes
        track['class_name'] = detection.get('class_name', track['class_name'])
        track['confidence'] = detection.get('confidence', track['confidence'])
        track['last_update'] = timestamp
        track['age'] += dt
        track['active'] = True

    def _create_new_track(self, detection: Dict, timestamp: rospy.Time) -> int:
        """Create a new track for detection"""
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            'id': track_id,
            'position': detection.get('position', [0, 0, 0]),
            'class_name': detection.get('class_name', 'unknown'),
            'confidence': detection.get('confidence', 0.8),
            'first_seen': timestamp,
            'last_update': timestamp,
            'age': 0,
            'active': True,
            'detections': [detection]
        }

        return track_id

    def _cleanup_tracks(self, timestamp: rospy.Time):
        """Remove old or inactive tracks"""
        current_time = timestamp.to_sec()

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]

            # Remove tracks that haven't been updated recently
            time_since_update = current_time - track['last_update'].to_sec()
            if time_since_update > self.max_time_gap:
                track['active'] = False

            # Remove very old inactive tracks
            if not track['active'] and track['age'] > 10:  # 10 seconds
                del self.tracks[track_id]
```

### Cognition System Implementation

#### Natural Language Understanding

```python
# Natural Language Understanding Component
import spacy
import torch
import transformers
from typing import Dict, List, Tuple
import rospy
from std_msgs.msg import String

class NaturalLanguageUnderstanding:
    """
    Natural language understanding component for humanoid robot
    """

    def __init__(self, config: Dict):
        self.config = config

        # Load NLP models
        self.nlp_model = spacy.load("en_core_web_sm")
        self.transformer_model = transformers.pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium"
        )

        # Initialize ROS interfaces
        self.command_sub = rospy.Subscriber(
            config['command_topic'],
            String,
            self._command_callback
        )

        # Task vocabulary and parsing rules
        self.task_vocabulary = self._load_task_vocabulary()
        self.action_templates = self._load_action_templates()

    def _load_task_vocabulary(self) -> Dict:
        """Load task vocabulary and semantic mappings"""
        return {
            'navigation': ['go to', 'move to', 'navigate to', 'walk to', 'drive to'],
            'manipulation': ['pick up', 'grasp', 'take', 'lift', 'hold', 'place', 'put'],
            'interaction': ['greet', 'say hello', 'introduce', 'talk to', 'meet'],
            'observation': ['find', 'look for', 'search for', 'locate', 'see']
        }

    def _load_action_templates(self) -> Dict:
        """Load action templates for command parsing"""
        return {
            'navigation': {
                'template': 'navigate_to(location)',
                'required_args': ['location'],
                'optional_args': ['speed', 'avoid_obstacles']
            },
            'manipulation': {
                'template': 'manipulate_object(object, action, location)',
                'required_args': ['object', 'action'],
                'optional_args': ['location', 'force']
            },
            'interaction': {
                'template': 'interact_with(person, action)',
                'required_args': ['person', 'action'],
                'optional_args': ['duration']
            }
        }

    def _command_callback(self, msg: String):
        """ROS callback for natural language commands"""
        try:
            # Parse the command
            parsed_command = self.parse_command(msg.data)

            # Validate command
            if self.validate_command(parsed_command):
                # Publish parsed command to action system
                self._publish_parsed_command(parsed_command)
            else:
                rospy.logwarn(f"Invalid command: {msg.data}")

        except Exception as e:
            rospy.logerr(f"Error parsing command: {e}")

    def parse_command(self, command: str) -> Dict:
        """Parse natural language command into structured format"""
        # Process with spaCy
        doc = self.nlp_model(command.lower())

        # Extract entities and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [token.text for token in doc]

        # Determine command type
        command_type = self._determine_command_type(command)

        # Extract arguments based on command type
        arguments = self._extract_arguments(command, command_type)

        return {
            'raw_command': command,
            'command_type': command_type,
            'arguments': arguments,
            'entities': entities,
            'tokens': tokens,
            'confidence': self._calculate_parse_confidence(command)
        }

    def _determine_command_type(self, command: str) -> str:
        """Determine the type of command from vocabulary"""
        command_lower = command.lower()

        for cmd_type, keywords in self.task_vocabulary.items():
            for keyword in keywords:
                if keyword in command_lower:
                    return cmd_type

        return 'unknown'

    def _extract_arguments(self, command: str, command_type: str) -> Dict:
        """Extract arguments from command based on type"""
        doc = self.nlp_model(command)
        arguments = {}

        if command_type == 'navigation':
            # Look for location entities (GPE, LOC, etc.)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical entity, location, facility
                    arguments['location'] = ent.text
                    break
            else:
                # If no named entity, look for noun phrases that might be locations
                for token in doc:
                    if token.pos_ == 'NOUN' and token.dep_ == 'pobj':
                        arguments['location'] = token.text
                        break

        elif command_type == 'manipulation':
            # Extract object and action
            for token in doc:
                if token.pos_ == 'VERB':
                    arguments['action'] = token.lemma_

            for ent in doc.ents:
                if ent.label_ in ['OBJECT', 'PRODUCT']:  # Custom object labels
                    arguments['object'] = ent.text
                    break
            else:
                # Look for direct objects
                for token in doc:
                    if token.dep_ == 'dobj':
                        arguments['object'] = token.text
                        break

        elif command_type == 'interaction':
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG']:  # Person or organization
                    arguments['person'] = ent.text
                    break
            else:
                # Look for person references
                for token in doc:
                    if token.text in ['person', 'someone', 'human', 'you'] or token.pos_ == 'PROPN':
                        arguments['person'] = token.text
                        break

        return arguments

    def _calculate_parse_confidence(self, command: str) -> float:
        """Calculate confidence in command parsing"""
        # Simple confidence based on entity recognition
        doc = self.nlp_model(command)
        entities_found = len(doc.ents)
        tokens = len([t for t in doc if not t.is_punct])

        confidence = min(1.0, entities_found * 0.3 + 0.4)  # Base confidence
        return confidence

    def validate_command(self, parsed_command: Dict) -> bool:
        """Validate parsed command against template requirements"""
        cmd_type = parsed_command['command_type']

        if cmd_type in self.action_templates:
            template = self.action_templates[cmd_type]
            required_args = template['required_args']
            provided_args = list(parsed_command['arguments'].keys())

            # Check if all required arguments are provided
            for required_arg in required_args:
                if required_arg not in provided_args:
                    return False

            return True

        return False

    def _publish_parsed_command(self, parsed_command: Dict):
        """Publish parsed command to action system"""
        # This would publish to a ROS topic for the action system
        pass

class TaskPlanner:
    """
    Task planning component for high-level goal achievement
    """

    def __init__(self, config: Dict):
        self.config = config
        self.knowledge_base = self._load_knowledge_base()
        self.planning_graph = None

    def _load_knowledge_base(self) -> Dict:
        """Load domain knowledge for planning"""
        return {
            'objects': {
                'cup': {'graspable': True, 'pourable': False, 'size': 'small'},
                'bottle': {'graspable': True, 'pourable': True, 'size': 'medium'},
                'table': {'graspable': False, 'surface': True, 'size': 'large'},
                'chair': {'graspable': False, 'sittable': True, 'size': 'medium'}
            },
            'locations': {
                'kitchen': {'accessible': True, 'contains': ['cup', 'bottle']},
                'living_room': {'accessible': True, 'contains': ['chair', 'table']},
                'bedroom': {'accessible': True, 'contains': ['bed', 'dresser']}
            },
            'actions': {
                'navigate_to': {'preconditions': [], 'effects': ['at_location']},
                'pick_up': {'preconditions': ['at_location', 'object_present'], 'effects': ['holding_object']},
                'place': {'preconditions': ['holding_object'], 'effects': ['object_placed']},
                'pour': {'preconditions': ['holding_pourable'], 'effects': ['liquid_poured']}
            }
        }

    def plan_task(self, goal: Dict, current_state: Dict) -> List[Dict]:
        """Plan a sequence of actions to achieve the goal"""
        # Use a planning algorithm (e.g., A* search, STRIPS)
        plan = self._search_plan(goal, current_state)
        return plan

    def _search_plan(self, goal: Dict, current_state: Dict) -> List[Dict]:
        """Search for a plan using A* algorithm"""
        # Define state representation
        start_state = self._encode_state(current_state)
        goal_state = self._encode_state(goal)

        # Define heuristic function
        def heuristic(state):
            return self._calculate_heuristic(state, goal_state)

        # A* search implementation
        open_set = [(0, start_state, [])]  # (f_score, state, path)
        closed_set = set()

        while open_set:
            open_set.sort(key=lambda x: x[0])  # Sort by f_score
            current_f, current_state, current_path = open_set.pop(0)

            if self._is_goal_state(current_state, goal_state):
                return current_path

            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            # Generate possible next states
            for action in self._get_applicable_actions(current_state):
                next_state = self._apply_action(current_state, action)
                next_path = current_path + [action]

                if next_state not in closed_set:
                    g_score = len(next_path)  # Simple step count
                    h_score = heuristic(next_state)
                    f_score = g_score + h_score

                    open_set.append((f_score, next_state, next_path))

        return []  # No plan found

    def _encode_state(self, state_dict: Dict) -> str:
        """Encode state dictionary as string for search"""
        # Simple encoding - in practice, this would be more sophisticated
        return str(sorted(state_dict.items()))

    def _is_goal_state(self, current_state: str, goal_state: str) -> bool:
        """Check if current state satisfies goal conditions"""
        # Check if goal conditions are met in current state
        # This is a simplified implementation
        return goal_state in current_state

    def _get_applicable_actions(self, state: str) -> List[Dict]:
        """Get actions applicable in current state"""
        # Return list of actions that can be applied in current state
        # This would check preconditions against current state
        applicable_actions = []

        for action_name, action_def in self.knowledge_base['actions'].items():
            # Check if action preconditions are satisfied
            if self._check_preconditions(state, action_def['preconditions']):
                applicable_actions.append({
                    'name': action_name,
                    'preconditions': action_def['preconditions'],
                    'effects': action_def['effects']
                })

        return applicable_actions

    def _check_preconditions(self, state: str, preconditions: List[str]) -> bool:
        """Check if preconditions are satisfied in state"""
        # This would be more sophisticated in practice
        return True

    def _apply_action(self, state: str, action: Dict) -> str:
        """Apply action to state and return new state"""
        # Apply action effects to current state
        # This would update the state representation
        return state + f"_{action['name']}"
```

### Action System Implementation

#### Motion Control and Navigation

```python
# Motion Control and Navigation Component
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class MotionController:
    """
    Motion control component for humanoid robot navigation and locomotion
    """

    def __init__(self, config: Dict):
        self.config = config

        # Initialize ROS interfaces
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self._scan_callback)

        # Initialize move_base client for navigation
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        # Robot state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.scan_data = None
        self.is_moving = False

    def _odom_callback(self, msg: Odometry):
        """Odometry callback to update robot pose"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def _scan_callback(self, msg: LaserScan):
        """Laser scan callback for obstacle detection"""
        self.scan_data = msg

    def navigate_to_pose(self, target_pose: Pose,
                        avoid_obstacles: bool = True) -> bool:
        """Navigate to target pose with obstacle avoidance"""
        try:
            if avoid_obstacles:
                # Use move_base for navigation with obstacle avoidance
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose = target_pose

                # Send goal to move_base
                self.move_base_client.send_goal(goal)

                # Wait for result with timeout
                finished_within_time = self.move_base_client.wait_for_result(
                    rospy.Duration(60.0)  # 60 second timeout
                )

                if not finished_within_time:
                    self.move_base_client.cancel_goal()
                    rospy.logwarn("Navigation timed out")
                    return False

                state = self.move_base_client.get_state()
                success = (state == actionlib.GoalStatus.SUCCEEDED)

                return success
            else:
                # Direct movement without obstacle avoidance
                return self._direct_movement(target_pose)

        except Exception as e:
            rospy.logerr(f"Navigation error: {e}")
            return False

    def _direct_movement(self, target_pose: Pose) -> bool:
        """Direct movement to target pose without obstacle avoidance"""
        # Calculate desired movement
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Calculate desired orientation
        desired_yaw = np.arctan2(dy, dx)

        # Move towards target
        cmd_vel = Twist()
        cmd_vel.linear.x = min(distance, self.config.get('max_linear_speed', 0.5))
        cmd_vel.angular.z = desired_yaw  # Simplified orientation control

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        return True

    def stop_motion(self):
        """Stop all robot motion"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        self.is_moving = False

    def check_obstacles(self, min_distance: float = 0.5) -> bool:
        """Check for obstacles in front of robot"""
        if self.scan_data is None:
            return False

        # Check forward-facing range
        front_ranges = self.scan_data.ranges[
            len(self.scan_data.ranges)//2 - 30:
            len(self.scan_data.ranges)//2 + 30
        ]

        for distance in front_ranges:
            if 0 < distance < min_distance:
                return True

        return False

class ManipulationController:
    """
    Manipulation control component for humanoid robot
    """

    def __init__(self, config: Dict):
        self.config = config

        # Initialize manipulator interfaces
        self.joint_state_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self._joint_state_callback
        )

        # Joint command publisher
        self.joint_cmd_pub = rospy.Publisher(
            '/joint_commands',
            JointState,
            queue_size=10
        )

        # Gripper control
        self.gripper_pub = rospy.Publisher(
            '/gripper_command',
            Float64,
            queue_size=10
        )

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

    def _joint_state_callback(self, msg: JointState):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def move_to_joint_positions(self, joint_positions: Dict[str, float],
                               duration: float = 5.0) -> bool:
        """Move manipulator to specified joint positions"""
        try:
            # Create joint command message
            cmd_msg = JointState()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.header.frame_id = "base_link"

            # Set joint names and positions
            for joint_name, position in joint_positions.items():
                cmd_msg.name.append(joint_name)
                cmd_msg.position.append(position)

            # Publish command
            self.joint_cmd_pub.publish(cmd_msg)

            # Wait for movement to complete (simplified)
            rospy.sleep(duration)

            return True

        except Exception as e:
            rospy.logerr(f"Joint movement error: {e}")
            return False

    def grasp_object(self, object_pose: Pose, grasp_type: str = "power") -> bool:
        """Grasp an object at the specified pose"""
        try:
            # Plan grasp trajectory
            grasp_trajectory = self._plan_grasp_trajectory(object_pose, grasp_type)

            # Execute trajectory
            for waypoint in grasp_trajectory:
                self.move_to_joint_positions(waypoint)

            # Close gripper
            self._close_gripper()

            # Verify grasp success
            return self._verify_grasp_success()

        except Exception as e:
            rospy.logerr(f"Grasp error: {e}")
            return False

    def _plan_grasp_trajectory(self, object_pose: Pose, grasp_type: str) -> List[Dict]:
        """Plan trajectory for grasping object"""
        # Simplified trajectory planning
        trajectory = []

        # Pre-grasp position (approach from above)
        pre_grasp = self.joint_positions.copy()
        # Adjust joint positions to approach object from above
        trajectory.append(pre_grasp)

        # Grasp position
        grasp_pose = self.joint_positions.copy()
        # Adjust to actual grasp position
        trajectory.append(grasp_pose)

        return trajectory

    def _close_gripper(self):
        """Close the robot gripper"""
        gripper_cmd = Float64()
        gripper_cmd.data = 0.0  # Closed position
        self.gripper_pub.publish(gripper_cmd)

    def _open_gripper(self):
        """Open the robot gripper"""
        gripper_cmd = Float64()
        gripper_cmd.data = 1.0  # Open position
        self.gripper_pub.publish(gripper_cmd)

    def _verify_grasp_success(self) -> bool:
        """Verify that grasp was successful"""
        # This would check force sensors, tactile sensors, or visual confirmation
        # For now, return True (in practice, implement proper verification)
        return True

    def place_object(self, place_pose: Pose) -> bool:
        """Place held object at specified location"""
        try:
            # Plan placement trajectory
            place_trajectory = self._plan_placement_trajectory(place_pose)

            # Execute trajectory
            for waypoint in place_trajectory:
                self.move_to_joint_positions(waypoint)

            # Open gripper
            self._open_gripper()

            # Retract
            self._retract_from_place()

            return True

        except Exception as e:
            rospy.logerr(f"Place error: {e}")
            return False

    def _retract_from_place(self):
        """Retract manipulator after placing object"""
        # Move manipulator away from placed object
        retract_pos = self.joint_positions.copy()
        # Adjust joint positions for retraction
        self.move_to_joint_positions(retract_pos)
```

## Phase 2: System Integration

### Integration Framework

```python
# System Integration Framework
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import threading
import time
from typing import Dict, Any, List

class SystemIntegrator:
    """
    System integration framework for autonomous humanoid system
    """

    def __init__(self, config: Dict):
        self.config = config

        # Initialize component interfaces
        self.vision_processor = VisionProcessor(config['vision'])
        self.nlu_component = NaturalLanguageUnderstanding(config['language'])
        self.motion_controller = MotionController(config['motion'])
        self.manipulation_controller = ManipulationController(config['manipulation'])
        self.sensor_fusion = SensorFusion(config['fusion'])

        # ROS interfaces for integration
        self.task_command_sub = rospy.Subscriber(
            '/task_command', String, self._task_command_callback
        )
        self.system_status_pub = rospy.Publisher(
            '/system_status', String, queue_size=10
        )
        self.emergency_stop_sub = rospy.Subscriber(
            '/emergency_stop', Bool, self._emergency_stop_callback
        )

        # System state
        self.system_state = 'idle'  # idle, processing, executing, error
        self.current_task = None
        self.emergency_stop_active = False

        # Integration threads
        self.perception_thread = threading.Thread(target=self._perception_loop)
        self.cognition_thread = threading.Thread(target=self._cognition_loop)
        self.action_thread = threading.Thread(target=self._action_loop)

        # Task queue
        self.task_queue = []
        self.current_behavior = None

    def _task_command_callback(self, msg: String):
        """Handle incoming task commands"""
        if self.emergency_stop_active:
            rospy.logwarn("Emergency stop active - ignoring command")
            return

        try:
            # Parse and validate command
            parsed_command = self.nlu_component.parse_command(msg.data)

            if self.nlu_component.validate_command(parsed_command):
                # Add to task queue
                self.task_queue.append(parsed_command)
                rospy.loginfo(f"Task added to queue: {parsed_command['command_type']}")
            else:
                rospy.logwarn(f"Invalid command: {msg.data}")

        except Exception as e:
            rospy.logerr(f"Error processing task command: {e}")

    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop commands"""
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self._trigger_emergency_stop()
        else:
            self._reset_emergency_stop()

    def _trigger_emergency_stop(self):
        """Execute emergency stop procedures"""
        rospy.logerr("EMERGENCY STOP ACTIVATED")

        # Stop all motion
        self.motion_controller.stop_motion()

        # Open gripper if holding object
        if hasattr(self, 'manipulation_controller'):
            self.manipulation_controller._open_gripper()

        # Update system state
        self.system_state = 'emergency_stop'

        # Publish status
        status_msg = String()
        status_msg.data = 'EMERGENCY_STOP'
        self.system_status_pub.publish(status_msg)

    def _reset_emergency_stop(self):
        """Reset from emergency stop state"""
        rospy.loginfo("Emergency stop reset")
        self.system_state = 'idle'

        status_msg = String()
        status_msg.data = 'READY'
        self.system_status_pub.publish(status_msg)

    def _perception_loop(self):
        """Continuous perception processing loop"""
        rate = rospy.Rate(self.config['perception_rate'])

        while not rospy.is_shutdown() and not self.emergency_stop_active:
            try:
                # Process vision data
                vision_results = self.vision_processor.process_frame()

                # Process other sensors (LIDAR, etc.)
                # This would be implemented based on available sensors

                # Perform sensor fusion
                if vision_results:
                    fused_data = self.sensor_fusion.fuse_sensors(
                        vision_results, {}, {}  # Add other sensor data
                    )

                    # Update world model
                    self._update_world_model(fused_data)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Perception loop error: {e}")
                time.sleep(0.1)  # Brief pause before continuing

    def _cognition_loop(self):
        """Continuous cognition processing loop"""
        rate = rospy.Rate(self.config['cognition_rate'])

        while not rospy.is_shutdown() and not self.emergency_stop_active:
            try:
                # Process new tasks from queue
                if self.task_queue and self.system_state == 'idle':
                    task = self.task_queue.pop(0)
                    self._execute_task(task)

                # Update system state based on current behavior
                if self.current_behavior:
                    self._update_behavior_state()

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Cognition loop error: {e}")
                time.sleep(0.1)

    def _action_loop(self):
        """Continuous action execution loop"""
        rate = rospy.Rate(self.config['action_rate'])

        while not rospy.is_shutdown() and not self.emergency_stop_active:
            try:
                # Execute current behavior
                if self.current_behavior:
                    success = self._execute_behavior_step(self.current_behavior)
                    if not success:
                        self._handle_behavior_failure(self.current_behavior)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Action loop error: {e}")
                time.sleep(0.1)

    def _execute_task(self, task: Dict):
        """Execute a parsed task"""
        self.system_state = 'processing'
        rospy.loginfo(f"Processing task: {task['command_type']}")

        try:
            # Plan the task
            plan = self._plan_task(task)

            if plan:
                # Execute the plan
                self.current_behavior = self._create_behavior_from_plan(plan)
                self.system_state = 'executing'

                rospy.loginfo(f"Task plan created with {len(plan)} steps")
            else:
                rospy.logwarn("No plan could be generated for task")
                self.system_state = 'idle'

        except Exception as e:
            rospy.logerr(f"Task execution error: {e}")
            self.system_state = 'error'

    def _plan_task(self, task: Dict) -> List[Dict]:
        """Plan a sequence of actions for the task"""
        # Use the task planner component
        task_planner = TaskPlanner(self.config['planning'])

        # Get current state from world model
        current_state = self._get_current_world_state()

        # Create goal from task
        goal = self._create_goal_from_task(task)

        # Plan the task
        plan = task_planner.plan_task(goal, current_state)

        return plan

    def _create_behavior_from_plan(self, plan: List[Dict]) -> Dict:
        """Create behavior from planned actions"""
        return {
            'plan': plan,
            'current_step': 0,
            'start_time': rospy.Time.now(),
            'status': 'running'
        }

    def _execute_behavior_step(self, behavior: Dict) -> bool:
        """Execute one step of the current behavior"""
        if behavior['current_step'] >= len(behavior['plan']):
            # Behavior completed
            behavior['status'] = 'completed'
            self.current_behavior = None
            self.system_state = 'idle'
            return True

        current_action = behavior['plan'][behavior['current_step']]

        # Execute the action based on type
        success = self._execute_action(current_action)

        if success:
            behavior['current_step'] += 1
            return True
        else:
            return False

    def _execute_action(self, action: Dict) -> bool:
        """Execute a specific action"""
        action_type = action.get('name', '')

        if action_type == 'navigate_to':
            target_pose = self._parse_pose_from_action(action)
            return self.motion_controller.navigate_to_pose(target_pose)

        elif action_type == 'pick_up':
            object_pose = self._parse_object_pose(action)
            return self.manipulation_controller.grasp_object(object_pose)

        elif action_type == 'place':
            place_pose = self._parse_pose_from_action(action)
            return self.manipulation_controller.place_object(place_pose)

        else:
            rospy.logwarn(f"Unknown action type: {action_type}")
            return False

    def _update_world_model(self, fused_data: Dict):
        """Update the world model with fused sensor data"""
        # This would update an internal representation of the world
        # For now, just store the data
        self.world_model = fused_data

    def _get_current_world_state(self) -> Dict:
        """Get current world state for planning"""
        # Return current world state for the planner
        if hasattr(self, 'world_model'):
            return {
                'objects': self.world_model.get('fused_objects', []),
                'robot_pose': self.motion_controller.current_pose if hasattr(self.motion_controller, 'current_pose') else None,
                'environment_map': self.world_model.get('environment_map', {})
            }
        else:
            return {}

    def _create_goal_from_task(self, task: Dict) -> Dict:
        """Create goal representation from task"""
        return {
            'task_type': task['command_type'],
            'arguments': task['arguments'],
            'entities': task['entities']
        }

    def _parse_pose_from_action(self, action: Dict) -> Pose:
        """Parse pose from action arguments"""
        # This would extract pose information from action arguments
        pose = Pose()
        # Set default values
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        return pose

    def start_system(self):
        """Start the integrated system"""
        rospy.loginfo("Starting autonomous humanoid system integration")

        # Start integration threads
        self.perception_thread.start()
        self.cognition_thread.start()
        self.action_thread.start()

        # Set initial status
        status_msg = String()
        status_msg.data = 'READY'
        self.system_status_pub.publish(status_msg)

        rospy.loginfo("System integration started successfully")

    def stop_system(self):
        """Stop the integrated system"""
        rospy.loginfo("Stopping autonomous humanoid system")

        # Stop threads
        # In a real implementation, you'd have flags to signal threads to stop
        # For now, we'll just log
        rospy.loginfo("System stopped")
```

## Week Summary

This implementation phase document provides detailed technical implementations for the autonomous humanoid system's core components: perception, cognition, and action systems. It includes code examples for vision processing, natural language understanding, task planning, motion control, manipulation, and system integration. The implementation follows a modular architecture that enables systematic development and testing of each component before integration into the complete system.