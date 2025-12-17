"""
Enhanced action module for Vision-Language-Action (VLA) systems
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class ActionFeatures:
    """Enhanced action features representation"""
    action_type: str  # "symbolic", "continuous", "parameterized", "hierarchical"
    action_parameters: Dict[str, Any]  # Parameters for the action
    confidence: float  # Confidence in the action
    constraints: Optional[Dict[str, Any]] = None  # Constraints for execution
    execution_plan: Optional[List[Dict[str, Any]]] = None  # Detailed execution plan
    safety_score: Optional[float] = None  # Safety assessment
    expected_outcome: Optional[Dict[str, Any]] = None  # Predicted result
    skill_sequence: Optional[List[str]] = None  # Sequence of skills to execute


class ActionModule:
    """Enhanced action module for VLA systems with advanced planning and execution capabilities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define action space and capabilities
        self._initialize_action_space()

    def _initialize_action_space(self):
        """Initialize the action space and capabilities"""
        # Define basic action types
        self.action_types = [
            "grasp", "place", "move", "navigate", "lift", "release",
            "push", "pull", "rotate", "align", "open", "close",
            "pour", "scoop", "wipe", "assemble", "disassemble"
        ]

        # Define robot capabilities
        self.robot_capabilities = {
            "max_force": 50.0,  # Newtons
            "max_velocity": 1.0,  # m/s
            "max_acceleration": 2.0,  # m/s^2
            "gripper_range": [0.01, 0.1],  # meters
            "workspace_bounds": [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],  # meters
            "precision": 0.001  # meters
        }

        # Define action preconditions
        self.action_preconditions = {
            "grasp": ["object_detected", "reachable", "not_grasping"],
            "place": ["grasping_object", "valid_location"],
            "move": ["collision_free_path", "valid_destination"],
            "navigate": ["known_environment", "traversable_path"],
            "push": ["contact_object", "stable_surface"],
            "pour": ["grasping_container", "valid_target"]
        }

        # Define action effects
        self.action_effects = {
            "grasp": ["object_attached", "hand_occupied"],
            "place": ["object_released", "hand_free"],
            "move": ["position_changed"],
            "navigate": ["location_changed"],
            "push": ["object_moved", "force_applied"],
            "pour": ["contents_transferred"]
        }

    def generate_action(self, vision_features: Dict[str, Any],
                       language_features: Dict[str, Any],
                       robot_state: Optional[Dict[str, Any]] = None,
                       execution_history: Optional[List[Dict[str, Any]]] = None) -> ActionFeatures:
        """Generate action based on vision and language features"""
        try:
            # Determine action type based on language intent
            intent = language_features.get("intent", "unknown")
            action_type = self._map_intent_to_action(intent)

            # Generate parameters based on visual context and language
            parameters = self._generate_parameters(
                intent, vision_features, language_features, robot_state
            )

            # Calculate confidence based on feature alignment and robot state
            confidence = self._calculate_confidence(
                vision_features, language_features, robot_state
            )

            # Generate execution plan
            execution_plan = self._generate_execution_plan(
                action_type, parameters, robot_state, execution_history
            )

            # Assess safety
            safety_score = self._assess_safety(action_type, parameters, robot_state)

            # Predict expected outcome
            expected_outcome = self._predict_outcome(
                action_type, parameters, robot_state
            )

            # Generate skill sequence
            skill_sequence = self._generate_skill_sequence(action_type, parameters)

            # Check preconditions
            preconditions_met = self._check_preconditions(
                action_type, parameters, robot_state
            )

            if not preconditions_met:
                # Adjust action or parameters to meet preconditions
                parameters = self._adjust_parameters_for_preconditions(
                    action_type, parameters, robot_state
                )

            return ActionFeatures(
                action_type=action_type,
                action_parameters=parameters,
                confidence=confidence,
                execution_plan=execution_plan,
                safety_score=safety_score,
                expected_outcome=expected_outcome,
                skill_sequence=skill_sequence
            )
        except Exception as e:
            self.logger.error(f"Error generating action: {str(e)}")
            raise

    def _map_intent_to_action(self, intent: str) -> str:
        """Map language intent to specific action type"""
        intent_to_action = {
            "grasp_object": "grasp",
            "place_object": "place",
            "navigate": "navigate",
            "manipulate": "manipulate",
            "household_task": "complex_task"
        }
        return intent_to_action.get(intent, "unknown")

    def _generate_parameters(self, intent: str, vision_features: Dict[str, Any],
                           language_features: Dict[str, Any],
                           robot_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate action parameters based on context"""
        params = {
            "force_limit": self.robot_capabilities["max_force"] * 0.2,  # 20% of max force
            "velocity": self.robot_capabilities["max_velocity"] * 0.3,  # 30% of max velocity
            "acceleration": self.robot_capabilities["max_acceleration"] * 0.5,  # 50% of max acceleration
            "precision": self.robot_capabilities["precision"],
            "timeout": 10.0  # seconds
        }

        # Extract relevant visual information
        relevant_objects = vision_features.get("relevant_objects", [])
        object_detections = vision_features.get("object_detections", [])

        # Extract relevant language information
        entities = language_features.get("entities", [])
        target_objects = [e["value"] for e in entities if e["type"] == "object"]
        target_locations = [e["value"] for e in entities if e["type"] == "location"]

        if intent == "grasp_object":
            # Find target object from language or visual context
            target_obj = None
            if target_objects:
                # Find the most relevant object
                for obj in object_detections:
                    if obj["class"] in target_objects:
                        target_obj = obj
                        break
            elif relevant_objects:
                target_obj = relevant_objects[0]["object"]

            if target_obj:
                params.update({
                    "target_object": target_obj["class"],
                    "target_position": target_obj["center"],
                    "object_size": target_obj.get("size", 0.1),
                    "approach_vector": [0, 0, 1],  # Approach from above
                    "gripper_width": min(0.08, max(0.02, target_obj.get("size", 0.1) * 0.5))
                })

        elif intent == "place_object":
            # Find target location
            params["target_location"] = target_locations[0] if target_locations else "default_location"
            params["placement_orientation"] = [0, 0, 1]  # Upright placement

        elif intent == "navigate":
            # Find target location
            params["target_location"] = target_locations[0] if target_locations else "default_location"
            params["navigation_strategy"] = "path_planning"

        elif intent == "manipulate":
            # Generic manipulation parameters
            params["manipulation_type"] = "push"  # Default
            params["force_direction"] = [1, 0, 0]  # Default direction

        # Adjust parameters based on language modifiers
        modifiers = [e["value"] for e in entities if e["type"] == "modifier"]
        if "gently" in modifiers or "carefully" in modifiers:
            params["force_limit"] *= 0.3
            params["velocity"] *= 0.5
        elif "firmly" in modifiers or "strongly" in modifiers:
            params["force_limit"] *= 2.0

        if "precisely" in modifiers or "accurately" in modifiers:
            params["precision"] *= 0.1  # Higher precision

        # Adjust based on robot state if provided
        if robot_state:
            current_gripper_state = robot_state.get("gripper_state", "open")
            params["current_gripper_state"] = current_gripper_state

        return params

    def _calculate_confidence(self, vision_features: Dict[str, Any],
                            language_features: Dict[str, Any],
                            robot_state: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in action based on feature alignment"""
        # Base confidence from language processing
        base_confidence = language_features.get("uncertainty_score", 0.1)
        base_confidence = 1.0 - base_confidence  # Convert uncertainty to confidence

        # Boost confidence based on visual grounding
        relevant_objects = vision_features.get("relevant_objects", [])
        if relevant_objects:
            # Average relevance scores
            relevance_scores = [obj.get("relevance_score", 0.5) for obj in relevant_objects]
            visual_confidence = np.mean(relevance_scores) if relevance_scores else 0.5
            base_confidence = 0.6 * base_confidence + 0.4 * visual_confidence

        # Adjust based on robot state validity
        if robot_state:
            state_validity = robot_state.get("state_validity", 1.0)
            base_confidence *= state_validity

        return min(1.0, max(0.1, base_confidence))

    def _generate_execution_plan(self, action_type: str, parameters: Dict[str, Any],
                               robot_state: Optional[Dict[str, Any]],
                               execution_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Generate detailed execution plan for the action"""
        plan = []

        if action_type == "grasp":
            plan = [
                {
                    "step": "approach_object",
                    "description": "Move gripper above target object",
                    "parameters": {
                        "position": [parameters.get("target_position", [0.5, 0.5])[0],
                                   parameters.get("target_position", [0.5, 0.5])[1],
                                   0.2],  # 20cm above object
                        "velocity": parameters.get("velocity", 0.2)
                    },
                    "constraints": ["collision_free", "within_workspace"]
                },
                {
                    "step": "descend_to_object",
                    "description": "Lower gripper to object level",
                    "parameters": {
                        "position": [parameters.get("target_position", [0.5, 0.5])[0],
                                   parameters.get("target_position", [0.5, 0.5])[1],
                                   0.05],  # 5cm above surface
                        "velocity": parameters.get("velocity", 0.2) * 0.5
                    },
                    "constraints": ["force_control", "precision"]
                },
                {
                    "step": "grasp_object",
                    "description": "Close gripper to grasp object",
                    "parameters": {
                        "gripper_width": parameters.get("gripper_width", 0.05),
                        "force": parameters.get("force_limit", 10.0) * 0.5
                    },
                    "constraints": ["force_feedback", "object_attached"]
                },
                {
                    "step": "lift_object",
                    "description": "Lift object to safe height",
                    "parameters": {
                        "position": [parameters.get("target_position", [0.5, 0.5])[0],
                                   parameters.get("target_position", [0.5, 0.5])[1],
                                   0.15],  # 15cm above surface
                        "velocity": parameters.get("velocity", 0.2)
                    },
                    "constraints": ["maintain_grasp", "collision_free"]
                }
            ]

        elif action_type == "place":
            plan = [
                {
                    "step": "navigate_to_location",
                    "description": "Move to placement location",
                    "parameters": {
                        "target_location": parameters.get("target_location", "default"),
                        "velocity": parameters.get("velocity", 0.2)
                    },
                    "constraints": ["collision_free", "traversable_path"]
                },
                {
                    "step": "align_for_placement",
                    "description": "Align object for placement",
                    "parameters": {
                        "orientation": parameters.get("placement_orientation", [0, 0, 1]),
                        "position_precision": parameters.get("precision", 0.001)
                    },
                    "constraints": ["orientation_alignment", "position_accuracy"]
                },
                {
                    "step": "place_object",
                    "description": "Release object at placement location",
                    "parameters": {
                        "force": 2.0  # Gentle release
                    },
                    "constraints": ["object_released", "stable_placement"]
                },
                {
                    "step": "retract_gripper",
                    "description": "Retract gripper to safe position",
                    "parameters": {
                        "clearance_height": 0.1,
                        "velocity": parameters.get("velocity", 0.2)
                    },
                    "constraints": ["collision_free"]
                }
            ]

        elif action_type == "navigate":
            plan = [
                {
                    "step": "localize",
                    "description": "Determine current position",
                    "parameters": {},
                    "constraints": ["position_accuracy"]
                },
                {
                    "step": "plan_path",
                    "description": "Plan collision-free path to destination",
                    "parameters": {
                        "target_location": parameters.get("target_location", "default"),
                        "navigation_strategy": parameters.get("navigation_strategy", "path_planning")
                    },
                    "constraints": ["collision_free", "traversable"]
                },
                {
                    "step": "execute_navigation",
                    "description": "Follow planned path",
                    "parameters": {
                        "velocity": parameters.get("velocity", 0.2),
                        "safety_margin": 0.1
                    },
                    "constraints": ["obstacle_avoidance", "position_tracking"]
                },
                {
                    "step": "verify_arrival",
                    "description": "Confirm arrival at destination",
                    "parameters": {},
                    "constraints": ["position_verification"]
                }
            ]

        else:
            # Default plan for unknown actions
            plan = [{
                "step": "analyze_task",
                "description": "Analyze the requested task",
                "parameters": {"task": action_type},
                "constraints": ["task_understanding"]
            }]

        # Add error handling steps
        error_handling = {
            "step": "error_recovery",
            "description": "Handle potential errors during execution",
            "parameters": {"recovery_strategy": "abort_and_report"},
            "constraints": ["safe_state", "human_intervention"]
        }
        plan.append(error_handling)

        return plan

    def _assess_safety(self, action_type: str, parameters: Dict[str, Any],
                      robot_state: Optional[Dict[str, Any]]) -> float:
        """Assess safety of the planned action"""
        safety_score = 1.0  # Start with maximum safety

        # Check force limits
        force_limit = parameters.get("force_limit", 10.0)
        max_force = self.robot_capabilities["max_force"]
        if force_limit > max_force:
            safety_score *= 0.5  # Reduce safety if force exceeds limits

        # Check velocity limits
        velocity = parameters.get("velocity", 0.2)
        max_velocity = self.robot_capabilities["max_velocity"]
        if velocity > max_velocity:
            safety_score *= 0.6

        # Check workspace bounds if position specified
        target_pos = parameters.get("target_position")
        if target_pos:
            workspace_bounds = self.robot_capabilities["workspace_bounds"]
            for i, (min_val, max_val) in enumerate(workspace_bounds):
                if not (min_val <= target_pos[i] <= max_val if i < len(target_pos) else True):
                    safety_score *= 0.3

        # Check for human presence (simulated)
        if robot_state and robot_state.get("human_proximity", 0) > 0.5:
            safety_score *= 0.7  # Reduce safety if human is close

        # Adjust for action type risk
        high_risk_actions = ["pour", "manipulate", "rotate"]
        if action_type in high_risk_actions:
            safety_score *= 0.8

        return max(0.1, safety_score)  # Minimum safety score

    def _predict_outcome(self, action_type: str, parameters: Dict[str, Any],
                        robot_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict expected outcome of the action"""
        outcome = {
            "success_probability": 0.8,  # Base success probability
            "expected_state_change": {},
            "side_effects": [],
            "verification_criteria": []
        }

        if action_type == "grasp":
            outcome["expected_state_change"] = {
                "gripper_state": "closed",
                "object_attached": True,
                "end_effector_load": parameters.get("object_size", 0.1) * 5.0  # Simulated weight
            }
            outcome["verification_criteria"] = [
                "force_sensor_confirmation",
                "object_attached_feedback"
            ]

        elif action_type == "place":
            outcome["expected_state_change"] = {
                "gripper_state": "open",
                "object_attached": False,
                "end_effector_load": 0.0
            }
            outcome["verification_criteria"] = [
                "force_sensor_confirmation",
                "object_placement_verification"
            ]

        elif action_type == "navigate":
            outcome["expected_state_change"] = {
                "position": parameters.get("target_location", "unknown")
            }
            outcome["verification_criteria"] = [
                "position_verification",
                "destination_confirmation"
            ]

        # Adjust success probability based on safety score
        safety_score = self._assess_safety(action_type, parameters, robot_state)
        outcome["success_probability"] = min(0.95, outcome["success_probability"] * safety_score)

        # Add potential side effects
        if action_type in ["grasp", "move"]:
            outcome["side_effects"].append("potential_object_displacement")

        return outcome

    def _generate_skill_sequence(self, action_type: str, parameters: Dict[str, Any]) -> List[str]:
        """Generate sequence of skills needed to execute the action"""
        skills = []

        if action_type == "grasp":
            skills = ["approach", "descend", "grasp", "lift"]
        elif action_type == "place":
            skills = ["navigate", "align", "place", "retract"]
        elif action_type == "navigate":
            skills = ["localize", "path_plan", "move_base", "verify"]
        elif action_type == "manipulate":
            skills = ["approach", "contact", "apply_force", "verify"]
        else:
            skills = ["analyze", "plan", "execute"]

        return skills

    def _check_preconditions(self, action_type: str, parameters: Dict[str, Any],
                           robot_state: Optional[Dict[str, Any]]) -> bool:
        """Check if action preconditions are met"""
        preconditions = self.action_preconditions.get(action_type, [])
        met_conditions = []

        for condition in preconditions:
            if condition == "object_detected":
                # Check if target object is in visual features (simulated)
                met_conditions.append(True)  # Assume object is detected
            elif condition == "reachable":
                # Check if target position is within workspace (simulated)
                target_pos = parameters.get("target_position")
                if target_pos:
                    workspace_bounds = self.robot_capabilities["workspace_bounds"]
                    is_reachable = all(
                        workspace_bounds[i][0] <= target_pos[i] <= workspace_bounds[i][1]
                        for i in range(min(len(target_pos), len(workspace_bounds)))
                    )
                    met_conditions.append(is_reachable)
                else:
                    met_conditions.append(True)
            elif condition == "not_grasping":
                # Check if robot is not currently grasping (simulated)
                current_state = robot_state or {}
                met_conditions.append(current_state.get("gripper_state", "open") == "open")
            elif condition == "grasping_object":
                # Check if robot is currently grasping (simulated)
                current_state = robot_state or {}
                met_conditions.append(current_state.get("gripper_state", "open") == "closed")
            elif condition == "valid_location":
                # Check if location is valid (simulated)
                met_conditions.append(True)
            else:
                # Default: assume condition is met
                met_conditions.append(True)

        return all(met_conditions)

    def _adjust_parameters_for_preconditions(self, action_type: str, parameters: Dict[str, Any],
                                           robot_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust parameters to meet action preconditions"""
        adjusted_params = parameters.copy()

        # For now, we'll just log that preconditions weren't met
        # In a real implementation, this would adjust parameters accordingly
        self.logger.warning(f"Preconditions not fully met for action {action_type}, proceeding with adjusted parameters")

        return adjusted_params

    def generate_hierarchical_action(self, high_level_task: str,
                                   visual_context: Dict[str, Any],
                                   language_context: Dict[str, Any]) -> ActionFeatures:
        """Generate hierarchical action for complex tasks"""
        try:
            # Decompose high-level task into subtasks
            subtasks = self._decompose_task(high_level_task)

            # Generate action for each subtask
            hierarchical_plan = []
            for subtask in subtasks:
                # Create context for this subtask
                subtask_context = {
                    "current_subtask": subtask,
                    "overall_task": high_level_task,
                    "visual_context": visual_context,
                    "language_context": language_context
                }

                # Generate action for this subtask
                subtask_action = self.generate_action(
                    vision_features=visual_context,
                    language_features=language_context,
                    execution_history=hierarchical_plan
                )

                hierarchical_plan.append({
                    "subtask": subtask,
                    "action": subtask_action,
                    "dependencies": self._get_subtask_dependencies(subtask, subtasks)
                })

            # Calculate overall confidence as average of subtask confidences
            overall_confidence = np.mean([plan["action"].confidence for plan in hierarchical_plan])

            # Generate overall execution plan
            overall_plan = [plan["action"].execution_plan for plan in hierarchical_plan]
            flattened_plan = [step for plan in overall_plan for step in plan]

            return ActionFeatures(
                action_type="hierarchical",
                action_parameters={
                    "high_level_task": high_level_task,
                    "subtask_plan": [plan["subtask"] for plan in hierarchical_plan],
                    "subtask_actions": [plan["action"].action_type for plan in hierarchical_plan]
                },
                confidence=overall_confidence,
                execution_plan=flattened_plan,
                safety_score=min([plan["action"].safety_score or 1.0 for plan in hierarchical_plan]),
                expected_outcome={"hierarchical_task_completed": True},
                skill_sequence=[skill for plan in hierarchical_plan for skill in plan["action"].skill_sequence or []]
            )
        except Exception as e:
            self.logger.error(f"Error generating hierarchical action: {str(e)}")
            raise

    def _decompose_task(self, high_level_task: str) -> List[str]:
        """Decompose high-level task into subtasks"""
        # This is a simplified decomposition - in practice, this would use more sophisticated NLP
        task_lower = high_level_task.lower()

        if "clean" in task_lower or "organize" in task_lower:
            return [
                "analyze_workspace",
                "identify_objects_to_move",
                "plan_object_movements",
                "execute_movements",
                "verify_completion"
            ]
        elif "assemble" in task_lower or "build" in task_lower:
            return [
                "identify_components",
                "plan_assembly_sequence",
                "pick_components",
                "assemble_parts",
                "verify_assembly"
            ]
        elif "cook" in task_lower or "prepare" in task_lower:
            return [
                "identify_ingredients",
                "prepare_workspace",
                "execute_cooking_steps",
                "monitor_progress",
                "complete_dish"
            ]
        else:
            # Default decomposition for unknown tasks
            return ["analyze_task", "plan_execution", "execute_action", "verify_result"]

    def _get_subtask_dependencies(self, subtask: str, all_subtasks: List[str]) -> List[str]:
        """Get dependencies for a subtask"""
        # Define simple dependency rules
        dependencies = []

        # Some tasks must follow others
        if subtask == "execute_movements":
            dependencies.append("plan_object_movements")
        elif subtask == "verify_completion":
            dependencies.append("execute_movements")

        return dependencies

    def adapt_action_to_robot(self, action: ActionFeatures, robot_config: Dict[str, Any]) -> ActionFeatures:
        """Adapt action to specific robot configuration"""
        try:
            # Update parameters based on robot capabilities
            adapted_params = action.action_parameters.copy()

            # Adjust based on robot-specific limits
            if "max_force" in robot_config:
                adapted_params["force_limit"] = min(
                    adapted_params.get("force_limit", 10.0),
                    robot_config["max_force"]
                )

            if "max_velocity" in robot_config:
                adapted_params["velocity"] = min(
                    adapted_params.get("velocity", 0.2),
                    robot_config["max_velocity"]
                )

            if "gripper_range" in robot_config:
                if "gripper_width" in adapted_params:
                    adapted_params["gripper_width"] = min(
                        max(adapted_params["gripper_width"], robot_config["gripper_range"][0]),
                        robot_config["gripper_range"][1]
                    )

            # Update execution plan with robot-specific constraints
            adapted_plan = []
            for step in action.execution_plan or []:
                adapted_step = step.copy()
                # Add robot-specific constraints
                if "constraints" not in adapted_step:
                    adapted_step["constraints"] = []
                adapted_step["constraints"].extend(robot_config.get("safety_constraints", []))
                adapted_plan.append(adapted_step)

            return ActionFeatures(
                action_type=action.action_type,
                action_parameters=adapted_params,
                confidence=action.confidence,
                constraints=action.constraints,
                execution_plan=adapted_plan,
                safety_score=action.safety_score,
                expected_outcome=action.expected_outcome,
                skill_sequence=action.skill_sequence
            )
        except Exception as e:
            self.logger.error(f"Error adapting action to robot: {str(e)}")
            return action  # Return original action if adaptation fails


# Singleton instance
action_module = ActionModule()