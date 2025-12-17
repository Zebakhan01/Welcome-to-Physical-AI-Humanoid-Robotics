"""
Enhanced language module for Vision-Language-Action (VLA) systems
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re


@dataclass
class LanguageFeatures:
    """Enhanced language features representation"""
    text_embeddings: np.ndarray  # Text embeddings from language model
    parsed_commands: List[Dict[str, Any]]  # Parsed natural language commands
    intent: str  # Detected intent
    entities: List[Dict[str, str]]  # Named entities
    action_sequence: Optional[List[str]] = None  # Sequence of actions to execute
    semantic_meaning: Optional[Dict[str, Any]] = None  # Semantic interpretation
    discourse_structure: Optional[Dict[str, Any]] = None  # Sentence relationships
    uncertainty_score: Optional[float] = None  # Uncertainty in interpretation


class LanguageModule:
    """Enhanced language module for VLA systems with advanced NLP capabilities"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)

        # Initialize language processing components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize language processing components"""
        # Define common action verbs and their mappings
        self.action_verbs = {
            "pick": ["pick", "grasp", "grab", "take", "lift"],
            "place": ["place", "put", "set", "position", "drop"],
            "move": ["move", "go", "navigate", "travel", "approach"],
            "push": ["push", "press", "apply_force"],
            "pull": ["pull", "drag", "tug"],
            "rotate": ["rotate", "turn", "spin", "orient"],
            "align": ["align", "adjust", "position", "match"],
            "open": ["open", "uncover", "reveal"],
            "close": ["close", "shut", "cover"],
            "clean": ["clean", "wipe", "scrub", "dust"]
        }

        # Define common objects and locations
        self.common_objects = [
            "cup", "bottle", "box", "table", "chair", "laptop", "phone",
            "book", "plate", "bowl", "knife", "spoon", "fork", "glass"
        ]
        self.common_locations = [
            "kitchen", "table", "shelf", "counter", "cabinet", "drawer",
            "bedroom", "living_room", "office", "dining_room"
        ]
        self.common_adjectives = [
            "red", "blue", "green", "large", "small", "heavy", "light",
            "fragile", "hard", "soft", "round", "square", "cylindrical"
        ]

        # Define intent patterns
        self.intent_patterns = {
            "grasp_object": [
                r"pick.*up", r"grasp.*", r"grab.*", r"take.*",
                r"lift.*", r"get.*"
            ],
            "place_object": [
                r"place.*", r"put.*", r"set.*down", r"drop.*",
                r"position.*", r"lay.*"
            ],
            "navigate": [
                r"go.*to", r"move.*to", r"navigate.*", r"approach.*",
                r"travel.*to", r"walk.*to"
            ],
            "manipulate": [
                r"push.*", r"pull.*", r"rotate.*", r"turn.*",
                r"align.*", r"adjust.*"
            ],
            "household_task": [
                r"clean.*", r"organize.*", r"tidy.*", r"arrange.*",
                r"sort.*", r"set.*table"
            ]
        }

    def extract_features(self, text: str) -> LanguageFeatures:
        """Extract comprehensive language features from text"""
        try:
            # Generate text embeddings (simulated)
            text_embeddings = np.random.random(self.embedding_dim).astype(np.float32)

            # Parse commands
            parsed_commands = self._parse_commands(text)

            # Detect intent
            intent = self._detect_intent(text)

            # Extract entities
            entities = self._extract_entities(text)

            # Generate action sequence
            action_sequence = self._generate_action_sequence(text, intent)

            # Extract semantic meaning
            semantic_meaning = self._extract_semantic_meaning(text)

            # Analyze discourse structure
            discourse_structure = self._analyze_discourse(text)

            # Calculate uncertainty score
            uncertainty_score = self._calculate_uncertainty(text, entities)

            return LanguageFeatures(
                text_embeddings=text_embeddings,
                parsed_commands=parsed_commands,
                intent=intent,
                entities=entities,
                action_sequence=action_sequence,
                semantic_meaning=semantic_meaning,
                discourse_structure=discourse_structure,
                uncertainty_score=uncertainty_score
            )
        except Exception as e:
            self.logger.error(f"Error extracting language features: {str(e)}")
            raise

    def _parse_commands(self, text: str) -> List[Dict[str, Any]]:
        """Parse natural language commands with enhanced understanding"""
        commands = []
        text_lower = text.lower()

        # Identify action verbs and map to standardized actions
        for action_type, verbs in self.action_verbs.items():
            for verb in verbs:
                if verb in text_lower:
                    # Extract target object
                    target_object = self._extract_target_object(text)

                    # Extract target location if applicable
                    target_location = self._extract_target_location(text)

                    # Extract modifiers (adjectives, etc.)
                    modifiers = self._extract_modifiers(text)

                    command = {
                        "type": action_type,
                        "verb": verb,
                        "target_object": target_object,
                        "target_location": target_location,
                        "modifiers": modifiers,
                        "confidence": float(np.random.uniform(0.7, 0.95)),
                        "parameters": self._generate_action_parameters(action_type, text)
                    }
                    commands.append(command)

        # If no specific commands found, add a general analysis command
        if not commands:
            commands.append({
                "type": "analyze",
                "verb": "analyze",
                "target_object": "scene",
                "target_location": "current",
                "modifiers": [],
                "confidence": 0.8,
                "parameters": {"detailed_analysis": True}
            })

        return commands

    def _detect_intent(self, text: str) -> str:
        """Detect intent with pattern matching and context analysis"""
        text_lower = text.lower()

        # Check pattern-based intents first
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        # If no pattern matches, use keyword-based detection
        if any(word in text_lower for word in ["pick", "grasp", "grab", "take"]):
            return "grasp_object"
        elif any(word in text_lower for word in ["place", "put", "set", "drop"]):
            return "place_object"
        elif any(word in text_lower for word in ["move", "go", "navigate", "approach"]):
            return "navigate"
        elif any(word in text_lower for word in ["clean", "organize", "tidy"]):
            return "household_task"
        elif any(word in text_lower for word in ["push", "pull", "rotate", "turn"]):
            return "manipulate"
        else:
            return "unknown"

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities with enhanced classification"""
        entities = []
        text_lower = text.lower()

        # Extract objects
        for obj in self.common_objects:
            if obj in text_lower:
                entities.append({
                    "type": "object",
                    "value": obj,
                    "confidence": float(np.random.uniform(0.6, 0.9))
                })

        # Extract locations
        for loc in self.common_locations:
            if loc.replace("_", " ") in text_lower:
                entities.append({
                    "type": "location",
                    "value": loc,
                    "confidence": float(np.random.uniform(0.6, 0.9))
                })

        # Extract adjectives/modifiers
        for adj in self.common_adjectives:
            if adj in text_lower:
                entities.append({
                    "type": "modifier",
                    "value": adj,
                    "confidence": float(np.random.uniform(0.5, 0.8))
                })

        # Extract numbers/quantities
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text_lower)
        for num in numbers:
            entities.append({
                "type": "quantity",
                "value": num,
                "confidence": 0.9
            })

        # Extract spatial relations
        spatial_relations = ["on", "in", "under", "above", "next to", "behind", "in front of"]
        for relation in spatial_relations:
            if relation in text_lower:
                entities.append({
                    "type": "spatial_relation",
                    "value": relation,
                    "confidence": float(np.random.uniform(0.6, 0.8))
                })

        return entities

    def _generate_action_sequence(self, text: str, intent: str) -> List[str]:
        """Generate sequence of actions based on text and intent"""
        if intent == "grasp_object":
            return ["approach_object", "identify_grasp_point", "grasp", "lift", "transport"]
        elif intent == "place_object":
            return ["navigate_to_location", "align_object", "place", "release", "retract"]
        elif intent == "navigate":
            return ["localize", "plan_path", "execute_navigation", "verify_arrival"]
        elif intent == "manipulate":
            return ["approach_object", "identify_manipulation_point", "manipulate", "verify_result"]
        elif intent == "household_task":
            return ["analyze_task", "gather_objects", "execute_task", "verify_completion"]
        else:
            return ["analyze_instruction", "plan_actions", "execute_planned_actions"]

    def _extract_semantic_meaning(self, text: str) -> Dict[str, Any]:
        """Extract semantic meaning from text (simulated)"""
        # Simulate semantic parsing
        return {
            "action_verb": self._extract_action_verb(text),
            "patient": self._extract_patient(text),
            "goal": self._extract_goal(text),
            "constraints": self._extract_constraints(text),
            "temporal_aspects": self._extract_temporal_aspects(text),
            "causal_relations": self._extract_causal_relations(text)
        }

    def _extract_action_verb(self, text: str) -> str:
        """Extract main action verb from text"""
        for action_type, verbs in self.action_verbs.items():
            for verb in verbs:
                if verb in text.lower():
                    return verb
        return "unknown"

    def _extract_patient(self, text: str) -> str:
        """Extract the object being acted upon"""
        entities = self._extract_entities(text)
        objects = [e["value"] for e in entities if e["type"] == "object"]
        return objects[0] if objects else "object"

    def _extract_goal(self, text: str) -> str:
        """Extract the goal or desired outcome"""
        if "to" in text.lower():
            # Extract text after "to"
            parts = text.lower().split("to")
            if len(parts) > 1:
                return parts[1].strip()
        return text

    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints or limitations"""
        constraints = []
        text_lower = text.lower()

        if "carefully" in text_lower or "gently" in text_lower or "softly" in text_lower:
            constraints.append("use_low_force")
        if "quickly" in text_lower or "fast" in text_lower:
            constraints.append("high_velocity")
        if "precisely" in text_lower or "accurately" in text_lower:
            constraints.append("high_precision")

        return constraints

    def _extract_temporal_aspects(self, text: str) -> Dict[str, Any]:
        """Extract temporal information"""
        temporal_info = {}
        text_lower = text.lower()

        if "before" in text_lower:
            temporal_info["before"] = True
        if "after" in text_lower:
            temporal_info["after"] = True
        if "while" in text_lower:
            temporal_info["during"] = True

        return temporal_info

    def _extract_causal_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract causal relationships"""
        causal_relations = []
        text_lower = text.lower()

        if "so that" in text_lower or "in order to" in text_lower:
            causal_relations.append({
                "cause": "preceding_action",
                "effect": "following_action"
            })

        return causal_relations

    def _analyze_discourse(self, text: str) -> Dict[str, Any]:
        """Analyze discourse structure (simulated)"""
        sentences = re.split(r'[.!?]+', text)
        discourse_structure = {
            "sentence_count": len([s for s in sentences if s.strip()]),
            "coherence_score": float(np.random.uniform(0.7, 0.95)),
            "cohesion_relations": self._extract_cohesion_relations(text),
            "discourse_markers": self._extract_discourse_markers(text)
        }
        return discourse_structure

    def _extract_cohesion_relations(self, text: str) -> List[str]:
        """Extract cohesion relations between parts of text"""
        relations = []
        text_lower = text.lower()

        if "then" in text_lower:
            relations.append("temporal_sequence")
        if "but" in text_lower or "however" in text_lower:
            relations.append("contrast")
        if "and" in text_lower or "also" in text_lower:
            relations.append("addition")
        if "because" in text_lower or "since" in text_lower:
            relations.append("causation")

        return relations

    def _extract_discourse_markers(self, text: str) -> List[str]:
        """Extract discourse markers"""
        markers = []
        text_lower = text.lower()

        discourse_markers = [
            "first", "then", "next", "finally", "however", "but", "and", "or",
            "because", "since", "so", "therefore", "thus", "then"
        ]

        for marker in discourse_markers:
            if marker in text_lower:
                markers.append(marker)

        return markers

    def _calculate_uncertainty(self, text: str, entities: List[Dict[str, str]]) -> float:
        """Calculate uncertainty in language interpretation"""
        uncertainty = 0.1  # Base uncertainty

        # Increase uncertainty for ambiguous terms
        ambiguous_terms = ["it", "that", "there", "this", "something", "thing"]
        for term in ambiguous_terms:
            if term in text.lower():
                uncertainty += 0.15

        # Increase uncertainty if few entities detected
        if len(entities) < 2:
            uncertainty += 0.2

        # Decrease uncertainty if specific details provided
        specific_indicators = ["red", "blue", "left", "right", "small", "large", "heavy", "light"]
        for indicator in specific_indicators:
            if indicator in text.lower():
                uncertainty = max(0.05, uncertainty - 0.05)

        return min(0.95, uncertainty)

    def _extract_target_object(self, text: str) -> str:
        """Extract target object from text"""
        entities = self._extract_entities(text)
        objects = [e["value"] for e in entities if e["type"] == "object"]
        return objects[0] if objects else "object"

    def _extract_target_location(self, text: str) -> str:
        """Extract target location from text"""
        entities = self._extract_entities(text)
        locations = [e["value"] for e in entities if e["type"] == "location"]
        return locations[0] if locations else "location"

    def _extract_modifiers(self, text: str) -> List[str]:
        """Extract modifying words from text"""
        entities = self._extract_entities(text)
        modifiers = [e["value"] for e in entities if e["type"] == "modifier"]
        return modifiers

    def _generate_action_parameters(self, action_type: str, text: str) -> Dict[str, Any]:
        """Generate action-specific parameters"""
        params = {
            "force_limit": 10.0,  # N
            "velocity": 0.2,  # m/s
            "acceleration": 0.5,  # m/s^2
            "precision": 0.9  # High precision by default
        }

        # Adjust parameters based on action type and text modifiers
        if action_type == "grasp":
            params.update({
                "grasp_type": "precision",
                "approach_vector": [0, 0, 1],
                "gripper_width": 0.05
            })
        elif action_type == "place":
            params.update({
                "placement_orientation": [0, 0, 1],
                "stability_threshold": 0.8
            })
        elif action_type == "move":
            params.update({
                "path_type": "straight",
                "obstacle_avoidance": True
            })

        # Adjust based on text modifiers
        text_lower = text.lower()
        if "gently" in text_lower or "carefully" in text_lower:
            params["force_limit"] = 5.0
            params["velocity"] = 0.1
        elif "quickly" in text_lower or "fast" in text_lower:
            params["velocity"] = 0.5
            params["acceleration"] = 1.0

        if "precisely" in text_lower or "accurately" in text_lower:
            params["precision"] = 0.95

        return params

    def process_multimodal_context(self, text: str, visual_context: Dict[str, Any]) -> LanguageFeatures:
        """Process text with visual context for grounding"""
        try:
            # Extract basic features
            base_features = self.extract_features(text)

            # Enhance with visual context
            enhanced_entities = self._ground_entities_in_visual_context(
                base_features.entities, visual_context
            )

            # Update semantic meaning with visual grounding
            enhanced_semantic = self._ground_semantics_in_visual_context(
                base_features.semantic_meaning, visual_context
            )

            return LanguageFeatures(
                text_embeddings=base_features.text_embeddings,
                parsed_commands=base_features.parsed_commands,
                intent=base_features.intent,
                entities=enhanced_entities,
                action_sequence=base_features.action_sequence,
                semantic_meaning=enhanced_semantic,
                discourse_structure=base_features.discourse_structure,
                uncertainty_score=min(base_features.uncertainty_score, 0.3)  # Reduced uncertainty with visual context
            )
        except Exception as e:
            self.logger.error(f"Error processing multimodal context: {str(e)}")
            raise

    def _ground_entities_in_visual_context(self, entities: List[Dict[str, str]],
                                         visual_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Ground language entities in visual context"""
        grounded_entities = []
        visual_objects = visual_context.get("object_detections", [])

        for entity in entities:
            grounded_entity = entity.copy()

            if entity["type"] == "object":
                # Find matching visual object
                for vis_obj in visual_objects:
                    if vis_obj["class"] == entity["value"]:
                        grounded_entity["visual_id"] = vis_obj.get("id", "unknown")
                        grounded_entity["location"] = vis_obj.get("center", [0.5, 0.5])
                        grounded_entity["confidence"] = min(
                            entity["confidence"],
                            vis_obj.get("confidence", 0.8)
                        )
                        break

            grounded_entities.append(grounded_entity)

        return grounded_entities

    def _ground_semantics_in_visual_context(self, semantic_meaning: Dict[str, Any],
                                          visual_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ground semantic meaning in visual context"""
        if not semantic_meaning:
            return semantic_meaning

        grounded_semantic = semantic_meaning.copy()

        # Enhance patient with visual information
        if "patient" in grounded_semantic:
            visual_objects = visual_context.get("object_detections", [])
            for vis_obj in visual_objects:
                if vis_obj["class"] == grounded_semantic["patient"]:
                    grounded_semantic["patient_visual_info"] = {
                        "location": vis_obj.get("center", [0.5, 0.5]),
                        "size": vis_obj.get("size", 0.1),
                        "confidence": vis_obj.get("confidence", 0.8)
                    }
                    break

        return grounded_semantic

    def decompose_complex_instruction(self, text: str) -> List[Dict[str, Any]]:
        """Decompose complex instructions into simpler subtasks"""
        try:
            # Identify subtasks based on conjunctions and sequence markers
            subtasks = []

            # Split by common sequence markers
            sequence_parts = re.split(r'(then|next|after that|finally)', text, flags=re.IGNORECASE)

            current_action = ""
            for i, part in enumerate(sequence_parts):
                if part.strip().lower() in ['then', 'next', 'after that', 'finally']:
                    # This is a sequence marker, combine with next part
                    if i + 1 < len(sequence_parts):
                        current_action += " " + sequence_parts[i+1].strip()
                else:
                    if current_action:
                        current_action += " " + part.strip()
                    else:
                        current_action = part.strip()

                    if current_action and not part.strip().lower() in ['then', 'next', 'after that', 'finally']:
                        # Extract features for this subtask
                        subtask_features = self.extract_features(current_action)
                        subtasks.append({
                            "instruction": current_action,
                            "intent": subtask_features.intent,
                            "entities": subtask_features.entities,
                            "action_sequence": subtask_features.action_sequence,
                            "dependencies": [len(subtasks) - 1] if len(subtasks) > 0 else []
                        })
                        current_action = ""

            # If we have a remaining action
            if current_action.strip():
                subtask_features = self.extract_features(current_action)
                subtasks.append({
                    "instruction": current_action,
                    "intent": subtask_features.intent,
                    "entities": subtask_features.entities,
                    "action_sequence": subtask_features.action_sequence,
                    "dependencies": [len(subtasks) - 1] if len(subtasks) > 0 else []
                })

            return subtasks if subtasks else [{"instruction": text, "intent": self._detect_intent(text), "entities": self._extract_entities(text), "action_sequence": self._generate_action_sequence(text, self._detect_intent(text)), "dependencies": []}]
        except Exception as e:
            self.logger.error(f"Error decomposing instruction: {str(e)}")
            # Return a single subtask if decomposition fails
            features = self.extract_features(text)
            return [{
                "instruction": text,
                "intent": features.intent,
                "entities": features.entities,
                "action_sequence": features.action_sequence,
                "dependencies": []
            }]


# Singleton instance
language_module = LanguageModule()