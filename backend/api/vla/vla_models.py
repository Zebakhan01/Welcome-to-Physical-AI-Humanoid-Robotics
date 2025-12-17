"""
Enhanced data models for Vision-Language-Action (VLA) module
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import numpy as np
from enum import Enum


class FusionMethod(str, Enum):
    """Available fusion methods for multimodal integration"""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    CROSS_ATTENTION = "cross_attention"
    CONCATENATION = "concatenation"
    DYNAMIC_FUSION = "dynamic_fusion"


class VLAComponentType(str, Enum):
    """Types of components in Vision-Language-Action systems"""
    VISION = "vision"
    LANGUAGE = "language"
    ACTION = "action"
    FUSION = "fusion"
    MEMORY = "memory"


class VLARequest(BaseModel):
    """Comprehensive request model for VLA processing"""
    image: str  # Base64 encoded image
    instruction: str
    context: Optional[Dict[str, Any]] = None  # Additional context information
    fusion_method: FusionMethod = FusionMethod.CROSS_ATTENTION
    task_type: Optional[str] = None  # "grasp", "navigation", "manipulation", etc.
    robot_state: Optional[Dict[str, Any]] = None  # Current robot state
    execution_history: Optional[List[Dict[str, Any]]] = None  # Previous actions


class VLAResponse(BaseModel):
    """Comprehensive response model for VLA processing"""
    action_type: str
    action_parameters: Dict[str, Any]
    confidence: float
    execution_plan: List[Dict[str, Any]]
    multimodal_features: Optional[Dict[str, Any]] = None
    attention_weights: Optional[List[float]] = None
    success: bool
    reasoning_trace: Optional[List[Dict[str, Any]]] = None  # Step-by-step reasoning


class VisionAnalysisRequest(BaseModel):
    """Request model for vision analysis"""
    image: str  # Base64 encoded image
    query: Optional[str] = None  # Optional language query for grounding
    analysis_type: str = "full"  # "full", "objects", "scene", "affordances"
    return_features: bool = True


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis"""
    object_detections: List[Dict[str, Any]]
    spatial_features: List[float]
    semantic_features: List[float]
    affordance_features: Optional[List[float]] = None
    scene_context: Optional[Dict[str, Any]] = None
    relevant_objects: List[Dict[str, Any]]
    grounding_confidence: Optional[float] = None
    success: bool


class LanguageAnalysisRequest(BaseModel):
    """Request model for language analysis"""
    text: str
    analysis_type: str = "full"  # "full", "intent", "entities", "actions"
    return_embeddings: bool = True
    target_modality: Optional[str] = None  # "vision", "action", or both


class LanguageAnalysisResponse(BaseModel):
    """Response model for language analysis"""
    text_embeddings: Optional[List[float]] = None
    parsed_commands: List[Dict[str, Any]]
    intent: str
    entities: List[Dict[str, Any]]
    action_sequence: List[str]
    semantic_meaning: Optional[Dict[str, Any]] = None
    success: bool


class ActionGenerationRequest(BaseModel):
    """Request model for action generation"""
    vision_features: Dict[str, Any]
    language_features: Dict[str, Any]
    task_context: Optional[Dict[str, Any]] = None
    robot_capabilities: Optional[Dict[str, Any]] = None
    safety_constraints: Optional[Dict[str, Any]] = None
    execution_history: Optional[List[Dict[str, Any]]] = None


class ActionGenerationResponse(BaseModel):
    """Response model for action generation"""
    action_type: str
    action_parameters: Dict[str, Any]
    confidence: float
    execution_plan: List[Dict[str, Any]]
    safety_score: float
    expected_outcome: Optional[Dict[str, Any]] = None
    success: bool


class MultimodalFusionRequest(BaseModel):
    """Request model for multimodal fusion"""
    vision_features: List[float]
    language_features: List[float]
    fusion_method: FusionMethod = FusionMethod.CROSS_ATTENTION
    return_attention_weights: bool = True
    normalization: bool = True


class MultimodalFusionResponse(BaseModel):
    """Response model for multimodal fusion"""
    fused_features: List[float]
    attention_weights: Optional[List[float]] = None
    fusion_method_used: FusionMethod
    success: bool


class VLAEvaluationRequest(BaseModel):
    """Request model for VLA evaluation"""
    test_cases: List[Dict[str, Any]]  # List of test cases with image, instruction, expected_action
    evaluation_metrics: List[str] = ["success_rate", "accuracy", "efficiency"]
    robot_model: Optional[str] = None
    environment: Optional[str] = None


class VLAEvaluationResponse(BaseModel):
    """Response model for VLA evaluation"""
    task_success_rate: float
    language_accuracy: float
    vision_accuracy: float
    action_accuracy: float
    efficiency_metrics: Dict[str, float]
    detailed_results: List[Dict[str, Any]]
    failure_analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    success: bool


class MemoryReadRequest(BaseModel):
    """Request model for memory operations"""
    query: str
    memory_type: str = "episodic"  # "episodic", "semantic", "working"
    max_results: int = 5
    similarity_threshold: float = 0.5


class MemoryWriteRequest(BaseModel):
    """Request model for memory write operations"""
    content: Dict[str, Any]
    memory_type: str = "episodic"
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    retrieved_content: List[Dict[str, Any]]
    similarity_scores: List[float]
    memory_type: str
    success: bool


class ExecutionPlanRequest(BaseModel):
    """Request model for execution planning"""
    instruction: str
    environment_context: Dict[str, Any]
    robot_capabilities: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    optimization_criteria: List[str] = ["safety", "efficiency", "accuracy"]


class ExecutionPlanResponse(BaseModel):
    """Response model for execution planning"""
    execution_plan: List[Dict[str, Any]]
    estimated_completion_time: float
    risk_assessment: Dict[str, float]
    resource_requirements: Dict[str, Any]
    success: bool


class InteractiveVLARequest(BaseModel):
    """Request model for interactive VLA sessions"""
    current_image: Optional[str] = None
    user_input: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    current_task_state: Optional[Dict[str, Any]] = None


class InteractiveVLAResponse(BaseModel):
    """Response model for interactive VLA sessions"""
    action_or_response: str
    next_prompt: Optional[str] = None
    session_id: str
    task_progress: float
    success: bool


class MultiStepTaskRequest(BaseModel):
    """Request model for multi-step tasks"""
    high_level_instruction: str
    subtasks: List[Dict[str, Any]]
    environment_state: Dict[str, Any]
    available_resources: List[str]


class MultiStepTaskResponse(BaseModel):
    """Response model for multi-step tasks"""
    task_decomposition: List[Dict[str, Any]]
    execution_order: List[int]
    dependency_graph: List[Dict[str, Any]]
    success: bool


class DemonstrationLearningRequest(BaseModel):
    """Request model for learning from demonstrations"""
    demonstration_sequence: List[Dict[str, Any]]  # Sequence of (image, action) pairs
    task_description: str
    robot_configuration: Dict[str, Any]
    success_criteria: List[Dict[str, Any]]


class DemonstrationLearningResponse(BaseModel):
    """Response model for learning from demonstrations"""
    learned_policy: Optional[Dict[str, Any]] = None
    generalization_score: float
    confidence: float
    adaptation_requirements: List[str]
    success: bool


class ContextAdaptationRequest(BaseModel):
    """Request model for context adaptation"""
    source_context: Dict[str, Any]
    target_context: Dict[str, Any]
    task: str
    adaptation_method: str = "transfer"  # "transfer", "fine_tune", "prompt"


class ContextAdaptationResponse(BaseModel):
    """Response model for context adaptation"""
    adapted_model: Optional[str] = None
    adaptation_score: float
    required_training_steps: int
    estimated_performance: float
    success: bool


class GroundLanguageInVisionRequest(BaseModel):
    """Request model for grounding language in visual features"""
    image: str  # Base64 encoded image
    language_query: str
    confidence_threshold: float = 0.3


class GroundLanguageInVisionResponse(BaseModel):
    """Response model for grounding language in visual features"""
    relevant_objects: List[Dict[str, Any]]
    grounding_confidence: Optional[float] = None
    object_features: Optional[List[Dict[str, Any]]] = None
    success: bool


class GenerateExecutionPlanRequest(BaseModel):
    """Request model for generating execution plans"""
    instruction: str
    environment_context: Dict[str, Any]
    robot_capabilities: Dict[str, Any]
    task_constraints: Optional[Dict[str, Any]] = None
    optimization_criteria: List[str] = ["safety", "efficiency", "accuracy"]


class GenerateExecutionPlanResponse(BaseModel):
    """Response model for generating execution plans"""
    execution_plan: List[Dict[str, Any]]
    estimated_completion_time: float
    risk_assessment: Optional[Dict[str, float]] = None
    resource_requirements: Optional[Dict[str, Any]] = None
    success: bool