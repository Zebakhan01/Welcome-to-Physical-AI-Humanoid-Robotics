"""
Enhanced Vision-Language-Action (VLA) service for the Physical AI & Humanoid Robotics Textbook backend
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import numpy as np
import base64
from PIL import Image
import io
from backend.api.vla.vla_models import (
    VLARequest, VLAResponse, VisionAnalysisRequest, VisionAnalysisResponse,
    LanguageAnalysisRequest, LanguageAnalysisResponse, ActionGenerationRequest,
    ActionGenerationResponse, MultimodalFusionRequest, MultimodalFusionResponse,
    VLAEvaluationRequest, VLAEvaluationResponse, MemoryReadRequest, MemoryWriteRequest,
    MemoryResponse, ExecutionPlanRequest, ExecutionPlanResponse, InteractiveVLARequest,
    InteractiveVLAResponse, MultiStepTaskRequest, MultiStepTaskResponse,
    DemonstrationLearningRequest, DemonstrationLearningResponse, ContextAdaptationRequest,
    ContextAdaptationResponse, GroundLanguageInVisionRequest, GroundLanguageInVisionResponse,
    GenerateExecutionPlanRequest, GenerateExecutionPlanResponse, FusionMethod
)
from backend.api.vla.vla_components import (
    perception_module, language_module, action_module,
    fusion_module, memory_module
)
from backend.utils.logger import logger

router = APIRouter()


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array image"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array and normalize to [0,1]
        image_array = np.array(image).astype(np.float32) / 255.0

        # Ensure 3-channel (RGB)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]  # Convert to RGB

        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")


@router.post("/process", response_model=VLAResponse)
async def process_vla(request: VLARequest):
    """
    Process vision-language input and generate action
    """
    try:
        # Convert base64 image to numpy array
        image_array = base64_to_image(request.image)

        # Extract visual features
        visual_features = perception_module.extract_features(image_array)

        # Extract language features
        language_features = language_module.extract_features(request.instruction)

        # Convert fusion method string to fusion module's enum
        from backend.api.vla.vla_components.fusion_module import FusionMethod as InternalFusionMethod
        internal_fusion_method = InternalFusionMethod(request.fusion_method.value)

        # Fuse multimodal features
        fusion_result = fusion_module.fuse_features(
            visual_features.__dict__,
            language_features.__dict__,
            method=internal_fusion_method
        )

        # Generate action
        action_features = action_module.generate_action(
            vision_features=visual_features.__dict__,
            language_features=language_features.__dict__,
            robot_state=request.robot_state
        )

        # Convert numpy arrays to lists for JSON serialization
        multimodal_features_serializable = None
        if fusion_result is not None:
            multimodal_features_serializable = {}
            for key, value in fusion_result.items():
                if isinstance(value, np.ndarray):
                    multimodal_features_serializable[key] = value.tolist()
                else:
                    multimodal_features_serializable[key] = value

        attention_weights_serializable = None
        if fusion_result.get("attention_weights") is not None:
            if isinstance(fusion_result["attention_weights"], np.ndarray):
                attention_weights_serializable = fusion_result["attention_weights"].tolist()
            else:
                attention_weights_serializable = fusion_result["attention_weights"]

        response = VLAResponse(
            action_type=action_features.action_type,
            action_parameters=action_features.action_parameters,
            confidence=action_features.confidence,
            execution_plan=action_features.execution_plan or [],
            multimodal_features=multimodal_features_serializable,
            attention_weights=attention_weights_serializable,
            success=True
        )

        logger.info(f"VLA processing completed: {action_features.action_type} with confidence {action_features.confidence}")

        return response

    except Exception as e:
        logger.error(f"Error in VLA processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in VLA processing: {str(e)}")


@router.post("/analyze-vision", response_model=VisionAnalysisResponse)
async def analyze_vision(request: VisionAnalysisRequest):
    """
    Analyze visual input and extract features
    """
    try:
        # Convert base64 image to numpy array
        image_array = base64_to_image(request.image)

        # Extract features
        features = perception_module.extract_features(image_array)

        # If query is provided, ground it in vision
        relevant_objects = []
        grounding_confidence = None
        if request.query:
            relevant_objects = perception_module.ground_language_in_vision(features, request.query)
            if relevant_objects:
                grounding_confidence = np.mean([obj.get("relevance_score", 0) for obj in relevant_objects])

        response = VisionAnalysisResponse(
            object_detections=features.object_detections,
            spatial_features=features.spatial_features.tolist(),
            semantic_features=features.semantic_features.tolist(),
            affordance_features=features.affordance_features.tolist() if features.affordance_features is not None else None,
            scene_context=features.scene_graph,
            relevant_objects=relevant_objects,
            grounding_confidence=grounding_confidence,
            success=True
        )

        logger.info(f"Vision analysis completed for image with {len(features.object_detections)} objects detected")

        return response

    except Exception as e:
        logger.error(f"Error in vision analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in vision analysis: {str(e)}")


@router.post("/analyze-language", response_model=LanguageAnalysisResponse)
async def analyze_language(request: LanguageAnalysisRequest):
    """
    Analyze natural language input and extract features
    """
    try:
        # Extract features
        features = language_module.extract_features(request.text)

        response = LanguageAnalysisResponse(
            text_embeddings=features.text_embeddings.tolist() if features.text_embeddings is not None else None,
            parsed_commands=features.parsed_commands,
            intent=features.intent,
            entities=features.entities,
            action_sequence=features.action_sequence or [],
            semantic_meaning=features.semantic_meaning,
            success=True
        )

        logger.info(f"Language analysis completed: intent={features.intent}")

        return response

    except Exception as e:
        logger.error(f"Error in language analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in language analysis: {str(e)}")


@router.post("/generate-action", response_model=ActionGenerationResponse)
async def generate_action(request: ActionGenerationRequest):
    """
    Generate action based on vision and language features
    """
    try:
        # Generate action
        action = action_module.generate_action(
            vision_features=request.vision_features,
            language_features=request.language_features,
            robot_state=request.task_context
        )

        response = ActionGenerationResponse(
            action_type=action.action_type,
            action_parameters=action.action_parameters,
            confidence=action.confidence,
            execution_plan=action.execution_plan or [],
            safety_score=action.safety_score or 0.5,
            success=True
        )

        logger.info(f"Action generated: {action.action_type} with confidence {action.confidence}")

        return response

    except Exception as e:
        logger.error(f"Error in action generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in action generation: {str(e)}")


@router.post("/fuse-multimodal", response_model=MultimodalFusionResponse)
async def fuse_multimodal(request: MultimodalFusionRequest):
    """
    Fuse vision and language features using specified method
    """
    try:
        # Convert fusion method string to fusion module's enum
        from backend.api.vla.vla_components.fusion_module import FusionMethod as InternalFusionMethod
        internal_fusion_method = InternalFusionMethod(request.fusion_method.value)

        # Fuse features
        fusion_result = fusion_module.fuse_features(
            {"image_features": np.array(request.vision_features, dtype=np.float32)},
            {"text_embeddings": np.array(request.language_features, dtype=np.float32)},
            method=internal_fusion_method
        )

        response = MultimodalFusionResponse(
            fused_features=fusion_result["fused_features"].tolist(),
            attention_weights=fusion_result.get("attention_weights"),
            fusion_method_used=request.fusion_method,
            success=True
        )

        logger.info(f"Multimodal fusion completed using {request.fusion_method}")

        return response

    except Exception as e:
        logger.error(f"Error in multimodal fusion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in multimodal fusion: {str(e)}")


@router.post("/evaluate", response_model=VLAEvaluationResponse)
async def evaluate_vla(request: VLAEvaluationRequest):
    """
    Evaluate VLA system performance on test cases
    """
    try:
        # For this enhanced version, we'll simulate evaluation
        # In a real implementation, this would connect to a proper evaluation system
        task_success_rate = np.mean([np.random.random() for _ in range(len(request.test_cases))])
        language_accuracy = np.random.random()
        vision_accuracy = np.random.random()
        action_accuracy = np.random.random()

        detailed_results = []
        for i, test_case in enumerate(request.test_cases):
            success = np.random.random() > 0.3  # 70% success rate in simulation
            detailed_results.append({
                "test_id": i,
                "instruction": test_case.get("instruction", "unknown"),
                "expected_action": test_case.get("expected_action", "unknown"),
                "success": success,
                "confidence": float(np.random.random())
            })

        response = VLAEvaluationResponse(
            task_success_rate=float(task_success_rate),
            language_accuracy=float(language_accuracy),
            vision_accuracy=float(vision_accuracy),
            action_accuracy=float(action_accuracy),
            efficiency_metrics={"avg_time": 2.5, "throughput": 0.4},
            detailed_results=detailed_results,
            success=True
        )

        logger.info(f"VLA evaluation completed: success_rate={task_success_rate:.2f}")

        return response

    except Exception as e:
        logger.error(f"Error in VLA evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in VLA evaluation: {str(e)}")


@router.post("/ground-language", response_model=GroundLanguageInVisionResponse)
async def ground_language_in_vision(request: GroundLanguageInVisionRequest):
    """
    Ground language query in visual features
    """
    try:
        # Convert base64 image to numpy array
        image_array = base64_to_image(request.image)

        # Extract vision features
        vision_features = perception_module.extract_features(image_array)

        # Ground language in vision
        relevant_objects = perception_module.ground_language_in_vision(vision_features, request.language_query)

        # Calculate grounding confidence based on relevance scores
        grounding_confidence = np.mean([obj.get("relevance_score", 0) for obj in relevant_objects]) if relevant_objects else 0.0

        response = GroundLanguageInVisionResponse(
            relevant_objects=relevant_objects,
            grounding_confidence=grounding_confidence,
            success=True
        )

        logger.info(f"Language grounding completed: {len(relevant_objects)} relevant objects found")

        return response

    except Exception as e:
        logger.error(f"Error in language grounding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in language grounding: {str(e)}")


@router.post("/generate-plan", response_model=GenerateExecutionPlanResponse)
async def generate_execution_plan(request: GenerateExecutionPlanRequest):
    """
    Generate detailed execution plan for complex tasks
    """
    try:
        # Use language module to understand the instruction
        language_features = language_module.extract_features(request.instruction)

        # Create a basic execution plan based on the action sequence
        execution_plan = []
        if language_features.action_sequence:
            for i, action in enumerate(language_features.action_sequence):
                step = {
                    "step_number": i + 1,
                    "action": action,
                    "description": f"Execute {action} action",
                    "estimated_time": float(np.random.uniform(1.0, 3.0)),  # Random time for simulation
                    "dependencies": [i] if i > 0 else [],  # Each step depends on previous
                    "constraints": []
                }
                execution_plan.append(step)

        # Add environment-specific considerations
        if "fragile" in request.instruction.lower():
            for step in execution_plan:
                if "grasp" in step["action"] or "pick" in step["action"]:
                    step["constraints"].append("use_low_force")
                    step["estimated_time"] += 1.0  # Extra care takes time

        estimated_time = sum(step["estimated_time"] for step in execution_plan)

        response = GenerateExecutionPlanResponse(
            execution_plan=execution_plan,
            estimated_completion_time=estimated_time,
            success=True
        )

        logger.info(f"Execution plan generated with {len(execution_plan)} steps")

        return response

    except Exception as e:
        logger.error(f"Error in execution plan generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in execution plan generation: {str(e)}")


# Advanced VLA Endpoints
@router.post("/memory/read", response_model=MemoryResponse)
async def memory_read(request: MemoryReadRequest):
    """
    Read from external memory
    """
    try:
        results = memory_module.read_from_memory(
            query=request.query,
            memory_type=request.memory_type,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )

        response = MemoryResponse(
            retrieved_content=results,
            similarity_scores=[item.get("similarity", 0.0) for item in results],
            memory_type=request.memory_type,
            success=True
        )

        logger.info(f"Memory read completed: {len(results)} items retrieved")

        return response

    except Exception as e:
        logger.error(f"Error in memory read: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in memory read: {str(e)}")


@router.post("/memory/write", response_model=MemoryResponse)
async def memory_write(request: MemoryWriteRequest):
    """
    Write to external memory
    """
    try:
        memory_id = memory_module.write_to_memory(
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata,
            tags=request.tags
        )

        response = MemoryResponse(
            retrieved_content=[{
                "id": memory_id,
                "content": request.content,
                "memory_type": request.memory_type
            }],
            similarity_scores=[1.0],
            memory_type=request.memory_type,
            success=True
        )

        logger.info(f"Memory write completed: {memory_id}")

        return response

    except Exception as e:
        logger.error(f"Error in memory write: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in memory write: {str(e)}")


@router.post("/multi-step", response_model=MultiStepTaskResponse)
async def multi_step_task(request: MultiStepTaskRequest):
    """
    Process multi-step instructions
    """
    try:
        # Use language module to decompose the high-level instruction
        subtasks = language_module._decompose_task(request.high_level_instruction)

        # Create dependency graph
        dependencies = []
        for i, subtask in enumerate(subtasks):
            dependencies.append({
                "task_id": i,
                "task": subtask,
                "dependencies": language_module._get_subtask_dependencies(subtask, subtasks)
            })

        response = MultiStepTaskResponse(
            task_decomposition=subtasks,
            execution_order=list(range(len(subtasks))),
            dependency_graph=dependencies,
            success=True
        )

        logger.info(f"Multi-step task decomposition completed: {len(subtasks)} subtasks")

        return response

    except Exception as e:
        logger.error(f"Error in multi-step task processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in multi-step task processing: {str(e)}")


@router.post("/interactive", response_model=InteractiveVLAResponse)
async def interactive_vla(request: InteractiveVLARequest):
    """
    Interactive VLA session
    """
    try:
        # Process user input with language module
        language_features = language_module.extract_features(request.user_input)

        # Generate appropriate response based on intent
        if language_features.intent == "grasp_object":
            action_response = "I will grasp the object you indicated."
        elif language_features.intent == "navigate":
            action_response = "I will navigate to the specified location."
        elif language_features.intent == "unknown":
            action_response = "I didn't understand that. Could you please rephrase?"
        else:
            action_response = f"I will perform the {language_features.intent} action."

        # Generate next prompt based on current state
        next_prompt = "What would you like me to do next?" if language_features.intent != "unknown" else None

        response = InteractiveVLAResponse(
            action_or_response=action_response,
            next_prompt=next_prompt,
            session_id=request.session_id or "new_session",
            task_progress=0.0,  # This would be updated based on actual task progress
            success=True
        )

        logger.info(f"Interactive VLA response generated: {language_features.intent}")

        return response

    except Exception as e:
        logger.error(f"Error in interactive VLA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in interactive VLA: {str(e)}")


@router.post("/learn-from-demo", response_model=DemonstrationLearningResponse)
async def learn_from_demonstration(request: DemonstrationLearningRequest):
    """
    Learn from human demonstrations
    """
    try:
        # Simulate learning from demonstration
        # In a real implementation, this would train a policy based on the demonstration sequence
        learned_policy = {
            "demonstration_length": len(request.demonstration_sequence),
            "task_type": request.task_description,
            "robot_config": request.robot_configuration
        }

        response = DemonstrationLearningResponse(
            learned_policy=learned_policy,
            generalization_score=0.85,  # Simulated score
            confidence=0.9,
            adaptation_requirements=["kinematic calibration", "force control tuning"],
            success=True
        )

        logger.info(f"Demonstration learning completed for: {request.task_description}")

        return response

    except Exception as e:
        logger.error(f"Error in demonstration learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in demonstration learning: {str(e)}")


@router.post("/adapt-context", response_model=ContextAdaptationResponse)
async def adapt_context(request: ContextAdaptationRequest):
    """
    Adapt to new contexts
    """
    try:
        # Simulate context adaptation
        # In a real implementation, this would adapt the model to the new context
        adaptation_score = 0.75  # Simulated score
        required_steps = 50  # Simulated training steps
        estimated_performance = 0.82  # Simulated performance

        response = ContextAdaptationResponse(
            adapted_model=f"adapted_model_{request.adaptation_method}",
            adaptation_score=adaptation_score,
            required_training_steps=required_steps,
            estimated_performance=estimated_performance,
            success=True
        )

        logger.info(f"Context adaptation completed: {request.adaptation_method}")

        return response

    except Exception as e:
        logger.error(f"Error in context adaptation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in context adaptation: {str(e)}")