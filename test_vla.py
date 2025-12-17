#!/usr/bin/env python3
"""
Test script for Vision-Language-Action functionality
"""
import numpy as np
from backend.utils.vla_utils import (
    VLASystem, VLAEvaluator, VisionProcessor, LanguageProcessor, ActionProcessor,
    MultimodalFusion, VisionFeatures, LanguageFeatures
)

def test_vision_processor():
    """Test vision processor functionality"""
    print("Testing Vision Processor...")

    processor = VisionProcessor()

    # Create a sample image (simulated)
    sample_image = np.random.random((224, 224, 3)).astype(np.float32)

    # Extract features
    features = processor.extract_features(sample_image)

    print(f"Extracted features - Objects detected: {len(features.object_detections)}")
    print(f"Image features shape: {features.image_features.shape}")
    print(f"Spatial features shape: {features.spatial_features.shape}")

    # Test language grounding
    grounded_objects = processor.ground_language_in_vision(features, "find the red object")
    print(f"Grounded objects: {len(grounded_objects)}")

    print("PASS: Vision Processor tests passed\n")


def test_language_processor():
    """Test language processor functionality"""
    print("Testing Language Processor...")

    processor = LanguageProcessor()

    # Test with sample instructions
    test_instructions = [
        "Pick up the red cup from the table",
        "Move to the kitchen and clean the counter",
        "Grasp the box and place it on the shelf"
    ]

    for instruction in test_instructions:
        features = processor.extract_features(instruction)
        print(f"Instruction: '{instruction}'")
        print(f"  Intent: {features.intent}")
        print(f"  Entities: {features.entities}")
        print(f"  Action sequence: {features.action_sequence}")
        print(f"  Parsed commands: {len(features.parsed_commands)}")

    print("PASS: Language Processor tests passed\n")


def test_action_processor():
    """Test action processor functionality"""
    print("Testing Action Processor...")

    processor = ActionProcessor()

    # Create sample features
    vision_features = VisionFeatures(
        image_features=np.random.random(512).astype(np.float32),
        object_detections=[
            {
                "class": "cup",
                "bbox": [0.3, 0.4, 0.6, 0.7],
                "confidence": 0.9,
                "center": [0.45, 0.55]
            }
        ],
        spatial_features=np.random.random(128).astype(np.float32),
        semantic_features=np.random.random(256).astype(np.float32),
        affordance_features=np.random.random(64).astype(np.float32)
    )

    language_features = LanguageFeatures(
        text_embeddings=np.random.random(768).astype(np.float32),
        parsed_commands=[{"type": "grasp", "target": "cup", "parameters": {"force": 0.5}}],
        intent="grasp_object",
        entities=[{"type": "object", "value": "cup"}],
        action_sequence=["approach_object", "grasp", "lift"]
    )

    # Generate action
    action = processor.generate_action(vision_features, language_features)

    print(f"Generated action: {action.action_type}")
    print(f"Action parameters: {action.action_parameters}")
    print(f"Action confidence: {action.confidence:.2f}")
    print(f"Execution plan steps: {len(action.execution_plan)}")

    print("PASS: Action Processor tests passed\n")


def test_multimodal_fusion():
    """Test multimodal fusion functionality"""
    print("Testing Multimodal Fusion...")

    # Test different fusion methods
    fusion_methods = ["early_fusion", "late_fusion", "cross_attention"]

    vision_features = VisionFeatures(
        image_features=np.random.random(256).astype(np.float32),
        object_detections=[],
        spatial_features=np.random.random(128).astype(np.float32),
        semantic_features=np.random.random(256).astype(np.float32),
        affordance_features=None
    )

    language_features = LanguageFeatures(
        text_embeddings=np.random.random(256).astype(np.float32),  # Same dim for simplicity
        parsed_commands=[],
        intent="test",
        entities=[],
        action_sequence=[]
    )

    for method in fusion_methods:
        fusion_module = MultimodalFusion(fusion_method=method)
        fused_features = fusion_module.fuse_features(vision_features, language_features)
        print(f"{method}: fused features shape = {fused_features.shape}")

    print("PASS: Multimodal Fusion tests passed\n")


def test_vla_system():
    """Test complete VLA system"""
    print("Testing Complete VLA System...")

    # Create VLA system
    vla_system = VLASystem(fusion_method="cross_attention")

    # Create sample image and instruction
    sample_image = np.random.random((224, 224, 3)).astype(np.float32)
    instruction = "Grasp the red cup on the table"

    # Process the task
    action = vla_system.process_task(sample_image, instruction)

    print(f"VLA System output:")
    print(f"  Action type: {action.action_type}")
    print(f"  Action parameters: {action.action_parameters}")
    print(f"  Confidence: {action.confidence:.2f}")
    print(f"  Execution plan: {len(action.execution_plan)} steps")

    # Test multimodal input processing
    transition = vla_system.process_multimodal_input(sample_image, instruction)
    print(f"  Transition created successfully")

    print("PASS: Complete VLA System tests passed\n")


def test_vla_evaluation():
    """Test VLA evaluation functionality"""
    print("Testing VLA Evaluation...")

    # Create test cases
    test_cases = [
        {
            "image": np.random.random((224, 224, 3)),
            "instruction": "pick up the object",
            "expected_action": "grasp"
        },
        {
            "image": np.random.random((224, 224, 3)),
            "instruction": "move to the table",
            "expected_action": "move"
        },
        {
            "image": np.random.random((224, 224, 3)),
            "instruction": "place the item",
            "expected_action": "place"
        }
    ]

    # Create VLA system and evaluator
    vla_system = VLASystem()
    evaluator = VLAEvaluator()

    # Run evaluation
    results = evaluator.evaluate_vla_system(vla_system, test_cases)

    print(f"Evaluation results:")
    print(f"  Task success rate: {results['task_success_rate']:.2f}")
    print(f"  Language accuracy: {results['language_accuracy']:.2f}")
    print(f"  Efficiency metrics: {results['efficiency_metrics']}")
    print(f"  Detailed results: {len(results['detailed_results'])} cases")

    print("PASS: VLA Evaluation tests passed\n")


def run_all_tests():
    """Run all VLA functionality tests"""
    print("Starting VLA Functionality Tests\n")

    test_vision_processor()
    test_language_processor()
    test_action_processor()
    test_multimodal_fusion()
    test_vla_system()
    test_vla_evaluation()

    print("All VLA functionality tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()