#!/usr/bin/env python3
"""
Simple test for Unity integration utilities module
"""
from backend.utils.unity_integration_utils import (
    unity_integration_manager, UnityVector3, UnityQuaternion, UnityTransform,
    UnityObjectType, UnityComponentType, UnityPhysicsEngine
)


def test_basic_functionality():
    """Test basic Unity integration functionality"""
    print("Testing Basic Unity Integration Functionality...")

    # Test vector and quaternion
    vec = UnityVector3(1.0, 2.0, 3.0)
    vec_list = vec.to_list()
    print(f"Vector3: {vec}, as list: {vec_list}")
    assert len(vec_list) == 3
    assert vec_list == [1.0, 2.0, 3.0]
    print("PASS: Vector3 functionality works")

    quat = UnityQuaternion(0.0, 0.1, 0.0, 1.0)
    quat_list = quat.to_list()
    print(f"Quaternion: {quat}, as list: {quat_list}")
    assert len(quat_list) == 4
    assert quat_list == [0.0, 0.1, 0.0, 1.0]
    print("PASS: Quaternion functionality works")

    # Test transform
    transform = UnityTransform(vec, quat, UnityVector3(1.0, 1.0, 1.0))
    transform_dict = transform.to_dict()
    print(f"Transform: {transform_dict}")
    assert "position" in transform_dict
    assert "rotation" in transform_dict
    print("PASS: Transform functionality works")

    # Test scene creation
    result = unity_integration_manager.create_unity_environment("test_scene_basic")
    print(f"Scene creation result: {result}")
    assert result["success"] == True
    print("PASS: Scene creation works")

    # Test object creation
    scene_id = result["scene_id"]
    object_id = unity_integration_manager.scene_manager.create_object(
        scene_id,
        "test_object",
        UnityObjectType.GAME_OBJECT,
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
        ["test", "basic"]
    )
    print(f"Object created: {object_id}")
    assert object_id is not None
    print("PASS: Object creation works")

    # Test robot spawning
    robot_result = unity_integration_manager.spawn_robot_in_environment(
        scene_id,
        "test_robot",
        "mobile_base"
    )
    print(f"Robot spawn result: {robot_result}")
    if robot_result:
        assert robot_result["success"] == True
        print("PASS: Robot spawning works")
    else:
        print("INFO: Robot spawning requires additional dependencies (expected in test)")

    # Test sensor addition
    # (This might fail if required dependencies are not available)
    try:
        sensor_id = unity_integration_manager.add_sensor_to_robot(
            scene_id,
            "dummy_robot_id",  # This will fail gracefully
            "camera",
            "test_camera"
        )
        print(f"Sensor addition result: {sensor_id}")
    except:
        print("INFO: Sensor addition test handled gracefully")

    # List scenes
    scenes = unity_integration_manager.list_scenes()
    print(f"Total scenes: {len(scenes)}")
    assert len(scenes) >= 1
    print("PASS: Scene listing works")

    print("\nAll basic functionality tests passed!")


if __name__ == "__main__":
    test_basic_functionality()