#!/usr/bin/env python3
"""
Simple test for Gazebo utilities module
"""
from backend.utils.gazebo_utils import (
    GazeboPose, PhysicsEngine, ModelType, SDFWorld, SDFModel,
    gazebo_scene_manager
)


def test_basic_functionality():
    """Test basic Gazebo functionality"""
    print("Testing Basic Gazebo Functionality...")

    # Test GazeboPose
    pose = GazeboPose(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    pose_list = pose.to_list()
    print(f"GazeboPose: {pose}, as list: {pose_list}")
    assert len(pose_list) == 6

    reconstructed = GazeboPose.from_list(pose_list)
    assert abs(reconstructed.x - pose.x) < 1e-9
    print("PASS: GazeboPose functionality works")

    # Test SDF World creation
    world = SDFWorld(
        name="test_world",
        physics_engine=PhysicsEngine.ODE,
        gravity=(0.0, 0.0, -9.81),
        models=[],
        lights=[],
        plugins=[],
        scene_properties={"ambient": "0.4 0.4 0.4 1"}
    )
    assert world.name == "test_world"
    assert world.physics_engine == PhysicsEngine.ODE
    print("PASS: SDF World creation works")

    # Test SDF Model creation
    model = SDFModel(
        name="test_model",
        pose=GazeboPose(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        model_type=ModelType.DYNAMIC,
        links=[
            {
                "name": "base_link",
                "inertial": {"mass": 1.0, "inertia": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
                "visuals": [{"name": "base_visual", "geometry": {"type": "box", "box": {"size": [1, 1, 1]}}}],
                "collisions": [{"name": "base_collision", "geometry": {"type": "box", "box": {"size": [1, 1, 1]}}}]
            }
        ],
        joints=[],
        sensors=[],
        static=False
    )
    assert model.name == "test_model"
    assert len(model.links) == 1
    print("PASS: SDF Model creation works")

    # Test scene manager operations
    world_id = gazebo_scene_manager.create_world(world)
    assert world_id is not None
    print(f"PASS: World creation works, ID: {world_id}")

    # List worlds
    worlds = gazebo_scene_manager.list_worlds()
    assert len(worlds) >= 1
    print(f"PASS: World listing works, found {len(worlds)} worlds")

    # Get world
    retrieved_world = gazebo_scene_manager.get_world(world_id)
    assert retrieved_world is not None
    assert retrieved_world.world.name == "test_world"
    print("PASS: World retrieval works")

    # Clean up
    success = gazebo_scene_manager.delete_world(world_id)
    assert success
    print("PASS: World deletion works")

    print("\nAll basic functionality tests passed!")


if __name__ == "__main__":
    test_basic_functionality()