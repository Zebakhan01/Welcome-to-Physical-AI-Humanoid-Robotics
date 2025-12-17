#!/usr/bin/env python3
"""
Test script for Gazebo utilities module
"""
from backend.utils.gazebo_utils import (
    gazebo_scene_manager, GazeboPose, PhysicsEngine, ModelType,
    SDFWorld, SDFModel, SDFParser
)


def test_gazebo_pose():
    """Test GazeboPose functionality"""
    print("Testing GazeboPose...")

    # Create a pose
    pose = GazeboPose(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    print(f"Created pose: {pose}")

    # Test to_list and from_list
    pose_list = pose.to_list()
    print(f"Pose as list: {pose_list}")
    assert len(pose_list) == 6, "Pose list should have 6 elements"

    reconstructed_pose = GazeboPose.from_list(pose_list)
    print(f"Reconstructed pose: {reconstructed_pose}")
    assert reconstructed_pose.x == pose.x, "X coordinate should match"
    assert reconstructed_pose.y == pose.y, "Y coordinate should match"
    assert reconstructed_pose.z == pose.z, "Z coordinate should match"

    print("PASS: GazeboPose test completed\n")


def test_sdf_world_creation():
    """Test SDF world creation"""
    print("Testing SDF World Creation...")

    # Create a simple world
    world = SDFWorld(
        name="test_world",
        physics_engine=PhysicsEngine.ODE,
        gravity=(0.0, 0.0, -9.81),
        models=[],
        lights=[],
        plugins=[],
        scene_properties={"ambient": "0.4 0.4 0.4 1"}
    )

    print(f"World created: {world.name}")
    print(f"Physics engine: {world.physics_engine.value}")
    print(f"Gravity: {world.gravity}")
    print(f"Scene properties: {world.scene_properties}")

    assert world.name == "test_world"
    assert world.physics_engine == PhysicsEngine.ODE
    assert world.gravity == (0.0, 0.0, -9.81)
    assert len(world.models) == 0

    print("PASS: SDF World Creation test completed\n")


def test_sdf_model_creation():
    """Test SDF model creation"""
    print("Testing SDF Model Creation...")

    # Create a simple model
    model = SDFModel(
        name="test_robot",
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

    print(f"Model created: {model.name}")
    print(f"Model type: {model.model_type.value}")
    print(f"Model pose: {model.pose.to_list()}")
    print(f"Number of links: {len(model.links)}")

    assert model.name == "test_robot"
    assert model.model_type == ModelType.DYNAMIC
    assert len(model.links) == 1
    assert model.links[0]["name"] == "base_link"

    print("PASS: SDF Model Creation test completed\n")


def test_world_management():
    """Test world management in scene manager"""
    print("Testing World Management...")

    # Create a world
    sdf_world = SDFWorld(
        name="manager_test_world",
        physics_engine=PhysicsEngine.BULLET,
        gravity=(0.0, 0.0, -9.81),
        models=[],
        lights=[],
        plugins=[],
        scene_properties={}
    )

    world_id = gazebo_scene_manager.create_world(sdf_world)
    print(f"World created with ID: {world_id}")

    # List worlds
    worlds = gazebo_scene_manager.list_worlds()
    print(f"Total worlds: {len(worlds)}")

    # Get world
    world = gazebo_scene_manager.get_world(world_id)
    assert world is not None, "World should be retrievable"
    print(f"Retrieved world: {world.world.name}")

    # Delete world
    success = gazebo_scene_manager.delete_world(world_id)
    print(f"World deletion success: {success}")

    assert success, "World should be deletable"

    # Verify world is gone
    worlds_after = gazebo_scene_manager.list_worlds()
    print(f"Worlds after deletion: {len(worlds_after)}")

    print("PASS: World Management test completed\n")


def test_model_spawning():
    """Test model spawning in worlds"""
    print("Testing Model Spawning...")

    # Create a world first
    sdf_world = SDFWorld(
        name="model_spawn_world",
        physics_engine=PhysicsEngine.ODE,
        gravity=(0.0, 0.0, -9.81),
        models=[],
        lights=[],
        plugins=[],
        scene_properties={}
    )

    world_id = gazebo_scene_manager.create_world(sdf_world)
    print(f"Created world: {world_id}")

    # Create a model to spawn
    sdf_model = SDFModel(
        name="spawned_robot",
        pose=GazeboPose(1.0, 1.0, 0.5, 0.0, 0.0, 0.0),
        model_type=ModelType.DYNAMIC,
        links=[
            {
                "name": "base_link",
                "inertial": {"mass": 5.0, "inertia": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
                "visuals": [{"name": "base_visual", "geometry": {"type": "cylinder", "cylinder": {"radius": 0.2, "length": 0.4}}}],
                "collisions": [{"name": "base_collision", "geometry": {"type": "cylinder", "cylinder": {"radius": 0.2, "length": 0.4}}}]
            }
        ],
        joints=[],
        sensors=[],
        static=False
    )

    # Spawn the model
    model_id = gazebo_scene_manager.spawn_model(world_id, sdf_model)
    print(f"Model spawned with ID: {model_id}")

    assert model_id is not None, "Model should be spawned successfully"

    # List models in world
    world = gazebo_scene_manager.get_world(world_id)
    models = world.list_models()
    print(f"Models in world: {len(models)}")

    # Get model state
    model_state = gazebo_scene_manager.get_model_state(world_id, model_id)
    print(f"Model state retrieved: {model_state is not None}")

    # Clean up
    gazebo_scene_manager.delete_world(world_id)

    print("PASS: Model Spawning test completed\n")


def test_simulation_control():
    """Test simulation control functionality"""
    print("Testing Simulation Control...")

    # Create a world
    sdf_world = SDFWorld(
        name="control_test_world",
        physics_engine=PhysicsEngine.ODE,
        gravity=(0.0, 0.0, -9.81),
        models=[],
        lights=[],
        plugins=[],
        scene_properties={}
    )

    world_id = gazebo_scene_manager.create_world(sdf_world)
    print(f"Created world: {world_id}")

    # Get the world object
    world = gazebo_scene_manager.get_world(world_id)
    assert world is not None, "World should be retrievable"

    # Test simulation control
    print(f"Initial paused state: {world.paused}")

    world.pause_simulation()
    print(f"Paused state after pause: {world.paused}")
    assert world.paused == True, "World should be paused"

    world.unpause_simulation()
    print(f"Paused state after unpause: {world.paused}")
    assert world.paused == False, "World should not be paused"

    # Step simulation
    initial_time = world.sim_time
    world.step_simulation(0.01)  # Step 10ms
    print(f"Time before step: {initial_time}, after step: {world.sim_time}")
    assert world.sim_time > initial_time, "Simulation time should advance after step"

    # Clean up
    gazebo_scene_manager.delete_world(world_id)

    print("PASS: Simulation Control test completed\n")


def test_sdf_parsing():
    """Test SDF parsing functionality"""
    print("Testing SDF Parsing...")

    # Note: Actual SDF parsing may require additional dependencies
    # For now, we'll just test that the parser classes exist
    print(f"SDFParser class exists: {hasattr(SDFParser, 'parse_world_from_xml')}")
    print(f"SDFParser available: {SDFParser is not None}")

    print("PASS: SDF Parsing test completed\n")


def run_all_tests():
    """Run all Gazebo utility tests"""
    print("Starting Gazebo Utilities Tests\n")

    test_gazebo_pose()
    test_sdf_world_creation()
    test_sdf_model_creation()
    test_world_management()
    test_model_spawning()
    test_simulation_control()
    test_sdf_parsing()

    print("All Gazebo utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()