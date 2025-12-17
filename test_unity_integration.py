#!/usr/bin/env python3
"""
Test script for Unity integration utilities module
"""
from backend.utils.unity_integration_utils import (
    unity_integration_manager, UnityVector3, UnityQuaternion, UnityTransform,
    UnityObjectType, UnityComponentType, UnityPhysicsEngine
)


def test_unity_vectors():
    """Test Unity vector and quaternion functionality"""
    print("Testing Unity Vector3 and Quaternion...")

    # Test UnityVector3
    vec = UnityVector3(1.0, 2.0, 3.0)
    vec_list = vec.to_list()
    print(f"Vector3: {vec}, as list: {vec_list}")
    assert len(vec_list) == 3
    assert vec_list == [1.0, 2.0, 3.0]

    reconstructed_vec = UnityVector3.from_list(vec_list)
    print(f"Reconstructed vector: {reconstructed_vec}")
    assert reconstructed_vec.x == vec.x
    assert reconstructed_vec.y == vec.y
    assert reconstructed_vec.z == vec.z

    # Test UnityQuaternion
    quat = UnityQuaternion(0.0, 0.1, 0.0, 1.0)
    quat_list = quat.to_list()
    print(f"Quaternion: {quat}, as list: {quat_list}")
    assert len(quat_list) == 4
    assert quat_list == [0.0, 0.1, 0.0, 1.0]

    reconstructed_quat = UnityQuaternion.from_list(quat_list)
    print(f"Reconstructed quaternion: {reconstructed_quat}")
    assert reconstructed_quat.x == quat.x
    assert reconstructed_quat.y == quat.y
    assert reconstructed_quat.z == quat.z
    assert reconstructed_quat.w == quat.w

    print("PASS: Unity Vector3 and Quaternion test completed\n")


def test_unity_transform():
    """Test Unity Transform functionality"""
    print("Testing Unity Transform...")

    position = UnityVector3(1.0, 2.0, 3.0)
    rotation = UnityQuaternion(0.0, 0.0, 0.0, 1.0)
    scale = UnityVector3(1.0, 1.0, 1.0)

    transform = UnityTransform(position, rotation, scale)
    transform_dict = transform.to_dict()
    print(f"Transform as dict: {transform_dict}")

    assert transform_dict["position"] == [1.0, 2.0, 3.0]
    assert transform_dict["rotation"] == [0.0, 0.0, 0.0, 1.0]
    assert transform_dict["scale"] == [1.0, 1.0, 1.0]

    reconstructed_transform = UnityTransform.from_dict(transform_dict)
    print(f"Reconstructed transform: {reconstructed_transform.to_dict()}")
    assert reconstructed_transform.position.x == position.x
    assert reconstructed_transform.rotation.w == rotation.w

    print("PASS: Unity Transform test completed\n")


def test_scene_management():
    """Test Unity scene management functionality"""
    print("Testing Unity Scene Management...")

    # Create a scene
    result = unity_integration_manager.create_unity_environment("test_scene")
    scene_id = result["scene_id"]
    print(f"Scene created: {result['name']} with ID: {scene_id}")
    assert result["success"] == True

    # Get scene info
    scene_info = unity_integration_manager.get_scene_info(scene_id)
    print(f"Scene info retrieved, objects count: {scene_info['scene']['object_count']}")

    # List all scenes
    scenes = unity_integration_manager.list_scenes()
    print(f"Total scenes: {len(scenes)}")
    assert len(scenes) >= 1

    # Load the scene
    success = unity_integration_manager.scene_manager.load_scene(scene_id)
    print(f"Scene load success: {success}")
    assert success == True

    print("PASS: Unity Scene Management test completed\n")


def test_object_creation():
    """Test Unity object creation functionality"""
    print("Testing Unity Object Creation...")

    # Create a scene first
    result = unity_integration_manager.create_unity_environment("object_test_scene")
    scene_id = result["scene_id"]
    print(f"Created scene: {scene_id}")

    # Create an object in the scene
    object_id = unity_integration_manager.scene_manager.create_object(
        scene_id,
        "test_cube",
        UnityObjectType.GAME_OBJECT,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        ["test", "cube"]
    )
    print(f"Object created with ID: {object_id}")
    assert object_id is not None

    # List objects in scene
    objects = unity_integration_manager.scene_manager.list_objects(scene_id)
    print(f"Objects in scene: {len(objects)}")
    assert len(objects) == 1

    # Get specific object
    obj = unity_integration_manager.scene_manager.get_object(scene_id, object_id)
    print(f"Object name: {obj.name}, type: {obj.object_type.value}")
    assert obj.name == "test_cube"

    # Set object transform
    success = unity_integration_manager.scene_manager.set_transform(
        scene_id,
        object_id,
        position=(1.0, 1.0, 1.0)
    )
    print(f"Transform set success: {success}")
    assert success == True

    print("PASS: Unity Object Creation test completed\n")


def test_component_management():
    """Test Unity component management functionality"""
    print("Testing Unity Component Management...")

    # Create a scene and object
    result = unity_integration_manager.create_unity_environment("component_test_scene")
    scene_id = result["scene_id"]
    print(f"Created scene: {scene_id}")

    object_id = unity_integration_manager.scene_manager.create_object(
        scene_id,
        "component_test_obj",
        UnityObjectType.GAME_OBJECT,
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
        ["test"]
    )
    print(f"Created object: {object_id}")

    # Add a rigidbody component
    success = unity_integration_manager.scene_manager.add_component(
        scene_id,
        object_id,
        UnityComponentType.RIGIDBODY,
        {
            "mass": 1.0,
            "drag": 0.1,
            "angular_drag": 0.05,
            "use_gravity": True,
            "is_kinematic": False
        }
    )
    print(f"Rigidbody component added: {success}")
    assert success == True

    # Add a collider component
    success = unity_integration_manager.scene_manager.add_component(
        scene_id,
        object_id,
        UnityComponentType.COLLIDER,
        {
            "type": "box",
            "center": [0, 0, 0],
            "size": [1, 1, 1],
            "is_trigger": False
        }
    )
    print(f"Collider component added: {success}")
    assert success == True

    # Get components
    rigidbody_comp = unity_integration_manager.scene_manager.get_component(
        scene_id, object_id, UnityComponentType.RIGIDBODY
    )
    print(f"Rigidbody component: {rigidbody_comp}")
    assert rigidbody_comp is not None
    assert rigidbody_comp["mass"] == 1.0

    print("PASS: Unity Component Management test completed\n")


def test_robot_spawning():
    """Test Unity robot spawning functionality"""
    print("Testing Unity Robot Spawning...")

    # Create a scene
    result = unity_integration_manager.create_unity_environment("robot_test_scene")
    scene_id = result["scene_id"]
    print(f"Created scene: {scene_id}")

    # Spawn a robot
    spawn_result = unity_integration_manager.spawn_robot_in_environment(
        scene_id,
        "test_robot",
        "mobile_base"
    )
    print(f"Robot spawn result: {spawn_result}")
    assert spawn_result is not None
    assert spawn_result["success"] == True

    robot_object_id = spawn_result["robot_object_id"]
    print(f"Robot spawned with object ID: {robot_object_id}")

    # Get robot state
    robot_state = unity_integration_manager.get_robot_state(scene_id, robot_object_id)
    print(f"Robot state: {robot_state}")
    assert robot_state is not None
    assert robot_state["object_id"] == robot_object_id

    # List robots in scene
    robots = unity_integration_manager.list_robots(scene_id)
    print(f"Robots in scene: {len(robots)}")
    assert len(robots) == 1

    print("PASS: Unity Robot Spawning test completed\n")


def test_sensor_system():
    """Test Unity sensor system functionality"""
    print("Testing Unity Sensor System...")

    # Create a scene and robot
    result = unity_integration_manager.create_unity_environment("sensor_test_scene")
    scene_id = result["scene_id"]
    print(f"Created scene: {scene_id}")

    spawn_result = unity_integration_manager.spawn_robot_in_environment(
        scene_id,
        "sensor_robot",
        "mobile_base"
    )
    robot_object_id = spawn_result["robot_object_id"]
    print(f"Spawned robot: {robot_object_id}")

    # Add a camera sensor
    camera_sensor_id = unity_integration_manager.add_sensor_to_robot(
        scene_id,
        robot_object_id,
        "camera",
        "robot_camera"
    )
    print(f"Camera sensor added with ID: {camera_sensor_id}")
    assert camera_sensor_id is not None

    # Add a LIDAR sensor
    lidar_sensor_id = unity_integration_manager.add_sensor_to_robot(
        scene_id,
        robot_object_id,
        "lidar",
        "robot_lidar"
    )
    print(f"LIDAR sensor added with ID: {lidar_sensor_id}")
    assert lidar_sensor_id is not None

    # Add an IMU sensor
    imu_sensor_id = unity_integration_manager.add_sensor_to_robot(
        scene_id,
        robot_object_id,
        "imu",
        "robot_imu"
    )
    print(f"IMU sensor added with ID: {imu_sensor_id}")
    assert imu_sensor_id is not None

    # Get sensor data
    camera_data = unity_integration_manager.get_sensor_data(camera_sensor_id)
    print(f"Camera data retrieved: {camera_data is not None}")
    assert camera_data is not None

    lidar_data = unity_integration_manager.get_sensor_data(lidar_sensor_id)
    print(f"LIDAR data retrieved: {lidar_data is not None}")
    assert lidar_data is not None

    imu_data = unity_integration_manager.get_sensor_data(imu_sensor_id)
    print(f"IMU data retrieved: {imu_data is not None}")
    assert imu_data is not None

    # List all sensors
    sensors = unity_integration_manager.list_sensors()
    print(f"Total sensors in system: {len(sensors)}")
    assert len(sensors) >= 3

    print("PASS: Unity Sensor System test completed\n")


def test_physics_simulation():
    """Test Unity physics simulation functionality"""
    print("Testing Unity Physics Simulation...")

    # Create a scene
    result = unity_integration_manager.create_unity_environment("physics_test_scene")
    scene_id = result["scene_id"]
    print(f"Created scene: {scene_id}")

    # Create an object with physics components
    object_id = unity_integration_manager.scene_manager.create_object(
        scene_id,
        "physics_cube",
        UnityObjectType.MODEL,
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0, 1.0),
        ["physics", "test"]
    )
    print(f"Created physics object: {object_id}")

    # Add rigidbody component
    success = unity_integration_manager.scene_manager.add_component(
        scene_id,
        object_id,
        UnityComponentType.RIGIDBODY,
        {
            "mass": 1.0,
            "drag": 0.0,
            "angular_drag": 0.05,
            "use_gravity": True,
            "is_kinematic": False
        }
    )
    print(f"Rigidbody added: {success}")

    # Add collider component
    success = unity_integration_manager.scene_manager.add_component(
        scene_id,
        object_id,
        UnityComponentType.COLLIDER,
        {
            "type": "box",
            "center": [0, 0, 0],
            "size": [1, 1, 1],
            "is_trigger": False
        }
    )
    print(f"Collider added: {success}")

    # Start simulation
    unity_integration_manager.start_simulation()
    print("Simulation started")

    # Wait a bit for physics to run
    import time
    time.sleep(0.1)

    # Apply a force to the object
    success = unity_integration_manager.scene_manager.physics_simulator.apply_force(
        object_id,
        (5.0, 0.0, 0.0)
    )
    print(f"Force applied: {success}")

    # Get the object's state after physics update
    obj = unity_integration_manager.scene_manager.get_object(scene_id, object_id)
    print(f"Object position after force: {obj.transform.position.to_list()}")

    # Stop simulation
    unity_integration_manager.stop_simulation()
    print("Simulation stopped")

    print("PASS: Unity Physics Simulation test completed\n")


def run_all_tests():
    """Run all Unity integration utility tests"""
    print("Starting Unity Integration Utilities Tests\n")

    test_unity_vectors()
    test_unity_transform()
    test_scene_management()
    test_object_creation()
    test_component_management()
    test_robot_spawning()
    test_sensor_system()
    test_physics_simulation()

    print("All Unity integration utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()