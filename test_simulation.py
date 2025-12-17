#!/usr/bin/env python3
"""
Test script for simulation utilities module
"""
import numpy as np
from backend.utils.simulation_utils import (
    simulation_manager, SimulationPlatform, PhysicsEngine,
    PhysicsSimulator, SensorSimulator, SimulationEnvironment,
    DomainRandomization
)


def test_physics_simulator():
    """Test physics simulation functionality"""
    print("Testing Physics Simulator...")

    # Create physics simulator
    phys_sim = PhysicsSimulator(PhysicsEngine.BULLET)
    phys_sim.set_gravity((0.0, 0.0, -9.81))
    phys_sim.set_time_step(0.001)

    # Add a rigid body
    phys_sim.add_rigid_body(
        "test_box",
        mass=1.0,
        position=np.array([0.0, 0.0, 2.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])  # 4-element quaternion
    )

    # Step the simulation
    stats = phys_sim.step(0.01)
    print(f"Physics step stats: {stats}")

    # Check that object moved due to gravity
    obj_pos = phys_sim.objects["test_box"]["position"]
    print(f"Object position after gravity: {obj_pos}")
    assert obj_pos[2] < 2.0, "Object should have moved downward due to gravity"

    print("PASS: Physics Simulator test completed\n")


def test_sensor_simulator():
    """Test sensor simulation functionality"""
    print("Testing Sensor Simulator...")

    # Create sensor simulator
    sensor_sim = SensorSimulator()

    # Add different types of sensors
    sensor_sim.add_camera(
        "camera_1",
        position=(0.0, 0.0, 1.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        params={"width": 640, "height": 480}
    )

    sensor_sim.add_lidar(
        "lidar_1",
        position=(0.0, 0.0, 1.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        params={"num_points": 360}
    )

    sensor_sim.add_imu(
        "imu_1",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        params={}
    )

    # Generate sensor data
    camera_data = sensor_sim.generate_sensor_data("camera_1", {})
    lidar_data = sensor_sim.generate_sensor_data("lidar_1", {})
    imu_data = sensor_sim.generate_sensor_data("imu_1", {})

    print(f"Camera data keys: {list(camera_data.keys()) if camera_data else 'None'}")
    print(f"LIDAR data keys: {list(lidar_data.keys()) if lidar_data else 'None'}")
    print(f"IMU data keys: {list(imu_data.keys()) if imu_data else 'None'}")

    assert camera_data is not None, "Camera data should be generated"
    assert lidar_data is not None, "LIDAR data should be generated"
    assert imu_data is not None, "IMU data should be generated"

    print("PASS: Sensor Simulator test completed\n")


def test_simulation_environment():
    """Test simulation environment functionality"""
    print("Testing Simulation Environment...")

    # Create simulation environment
    env = SimulationEnvironment(SimulationPlatform.GAZEBO)

    # Create a world
    world = env.create_world("test_world", "Test simulation world")
    print(f"Created world: {world.name} with ID: {world.id}")

    # Load a robot
    robot = env.load_robot_model(
        world.id,
        "/path/to/robot.urdf",
        "test_robot",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    print(f"Loaded robot: {robot.name} with ID: {robot.id}")

    # Add a sensor to the robot
    sensor_config = {
        "type": "camera",
        "name": "robot_camera",
        "position": (0.1, 0.0, 0.5),
        "parameters": {"width": 320, "height": 240}
    }
    sensor = env.add_sensor_to_robot(robot.id, sensor_config)
    print(f"Added sensor: {sensor.name} to robot")

    # Start simulation
    env.start_simulation()
    print("Simulation started")

    # Step simulation
    env.step_simulation(0.01)
    print("Simulation stepped")

    # Get simulation state
    state = env.get_current_state()
    if state:
        print(f"Simulation time: {state.time}")
        print(f"Robot states count: {len(state.robot_states)}")
        print(f"Sensor data count: {len(state.sensor_data)}")

    # Stop simulation
    env.stop_simulation()
    print("Simulation stopped")

    print("PASS: Simulation Environment test completed\n")


def test_domain_randomization():
    """Test domain randomization functionality"""
    print("Testing Domain Randomization...")

    # Create domain randomizer
    domain_rand = DomainRandomization()

    # Add randomization parameters
    domain_rand.add_randomization_parameter("gravity_z", "float", -12.0, -6.0)
    domain_rand.add_randomization_parameter("friction", "float", 0.1, 1.0)
    domain_rand.add_randomization_parameter("mass_multiplier", "float", 0.8, 1.2)

    print("Added randomization parameters")

    # Get initial values
    initial_values = domain_rand.get_current_randomization_values()
    print(f"Initial values: {initial_values}")

    # Randomize a specific parameter
    new_gravity = domain_rand.randomize_parameter("gravity_z")
    print(f"Randomized gravity_z: {new_gravity}")

    # Randomize all parameters
    all_new_values = domain_rand.randomize_all_parameters()
    print(f"All randomized values: {all_new_values}")

    # Verify values are within bounds
    assert -12.0 <= all_new_values["gravity_z"] <= -6.0
    assert 0.1 <= all_new_values["friction"] <= 1.0
    assert 0.8 <= all_new_values["mass_multiplier"] <= 1.2

    print("PASS: Domain Randomization test completed\n")


def test_simulation_manager():
    """Test simulation manager functionality"""
    print("Testing Simulation Manager...")

    # Create an environment through the manager
    env = simulation_manager.create_environment("test_env")
    print(f"Created environment: test_env")

    # Create a world in the environment
    world = env.create_world("managed_world", "World created through manager")
    print(f"Created world through manager: {world.name}")

    # List environments
    envs = simulation_manager.list_environments()
    print(f"Total environments: {len(envs)}")

    # Verify the environment exists
    retrieved_env = simulation_manager.get_environment("test_env")
    assert retrieved_env is not None, "Environment should be retrievable"

    print("PASS: Simulation Manager test completed\n")


def run_all_tests():
    """Run all simulation utility tests"""
    print("Starting Simulation Utilities Tests\n")

    test_physics_simulator()
    test_sensor_simulator()
    test_simulation_environment()
    test_domain_randomization()
    test_simulation_manager()

    print("All simulation utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()