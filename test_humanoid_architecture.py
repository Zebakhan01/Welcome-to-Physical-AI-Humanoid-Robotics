#!/usr/bin/env python3
"""
Test script for humanoid architecture utilities module
"""
from backend.utils.humanoid_architecture_utils import (
    humanoid_manager, HumanoidPlatform, ActuatorType, ControlLevel,
    BalanceController, WholeBodyController, HumanoidController
)


def test_balance_controller():
    """Test balance controller functionality"""
    print("Testing Balance Controller...")

    balance_ctrl = BalanceController()

    # Test support polygon calculation
    foot_positions = [(-0.1, -0.1), (-0.1, 0.1), (0.1, -0.1), (0.1, 0.1)]
    support_polygon = balance_ctrl.calculate_support_polygon(foot_positions)
    print(f"Support polygon: {support_polygon}")
    assert len(support_polygon) > 0, "Support polygon should be calculated"

    # Test ZMP calculation (simplified)
    com_state = (0.0, 0.0, 0.8, 0.0, 0.0, 0.0)  # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
    zmp = balance_ctrl.calculate_zmp(com_state)
    print(f"ZMP position: {zmp}")

    # Test stability margin calculation
    stability_margin = balance_ctrl.calculate_stability_margin((0.0, 0.0), support_polygon)
    print(f"Stability margin: {stability_margin}")
    assert stability_margin >= 0, "Stability margin should be non-negative"

    print("PASS: Balance Controller test completed\n")


def test_whole_body_controller():
    """Test whole body controller functionality"""
    print("Testing Whole Body Controller...")

    wb_ctrl = WholeBodyController()

    # Add a task
    wb_ctrl.add_task("move_arm", 10, {"target": [0.5, 0.5, 0.5]})
    print("Task added to whole body controller")

    # Add constraints
    wb_ctrl.add_constraint("balance", {"enabled": True})
    wb_ctrl.add_constraint("joint_limits", {"enabled": True})
    print(f"Constraints added: {len(wb_ctrl.active_constraints)}")

    # Verify tasks are prioritized
    assert len(wb_ctrl.task_queue) == 1, "Task should be in queue"
    print(f"Task queue length: {len(wb_ctrl.task_queue)}")

    # Remove a constraint
    wb_ctrl.remove_constraint("joint_limits")
    print(f"Constraints after removal: {len(wb_ctrl.active_constraints)}")
    assert len(wb_ctrl.active_constraints) == 1, "Should have 1 constraint after removal"

    print("PASS: Whole Body Controller test completed\n")


def test_humanoid_controller():
    """Test humanoid controller functionality"""
    print("Testing Humanoid Controller...")

    # Create a simple body plan for testing
    from backend.utils.humanoid_architecture_utils import (
        HumanoidBodyPlan, JointConfiguration, LinkProperties
    )

    joint_configs = [
        JointConfiguration(
            name="hip_joint",
            position=0.0,
            velocity=0.0,
            effort=0.0,
            limits=(-1.57, 1.57)
        ),
        JointConfiguration(
            name="knee_joint",
            position=0.0,
            velocity=0.0,
            effort=0.0,
            limits=(-1.57, 1.57)
        )
    ]

    link_props = [
        LinkProperties(
            name="hip_link",
            mass=1.0,
            com=(0.0, 0.0, 0.0),
            inertia=(0.1, 0.1, 0.1, 0.0, 0.0, 0.0)
        )
    ]

    actuator_types = {"hip_joint": ActuatorType.SERVO, "knee_joint": ActuatorType.SERVO}

    body_plan = HumanoidBodyPlan(
        platform=HumanoidPlatform.CUSTOM,
        total_dof=2,
        joint_configurations=joint_configs,
        link_properties=link_props,
        actuator_types=actuator_types,
        sensor_configurations={"legs": ["force_torque"]}
    )

    # Create controller
    controller = HumanoidController(body_plan)
    print(f"Controller created with {controller.body_plan.total_dof} DOF")

    # Check initial state
    initial_state = controller.current_state
    print(f"Initial balance mode: {initial_state.balance_state.balance_mode}")
    print(f"Initial stability margin: {initial_state.balance_state.stability_margin}")
    assert initial_state.is_safe, "Initial state should be safe"

    # Update state with sensor data
    sensor_data = {
        "imu": {"orientation": [0, 0, 0, 1], "angular_velocity": [0, 0, 0], "linear_acceleration": [0, 0, 9.81]},
        "ft_sensors": {"left_foot": [0, 0, -50, 0, 0, 0]}
    }

    updated_state = controller.update_state(sensor_data)
    print(f"State updated, safe: {updated_state.is_safe}")

    # Compute control commands
    commands = controller.compute_control_commands()
    print(f"Control commands computed, joint commands: {len(commands['joint_commands'])}")
    assert len(commands["joint_commands"]) == 2, "Should have commands for both joints"

    print("PASS: Humanoid Controller test completed\n")


def test_humanoid_platform_manager():
    """Test humanoid platform manager functionality"""
    print("Testing Humanoid Platform Manager...")

    # Create different platform types
    nao_controller = humanoid_manager.create_platform(HumanoidPlatform.NAO, "test_nao")
    atlas_controller = humanoid_manager.create_platform(HumanoidPlatform.ATLAS, "test_atlas")

    print(f"NAO controller created with {nao_controller.body_plan.total_dof} DOF")
    print(f"ATLAS controller created with {atlas_controller.body_plan.total_dof} DOF")

    # Verify DOF counts
    assert nao_controller.body_plan.total_dof == 25, "NAO should have 25 DOF"
    assert atlas_controller.body_plan.total_dof == 28, "ATLAS should have 28 DOF"

    # Get robots
    nao_robot = humanoid_manager.get_robot("test_nao")
    atlas_robot = humanoid_manager.get_robot("test_atlas")

    assert nao_robot is not None, "NAO robot should be retrievable"
    assert atlas_robot is not None, "ATLAS robot should be retrievable"

    # List platforms
    platforms = humanoid_manager.list_platforms()
    print(f"Available platforms: {platforms}")
    assert len(platforms) >= 2, "Should have at least 2 platforms"

    # List robots
    robots = list(humanoid_manager.active_robots.keys())
    print(f"Active robots: {robots}")
    assert len(robots) >= 2, "Should have at least 2 robots"

    print("PASS: Humanoid Platform Manager test completed\n")


def test_performance_evaluation():
    """Test performance evaluation functionality"""
    print("Testing Performance Evaluation...")

    # Create a robot for testing
    robot = humanoid_manager.create_platform(HumanoidPlatform.NAO, "eval_test_robot")

    # Evaluate different tasks
    walking_eval = humanoid_manager.evaluate_performance("eval_test_robot", "walking")
    manipulation_eval = humanoid_manager.evaluate_performance("eval_test_robot", "manipulation")
    balance_eval = humanoid_manager.evaluate_performance("eval_test_robot", "balance")

    print(f"Walking evaluation success: {walking_eval['success']}")
    print(f"Manipulation evaluation success: {manipulation_eval['success']}")
    print(f"Balance evaluation success: {balance_eval['success']}")

    assert walking_eval["success"], "Walking evaluation should succeed"
    assert manipulation_eval["success"], "Manipulation evaluation should succeed"
    assert balance_eval["success"], "Balance evaluation should succeed"

    # Check that metrics are present
    assert "metrics" in walking_eval, "Walking evaluation should have metrics"
    assert "metrics" in manipulation_eval, "Manipulation evaluation should have metrics"
    assert "metrics" in balance_eval, "Balance evaluation should have metrics"

    print("PASS: Performance Evaluation test completed\n")


def test_control_modes():
    """Test control mode functionality"""
    print("Testing Control Modes...")

    # Create a robot
    robot = humanoid_manager.create_platform(HumanoidPlatform.NAO, "mode_test_robot")

    # Test different control modes
    robot.set_control_mode("walking")
    assert robot.control_mode == "walking", "Control mode should be walking"
    print(f"Set to walking mode: {robot.control_mode}")

    robot.set_control_mode("standing")
    assert robot.control_mode == "standing", "Control mode should be standing"
    print(f"Set to standing mode: {robot.control_mode}")

    robot.set_control_mode("manipulation")
    assert robot.control_mode == "manipulation", "Control mode should be manipulation"
    print(f"Set to manipulation mode: {robot.control_mode}")

    robot.set_control_mode("idle")
    assert robot.control_mode == "idle", "Control mode should be idle"
    print(f"Set to idle mode: {robot.control_mode}")

    print("PASS: Control Modes test completed\n")


def run_all_tests():
    """Run all humanoid architecture utility tests"""
    print("Starting Humanoid Architecture Utilities Tests\n")

    test_balance_controller()
    test_whole_body_controller()
    test_humanoid_controller()
    test_humanoid_platform_manager()
    test_performance_evaluation()
    test_control_modes()

    print("All humanoid architecture utility tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()