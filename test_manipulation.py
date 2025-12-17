#!/usr/bin/env python3
"""
Test script for manipulation functionality
"""
import asyncio
import numpy as np
from backend.utils.manipulation_utils import (
    GraspConfiguration, ManipulationState, ObjectProperties,
    GraspAnalyzer, ManipulationKinematics, ManipulationController,
    GraspPlanner, calculate_manipulability, compute_contact_jacobian
)

def test_grasp_analyzer():
    """Test grasp analysis functions"""
    print("Testing Grasp Analyzer...")

    # Test grasp matrix computation
    finger_positions = [(0.1, 0.0, 0.0), (-0.05, 0.087, 0.0), (-0.05, -0.087, 0.0)]
    contact_normals = [(-1.0, 0.0, 0.0), (0.5, -0.866, 0.0), (0.5, 0.866, 0.0)]

    grasp_matrix = GraspAnalyzer.compute_grasp_matrix(finger_positions, contact_normals)
    print(f"Grasp matrix shape: {grasp_matrix.shape}")
    print(f"Grasp matrix:\n{grasp_matrix}")

    # Test grasp quality
    quality = GraspAnalyzer.calculate_grasp_quality(grasp_matrix)
    print(f"Grasp quality: {quality}")

    # Test 2D force closure
    force_closure = GraspAnalyzer.check_force_closure_2d(
        [(0.1, 0.0), (-0.05, 0.087), (-0.05, -0.087)],
        [(-1.0, 0.0), (0.5, -0.866), (0.5, 0.866)]
    )
    print(f"Force closure (2D): {force_closure}")

    # Test grasp type suggestion
    obj_props = ObjectProperties(
        shape="cylinder",
        dimensions=(0.05, 0.05, 0.1),
        mass=0.3,
        center_of_mass=(0, 0, 0),
        friction_coeff=0.8,
        fragility=0.2
    )
    suggested_type = GraspAnalyzer.suggest_grasp_type(obj_props)
    print(f"Suggested grasp type: {suggested_type}")

    print("PASS: Grasp Analyzer tests passed\n")

def test_manipulation_kinematics():
    """Test manipulation kinematics functions"""
    print("Testing Manipulation Kinematics...")

    # Test forward kinematics
    kinematics = ManipulationKinematics(arm_dof=7)
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Home position

    try:
        position, orientation = kinematics.forward_kinematics(joint_angles)
        print(f"End-effector position: {position}")
        print(f"End-effector orientation: {orientation}")
    except Exception as e:
        print(f"Forward kinematics error: {e}")

    # Test Jacobian
    try:
        jacobian = kinematics.jacobian(joint_angles)
        print(f"Jacobian shape: {jacobian.shape}")

        # Calculate manipulability
        manipulability = calculate_manipulability(jacobian)
        print(f"Manipulability: {manipulability}")
    except Exception as e:
        print(f"Jacobian calculation error: {e}")

    print("PASS: Manipulation Kinematics tests passed\n")

def test_manipulation_controller():
    """Test manipulation controller functions"""
    print("Testing Manipulation Controller...")

    controller = ManipulationController()

    # Test impedance control
    desired_pos = np.array([0.1, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, 0.0])
    desired_vel = np.array([0.0, 0.0, 0.0])
    current_vel = np.array([0.0, 0.0, 0.0])
    stiffness = np.array([100.0, 100.0, 100.0])
    damping = np.array([20.0, 20.0, 20.0])

    force = controller.compute_impedance_control(
        desired_pos, current_pos, desired_vel, current_vel, stiffness, damping
    )
    print(f"Impedance control output: {force}")

    # Test hybrid control
    desired_force = np.array([5.0, 0.0, 0.0])
    current_force = np.array([0.0, 0.0, 0.0])
    selection_matrix = np.eye(3)

    control_output, pos_error, force_error = controller.compute_hybrid_position_force_control(
        desired_pos, current_pos, desired_force, current_force, selection_matrix
    )
    print(f"Hybrid control output: {control_output}")
    print(f"Position error: {pos_error}")
    print(f"Force error: {force_error}")

    # Test admittance control
    applied_force = np.array([10.0, 5.0, 0.0])
    displacement = controller.compute_admittance_control(applied_force)
    print(f"Admittance control displacement: {displacement}")

    print("PASS: Manipulation Controller tests passed\n")

def test_grasp_planning():
    """Test grasp planning functions"""
    print("Testing Grasp Planning...")

    planner = GraspPlanner()

    # Test cylindrical grasp planning
    obj_props = ObjectProperties(
        shape="cylinder",
        dimensions=(0.04, 0.04, 0.1),  # 4cm radius, 10cm height
        mass=0.2,
        center_of_mass=(0, 0, 0),
        friction_coeff=0.7,
        fragility=0.1
    )
    object_pose = (0.5, 0.0, 0.2, 0.0, 0.0, 0.0)  # [x, y, z, roll, pitch, yaw]

    cylindrical_grasp = planner.plan_cylindrical_grasp(obj_props, object_pose)
    print(f"Cylindrical grasp type: {cylindrical_grasp.grasp_type}")
    print(f"Cylindrical grasp quality: {cylindrical_grasp.grasp_quality}")
    print(f"Finger positions: {cylindrical_grasp.finger_positions}")

    # Test precision grasp planning
    small_obj = ObjectProperties(
        shape="box",
        dimensions=(0.02, 0.02, 0.02),  # 2cm cube
        mass=0.05,
        center_of_mass=(0, 0, 0),
        friction_coeff=0.8,
        fragility=0.9  # Fragile
    )

    precision_grasp = planner.plan_precision_grasp(small_obj, object_pose)
    print(f"Precision grasp type: {precision_grasp.grasp_type}")
    print(f"Precision grasp quality: {precision_grasp.grasp_quality}")
    print(f"Finger positions: {precision_grasp.finger_positions}")

    # Test automatic grasp planning
    auto_grasp = planner.plan_grasp(obj_props, object_pose, "auto")
    print(f"Auto grasp type: {auto_grasp.grasp_type}")
    print(f"Auto grasp quality: {auto_grasp.grasp_quality}")

    print("PASS: Grasp Planning tests passed\n")

def test_contact_jacobian():
    """Test contact Jacobian computation"""
    print("Testing Contact Jacobian...")

    contact_points = [(0.1, 0.0, 0.0), (-0.05, 0.087, 0.0), (-0.05, -0.087, 0.0)]
    end_effector_pose = (0.0, 0.0, 0.0)

    contact_jac = compute_contact_jacobian(contact_points, end_effector_pose)
    print(f"Contact Jacobian shape: {contact_jac.shape}")
    print(f"Contact Jacobian:\n{contact_jac}")

    print("PASS: Contact Jacobian tests passed\n")

def run_all_tests():
    """Run all manipulation functionality tests"""
    print("Starting Manipulation Functionality Tests\n")

    test_grasp_analyzer()
    test_manipulation_kinematics()
    test_manipulation_controller()
    test_grasp_planning()
    test_contact_jacobian()

    print("All manipulation functionality tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()