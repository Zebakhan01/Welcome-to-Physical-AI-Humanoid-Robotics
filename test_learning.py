#!/usr/bin/env python3
"""
Test script for learning functionality
"""
import numpy as np
from backend.utils.learning_utils import (
    DQNAgent, PolicyGradientAgent, ActorCriticAgent, BehavioralCloning,
    RobotState, RobotAction, Demonstration, RewardFunction, LearningAnalyzer
)

def test_dqn_agent():
    """Test DQN agent functionality"""
    print("Testing DQN Agent...")

    # Create DQN agent
    agent = DQNAgent(state_dim=10, action_dim=4, lr=0.001)

    # Test action selection
    test_state = np.random.random(10)
    action = agent.act(test_state)
    print(f"Selected action: {action} for state of shape {test_state.shape}")

    # Test experience replay
    for i in range(50):  # Add some experiences
        state = np.random.random(10)
        next_state = np.random.random(10)
        action = np.random.randint(0, 4)
        reward = np.random.random()
        done = i == 49
        agent.remember(state, action, reward, next_state, done)

    # Test training step
    agent.replay(batch_size=16)
    print("DQN training step completed")

    # Update target network
    agent.update_target_network()
    print("Target network updated")
    print("PASS: DQN Agent tests passed\n")


def test_policy_gradient_agent():
    """Test Policy Gradient agent functionality"""
    print("Testing Policy Gradient Agent...")

    # Create Policy Gradient agent
    agent = PolicyGradientAgent(state_dim=8, action_dim=3, lr=0.001)

    # Test action selection
    test_state = np.random.random(8)
    action = agent.act(test_state)
    print(f"Selected action: {action} for state of shape {test_state.shape}")

    # Test update with sample trajectory
    states = [np.random.random(8) for _ in range(10)]
    actions = [np.random.randint(0, 3) for _ in range(10)]
    rewards = [np.random.random() for _ in range(10)]

    agent.update(states, actions, rewards)
    print("Policy gradient update completed")
    print("PASS: Policy Gradient Agent tests passed\n")


def test_actor_critic_agent():
    """Test Actor-Critic agent functionality"""
    print("Testing Actor-Critic Agent...")

    # Create Actor-Critic agent
    agent = ActorCriticAgent(state_dim=6, action_dim=2, lr=0.001)

    # Test action selection
    test_state = np.random.random(6)
    action = agent.act(test_state)
    print(f"Selected action: {action} for state of shape {test_state.shape}")

    # Test update
    state = np.random.random(6)
    next_state = np.random.random(6)
    action = np.random.randint(0, 2)
    reward = np.random.random()
    done = False

    agent.update(state, action, reward, next_state, done)
    print("Actor-Critic update completed")
    print("PASS: Actor-Critic Agent tests passed\n")


def test_behavioral_cloning():
    """Test Behavioral Cloning functionality"""
    print("Testing Behavioral Cloning...")

    # Create Behavioral Cloning model
    bc = BehavioralCloning(state_dim=23, action_dim=5)

    # Create sample demonstrations
    demonstrations = []
    for demo_idx in range(3):
        states = []
        actions = []
        for step in range(5):
            # Create sample robot states
            state = RobotState(
                position=(np.random.random(), np.random.random(), np.random.random()),
                orientation=(0, 0, 0, 1),
                joint_angles=[np.random.random() for _ in range(3)],
                joint_velocities=[np.random.random() for _ in range(3)],
                sensor_readings=[np.random.random() for _ in range(4)],
                velocity=(0, 0, 0),
                angular_velocity=(0, 0, 0)
            )
            states.append(state)

            # Create sample robot actions
            action = RobotAction(
                joint_commands=[np.random.random() for _ in range(2)],
                velocity_commands=(np.random.random(), 0, 0),
                gripper_commands=[np.random.random()]
            )
            actions.append(action)

        demo = Demonstration(
            states=states,
            actions=actions,
            rewards=[np.random.random() for _ in range(5)]
        )
        demonstrations.append(demo)

    # Train the model
    bc.train(demonstrations, epochs=10)
    print("Behavioral cloning training completed")

    # Test prediction
    test_state = RobotState(
        position=(0.5, 0.5, 0.5),
        orientation=(0, 0, 0, 1),
        joint_angles=[0.1, 0.2, 0.3],
        joint_velocities=[0.0, 0.0, 0.0],
        sensor_readings=[0.5, 0.5, 0.5, 0.5],
        velocity=(0, 0, 0),
        angular_velocity=(0, 0, 0)
    )
    predicted_action = bc.predict(test_state)
    print(f"Predicted action: {predicted_action}")
    print("PASS: Behavioral Cloning tests passed\n")


def test_reward_functions():
    """Test reward functions"""
    print("Testing Reward Functions...")

    # Test reach target reward
    current_pos = (0.0, 0.0, 0.0)
    target_pos = (1.0, 1.0, 1.0)
    reward = RewardFunction.reach_target(current_pos, target_pos)
    print(f"Reach target reward: {reward}")

    # Test avoid obstacles reward
    obstacles = [(0.5, 0.5, 0.5), (2.0, 2.0, 2.0)]
    reward = RewardFunction.avoid_obstacles(current_pos, obstacles)
    print(f"Avoid obstacles reward: {reward}")

    # Test smooth control reward
    reward = RewardFunction.smooth_control(0.5)
    print(f"Smooth control reward: {reward}")
    print("PASS: Reward Functions tests passed\n")


def test_learning_analyzer():
    """Test learning analyzer"""
    print("Testing Learning Analyzer...")

    # Generate sample episode rewards
    episode_rewards = [np.random.normal(5, 2) for _ in range(50)]

    # Calculate convergence metrics
    analyzer = LearningAnalyzer()
    metrics = analyzer.calculate_convergence_metrics(episode_rewards, window_size=20)
    print(f"Convergence metrics: {metrics}")
    print("PASS: Learning Analyzer tests passed\n")


def run_all_tests():
    """Run all learning functionality tests"""
    print("Starting Learning Functionality Tests\n")

    test_dqn_agent()
    test_policy_gradient_agent()
    test_actor_critic_agent()
    test_behavioral_cloning()
    test_reward_functions()
    test_learning_analyzer()

    print("All learning functionality tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()