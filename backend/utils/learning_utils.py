import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque
import pickle


class LearningType(Enum):
    """Types of learning approaches for robotics"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    SUPERVISED_LEARNING = "supervised_learning"
    UNSUPERVISED_LEARNING = "unsupervised_learning"


@dataclass
class RobotState:
    """Represents the state of a robot"""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    joint_angles: List[float]
    joint_velocities: List[float]
    sensor_readings: List[float]
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]


@dataclass
class RobotAction:
    """Represents an action that can be taken by a robot"""
    joint_commands: List[float]
    velocity_commands: Tuple[float, float, float]
    gripper_commands: List[float]  # for manipulation tasks


@dataclass
class Transition:
    """Represents a state-action-reward transition for RL"""
    state: RobotState
    action: RobotAction
    reward: float
    next_state: RobotState
    done: bool


@dataclass
class Demonstration:
    """Represents a human demonstration for imitation learning"""
    states: List[RobotState]
    actions: List[RobotAction]
    rewards: List[float]


class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: RobotState, action: RobotAction, reward: float,
             next_state: RobotState, done: bool):
        """Add a transition to the buffer"""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions"""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """Simple Q-Network using numpy (no PyTorch dependency)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        # Initialize weights randomly
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # ReLU activation
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        output = np.dot(h2, self.W3) + self.b3
        return output


class PolicyNetwork:
    """Simple Policy network using numpy (no PyTorch dependency)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        # Initialize weights randomly
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through the network - returns action probabilities"""
        # ReLU activation
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        logits = np.dot(h2, self.W3) + self.b3
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)


class ActorCriticNetwork:
    """Actor-Critic network combining policy and value functions using numpy"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        # Shared feature extractor
        self.W_shared = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b_shared = np.zeros(hidden_dim)

        # Actor (policy) network
        self.W_actor = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_actor = np.zeros(hidden_dim)
        self.W_actor_out = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b_actor_out = np.zeros(action_dim)

        # Critic (value) network
        self.W_critic = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_critic = np.zeros(hidden_dim)
        self.W_critic_out = np.random.randn(hidden_dim, 1) * 0.1
        self.b_critic_out = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass - returns action probabilities and value"""
        # Shared features
        h_shared = np.maximum(0, np.dot(state, self.W_shared) + self.b_shared)

        # Actor
        h_actor = np.maximum(0, np.dot(h_shared, self.W_actor) + self.b_actor)
        logits = np.dot(h_actor, self.W_actor_out) + self.b_actor_out
        action_probs = np.exp(logits - np.max(logits))
        action_probs = action_probs / np.sum(action_probs)

        # Critic
        h_critic = np.maximum(0, np.dot(h_shared, self.W_critic) + self.b_critic)
        value = np.dot(h_critic, self.W_critic_out) + self.b_critic_out

        return action_probs.flatten(), value.flatten()


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.update_target_network()  # Initialize target network

        self.memory = ReplayBuffer(10000)

    def act(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.push(
            RobotState(position=(0,0,0), orientation=(0,0,0,1), joint_angles=[],
                      joint_velocities=[], sensor_readings=state.tolist(),
                      velocity=(0,0,0), angular_velocity=(0,0,0)),
            RobotAction(joint_commands=[], velocity_commands=(0,0,0),
                       gripper_commands=[action]),
            reward,
            RobotState(position=(0,0,0), orientation=(0,0,0,1), joint_angles=[],
                      joint_velocities=[], sensor_readings=next_state.tolist(),
                      velocity=(0,0,0), angular_velocity=(0,0,0)),
            done
        )

    def replay(self, batch_size: int = 32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)

        # Extract batch data
        states = np.array([t.state.sensor_readings for t in batch])
        actions = np.array([t.action.gripper_commands[0] if t.action.gripper_commands else 0 for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state.sensor_readings for t in batch])
        dones = np.array([t.done for t in batch])

        # Compute target Q values
        next_q_values = np.array([self.target_network.forward(ns) for ns in next_states])
        max_next_q = np.max(next_q_values, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones.astype(int))

        # Update network using gradient descent (simplified)
        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            target = target_q[i]

            # Get current Q values
            current_q = self.q_network.forward(state)

            # Create target vector (only update the action that was taken)
            target_vec = current_q.copy()
            target_vec[action] = target

            # Update network weights (simple gradient update)
            self._update_weights(state, target_vec)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _update_weights(self, state: np.ndarray, target: np.ndarray):
        """Update network weights using a simple gradient descent step"""
        # Forward pass to get current output
        h1 = np.maximum(0, np.dot(state, self.q_network.W1) + self.q_network.b1)
        h2 = np.maximum(0, np.dot(h1, self.q_network.W2) + self.q_network.b2)
        output = np.dot(h2, self.q_network.W3) + self.q_network.b3

        # Compute error
        error = target - output

        # Backpropagate (simplified)
        d_output = -2 * error
        d_W3 = np.outer(h2, d_output)
        d_b3 = d_output
        d_h2 = np.dot(d_output, self.q_network.W3.T)
        d_h2[h2 <= 0] = 0  # ReLU derivative
        d_W2 = np.outer(h1, d_h2)
        d_b2 = d_h2
        d_h1 = np.dot(d_h2, self.q_network.W2.T)
        d_h1[h1 <= 0] = 0  # ReLU derivative
        d_W1 = np.outer(state, d_h1)
        d_b1 = d_h1

        # Update weights
        self.q_network.W3 -= self.lr * d_W3
        self.q_network.b3 -= self.lr * d_b3
        self.q_network.W2 -= self.lr * d_W2
        self.q_network.b2 -= self.lr * d_b2
        self.q_network.W1 -= self.lr * d_W1
        self.q_network.b1 -= self.lr * d_b1

    def update_target_network(self):
        """Update target network with current network weights"""
        # Copy weights from main network to target network
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
        self.target_network.W3 = self.q_network.W3.copy()
        self.target_network.b3 = self.q_network.b3.copy()


class PolicyGradientAgent:
    """Policy Gradient agent for reinforcement learning"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        self.policy_network = PolicyNetwork(state_dim, action_dim)

    def act(self, state: np.ndarray) -> int:
        """Select an action based on current policy"""
        action_probs = self.policy_network.forward(state)
        action = np.random.choice(len(action_probs), p=action_probs)
        return int(action)

    def update(self, states: List[np.ndarray], actions: List[int],
               rewards: List[float]):
        """Update policy using policy gradient"""
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Update policy for each state-action pair
        for i, (state, action, ret) in enumerate(zip(states, actions, returns)):
            # Get current policy for this state
            action_probs = self.policy_network.forward(state)

            # Compute gradient for this action (REINFORCE gradient)
            # Gradient is (log_prob_action) * advantage
            log_prob_action = np.log(action_probs[action] + 1e-8)  # Add small value to avoid log(0)

            # Update policy using gradient ascent
            # Increase probability of action if return was positive, decrease if negative
            gradient = ret  # Advantage is just the return in this simplified version

            # Simple approach: adjust the weights based on the advantage
            # This is a simplified version of the actual policy gradient update
            self._update_policy_weights(state, action, gradient)

    def _update_policy_weights(self, state: np.ndarray, action: int, advantage: float):
        """Update policy network weights based on advantage"""
        # Forward pass
        h1 = np.maximum(0, np.dot(state, self.policy_network.W1) + self.policy_network.b1)
        h2 = np.maximum(0, np.dot(h1, self.policy_network.W2) + self.policy_network.b2)
        logits = np.dot(h2, self.policy_network.W3) + self.policy_network.b3

        # Compute action probabilities
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        # Create target logits that encourage the selected action
        target_logits = logits.copy()
        target_logits[action] += advantage  # Increase logit for selected action based on advantage

        # Compute error
        error = target_logits - logits

        # Backpropagate (simplified)
        d_output = error
        d_W3 = np.outer(h2, d_output)
        d_b3 = d_output
        d_h2 = np.dot(d_output, self.policy_network.W3.T)
        d_h2[h2 <= 0] = 0  # ReLU derivative
        d_W2 = np.outer(h1, d_h2)
        d_b2 = d_h2
        d_h1 = np.dot(d_h2, self.policy_network.W2.T)
        d_h1[h1 <= 0] = 0  # ReLU derivative
        d_W1 = np.outer(state, d_h1)
        d_b1 = d_h1

        # Update weights
        self.policy_network.W3 += self.lr * d_W3
        self.policy_network.b3 += self.lr * d_b3
        self.policy_network.W2 += self.lr * d_W2
        self.policy_network.b2 += self.lr * d_b2
        self.policy_network.W1 += self.lr * d_W1
        self.policy_network.b1 += self.lr * d_b1


class ActorCriticAgent:
    """Actor-Critic agent for reinforcement learning"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        self.network = ActorCriticNetwork(state_dim, action_dim)

    def act(self, state: np.ndarray) -> int:
        """Select an action based on current policy"""
        action_probs, _ = self.network.forward(state)
        action = np.random.choice(len(action_probs), p=action_probs)
        return int(action)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Update actor and critic networks"""
        _, current_value = self.network.forward(state)
        if not done:
            _, next_value = self.network.forward(next_state)
            next_value_scalar = next_value[0]
        else:
            next_value_scalar = 0.0  # Value is 0 for terminal states

        # Compute advantage
        target = reward + self.gamma * next_value_scalar
        advantage = target - current_value[0]

        # Update networks (simplified)
        self._update_actor_critic(state, action, advantage, target)

    def _update_actor_critic(self, state: np.ndarray, action: int, advantage: float, target: float):
        """Update both actor and critic networks"""
        # Get current action probabilities and value
        action_probs, value = self.network.forward(state)

        # Update critic (value network)
        # Compute error for value prediction
        value_error = target - value[0]

        # Update actor (policy network) - encourage actions with positive advantage
        target_probs = action_probs.copy()
        target_probs[action] += advantage * 0.1  # Small learning rate for policy update

        # Normalize to maintain valid probability distribution
        target_probs = np.maximum(target_probs, 0)  # Ensure non-negative
        if np.sum(target_probs) > 0:
            target_probs = target_probs / np.sum(target_probs)
        else:
            target_probs = np.ones(len(target_probs)) / len(target_probs)  # Uniform if all zeros

        # Backpropagate through both networks (simplified)
        # Critic update
        h_shared = np.maximum(0, np.dot(state, self.network.W_shared) + self.network.b_shared)
        h_critic = np.maximum(0, np.dot(h_shared, self.network.W_critic) + self.network.b_critic)
        critic_output = np.dot(h_critic, self.network.W_critic_out) + self.network.b_critic_out

        d_critic_out = (critic_output[0] - target)  # Error for value prediction
        d_W_critic_out = np.outer(h_critic, [d_critic_out])
        d_b_critic_out = np.array([d_critic_out])  # Ensure it's a numpy array
        d_h_critic = np.dot([d_critic_out], self.network.W_critic_out.T)
        d_h_critic[h_critic <= 0] = 0  # ReLU derivative
        d_W_critic = np.outer(h_shared, d_h_critic)
        d_b_critic = d_h_critic
        d_h_shared_critic = np.dot(d_h_critic, self.network.W_critic.T)

        # Actor update
        h_actor = np.maximum(0, np.dot(h_shared, self.network.W_actor) + self.network.b_actor)
        actor_output = np.dot(h_actor, self.network.W_actor_out) + self.network.b_actor_out
        exp_logits = np.exp(actor_output - np.max(actor_output))
        current_probs = exp_logits / np.sum(exp_logits)

        # Compute error based on target probabilities
        prob_error = current_probs - target_probs
        d_W_actor_out = np.outer(h_actor, prob_error)
        d_b_actor_out = prob_error
        d_h_actor = np.dot(prob_error, self.network.W_actor_out.T)
        d_h_actor[h_actor <= 0] = 0  # ReLU derivative
        d_W_actor = np.outer(h_shared, d_h_actor)
        d_b_actor = d_h_actor
        d_h_shared_actor = np.dot(d_h_actor, self.network.W_actor.T)

        # Combine gradients for shared layer
        d_h_shared = d_h_shared_critic + d_h_shared_actor

        # Update all weights
        self.network.W_shared -= self.lr * (np.outer(state, d_h_shared))
        self.network.b_shared -= self.lr * d_h_shared
        self.network.W_critic -= self.lr * d_W_critic
        self.network.b_critic -= self.lr * d_b_critic
        self.network.W_critic_out -= self.lr * d_W_critic_out
        self.network.b_critic_out -= self.lr * d_b_critic_out.flatten()  # Ensure it matches the shape
        self.network.W_actor -= self.lr * d_W_actor
        self.network.b_actor -= self.lr * d_b_actor
        self.network.W_actor_out -= self.lr * d_W_actor_out
        self.network.b_actor_out -= self.lr * d_b_actor_out


class BehavioralCloning:
    """Behavioral cloning for imitation learning"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple neural network for behavioral cloning
        self.network = PolicyNetwork(state_dim, action_dim)

    def train(self, demonstrations: List[Demonstration], epochs: int = 100):
        """Train the behavioral cloning network using supervised learning"""
        # Flatten demonstrations into state-action pairs
        all_states = []
        all_actions = []

        for demo in demonstrations:
            for state, action in zip(demo.states, demo.actions):
                # Convert state to feature vector
                state_features = self._state_to_features(state)
                all_states.append(state_features)

                # Convert action to action vector and extract the first action dimension
                action_vector = self._action_to_vector(action)
                # For behavioral cloning, we want to learn a mapping to action probabilities
                # So we'll convert the action to a one-hot encoded format
                action_idx = action_vector[0] if action_vector else 0
                # Ensure action_idx is valid
                action_idx = max(0, min(int(action_idx), self.action_dim - 1))
                all_actions.append(action_idx)

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        # Train using supervised learning (imitation of expert actions)
        for epoch in range(epochs):
            for state, action in zip(all_states, all_actions):
                # Create target probabilities (one-hot for the expert action)
                target_probs = np.zeros(self.action_dim)
                target_probs[action] = 1.0

                # Get current predictions
                current_probs = self.network.forward(state)

                # Compute error
                error = target_probs - current_probs

                # Update network weights (simplified gradient update)
                self._update_network_weights(state, error)

    def _update_network_weights(self, state: np.ndarray, error: np.ndarray):
        """Update network weights based on error"""
        # Forward pass
        h1 = np.maximum(0, np.dot(state, self.network.W1) + self.network.b1)
        h2 = np.maximum(0, np.dot(h1, self.network.W2) + self.network.b2)
        logits = np.dot(h2, self.network.W3) + self.network.b3

        # Compute softmax gradient
        exp_logits = np.exp(logits - np.max(logits))
        current_probs = exp_logits / np.sum(exp_logits)

        # Compute gradient (derivative of cross-entropy loss)
        grad_output = current_probs - error  # Simplified gradient calculation

        # Backpropagate
        d_W3 = np.outer(h2, grad_output)
        d_b3 = grad_output
        d_h2 = np.dot(grad_output, self.network.W3.T)
        d_h2[h2 <= 0] = 0  # ReLU derivative
        d_W2 = np.outer(h1, d_h2)
        d_b2 = d_h2
        d_h1 = np.dot(d_h2, self.network.W2.T)
        d_h1[h1 <= 0] = 0  # ReLU derivative
        d_W1 = np.outer(state, d_h1)
        d_b1 = d_h1

        # Update weights
        self.network.W3 -= 0.01 * d_W3  # Learning rate for behavioral cloning
        self.network.b3 -= 0.01 * d_b3
        self.network.W2 -= 0.01 * d_W2
        self.network.b2 -= 0.01 * d_b2
        self.network.W1 -= 0.01 * d_W1
        self.network.b1 -= 0.01 * d_b1

    def predict(self, state: RobotState) -> RobotAction:
        """Predict action for given state"""
        state_features = self._state_to_features(state)

        action_probs = self.network.forward(state_features)

        # Sample action according to learned policy
        action = np.random.choice(len(action_probs), p=action_probs)

        # Convert action index back to RobotAction
        return self._index_to_action(action)

    def _state_to_features(self, state: RobotState) -> np.ndarray:
        """Convert RobotState to feature vector"""
        features = []
        features.extend(state.position)
        features.extend(state.orientation)
        features.extend(state.joint_angles)
        features.extend(state.joint_velocities)
        features.extend(state.sensor_readings)
        features.extend(state.velocity)
        features.extend(state.angular_velocity)
        return np.array(features)

    def _action_to_vector(self, action: RobotAction) -> List[float]:
        """Convert RobotAction to action vector"""
        vector = []
        vector.extend(action.joint_commands)
        vector.extend(action.velocity_commands)
        vector.extend(action.gripper_commands)
        return vector

    def _index_to_action(self, action_idx: int) -> RobotAction:
        """Convert action index back to RobotAction (simplified)"""
        # This is a simplified conversion - in practice, you'd need to know the dimensions
        # For now, we'll return a basic action with the action_idx as the first joint command
        return RobotAction(
            joint_commands=[float(action_idx)] + [0.0] * 6,  # 7 joint commands
            velocity_commands=(0.0, 0.0, 0.0),
            gripper_commands=[float(action_idx)]
        )


class RewardFunction:
    """Utility class for defining reward functions in robotics"""

    @staticmethod
    def reach_target(current_pos: Tuple[float, float, float],
                     target_pos: Tuple[float, float, float],
                     threshold: float = 0.1) -> float:
        """Reward function for reaching a target position"""
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)))
        if distance < threshold:
            return 10.0  # High reward for reaching target
        else:
            return -distance  # Negative reward proportional to distance

    @staticmethod
    def avoid_obstacles(current_pos: Tuple[float, float, float],
                        obstacles: List[Tuple[float, float, float]],
                        safe_distance: float = 0.2) -> float:
        """Reward function for avoiding obstacles"""
        for obs in obstacles:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, obs)))
            if distance < safe_distance:
                return -5.0  # Penalty for being too close to obstacle
        return 1.0  # Reward for staying safe

    @staticmethod
    def smooth_control(control_effort: float, max_effort: float = 1.0) -> float:
        """Reward function for smooth control (penalize large control efforts)"""
        normalized_effort = abs(control_effort) / max_effort
        return 1.0 - normalized_effort


class LearningEnvironment:
    """Base class for learning environments in robotics"""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = None
        self.episode_step = 0
        self.max_steps = 1000

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state"""
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        raise NotImplementedError

    def get_state_vector(self, state: RobotState) -> np.ndarray:
        """Convert RobotState to state vector for learning algorithms"""
        # Flatten state into a vector
        state_vec = []
        state_vec.extend(state.position)
        state_vec.extend(state.orientation)
        state_vec.extend(state.joint_angles)
        state_vec.extend(state.joint_velocities)
        state_vec.extend(state.sensor_readings)
        state_vec.extend(state.velocity)
        state_vec.extend(state.angular_velocity)
        return np.array(state_vec)


class LearningAnalyzer:
    """Analyzer for learning performance in robotics"""

    @staticmethod
    def calculate_convergence_metrics(episodes_rewards: List[float],
                                    window_size: int = 100) -> Dict[str, float]:
        """Calculate metrics to assess learning convergence"""
        if len(episodes_rewards) < window_size:
            return {"mean_reward": np.mean(episodes_rewards) if episodes_rewards else 0}

        recent_rewards = episodes_rewards[-window_size:]
        return {
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "trend": np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0] if len(recent_rewards) > 1 else 0,
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards)
        }

    @staticmethod
    def evaluate_policy(agent, env: LearningEnvironment,
                       num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the learned policy"""
        total_rewards = []

        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards)
        }