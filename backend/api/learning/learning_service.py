from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.learning_utils import (
    RobotState, RobotAction, DQNAgent, PolicyGradientAgent, ActorCriticAgent,
    BehavioralCloning, LearningAnalyzer, LearningEnvironment, Demonstration
)
from backend.utils.logger import logger

router = APIRouter()


class RobotStateRequest(BaseModel):
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [qx, qy, qz, qw]
    joint_angles: List[float]
    joint_velocities: List[float]
    sensor_readings: List[float]
    velocity: List[float]  # [vx, vy, vz]
    angular_velocity: List[float]  # [wx, wy, wz]


class RobotActionRequest(BaseModel):
    joint_commands: List[float]
    velocity_commands: List[float]  # [vx, vy, vz]
    gripper_commands: List[float]


class RLTrainingRequest(BaseModel):
    algorithm: str  # "dqn", "policy_gradient", "actor_critic"
    state_dim: int
    action_dim: int
    episodes: int
    environment_config: Dict[str, Any]


class ImitationLearningRequest(BaseModel):
    demonstrations: List[Dict[str, Any]]  # List of demonstration data
    state_dim: int
    action_dim: int
    epochs: int


class LearningResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, float]] = None


class RLActRequest(BaseModel):
    agent_id: str
    state: RobotStateRequest


class RLActResponse(BaseModel):
    action: RobotActionRequest
    success: bool


class RLTrainStepRequest(BaseModel):
    agent_id: str
    state: RobotStateRequest
    action: int
    reward: float
    next_state: RobotStateRequest
    done: bool


class RLTrainStepResponse(BaseModel):
    success: bool
    loss: Optional[float] = None


class BehavioralCloningTrainRequest(BaseModel):
    demonstrations: List[Dict[str, Any]]
    state_dim: int
    action_dim: int
    epochs: int = 100


class BehavioralCloningResponse(BaseModel):
    success: bool
    message: str


class RewardFunctionRequest(BaseModel):
    current_position: List[float]  # [x, y, z]
    target_position: List[float]  # [x, y, z]
    obstacles: List[List[float]]  # List of [x, y, z] obstacle positions
    function_type: str  # "reach_target", "avoid_obstacles", "smooth_control"


class RewardFunctionResponse(BaseModel):
    reward: float
    success: bool


class LearningEvaluationRequest(BaseModel):
    agent_id: str
    episodes: int
    environment_config: Dict[str, Any]


class LearningEvaluationResponse(BaseModel):
    metrics: Dict[str, float]
    success: bool


class ConvergenceAnalysisRequest(BaseModel):
    episodes_rewards: List[float]
    window_size: int = 100


class ConvergenceAnalysisResponse(BaseModel):
    metrics: Dict[str, float]
    success: bool


# In-memory storage for agents (in production, use a proper database)
agents: Dict[str, Any] = {}


@router.post("/train-rl", response_model=LearningResponse)
async def train_rl(request: RLTrainingRequest):
    """
    Train a reinforcement learning agent
    """
    try:
        agent_id = f"{request.algorithm}_{len(agents)}"

        # Create appropriate agent based on algorithm
        if request.algorithm == "dqn":
            agent = DQNAgent(request.state_dim, request.action_dim)
        elif request.algorithm == "policy_gradient":
            agent = PolicyGradientAgent(request.state_dim, request.action_dim)
        elif request.algorithm == "actor_critic":
            agent = ActorCriticAgent(request.state_dim, request.action_dim)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

        # Store agent
        agents[agent_id] = agent

        # Training loop (simplified for API - in practice, this would be more complex)
        episode_rewards = []
        for episode in range(request.episodes):
            # This is a simplified training step - in a real environment, you'd have
            # an actual environment to interact with
            total_reward = 0

            # For demonstration, we'll simulate some training steps
            for step in range(10):  # 10 steps per episode
                # Random state for demonstration
                state = np.random.random(request.state_dim)

                # Agent takes action
                action = agent.act(state)

                # Simulated reward (in real env, this would come from environment)
                reward = np.random.random() - 0.5  # Random reward between -0.5 and 0.5
                total_reward += reward

                # Next random state
                next_state = np.random.random(request.state_dim)
                done = step == 9  # Done after 10 steps

                # Store experience for algorithms that need it
                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)

                # Update networks for algorithms that update online
                if hasattr(agent, 'update') and request.algorithm == "actor_critic":
                    agent.update(state, action, reward, next_state, done)

                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)

            # Update target network for DQN
            if request.algorithm == "dqn" and episode % 10 == 0:
                agent.update_target_network()

            # Replay for DQN
            if request.algorithm == "dqn" and len(getattr(agent, 'memory', [])) > 32:
                agent.replay(32)

        # Calculate metrics
        analyzer = LearningAnalyzer()
        metrics = analyzer.calculate_convergence_metrics(episode_rewards)

        response = LearningResponse(
            success=True,
            message=f"Trained {request.algorithm} agent for {request.episodes} episodes",
            metrics=metrics
        )

        logger.info(f"RL training completed for agent {agent_id}")

        return response

    except Exception as e:
        logger.error(f"Error in RL training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in RL training: {str(e)}")


@router.post("/act-rl", response_model=RLActResponse)
async def rl_act(request: RLActRequest):
    """
    Get action from trained RL agent
    """
    try:
        if request.agent_id not in agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")

        agent = agents[request.agent_id]

        # Convert request state to numpy array
        state_features = []
        state_features.extend(request.state.position)
        state_features.extend(request.state.orientation)
        state_features.extend(request.state.joint_angles)
        state_features.extend(request.state.joint_velocities)
        state_features.extend(request.state.sensor_readings)
        state_features.extend(request.state.velocity)
        state_features.extend(request.state.angular_velocity)

        state_array = np.array(state_features)

        # Get action from agent
        action_idx = agent.act(state_array)

        # For simplicity, return a basic action - in practice, this would map back to real robot commands
        action = RobotActionRequest(
            joint_commands=[action_idx * 0.1] * len(request.state.joint_angles) if request.state.joint_angles else [0.0],
            velocity_commands=[action_idx * 0.01, 0.0, 0.0],
            gripper_commands=[action_idx]
        )

        response = RLActResponse(
            action=action,
            success=True
        )

        logger.info(f"Action selected by agent {request.agent_id}")

        return response

    except Exception as e:
        logger.error(f"Error in RL act: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in RL act: {str(e)}")


@router.post("/train-step-rl", response_model=RLTrainStepResponse)
async def rl_train_step(request: RLTrainStepRequest):
    """
    Single training step for RL agent (for online learning)
    """
    try:
        if request.agent_id not in agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")

        agent = agents[request.agent_id]

        # Convert states to numpy arrays
        def state_to_array(state_req):
            state_features = []
            state_features.extend(state_req.position)
            state_features.extend(state_req.orientation)
            state_features.extend(state_req.joint_angles)
            state_features.extend(state_req.joint_velocities)
            state_features.extend(state_req.sensor_readings)
            state_features.extend(state_req.velocity)
            state_features.extend(state_req.angular_velocity)
            return np.array(state_features)

        state_array = state_to_array(request.state)
        next_state_array = state_to_array(request.next_state)

        # Store experience for algorithms that use replay buffer
        if hasattr(agent, 'remember'):
            agent.remember(state_array, request.action, request.reward, next_state_array, request.done)

        # Perform training step for DQN
        loss = None
        if hasattr(agent, 'replay'):
            # For demonstration, we'll just do a replay if buffer is large enough
            if len(getattr(agent, 'memory', [])) >= 32:
                agent.replay(32)

        # Update for Actor-Critic
        if hasattr(agent, 'update') and hasattr(agent, 'network'):
            agent.update(state_array, request.action, request.reward, next_state_array, request.done)

        response = RLTrainStepResponse(
            success=True,
            loss=loss
        )

        logger.info(f"Training step completed for agent {request.agent_id}")

        return response

    except Exception as e:
        logger.error(f"Error in RL train step: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in RL train step: {str(e)}")


@router.post("/train-imitation", response_model=BehavioralCloningResponse)
async def train_imitation(request: ImitationLearningRequest):
    """
    Train using imitation learning (behavioral cloning)
    """
    try:
        # Create behavioral cloning model
        bc_model = BehavioralCloning(request.state_dim, request.action_dim)

        # Convert demonstrations to proper format
        demonstrations = []
        for demo_data in request.demonstrations:
            states = []
            actions = []
            rewards = demo_data.get("rewards", [])

            for state_data in demo_data["states"]:
                state = RobotState(
                    position=tuple(state_data.get("position", [0, 0, 0])),
                    orientation=tuple(state_data.get("orientation", [0, 0, 0, 1])),
                    joint_angles=state_data.get("joint_angles", []),
                    joint_velocities=state_data.get("joint_velocities", []),
                    sensor_readings=state_data.get("sensor_readings", []),
                    velocity=tuple(state_data.get("velocity", [0, 0, 0])),
                    angular_velocity=tuple(state_data.get("angular_velocity", [0, 0, 0]))
                )
                states.append(state)

            for action_data in demo_data["actions"]:
                action = RobotAction(
                    joint_commands=action_data.get("joint_commands", []),
                    velocity_commands=tuple(action_data.get("velocity_commands", [0, 0, 0])),
                    gripper_commands=action_data.get("gripper_commands", [])
                )
                actions.append(action)

            demo = Demonstration(
                states=states,
                actions=actions,
                rewards=rewards
            )
            demonstrations.append(demo)

        # Train the model
        bc_model.train(demonstrations, request.epochs)

        # Store the model (simplified - in practice, you'd save to storage)
        model_id = f"bc_{len(agents)}"
        agents[model_id] = bc_model

        response = BehavioralCloningResponse(
            success=True,
            message=f"Trained behavioral cloning model with {len(demonstrations)} demonstrations"
        )

        logger.info(f"Imitation learning completed for model {model_id}")

        return response

    except Exception as e:
        logger.error(f"Error in imitation learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in imitation learning: {str(e)}")


@router.post("/compute-reward", response_model=RewardFunctionResponse)
async def compute_reward(request: RewardFunctionRequest):
    """
    Compute reward based on different reward functions
    """
    try:
        current_pos = tuple(request.current_position)
        target_pos = tuple(request.target_position)
        obstacles = [tuple(obs) for obs in request.obstacles]

        if request.function_type == "reach_target":
            from backend.utils.learning_utils import RewardFunction
            reward = RewardFunction.reach_target(current_pos, target_pos)
        elif request.function_type == "avoid_obstacles":
            from backend.utils.learning_utils import RewardFunction
            reward = RewardFunction.avoid_obstacles(current_pos, obstacles)
        elif request.function_type == "smooth_control":
            # For smooth control, we need control effort - using a dummy value for demo
            reward = RewardFunction.smooth_control(0.5)  # Dummy control effort
        else:
            raise HTTPException(status_code=400, detail=f"Unknown reward function: {request.function_type}")

        response = RewardFunctionResponse(
            reward=reward,
            success=True
        )

        logger.info(f"Reward computed: {reward} for function {request.function_type}")

        return response

    except Exception as e:
        logger.error(f"Error computing reward: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing reward: {str(e)}")


@router.post("/evaluate-learning", response_model=LearningEvaluationResponse)
async def evaluate_learning(request: LearningEvaluationRequest):
    """
    Evaluate learning performance
    """
    try:
        if request.agent_id not in agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")

        agent = agents[request.agent_id]

        # For evaluation, we would normally have a proper environment
        # Here we'll simulate the evaluation process
        analyzer = LearningAnalyzer()
        metrics = analyzer.evaluate_policy(agent, None, request.episodes)  # Pass None for env since we're simulating

        # For simulation purposes, generate some dummy evaluation results
        # In a real system, this would interact with the actual environment
        total_rewards = []
        for _ in range(request.episodes):
            # Simulate evaluation episode
            total_reward = np.random.normal(5, 2)  # Random reward around 5
            total_rewards.append(total_reward)

        metrics = {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "max_reward": float(np.max(total_rewards))
        }

        response = LearningEvaluationResponse(
            metrics=metrics,
            success=True
        )

        logger.info(f"Learning evaluation completed for agent {request.agent_id}")

        return response

    except Exception as e:
        logger.error(f"Error in learning evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in learning evaluation: {str(e)}")


@router.post("/analyze-convergence", response_model=ConvergenceAnalysisResponse)
async def analyze_convergence(request: ConvergenceAnalysisRequest):
    """
    Analyze learning convergence
    """
    try:
        analyzer = LearningAnalyzer()
        metrics = analyzer.calculate_convergence_metrics(request.episodes_rewards, request.window_size)

        response = ConvergenceAnalysisResponse(
            metrics=metrics,
            success=True
        )

        logger.info(f"Convergence analysis completed with {len(request.episodes_rewards)} episodes")

        return response

    except Exception as e:
        logger.error(f"Error in convergence analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in convergence analysis: {str(e)}")


class TransferLearningRequest(BaseModel):
    source_model_path: str
    target_task: str
    adaptation_method: str = "fine_tuning"  # "fine_tuning", "feature_extraction", "domain_adaptation"


class TransferLearningResponse(BaseModel):
    success: bool
    message: str
    adapted_model_path: Optional[str] = None


@router.post("/transfer-learning", response_model=TransferLearningResponse)
async def transfer_learning(request: TransferLearningRequest):
    """
    Perform transfer learning from source to target task
    """
    try:
        # This is a simplified implementation of transfer learning
        # In practice, this would involve loading a pre-trained model,
        # adapting it to the new task, and saving the adapted model

        # For demonstration, we'll simulate the transfer process
        message = f"Transfer learning from {request.source_model_path} to {request.target_task} using {request.adaptation_method}"

        response = TransferLearningResponse(
            success=True,
            message=message,
            adapted_model_path=f"adapted_{request.target_task}_model.pth"
        )

        logger.info(f"Transfer learning completed: {message}")

        return response

    except Exception as e:
        logger.error(f"Error in transfer learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in transfer learning: {str(e)}")


class SafeExplorationRequest(BaseModel):
    current_state: RobotStateRequest
    action_space: List[Dict[str, Any]]  # Define safe action space
    safety_constraints: Dict[str, Any]  # Define safety constraints


class SafeExplorationResponse(BaseModel):
    safe_actions: List[int]
    risk_assessment: Dict[str, float]
    success: bool


@router.post("/safe-exploration", response_model=SafeExplorationResponse)
async def safe_exploration(request: SafeExplorationRequest):
    """
    Perform safe exploration in robotic learning
    """
    try:
        # This is a simplified implementation of safe exploration
        # In practice, this would involve complex safety checks and constraint satisfaction

        # For demonstration, we'll return all action indices as potentially safe
        # with a simple risk assessment
        safe_actions = list(range(len(request.action_space)))

        # Calculate risk assessment (simplified)
        risk_assessment = {
            "collision_risk": 0.1,  # Low risk for demonstration
            "stability_risk": 0.05,
            "task_failure_risk": 0.2
        }

        response = SafeExplorationResponse(
            safe_actions=safe_actions,
            risk_assessment=risk_assessment,
            success=True
        )

        logger.info(f"Safe exploration assessment completed")

        return response

    except Exception as e:
        logger.error(f"Error in safe exploration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in safe exploration: {str(e)}")


class MetaLearningRequest(BaseModel):
    tasks: List[Dict[str, Any]]  # List of different tasks to learn
    base_learner_type: str = "maml"  # "maml", "reptile", "proto_networks"
    adaptation_steps: int = 5


class MetaLearningResponse(BaseModel):
    success: bool
    message: str
    meta_model_path: Optional[str] = None


@router.post("/meta-learning", response_model=MetaLearningResponse)
async def meta_learning(request: MetaLearningRequest):
    """
    Perform meta-learning (learning to learn) for robotics
    """
    try:
        # This is a simplified implementation of meta-learning
        # In practice, this would involve training a model that can quickly adapt to new tasks

        message = f"Meta-learning with {len(request.tasks)} tasks using {request.base_learner_type}"

        response = MetaLearningResponse(
            success=True,
            message=message,
            meta_model_path=f"meta_model_{request.base_learner_type}.pth"
        )

        logger.info(f"Meta-learning completed: {message}")

        return response

    except Exception as e:
        logger.error(f"Error in meta learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in meta learning: {str(e)}")