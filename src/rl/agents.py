"""RL agents for dispatch optimization.

Implements various RL agents:
- RandomAgent: Baseline random policy
- PPOAgent: Proximal Policy Optimization
- DQNAgent: Deep Q-Network (discretized actions)

Note: Full training requires stable-baselines3 or similar libraries.
This module provides a simplified implementation for demonstration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from src.rl.environment import CurtailmentEnv


@dataclass
class AgentConfig:
    """Configuration for RL agents.

    Attributes:
        learning_rate: Learning rate for gradient updates.
        gamma: Discount factor for future rewards.
        epsilon: Exploration rate (for epsilon-greedy).
        epsilon_decay: Rate of epsilon decay.
        epsilon_min: Minimum exploration rate.
        hidden_sizes: Hidden layer sizes for neural networks.
        buffer_size: Experience replay buffer size.
        batch_size: Training batch size.
        seed: Random seed.
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    buffer_size: int = 10000
    batch_size: int = 64
    seed: int = 42


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, env: CurtailmentEnv, config: AgentConfig | None = None) -> None:
        """Initialize the agent.

        Args:
            env: The environment to learn in.
            config: Agent configuration.
        """
        self.env = env
        self.config = config or AgentConfig()
        self._rng: Generator = np.random.default_rng(self.config.seed)
        self._training_step = 0

    @abstractmethod
    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select an action given an observation.

        Args:
            observation: Current environment observation.
            deterministic: If True, select the best action (no exploration).

        Returns:
            Action to take.
        """
        pass

    @abstractmethod
    def learn(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Update the agent from a single experience.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.

        Returns:
            Dictionary of training metrics.
        """
        pass

    def save(self, path: str) -> None:  # noqa: B027
        """Save agent to file.

        Args:
            path: Path to save to.
        """
        pass

    def load(self, path: str) -> None:  # noqa: B027
        """Load agent from file.

        Args:
            path: Path to load from.
        """
        pass


class RandomAgent(BaseAgent):
    """Random baseline agent.

    Selects actions uniformly at random. Useful for baseline comparisons.
    """

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False  # noqa: ARG002
    ) -> np.ndarray:
        """Select a random action.

        Args:
            observation: Current observation (ignored).
            deterministic: Ignored for random agent.

        Returns:
            Random action in action space.
        """
        # Random fractions that sum to <= 1
        sell_frac = self._rng.uniform(0, 1)
        store_frac = self._rng.uniform(0, 1 - sell_frac)
        return np.array([sell_frac, store_frac], dtype=np.float32)

    def learn(
        self,
        observation: np.ndarray,  # noqa: ARG002
        action: np.ndarray,  # noqa: ARG002
        reward: float,  # noqa: ARG002
        next_observation: np.ndarray,  # noqa: ARG002
        done: bool,  # noqa: ARG002
    ) -> dict[str, float]:
        """Random agent doesn't learn."""
        return {}


class HeuristicAgent(BaseAgent):
    """Heuristic-based agent using simple rules.

    Strategy:
    - High price (>$80): Sell as much as possible, discharge battery
    - Low/negative price (<$20): Store energy, don't sell
    - Medium price: Sell generation, store excess
    """

    def __init__(
        self,
        env: CurtailmentEnv,
        config: AgentConfig | None = None,
        high_price_threshold: float = 80.0,
        low_price_threshold: float = 20.0,
    ) -> None:
        """Initialize with price thresholds.

        Args:
            env: Environment.
            config: Agent configuration.
            high_price_threshold: Price above which to maximize selling.
            low_price_threshold: Price below which to maximize storage.
        """
        super().__init__(env, config)
        self.high_price_threshold = high_price_threshold
        self.low_price_threshold = low_price_threshold

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False  # noqa: ARG002
    ) -> np.ndarray:
        """Select action based on price heuristics.

        Args:
            observation: Current observation (contains price info).
            deterministic: Ignored (always deterministic).

        Returns:
            Action based on price-driven heuristics.
        """
        # Extract normalized price from observation
        # Observation: [soc, gen[0:lookahead], grid_cap, price, hour]
        price_idx = -2  # Second to last element
        normalized_price = observation[price_idx]

        # Denormalize price
        price_range = (
            self.env.config.max_price_per_mwh - self.env.config.min_price_per_mwh
        )
        price = normalized_price * price_range + self.env.config.min_price_per_mwh

        # Get SOC for storage decisions
        soc = observation[0]

        if price > self.high_price_threshold:
            # High price: sell everything, don't store
            return np.array([1.0, 0.0], dtype=np.float32)
        elif price < self.low_price_threshold:
            # Low/negative price: store if SOC allows, curtail rest
            if soc < 0.8:
                return np.array([0.0, 1.0], dtype=np.float32)
            else:
                return np.array([0.0, 0.0], dtype=np.float32)  # Curtail
        else:
            # Medium price: sell mostly, store some
            return np.array([0.8, 0.2], dtype=np.float32)

    def learn(
        self,
        observation: np.ndarray,  # noqa: ARG002
        action: np.ndarray,  # noqa: ARG002
        reward: float,  # noqa: ARG002
        next_observation: np.ndarray,  # noqa: ARG002
        done: bool,  # noqa: ARG002
    ) -> dict[str, float]:
        """Heuristic agent doesn't learn."""
        return {}


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent.

    Simplified PPO implementation for demonstration. For production use,
    consider stable-baselines3 or similar libraries.

    PPO uses a clipped surrogate objective to ensure stable policy updates:
        L^CLIP = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
    """

    def __init__(self, env: CurtailmentEnv, config: AgentConfig | None = None) -> None:
        """Initialize PPO agent.

        Args:
            env: Environment.
            config: Agent configuration.
        """
        super().__init__(env, config)

        # Get dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Initialize policy network weights (simple linear for demo)
        self._policy_weights = self._rng.normal(
            0, 0.1, size=(obs_dim, action_dim)
        ).astype(np.float32)
        self._policy_bias = np.zeros(action_dim, dtype=np.float32)

        # Value network weights
        self._value_weights = self._rng.normal(0, 0.1, size=(obs_dim, 1)).astype(
            np.float32
        )
        self._value_bias = np.zeros(1, dtype=np.float32)

        # Experience buffer for PPO updates
        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._values: list[float] = []

        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def _policy_forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through policy network.

        Args:
            obs: Observation.

        Returns:
            Tuple of (mean action, log std).
        """
        mean = np.tanh(obs @ self._policy_weights + self._policy_bias)
        # Squash to [0, 1] for action space
        mean = (mean + 1) / 2
        log_std = np.full_like(mean, -1.0)  # Fixed std for simplicity
        return mean, log_std

    def _value_forward(self, obs: np.ndarray) -> float:
        """Forward pass through value network.

        Args:
            obs: Observation.

        Returns:
            Estimated state value.
        """
        result = obs @ self._value_weights + self._value_bias
        # Handle both scalar and array results
        if isinstance(result, np.ndarray):
            return float(result.item() if result.size == 1 else result[0])
        return float(result)

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action using current policy.

        Args:
            observation: Current observation.
            deterministic: If True, return mean action.

        Returns:
            Action array.
        """
        mean, log_std = self._policy_forward(observation)

        if deterministic:
            action = mean
        else:
            std = np.exp(log_std)
            noise = self._rng.normal(0, 1, size=mean.shape)
            action = mean + std * noise

        # Clip to valid action range
        action = np.clip(action, 0.0, 1.0)

        # Ensure fractions sum to at most 1
        if action.sum() > 1.0:
            action = action / action.sum()

        return action.astype(np.float32)

    def learn(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,  # noqa: ARG002
        done: bool,
    ) -> dict[str, float]:
        """Store experience and update on episode end.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.

        Returns:
            Training metrics (empty unless episode ended).
        """
        # Store experience
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(self._value_forward(observation))

        self._training_step += 1

        # Update at episode end
        if done and len(self._observations) > 0:
            metrics = self._update()
            self._clear_buffer()
            return metrics

        return {}

    def _update(self) -> dict[str, float]:
        """Perform PPO update.

        Returns:
            Training metrics.
        """
        if len(self._observations) == 0:
            return {}

        # Compute returns and advantages
        returns = self._compute_returns()
        values = np.array(self._values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Simple gradient update (simplified for demo)
        obs_array = np.array(self._observations)
        actions_array = np.array(self._actions)

        # Policy gradient
        for i in range(len(self._observations)):
            obs = obs_array[i]
            action = actions_array[i]
            adv = advantages[i]

            mean, _ = self._policy_forward(obs)
            error = action - mean
            grad = np.outer(obs, error * adv)
            self._policy_weights += self.config.learning_rate * grad
            self._policy_bias += self.config.learning_rate * error * adv

        # Value function update
        for i in range(len(self._observations)):
            obs = obs_array[i]
            target = returns[i]
            pred = self._value_forward(obs)
            error = target - pred
            self._value_weights += (
                self.config.learning_rate * self.value_coef * error * obs.reshape(-1, 1)
            )
            self._value_bias += self.config.learning_rate * self.value_coef * error

        return {
            "policy_loss": float(np.mean(advantages**2)),
            "value_loss": float(np.mean((returns - values) ** 2)),
            "mean_return": float(np.mean(returns)),
        }

    def _compute_returns(self) -> np.ndarray:
        """Compute discounted returns.

        Returns:
            Array of discounted returns.
        """
        returns = np.zeros(len(self._rewards))
        running_return = 0.0

        for t in reversed(range(len(self._rewards))):
            if self._dones[t]:
                running_return = 0.0
            running_return = self._rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def _clear_buffer(self) -> None:
        """Clear experience buffer."""
        self._observations = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._values = []


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with discretized actions.

    Since the original action space is continuous, we discretize it into
    a fixed set of action combinations for DQN.

    Discretization:
        - 5 levels each for sell_fraction and store_fraction
        - Total: 5x5 = 25 discrete actions (filtered to valid ones)
    """

    def __init__(self, env: CurtailmentEnv, config: AgentConfig | None = None) -> None:
        """Initialize DQN agent.

        Args:
            env: Environment.
            config: Agent configuration.
        """
        super().__init__(env, config)

        # Create discrete action mapping
        self._create_action_mapping()

        # Get dimensions
        obs_dim = env.observation_space.shape[0]
        n_actions = len(self._action_mapping)

        # Q-network weights (simple linear for demo)
        self._q_weights = self._rng.normal(0, 0.1, size=(obs_dim, n_actions)).astype(
            np.float32
        )
        self._q_bias = np.zeros(n_actions, dtype=np.float32)

        # Target network (for stability)
        self._target_weights = self._q_weights.copy()
        self._target_bias = self._q_bias.copy()

        # Experience replay buffer
        self._replay_buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

        # Exploration
        self._epsilon = self.config.epsilon
        self._update_counter = 0
        self._target_update_freq = 100

    def _create_action_mapping(self) -> None:
        """Create mapping from discrete actions to continuous actions."""
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._action_mapping = []

        for sell in levels:
            for store in levels:
                if sell + store <= 1.0:
                    self._action_mapping.append(
                        np.array([sell, store], dtype=np.float32)
                    )

    def _q_forward(self, obs: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through Q-network.

        Args:
            obs: Observation.
            use_target: If True, use target network.

        Returns:
            Q-values for all discrete actions.
        """
        if use_target:
            return obs @ self._target_weights + self._target_bias
        return obs @ self._q_weights + self._q_bias

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action using epsilon-greedy policy.

        Args:
            observation: Current observation.
            deterministic: If True, always select best action.

        Returns:
            Action array.
        """
        if not deterministic and self._rng.random() < self._epsilon:
            # Random action
            action_idx = self._rng.integers(0, len(self._action_mapping))
        else:
            # Greedy action
            q_values = self._q_forward(observation)
            action_idx = int(np.argmax(q_values))

        return self._action_mapping[action_idx]

    def learn(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """Learn from experience using Q-learning.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.

        Returns:
            Training metrics.
        """
        # Find discrete action index
        action_idx = self._find_closest_action(action)

        # Store in replay buffer
        self._replay_buffer.append(
            (observation, action_idx, reward, next_observation, done)
        )

        # Limit buffer size
        if len(self._replay_buffer) > self.config.buffer_size:
            self._replay_buffer.pop(0)

        # Learn from batch
        if len(self._replay_buffer) >= self.config.batch_size:
            metrics = self._update_from_batch()
        else:
            metrics = {}

        # Decay epsilon
        self._epsilon = max(
            self.config.epsilon_min, self._epsilon * self.config.epsilon_decay
        )

        # Update target network periodically
        self._update_counter += 1
        if self._update_counter % self._target_update_freq == 0:
            self._target_weights = self._q_weights.copy()
            self._target_bias = self._q_bias.copy()

        return metrics

    def _find_closest_action(self, action: np.ndarray) -> int:
        """Find the closest discrete action index.

        Args:
            action: Continuous action.

        Returns:
            Index of closest discrete action.
        """
        min_dist = float("inf")
        best_idx = 0

        for i, discrete_action in enumerate(self._action_mapping):
            dist = np.sum((action - discrete_action) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        return best_idx

    def _update_from_batch(self) -> dict[str, float]:
        """Update Q-network from a batch of experiences.

        Returns:
            Training metrics.
        """
        # Sample batch
        indices = self._rng.choice(
            len(self._replay_buffer), size=self.config.batch_size, replace=False
        )

        total_loss = 0.0

        for idx in indices:
            obs, action_idx, reward, next_obs, done = self._replay_buffer[idx]

            # Compute target
            if done:
                target = reward
            else:
                next_q = self._q_forward(next_obs, use_target=True)
                target = reward + self.config.gamma * np.max(next_q)

            # Current Q-value
            current_q = self._q_forward(obs)
            current_value = current_q[action_idx]

            # TD error
            td_error = target - current_value
            total_loss += td_error**2

            # Update weights
            grad = np.zeros_like(self._q_weights[:, action_idx])
            grad = obs * td_error
            self._q_weights[:, action_idx] += self.config.learning_rate * grad
            self._q_bias[action_idx] += self.config.learning_rate * td_error

        return {
            "q_loss": total_loss / self.config.batch_size,
            "epsilon": self._epsilon,
            "buffer_size": len(self._replay_buffer),
        }
