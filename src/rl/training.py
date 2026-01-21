"""Training utilities for RL agents.

Provides training loops, evaluation, and comparison with MILP baseline.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.rl.agents import BaseAgent, RandomAgent
from src.rl.environment import CurtailmentEnv, CurtailmentEnvConfig, EpisodeResult


@dataclass
class TrainingConfig:
    """Configuration for RL training.

    Attributes:
        n_episodes: Number of training episodes.
        eval_frequency: Evaluate every N episodes.
        eval_episodes: Number of episodes for evaluation.
        log_frequency: Log metrics every N episodes.
        save_frequency: Save model every N episodes.
        early_stopping_patience: Stop if no improvement for N evals.
        early_stopping_threshold: Minimum improvement threshold.
        seed: Random seed.
    """

    n_episodes: int = 1000
    eval_frequency: int = 50
    eval_episodes: int = 10
    log_frequency: int = 10
    save_frequency: int = 100
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01
    seed: int = 42


@dataclass
class TrainingResult:
    """Results from training run.

    Attributes:
        episode_rewards: Reward per episode.
        episode_revenues: Revenue per episode.
        eval_rewards: Evaluation rewards (at eval points).
        eval_revenues: Evaluation revenues.
        best_eval_reward: Best evaluation reward achieved.
        total_episodes: Total episodes trained.
        converged: Whether training converged.
        training_metrics: Per-episode training metrics.
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_revenues: list[float] = field(default_factory=list)
    eval_rewards: list[float] = field(default_factory=list)
    eval_revenues: list[float] = field(default_factory=list)
    best_eval_reward: float = float("-inf")
    total_episodes: int = 0
    converged: bool = False
    training_metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def final_eval_reward(self) -> float:
        """Final evaluation reward."""
        return self.eval_rewards[-1] if self.eval_rewards else 0.0

    @property
    def mean_training_reward(self) -> float:
        """Mean training reward."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0


class Trainer:
    """Trainer for RL agents.

    Handles training loop, evaluation, logging, and early stopping.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: CurtailmentEnv,
        config: TrainingConfig | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            agent: Agent to train.
            env: Environment to train in.
            config: Training configuration.
            progress_callback: Callback(episode, total, metrics) for progress.
        """
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        self.progress_callback = progress_callback

        # Tracking
        self._result = TrainingResult()
        self._best_agent_state: dict | None = None
        self._no_improvement_count = 0

    def train(self) -> TrainingResult:
        """Run the training loop.

        Returns:
            TrainingResult with metrics and outcomes.
        """
        for episode in range(self.config.n_episodes):
            # Run one training episode
            episode_result = self._run_episode(training=True)

            # Record metrics
            self._result.episode_rewards.append(episode_result.total_reward)
            self._result.episode_revenues.append(episode_result.total_revenue)
            self._result.total_episodes = episode + 1

            # Logging
            if (episode + 1) % self.config.log_frequency == 0:
                self._log_progress(episode)

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_reward, eval_revenue = self._evaluate()
                self._result.eval_rewards.append(eval_reward)
                self._result.eval_revenues.append(eval_revenue)

                # Early stopping check
                if (
                    eval_reward
                    > self._result.best_eval_reward
                    + self.config.early_stopping_threshold
                ):
                    self._result.best_eval_reward = eval_reward
                    self._no_improvement_count = 0
                else:
                    self._no_improvement_count += 1

                if self._no_improvement_count >= self.config.early_stopping_patience:
                    self._result.converged = True
                    break

            # Progress callback
            if self.progress_callback:
                metrics = {
                    "reward": episode_result.total_reward,
                    "revenue": episode_result.total_revenue,
                    "curtailed": episode_result.total_curtailed_mwh,
                }
                self.progress_callback(episode + 1, self.config.n_episodes, metrics)

        return self._result

    def _run_episode(self, training: bool = True) -> EpisodeResult:
        """Run one episode.

        Args:
            training: If True, agent learns from experience.

        Returns:
            Episode result.
        """
        obs, info = self.env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            action = self.agent.select_action(obs, deterministic=not training)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Learn (if training)
            if training:
                metrics = self.agent.learn(
                    obs, action, reward, next_obs, done or truncated
                )
                if metrics:
                    self._result.training_metrics.append(metrics)

            obs = next_obs

        return self.env.get_episode_result()

    def _evaluate(self) -> tuple[float, float]:
        """Evaluate the agent over multiple episodes.

        Returns:
            Tuple of (mean reward, mean revenue).
        """
        rewards = []
        revenues = []

        for _ in range(self.config.eval_episodes):
            result = self._run_episode(training=False)
            rewards.append(result.total_reward)
            revenues.append(result.total_revenue)

        return float(np.mean(rewards)), float(np.mean(revenues))

    def _log_progress(self, episode: int) -> None:
        """Log training progress.

        Args:
            episode: Current episode number.
        """
        recent_rewards = self._result.episode_rewards[-self.config.log_frequency :]
        recent_revenues = self._result.episode_revenues[-self.config.log_frequency :]

        print(
            f"Episode {episode + 1}/{self.config.n_episodes} | "
            f"Reward: {np.mean(recent_rewards):.2f} | "
            f"Revenue: ${np.mean(recent_revenues):,.0f}"
        )


def evaluate_agent(
    agent: BaseAgent,
    env: CurtailmentEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    """Evaluate an agent over multiple episodes.

    Args:
        agent: Agent to evaluate.
        env: Environment to evaluate in.
        n_episodes: Number of evaluation episodes.
        deterministic: If True, use deterministic policy.

    Returns:
        List of episode results.
    """
    results = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, deterministic=deterministic)
            obs, _, done, truncated, _ = env.step(action)

        results.append(env.get_episode_result())

    return results


def compare_agents(
    agents: dict[str, BaseAgent],
    env: CurtailmentEnv,
    n_episodes: int = 20,
) -> dict[str, dict[str, float]]:
    """Compare multiple agents on the same environment.

    Args:
        agents: Dictionary mapping agent names to agents.
        env: Environment to evaluate in.
        n_episodes: Number of evaluation episodes per agent.

    Returns:
        Dictionary mapping agent names to metrics.
    """
    results = {}

    for name, agent in agents.items():
        episode_results = evaluate_agent(agent, env, n_episodes)

        rewards = [r.total_reward for r in episode_results]
        revenues = [r.total_revenue for r in episode_results]
        curtailed = [r.curtailment_rate for r in episode_results]
        violations = [r.grid_violations + r.soc_violations for r in episode_results]

        results[name] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_revenue": float(np.mean(revenues)),
            "std_revenue": float(np.std(revenues)),
            "mean_curtailment_rate": float(np.mean(curtailed)),
            "mean_violations": float(np.mean(violations)),
        }

    return results


def quick_train(
    env_config: CurtailmentEnvConfig | None = None,
    n_episodes: int = 100,
    agent_type: str = "ppo",
    seed: int = 42,
) -> tuple[BaseAgent, TrainingResult]:
    """Quick training convenience function.

    Args:
        env_config: Environment configuration.
        n_episodes: Number of training episodes.
        agent_type: Type of agent ('ppo', 'dqn', 'random').
        seed: Random seed.

    Returns:
        Tuple of (trained agent, training result).
    """
    from src.rl.agents import AgentConfig, DQNAgent, PPOAgent

    # Create environment
    if env_config is None:
        env_config = CurtailmentEnvConfig(seed=seed)
    env = CurtailmentEnv(config=env_config)

    # Create agent
    agent_config = AgentConfig(seed=seed)

    if agent_type.lower() == "ppo":
        agent = PPOAgent(env, agent_config)
    elif agent_type.lower() == "dqn":
        agent = DQNAgent(env, agent_config)
    else:
        agent = RandomAgent(env, agent_config)

    # Train
    training_config = TrainingConfig(
        n_episodes=n_episodes,
        eval_frequency=max(1, n_episodes // 10),
        log_frequency=max(1, n_episodes // 20),
        seed=seed,
    )

    trainer = Trainer(agent, env, training_config)
    result = trainer.train()

    return agent, result
