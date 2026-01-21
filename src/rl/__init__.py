"""Reinforcement Learning module for dispatch optimization.

Phase 6: Implements RL-based decision making using Gymnasium:
- Custom environment for energy dispatch
- PPO and DQN agent training
- Policy evaluation and comparison with MILP baseline
"""

from src.rl.agents import (
    AgentConfig,
    BaseAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
)
from src.rl.environment import (
    CurtailmentEnv,
    CurtailmentEnvConfig,
    EpisodeResult,
)
from src.rl.training import (
    Trainer,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    "CurtailmentEnv",
    "CurtailmentEnvConfig",
    "EpisodeResult",
    "BaseAgent",
    "PPOAgent",
    "DQNAgent",
    "RandomAgent",
    "AgentConfig",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
]
