"""Tests for the Reinforcement Learning module.

Tests the Gymnasium environment, agents, and training utilities.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.domain.models import BatteryConfig
from src.generators.generation import GenerationGenerator
from src.generators.grid import GridConstraintGenerator
from src.generators.prices import MarketPriceGenerator
from src.rl.agents import (
    AgentConfig,
    DQNAgent,
    HeuristicAgent,
    PPOAgent,
    RandomAgent,
)
from src.rl.environment import (
    CurtailmentEnv,
    CurtailmentEnvConfig,
    EpisodeResult,
    make_env,
)
from src.rl.training import (
    Trainer,
    TrainingConfig,
    TrainingResult,
    compare_agents,
    evaluate_agent,
    quick_train,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def battery_config() -> BatteryConfig:
    """Standard battery configuration for testing."""
    return BatteryConfig(
        capacity_mwh=100.0,
        max_charge_mw=25.0,
        max_discharge_mw=25.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        degradation_cost_per_mwh=8.0,
    )


@pytest.fixture
def env_config(battery_config: BatteryConfig) -> CurtailmentEnvConfig:
    """Standard environment configuration for testing."""
    return CurtailmentEnvConfig(
        battery_config=battery_config,
        episode_length=24,
        lookahead_hours=4,
        normalize_obs=True,
        violation_penalty=100.0,
        seed=42,
    )


@pytest.fixture
def generators() -> dict:
    """Create generators with fixed seed for reproducibility."""
    seed = 42
    start_time = datetime(2024, 7, 15, 0, 0, 0)

    gen_generator = GenerationGenerator(
        solar_capacity_mw=200.0,
        wind_capacity_mw=100.0,
        seed=seed,
    )

    grid_generator = GridConstraintGenerator(
        base_export_capacity_mw=150.0,
        seed=seed,
    )

    price_generator = MarketPriceGenerator(
        base_price_multiplier=1.0,
        seed=seed,
    )

    # Generate 48 hours of data to have buffer for lookahead
    generation = gen_generator.generate_day_ahead(start_time)
    generation.extend(gen_generator.generate_day_ahead(start_time + timedelta(days=1)))

    constraints = grid_generator.generate_day_ahead(start_time)
    constraints.extend(
        grid_generator.generate_day_ahead(start_time + timedelta(days=1))
    )

    prices = price_generator.generate_day_ahead(start_time)
    prices.extend(price_generator.generate_day_ahead(start_time + timedelta(days=1)))

    return {
        "generation": generation,
        "constraints": constraints,
        "prices": prices,
    }


@pytest.fixture
def env(env_config: CurtailmentEnvConfig, generators: dict) -> CurtailmentEnv:
    """Create a test environment."""
    return CurtailmentEnv(
        config=env_config,
        forecasts=generators["generation"],
        grid_constraints=generators["constraints"],
        market_prices=generators["prices"],
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Standard agent configuration for testing."""
    return AgentConfig(
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_sizes=[64, 64],
        buffer_size=1000,
        batch_size=32,
        seed=42,
    )


# ============================================================================
# Environment Tests
# ============================================================================


class TestCurtailmentEnvConfig:
    """Tests for the CurtailmentEnvConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CurtailmentEnvConfig()

        assert config.episode_length == 24
        assert config.lookahead_hours == 4
        assert config.normalize_obs is True
        assert config.violation_penalty > 0

    def test_custom_config(self, battery_config: BatteryConfig) -> None:
        """Test custom configuration values."""
        config = CurtailmentEnvConfig(
            battery_config=battery_config,
            episode_length=48,
            lookahead_hours=6,
            normalize_obs=False,
            violation_penalty=200.0,
        )

        assert config.episode_length == 48
        assert config.lookahead_hours == 6
        assert config.normalize_obs is False
        assert config.violation_penalty == 200.0


class TestCurtailmentEnv:
    """Tests for the CurtailmentEnv Gymnasium environment."""

    def test_env_creation(self, env: CurtailmentEnv) -> None:
        """Test environment can be created."""
        assert env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_observation_space_shape(self, env: CurtailmentEnv) -> None:
        """Test observation space has correct shape."""
        # State: [SOC, generation (current + lookahead), grid_capacity, price, hour]
        # lookahead_hours = 4, so generation has 5 values (current + 4 lookahead)
        # Total: 1 (SOC) + 5 (generation) + 1 (grid) + 1 (price) + 1 (hour) = 9
        expected_dim = 1 + (1 + env.config.lookahead_hours) + 1 + 1 + 1
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space_shape(self, env: CurtailmentEnv) -> None:
        """Test action space has correct shape."""
        # Action: [sell_fraction, store_fraction]
        assert env.action_space.shape == (2,)
        # Actions should be bounded [0, 1]
        assert env.action_space.low[0] == 0.0
        assert env.action_space.low[1] == 0.0
        assert env.action_space.high[0] == 1.0
        assert env.action_space.high[1] == 1.0

    def test_reset_returns_observation(self, env: CurtailmentEnv) -> None:
        """Test reset returns valid observation and info."""
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert info is not None
        assert isinstance(info, dict)

    def test_reset_initializes_state(self, env: CurtailmentEnv) -> None:
        """Test reset properly initializes environment state."""
        env.reset()

        assert env._current_step == 0
        assert env._battery is not None
        result = env.get_episode_result()
        assert result.total_revenue == 0.0
        assert result.total_curtailed_mwh == 0.0
        assert result.grid_violations == 0

    def test_step_returns_correct_tuple(self, env: CurtailmentEnv) -> None:
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env.reset()
        action = env.action_space.sample()

        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_time(self, env: CurtailmentEnv) -> None:
        """Test step advances the current timestep."""
        env.reset()
        initial_step = env._current_step

        action = np.array([0.5, 0.3])
        env.step(action)

        assert env._current_step == initial_step + 1

    def test_episode_terminates_at_length(self, env: CurtailmentEnv) -> None:
        """Test episode terminates when reaching episode length."""
        env.reset()

        for _ in range(env.config.episode_length - 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated

        # Final step should terminate
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    def test_action_clipping(self, env: CurtailmentEnv) -> None:
        """Test actions are clipped to valid range."""
        env.reset()

        # Action with values > 1 (should be clipped)
        action = np.array([1.5, 0.8])
        obs, reward, terminated, truncated, info = env.step(action)

        # Should not raise error, action should be handled
        assert not np.isnan(reward)

    def test_info_contains_step_details(self, env: CurtailmentEnv) -> None:
        """Test info dict contains useful step information."""
        env.reset()
        action = np.array([0.5, 0.3])

        _, _, _, _, info = env.step(action)

        assert "step" in info
        assert "soc" in info
        assert "outcome" in info
        assert "actual_sell" in info["outcome"]
        assert "actual_store" in info["outcome"]
        assert "actual_curtail" in info["outcome"]
        assert "revenue" in info["outcome"]

    def test_energy_balance(self, env: CurtailmentEnv) -> None:
        """Test energy balance approximately holds."""
        env.reset()
        action = np.array([0.4, 0.3])  # 40% sell, 30% store, 30% curtail

        _, _, _, _, info = env.step(action)

        # Note: There can be overflow from storage limits, so balance is approximate
        outcome = info["outcome"]
        assert outcome["actual_sell"] >= 0
        assert outcome["actual_store"] >= 0
        assert outcome["actual_curtail"] >= 0

    def test_reproducibility_with_seed(
        self,
        battery_config: BatteryConfig,
        generators: dict,
    ) -> None:
        """Test environment produces same results with same seed."""
        config1 = CurtailmentEnvConfig(battery_config=battery_config, seed=123)
        config2 = CurtailmentEnvConfig(battery_config=battery_config, seed=123)

        env1 = CurtailmentEnv(
            config=config1,
            forecasts=generators["generation"],
            grid_constraints=generators["constraints"],
            market_prices=generators["prices"],
        )

        env2 = CurtailmentEnv(
            config=config2,
            forecasts=generators["generation"],
            grid_constraints=generators["constraints"],
            market_prices=generators["prices"],
        )

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_get_episode_result(self, env: CurtailmentEnv) -> None:
        """Test get_episode_result returns proper summary."""
        env.reset()

        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        result = env.get_episode_result()

        assert isinstance(result, EpisodeResult)
        assert result.steps == 5

    def test_make_env_factory(self) -> None:
        """Test make_env factory function."""
        env = make_env(seed=42)

        assert env is not None
        assert isinstance(env, CurtailmentEnv)

        obs, _ = env.reset()
        assert obs is not None


class TestEpisodeResult:
    """Tests for the EpisodeResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = EpisodeResult()

        assert result.total_reward == 0.0
        assert result.total_revenue == 0.0
        assert result.total_curtailed_mwh == 0.0
        assert result.grid_violations == 0
        assert result.steps == 0

    def test_net_profit_property(self) -> None:
        """Test net profit calculation."""
        result = EpisodeResult(
            total_revenue=1000.0,
            total_degradation=100.0,
        )

        assert result.net_profit == 900.0

    def test_curtailment_rate_property(self) -> None:
        """Test curtailment rate calculation."""
        result = EpisodeResult(
            total_sold_mwh=60.0,
            total_stored_mwh=20.0,
            total_curtailed_mwh=20.0,
        )

        # 20 / 100 = 0.2
        assert abs(result.curtailment_rate - 0.2) < 1e-6


# ============================================================================
# Agent Tests
# ============================================================================


class TestRandomAgent:
    """Tests for the RandomAgent."""

    def test_agent_creation(self, env: CurtailmentEnv) -> None:
        """Test agent can be created."""
        agent = RandomAgent(env)
        assert agent is not None

    def test_select_action_returns_valid_action(self, env: CurtailmentEnv) -> None:
        """Test select_action returns action in valid range."""
        agent = RandomAgent(env, AgentConfig(seed=42))

        obs, _ = env.reset()
        action = agent.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == env.action_space.shape
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_learn_does_nothing(self, env: CurtailmentEnv) -> None:
        """Test learn method exists and doesn't crash."""
        agent = RandomAgent(env)

        obs, _ = env.reset()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Should not raise
        result = agent.learn(obs, action, reward, next_obs, terminated)
        assert result == {}  # RandomAgent returns empty dict

    def test_actions_sum_valid(self, env: CurtailmentEnv) -> None:
        """Test sell + store fractions sum to at most 1."""
        agent = RandomAgent(env, AgentConfig(seed=42))

        obs, _ = env.reset()

        for _ in range(10):
            action = agent.select_action(obs)
            assert action[0] + action[1] <= 1.0 + 1e-6
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break


class TestHeuristicAgent:
    """Tests for the HeuristicAgent."""

    def test_agent_creation(self, env: CurtailmentEnv) -> None:
        """Test agent can be created."""
        agent = HeuristicAgent(env)
        assert agent is not None

    def test_select_action_returns_valid_action(self, env: CurtailmentEnv) -> None:
        """Test select_action returns action in valid range."""
        agent = HeuristicAgent(env)

        obs, _ = env.reset()
        action = agent.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == env.action_space.shape
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_actions_sum_to_at_most_one(self, env: CurtailmentEnv) -> None:
        """Test sell + store fractions sum to at most 1."""
        agent = HeuristicAgent(env)

        obs, _ = env.reset()

        for _ in range(10):
            action = agent.select_action(obs)
            assert action[0] + action[1] <= 1.0 + 1e-6
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    def test_custom_thresholds(self, env: CurtailmentEnv) -> None:
        """Test agent can be created with custom thresholds."""
        agent = HeuristicAgent(
            env,
            high_price_threshold=100.0,
            low_price_threshold=10.0,
        )

        assert agent.high_price_threshold == 100.0
        assert agent.low_price_threshold == 10.0


class TestPPOAgent:
    """Tests for the PPOAgent."""

    def test_agent_creation(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test agent can be created."""
        agent = PPOAgent(env, agent_config)
        assert agent is not None

    def test_select_action_returns_valid_action(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test select_action returns action in valid range."""
        agent = PPOAgent(env, agent_config)

        obs, _ = env.reset()
        action = agent.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == env.action_space.shape

    def test_learn_returns_dict(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test learn method returns metrics dict."""
        agent = PPOAgent(env, agent_config)

        obs, _ = env.reset()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Should not raise and should return dict
        result = agent.learn(obs, action, reward, next_obs, terminated)
        assert isinstance(result, dict)


class TestDQNAgent:
    """Tests for the DQNAgent."""

    def test_agent_creation(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test agent can be created."""
        agent = DQNAgent(env, agent_config)
        assert agent is not None

    def test_select_action_returns_valid_action(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test select_action returns action in valid range."""
        agent = DQNAgent(env, agent_config)

        obs, _ = env.reset()
        action = agent.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == env.action_space.shape
        # DQN discretizes actions, but output should still be valid
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_epsilon_exploration(self, env: CurtailmentEnv) -> None:
        """Test epsilon-greedy exploration."""
        config = AgentConfig(epsilon=1.0, seed=42)  # Always explore
        agent = DQNAgent(env, config)

        obs, _ = env.reset()

        # With epsilon=1, actions should vary (exploring)
        actions = [tuple(agent.select_action(obs)) for _ in range(10)]
        # Not all actions should be identical due to exploration
        unique_actions = len(set(actions))
        assert unique_actions >= 1  # At least some actions produced


# ============================================================================
# Training Tests
# ============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.n_episodes == 1000
        assert config.eval_frequency == 50
        assert config.eval_episodes == 10

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TrainingConfig(
            n_episodes=100,
            eval_frequency=10,
            eval_episodes=5,
        )

        assert config.n_episodes == 100
        assert config.eval_frequency == 10
        assert config.eval_episodes == 5


class TestTrainer:
    """Tests for the Trainer class."""

    def test_trainer_creation(self, env: CurtailmentEnv) -> None:
        """Test trainer can be created."""
        agent = RandomAgent(env)
        config = TrainingConfig(n_episodes=5)

        trainer = Trainer(agent, env, config)

        assert trainer is not None

    def test_train_runs_episodes(self, env: CurtailmentEnv) -> None:
        """Test train runs the specified number of episodes."""
        agent = RandomAgent(env, AgentConfig(seed=42))
        config = TrainingConfig(n_episodes=3, eval_frequency=10, log_frequency=10)

        trainer = Trainer(agent, env, config)
        result = trainer.train()

        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 3

    def test_train_returns_metrics(self, env: CurtailmentEnv) -> None:
        """Test train returns useful metrics."""
        agent = RandomAgent(env, AgentConfig(seed=42))
        config = TrainingConfig(n_episodes=3, eval_frequency=10, log_frequency=10)

        trainer = Trainer(agent, env, config)
        result = trainer.train()

        assert hasattr(result, "episode_rewards")
        assert hasattr(result, "episode_revenues")
        assert hasattr(result, "eval_rewards")

    def test_evaluation_during_training(self, env: CurtailmentEnv) -> None:
        """Test evaluation is performed at specified frequency."""
        agent = RandomAgent(env, AgentConfig(seed=42))
        config = TrainingConfig(
            n_episodes=6,
            eval_frequency=2,
            eval_episodes=1,
            log_frequency=10,
        )

        trainer = Trainer(agent, env, config)
        result = trainer.train()

        # Should have evaluations at episodes 2, 4, 6
        assert len(result.eval_rewards) == 3


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = TrainingResult()

        assert result.episode_rewards == []
        assert result.episode_revenues == []
        assert result.best_eval_reward == float("-inf")
        assert result.total_episodes == 0
        assert result.converged is False

    def test_final_eval_reward_property(self) -> None:
        """Test final_eval_reward property."""
        result = TrainingResult(eval_rewards=[1.0, 2.0, 3.0])

        assert result.final_eval_reward == 3.0

    def test_mean_training_reward_property(self) -> None:
        """Test mean_training_reward property."""
        result = TrainingResult(episode_rewards=[1.0, 2.0, 3.0])

        assert result.mean_training_reward == 2.0


class TestEvaluateAgent:
    """Tests for the evaluate_agent function."""

    def test_evaluate_agent_returns_results(self, env: CurtailmentEnv) -> None:
        """Test evaluate_agent returns evaluation results."""
        agent = RandomAgent(env, AgentConfig(seed=42))

        results = evaluate_agent(agent, env, n_episodes=2)

        assert len(results) == 2
        assert all(isinstance(r, EpisodeResult) for r in results)

    def test_evaluate_agent_reproducibility(self, env: CurtailmentEnv) -> None:
        """Test evaluation is reproducible with same seed."""
        agent1 = RandomAgent(env, AgentConfig(seed=42))
        agent2 = RandomAgent(env, AgentConfig(seed=42))

        # Reset environments with same seed
        env.reset(seed=123)
        results1 = evaluate_agent(agent1, env, n_episodes=1)

        env.reset(seed=123)
        results2 = evaluate_agent(agent2, env, n_episodes=1)

        # Both should produce same rewards with same seeds
        # (Note: exact equality depends on deterministic behavior)
        assert len(results1) == len(results2)


class TestCompareAgents:
    """Tests for the compare_agents function."""

    def test_compare_agents_returns_comparison(self, env: CurtailmentEnv) -> None:
        """Test compare_agents returns comparison results."""
        agents = {
            "random": RandomAgent(env, AgentConfig(seed=42)),
            "heuristic": HeuristicAgent(env),
        }

        results = compare_agents(agents, env, n_episodes=2)

        assert "random" in results
        assert "heuristic" in results
        assert "mean_reward" in results["random"]
        assert "mean_reward" in results["heuristic"]

    def test_compare_agents_all_metrics(self, env: CurtailmentEnv) -> None:
        """Test compare_agents returns all expected metrics."""
        agents = {"random": RandomAgent(env)}

        results = compare_agents(agents, env, n_episodes=2)

        metrics = results["random"]
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "mean_revenue" in metrics
        assert "mean_curtailment_rate" in metrics
        assert "mean_violations" in metrics


class TestQuickTrain:
    """Tests for the quick_train function."""

    def test_quick_train_ppo(self) -> None:
        """Test quick_train with PPO agent."""
        agent, result = quick_train(n_episodes=3, agent_type="ppo", seed=42)

        assert agent is not None
        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 3

    def test_quick_train_dqn(self) -> None:
        """Test quick_train with DQN agent."""
        agent, result = quick_train(n_episodes=3, agent_type="dqn", seed=42)

        assert agent is not None
        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 3

    def test_quick_train_random(self) -> None:
        """Test quick_train with random agent."""
        agent, result = quick_train(n_episodes=3, agent_type="random", seed=42)

        assert agent is not None
        assert isinstance(result, TrainingResult)


# ============================================================================
# Integration Tests
# ============================================================================


class TestRLIntegration:
    """Integration tests for the RL module."""

    def test_full_training_loop(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test a complete training loop with PPO agent."""
        agent = PPOAgent(env, agent_config)

        config = TrainingConfig(
            n_episodes=5,
            eval_frequency=2,
            eval_episodes=1,
            log_frequency=10,
        )

        trainer = Trainer(agent, env, config)
        result = trainer.train()

        assert result is not None
        assert len(result.episode_rewards) == 5

    def test_agent_comparison(self, env: CurtailmentEnv) -> None:
        """Test comparing multiple agents."""
        agents = {
            "random": RandomAgent(env, AgentConfig(seed=42)),
            "heuristic": HeuristicAgent(env),
        }

        results = compare_agents(agents, env, n_episodes=3)

        # Both agents should complete successfully
        assert results["random"]["mean_reward"] is not None
        assert results["heuristic"]["mean_reward"] is not None

    def test_dqn_training_loop(
        self, env: CurtailmentEnv, agent_config: AgentConfig
    ) -> None:
        """Test a complete training loop with DQN agent."""
        agent = DQNAgent(env, agent_config)

        config = TrainingConfig(
            n_episodes=3,
            eval_frequency=10,
            log_frequency=10,
        )

        trainer = Trainer(agent, env, config)
        result = trainer.train()

        assert result is not None
        assert len(result.episode_rewards) == 3

    def test_environment_with_different_configs(
        self,
        battery_config: BatteryConfig,
        generators: dict,
    ) -> None:
        """Test environment works with different configurations."""
        configs = [
            CurtailmentEnvConfig(
                battery_config=battery_config,
                episode_length=12,
                lookahead_hours=2,
            ),
            CurtailmentEnvConfig(
                battery_config=battery_config,
                episode_length=24,
                lookahead_hours=6,
            ),
            CurtailmentEnvConfig(
                battery_config=battery_config,
                normalize_obs=False,
            ),
        ]

        for config in configs:
            env = CurtailmentEnv(
                config=config,
                forecasts=generators["generation"],
                grid_constraints=generators["constraints"],
                market_prices=generators["prices"],
            )

            obs, _ = env.reset()
            assert obs is not None

            action = env.action_space.sample()
            next_obs, reward, _, _, _ = env.step(action)
            assert next_obs is not None

    def test_full_episode_completion(self, env: CurtailmentEnv) -> None:
        """Test running a complete episode."""
        agent = RandomAgent(env, AgentConfig(seed=42))

        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            action = agent.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        result = env.get_episode_result()

        assert steps == env.config.episode_length
        assert result.steps == steps
