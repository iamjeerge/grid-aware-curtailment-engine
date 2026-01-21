"""Gymnasium environment for grid-aware curtailment optimization.

Defines a custom Gymnasium environment where an RL agent learns to make
energy dispatch decisions (sell, store, curtail) to maximize profit while
respecting grid constraints and battery limits.

State Space:
    - SOC: Battery state of charge (normalized 0-1)
    - Generation forecast: Current and lookahead (normalized)
    - Grid capacity: Export limit (normalized)
    - Market price: Current price (normalized)
    - Hour of day: Time feature (normalized 0-1)

Action Space:
    - Continuous: [sell_fraction, store_fraction]
    - Curtail fraction = 1 - sell_fraction - store_fraction

Reward:
    - Revenue from selling energy
    - Minus degradation cost from battery cycling
    - Minus penalty for constraint violations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import Generator

from src.battery.physics import BatteryModel
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)


@dataclass
class CurtailmentEnvConfig:
    """Configuration for the curtailment environment.

    Attributes:
        battery_config: Battery configuration parameters.
        episode_length: Number of timesteps per episode (hours).
        lookahead_hours: Hours of forecast lookahead in state.
        reward_scale: Scale factor for rewards.
        violation_penalty: Penalty for constraint violations.
        normalize_obs: Whether to normalize observations.
        seed: Random seed for reproducibility.
    """

    battery_config: BatteryConfig = field(default_factory=BatteryConfig)
    episode_length: int = 24  # One day
    lookahead_hours: int = 4
    reward_scale: float = 0.001  # Scale down large dollar values
    violation_penalty: float = 100.0
    normalize_obs: bool = True
    seed: int = 42

    # Normalization bounds (CAISO-style values)
    max_generation_mw: float = 1000.0
    max_grid_capacity_mw: float = 500.0
    max_price_per_mwh: float = 200.0
    min_price_per_mwh: float = -50.0


@dataclass
class EpisodeResult:
    """Results from running one episode."""

    total_reward: float = 0.0
    total_revenue: float = 0.0
    total_degradation: float = 0.0
    total_curtailed_mwh: float = 0.0
    total_sold_mwh: float = 0.0
    total_stored_mwh: float = 0.0
    total_discharged_mwh: float = 0.0
    grid_violations: int = 0
    soc_violations: int = 0
    steps: int = 0
    final_soc: float = 0.0

    @property
    def net_profit(self) -> float:
        """Net profit after degradation."""
        return self.total_revenue - self.total_degradation

    @property
    def curtailment_rate(self) -> float:
        """Fraction of available energy curtailed."""
        total = self.total_sold_mwh + self.total_stored_mwh + self.total_curtailed_mwh
        return self.total_curtailed_mwh / max(total, 1e-6)


class CurtailmentEnv(gym.Env):
    """Gymnasium environment for energy dispatch optimization.

    The agent observes the current state (SOC, generation, price, grid capacity)
    and chooses how to dispatch available energy among selling, storing, and
    curtailing.

    This environment supports both continuous and discrete action spaces
    depending on configuration.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: CurtailmentEnvConfig | None = None,
        forecasts: list[GenerationForecast] | None = None,
        grid_constraints: list[GridConstraint] | None = None,
        market_prices: list[MarketPrice] | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            config: Environment configuration.
            forecasts: Pre-generated forecasts (or None to generate).
            grid_constraints: Pre-generated constraints (or None to generate).
            market_prices: Pre-generated prices (or None to generate).
            render_mode: Rendering mode ('human' or 'ansi').
        """
        super().__init__()

        self.config = config or CurtailmentEnvConfig()
        self.render_mode = render_mode

        # Random generator for scenario generation
        self._rng: Generator = np.random.default_rng(self.config.seed)

        # Episode data (can be provided or generated)
        self._forecasts = forecasts
        self._grid_constraints = grid_constraints
        self._market_prices = market_prices

        # State tracking
        self._battery: BatteryModel | None = None
        self._current_step = 0
        self._episode_result = EpisodeResult()

        # Define observation space
        # [SOC, generation (current + lookahead), grid_capacity, price, hour]
        obs_dim = 1 + (1 + self.config.lookahead_hours) + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Define action space: [sell_fraction, store_fraction]
        # Curtail = 1 - sell - store (implicitly)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for this episode.
            options: Additional options (e.g., 'initial_soc').

        Returns:
            Tuple of (initial observation, info dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Parse options
        options = options or {}
        initial_soc = options.get("initial_soc", 0.5)

        # Generate or use provided scenario data
        if self._forecasts is None or options.get("regenerate", False):
            self._generate_scenario()

        # Initialize battery
        self._battery = BatteryModel(self.config.battery_config)
        start_time = self._forecasts[0].timestamp
        self._battery.initialize(start_time, initial_soc)

        # Reset tracking
        self._current_step = 0
        self._episode_result = EpisodeResult()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Array [sell_fraction, store_fraction].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._battery is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0)
        sell_frac, store_frac = action[0], action[1]

        # Ensure fractions sum to at most 1
        total_frac = sell_frac + store_frac
        if total_frac > 1.0:
            sell_frac /= total_frac
            store_frac /= total_frac

        curtail_frac = 1.0 - sell_frac - store_frac

        # Get current state
        t = self._current_step
        forecast = self._forecasts[t]
        constraint = self._grid_constraints[t]
        price = self._market_prices[t]

        # Calculate available energy
        generation = forecast.total_generation(ForecastScenario.P50)
        grid_capacity = constraint.max_export_mw
        current_price = price.effective_price

        # Apply action fractions
        energy_to_sell = generation * sell_frac
        energy_to_store = generation * store_frac
        energy_to_curtail = generation * curtail_frac

        # Check and apply battery discharge (sell from battery if profitable)
        discharge_energy = 0.0
        if current_price > self.config.battery_config.degradation_cost_per_mwh * 2:
            # Worth discharging
            max_discharge = self._battery.get_max_discharge_power()
            discharge_headroom = max(0, grid_capacity - energy_to_sell)
            discharge_energy = min(max_discharge, discharge_headroom)

        # Enforce grid constraint
        actual_sell = min(energy_to_sell + discharge_energy, grid_capacity)
        grid_violation = (energy_to_sell + discharge_energy) > grid_capacity * 1.01

        # Enforce battery charge constraint
        max_charge = self._battery.get_max_charge_power()
        actual_store = min(energy_to_store, max_charge)
        store_overflow = energy_to_store - actual_store

        # Add overflow to curtailment
        actual_curtail = energy_to_curtail + store_overflow

        # Update battery
        timestamp = forecast.timestamp
        if actual_store > 0:
            self._battery.charge(actual_store, 1.0, timestamp)

        actual_discharge = 0.0
        if discharge_energy > 0:
            _, actual_discharge = self._battery.discharge(
                discharge_energy, 1.0, timestamp
            )

        # Calculate reward components
        revenue = actual_sell * current_price
        degradation_cost = (
            actual_store + actual_discharge
        ) * self.config.battery_config.degradation_cost_per_mwh

        # Penalty for violations
        penalty = 0.0
        if grid_violation:
            penalty += self.config.violation_penalty
            self._episode_result.grid_violations += 1

        # SOC violation check
        new_state = self._battery.current_state
        soc_violation = (
            new_state.soc_fraction < self.config.battery_config.min_soc_fraction * 0.99
            or new_state.soc_fraction
            > self.config.battery_config.max_soc_fraction * 1.01
        )
        if soc_violation:
            penalty += self.config.violation_penalty * 0.5
            self._episode_result.soc_violations += 1

        # Compute reward
        reward = (revenue - degradation_cost - penalty) * self.config.reward_scale

        # Update episode tracking
        self._episode_result.total_reward += reward
        self._episode_result.total_revenue += revenue
        self._episode_result.total_degradation += degradation_cost
        self._episode_result.total_sold_mwh += actual_sell
        self._episode_result.total_stored_mwh += actual_store
        self._episode_result.total_discharged_mwh += actual_discharge
        self._episode_result.total_curtailed_mwh += actual_curtail
        self._episode_result.steps += 1

        # Advance timestep
        self._current_step += 1

        # Check termination
        terminated = self._current_step >= self.config.episode_length
        truncated = self._current_step >= len(self._forecasts)

        if terminated or truncated:
            self._episode_result.final_soc = new_state.soc_mwh

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info["action"] = {
            "sell_frac": sell_frac,
            "store_frac": store_frac,
            "curtail_frac": curtail_frac,
        }
        info["outcome"] = {
            "actual_sell": actual_sell,
            "actual_store": actual_store,
            "actual_curtail": actual_curtail,
            "discharge": actual_discharge,
            "revenue": revenue,
            "degradation": degradation_cost,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        t = self._current_step
        battery_state = self._battery.current_state

        # SOC (normalized 0-1)
        soc = battery_state.soc_fraction if battery_state else 0.5

        # Generation forecast (current + lookahead)
        gen_values = []
        for i in range(1 + self.config.lookahead_hours):
            idx = min(t + i, len(self._forecasts) - 1)
            gen = self._forecasts[idx].total_generation(ForecastScenario.P50)
            if self.config.normalize_obs:
                gen = gen / self.config.max_generation_mw
            gen_values.append(np.clip(gen, 0.0, 1.0))

        # Grid capacity
        grid_cap = self._grid_constraints[t].max_export_mw
        if self.config.normalize_obs:
            grid_cap = grid_cap / self.config.max_grid_capacity_mw
        grid_cap = np.clip(grid_cap, 0.0, 1.0)

        # Price (normalized to 0-1 range)
        price = self._market_prices[t].effective_price
        if self.config.normalize_obs:
            price_range = self.config.max_price_per_mwh - self.config.min_price_per_mwh
            price = (price - self.config.min_price_per_mwh) / price_range
        price = np.clip(price, 0.0, 1.0)

        # Hour of day (normalized 0-1)
        hour = self._forecasts[t].timestamp.hour / 24.0

        # Combine into observation
        obs = np.array([soc] + gen_values + [grid_cap, price, hour], dtype=np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Construct the info dictionary."""
        t = self._current_step
        battery_state = self._battery.current_state

        return {
            "step": t,
            "soc": battery_state.soc_mwh if battery_state else 0.0,
            "soc_fraction": battery_state.soc_fraction if battery_state else 0.0,
            "episode_result": self._episode_result,
            "timestamp": (
                self._forecasts[t].timestamp
                if t < len(self._forecasts)
                else self._forecasts[-1].timestamp
            ),
        }

    def _generate_scenario(self) -> None:
        """Generate synthetic scenario data for the episode."""
        from src.generators.generation import GenerationGenerator
        from src.generators.grid import GridConstraintGenerator
        from src.generators.prices import MarketPriceGenerator

        # Use seeded generators
        seed = self._rng.integers(0, 2**31)

        gen_generator = GenerationGenerator(seed=seed)
        grid_generator = GridConstraintGenerator(seed=seed)
        price_generator = MarketPriceGenerator(seed=seed)

        # Generate for multiple days to cover episode length + lookahead
        start_time = datetime(2024, 6, 15, 0, 0)  # Summer day

        # Generate 2 days to ensure we have enough data
        self._forecasts = gen_generator.generate_day_ahead(start_time)
        self._forecasts.extend(
            gen_generator.generate_day_ahead(start_time + timedelta(days=1))
        )

        self._grid_constraints = grid_generator.generate_day_ahead(start_time)
        self._grid_constraints.extend(
            grid_generator.generate_day_ahead(start_time + timedelta(days=1))
        )

        self._market_prices = price_generator.generate_day_ahead(start_time)
        self._market_prices.extend(
            price_generator.generate_day_ahead(start_time + timedelta(days=1))
        )

    def render(self) -> str | None:
        """Render the environment state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            t = self._current_step
            state = self._battery.current_state
            result = self._episode_result

            output = f"""
Step {t}/{self.config.episode_length}
SOC: {state.soc_mwh:.1f} MWh ({state.soc_fraction*100:.1f}%)
Revenue: ${result.total_revenue:,.0f}
Curtailed: {result.total_curtailed_mwh:.1f} MWh
Violations: {result.grid_violations} grid, {result.soc_violations} SOC
"""
            if self.render_mode == "human":
                print(output)
            return output
        return None

    def get_episode_result(self) -> EpisodeResult:
        """Get the current episode result."""
        return self._episode_result


def make_env(
    config: CurtailmentEnvConfig | None = None,
    seed: int = 42,
) -> CurtailmentEnv:
    """Factory function to create environment.

    Args:
        config: Environment configuration.
        seed: Random seed.

    Returns:
        Configured CurtailmentEnv instance.
    """
    if config is None:
        config = CurtailmentEnvConfig(seed=seed)
    else:
        config.seed = seed

    return CurtailmentEnv(config=config)
