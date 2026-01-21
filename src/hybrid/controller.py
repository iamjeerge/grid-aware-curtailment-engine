"""Hybrid MILP + RL controller for energy dispatch optimization.

Combines the reliability of MILP optimization with the adaptability of
reinforcement learning agents. The MILP provides a baseline schedule,
while the RL agent can override decisions during real-time deviations
from forecasts.

Key features:
- MILP baseline with guaranteed constraint satisfaction
- RL override for real-time adaptation
- Override logging for analysis and debugging
- Configurable trust thresholds
- Fallback to MILP when RL produces invalid actions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from src.battery.physics import BatteryModel
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)
from src.optimization.milp import MILPOptimizer, OptimizationConfig
from src.rl.agents import BaseAgent, HeuristicAgent
from src.rl.environment import CurtailmentEnv, CurtailmentEnvConfig


class DecisionSource(str, Enum):
    """Source of the dispatch decision."""

    MILP = "milp"  # MILP optimizer baseline
    RL = "rl"  # RL agent override
    FALLBACK = "fallback"  # Fallback due to invalid RL action
    NAIVE = "naive"  # Naive heuristic (emergency fallback)


@dataclass
class DispatchDecision:
    """Simple dispatch decision for hybrid controller internal use.

    This is a lightweight dataclass used internally by the hybrid controller,
    separate from the more comprehensive OptimizationDecision domain model.

    Attributes:
        timestamp: When the decision was made.
        timestep: Time step index.
        energy_sold_mw: Energy sold to grid (MW).
        energy_stored_mw: Energy stored in battery (MW).
        energy_curtailed_mw: Energy curtailed (MW).
        battery_soc_mwh: Battery state of charge (MWh).
    """

    timestamp: datetime
    timestep: int
    energy_sold_mw: float
    energy_stored_mw: float
    energy_curtailed_mw: float
    battery_soc_mwh: float

    @property
    def total_mw(self) -> float:
        """Total energy dispatched."""
        return self.energy_sold_mw + self.energy_stored_mw + self.energy_curtailed_mw


@dataclass
class OverrideEvent:
    """Record of an RL override decision.

    Attributes:
        timestamp: When the override occurred.
        step: Time step index.
        milp_action: Original MILP decision.
        rl_action: RL agent's override decision.
        final_action: Actual action taken.
        source: Which source was used.
        deviation_reason: Why the override occurred.
        deviation_magnitude: How different was actual vs forecast.
        confidence: RL agent's confidence in override (0-1).
    """

    timestamp: datetime
    step: int
    milp_action: dict[str, float]
    rl_action: dict[str, float]
    final_action: dict[str, float]
    source: DecisionSource
    deviation_reason: str = ""
    deviation_magnitude: float = 0.0
    confidence: float = 0.0


@dataclass
class HybridControllerConfig:
    """Configuration for the hybrid controller.

    Attributes:
        battery_config: Battery configuration.
        enable_rl_override: Whether RL can override MILP.
        deviation_threshold: Forecast deviation threshold for RL override.
        confidence_threshold: Minimum RL confidence to accept override.
        max_override_fraction: Maximum fraction of decisions RL can override.
        fallback_on_violation: Fall back to MILP if RL violates constraints.
        log_all_decisions: Log all decisions, not just overrides.
        milp_config: MILP optimizer configuration.
        seed: Random seed for reproducibility.
    """

    battery_config: BatteryConfig = field(default_factory=BatteryConfig)
    enable_rl_override: bool = True
    deviation_threshold: float = 0.15  # 15% deviation triggers RL
    confidence_threshold: float = 0.6  # 60% confidence required
    max_override_fraction: float = 0.3  # Max 30% RL overrides
    fallback_on_violation: bool = True
    log_all_decisions: bool = False
    milp_config: OptimizationConfig | None = None
    seed: int = 42


@dataclass
class HybridResult:
    """Results from hybrid controller execution.

    Attributes:
        decisions: List of dispatch decisions.
        override_events: List of override events.
        total_revenue: Total revenue achieved.
        total_curtailment_mwh: Total energy curtailed.
        total_stored_mwh: Total energy stored.
        total_sold_mwh: Total energy sold.
        grid_violations: Number of grid constraint violations.
        milp_decision_count: Number of MILP decisions.
        rl_override_count: Number of RL overrides.
        fallback_count: Number of fallback events.
        override_rate: Fraction of decisions from RL.
    """

    decisions: list[DispatchDecision] = field(default_factory=list)
    override_events: list[OverrideEvent] = field(default_factory=list)
    total_revenue: float = 0.0
    total_curtailment_mwh: float = 0.0
    total_stored_mwh: float = 0.0
    total_sold_mwh: float = 0.0
    grid_violations: int = 0
    milp_decision_count: int = 0
    rl_override_count: int = 0
    fallback_count: int = 0

    @property
    def override_rate(self) -> float:
        """Fraction of decisions from RL."""
        total = self.milp_decision_count + self.rl_override_count + self.fallback_count
        return self.rl_override_count / max(total, 1)

    @property
    def total_decisions(self) -> int:
        """Total number of decisions made."""
        return self.milp_decision_count + self.rl_override_count + self.fallback_count


class HybridController:
    """Hybrid MILP + RL controller for energy dispatch.

    Combines MILP optimization for reliable baseline scheduling with
    RL agents for real-time adaptation to forecast deviations.

    The controller workflow:
    1. Generate MILP schedule for the planning horizon
    2. At each timestep, check for significant deviations from forecast
    3. If deviation exceeds threshold, consult RL agent for override
    4. Validate RL action against constraints
    5. Execute final action and log decisions

    Example:
        ```python
        controller = HybridController(config, rl_agent)
        result = controller.run(
            forecasts=forecasts,
            constraints=constraints,
            prices=prices,
            actuals=actual_generation,  # Real-time data
        )
        print(f"Revenue: ${result.total_revenue:,.0f}")
        print(f"Override rate: {result.override_rate:.1%}")
        ```
    """

    def __init__(
        self,
        config: HybridControllerConfig | None = None,
        rl_agent: BaseAgent | None = None,
    ) -> None:
        """Initialize the hybrid controller.

        Args:
            config: Controller configuration.
            rl_agent: Trained RL agent for overrides (or None for default).
        """
        self.config = config or HybridControllerConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Initialize MILP optimizer
        milp_config = self.config.milp_config or OptimizationConfig(
            solver_name="glpk",
            time_limit_seconds=60,
            mip_gap=0.01,
        )
        self._milp_optimizer = MILPOptimizer(milp_config)

        # Initialize RL agent (create default if not provided)
        self._rl_agent = rl_agent

        # Tracking
        self._override_count = 0
        self._total_decisions = 0

    def run(
        self,
        forecasts: list[GenerationForecast],
        constraints: list[GridConstraint],
        prices: list[MarketPrice],
        actuals: list[float] | None = None,
        start_soc: float = 0.5,
    ) -> HybridResult:
        """Run the hybrid controller over the planning horizon.

        Args:
            forecasts: Generation forecasts for each timestep.
            constraints: Grid constraints for each timestep.
            prices: Market prices for each timestep.
            actuals: Actual generation values (for real-time simulation).
                    If None, uses P50 forecast.
            start_soc: Initial battery state of charge (0-1).

        Returns:
            HybridResult with decisions, metrics, and override logs.
        """
        n_steps = len(forecasts)

        # Handle empty forecasts
        if n_steps == 0:
            return HybridResult()

        # Generate MILP baseline schedule
        milp_decisions = self._generate_milp_schedule(
            forecasts=forecasts,
            constraints=constraints,
            prices=prices,
            start_soc=start_soc,
        )

        # Initialize battery model for simulation
        battery = BatteryModel(self.config.battery_config)
        battery.initialize(forecasts[0].timestamp, start_soc)

        # Prepare RL environment if agent provided
        rl_env = None
        if self._rl_agent is not None and self.config.enable_rl_override:
            env_config = CurtailmentEnvConfig(
                battery_config=self.config.battery_config,
                episode_length=n_steps,
                seed=self.config.seed,
            )
            rl_env = CurtailmentEnv(
                config=env_config,
                forecasts=forecasts,
                grid_constraints=constraints,
                market_prices=prices,
            )
            rl_env.reset()

        # Initialize result
        result = HybridResult()

        # Actual generation values (use P50 if not provided)
        if actuals is None:
            actuals = [f.total_generation(ForecastScenario.P50) for f in forecasts]

        # Execute step by step
        for t in range(n_steps):
            forecast = forecasts[t]
            constraint = constraints[t]
            price = prices[t]
            actual_gen = actuals[t]
            forecast_gen = forecast.total_generation(ForecastScenario.P50)

            # Get MILP decision for this step
            milp_decision = milp_decisions[t] if t < len(milp_decisions) else None

            # Check for deviation from forecast
            deviation = abs(actual_gen - forecast_gen) / max(forecast_gen, 1.0)

            # Determine whether to use RL override
            source = DecisionSource.MILP
            final_decision = milp_decision
            rl_decision = None
            override_event = None

            if (
                self.config.enable_rl_override
                and self._rl_agent is not None
                and rl_env is not None
                and deviation >= self.config.deviation_threshold
                and self._can_override()
            ):
                # Get RL agent's recommendation
                rl_decision = self._get_rl_decision(
                    rl_env=rl_env,
                    actual_gen=actual_gen,
                    constraint=constraint,
                    price=price,
                    battery=battery,
                    timestep=t,
                    timestamp=forecast.timestamp,
                )

                # Validate RL decision
                if self._is_valid_decision(
                    rl_decision, actual_gen, constraint, battery
                ):
                    # Accept RL override
                    final_decision = rl_decision
                    source = DecisionSource.RL
                    self._override_count += 1
                elif self.config.fallback_on_violation:
                    # RL decision invalid, fall back to MILP
                    source = DecisionSource.FALLBACK
                    # fallback_count incremented in final metrics section

            # If no valid decision yet, use naive as last resort
            if final_decision is None:
                final_decision = self._get_naive_decision(
                    actual_gen=actual_gen,
                    constraint=constraint,
                    price=price,
                    battery=battery,
                    timestamp=forecast.timestamp,
                    timestep=t,
                )
                source = DecisionSource.NAIVE

            # Scale decision to actual generation (not forecast)
            final_decision = self._scale_decision_to_actual(
                decision=final_decision,
                actual_gen=actual_gen,
                constraint=constraint,
                battery=battery,
            )

            # Execute decision (update battery, compute revenue)
            revenue = self._execute_decision(
                decision=final_decision,
                price=price,
                battery=battery,
                timestamp=forecast.timestamp,
            )

            # Check for violations
            if final_decision.energy_sold_mw > constraint.max_export_mw * 1.01:
                result.grid_violations += 1

            # Record metrics
            result.decisions.append(final_decision)
            result.total_revenue += revenue
            result.total_sold_mwh += final_decision.energy_sold_mw
            result.total_stored_mwh += final_decision.energy_stored_mw
            result.total_curtailment_mwh += final_decision.energy_curtailed_mw

            if source == DecisionSource.MILP:
                result.milp_decision_count += 1
            elif source == DecisionSource.RL:
                result.rl_override_count += 1
            elif source in (DecisionSource.NAIVE, DecisionSource.FALLBACK):
                result.fallback_count += 1

            # Log override event
            if source != DecisionSource.MILP or self.config.log_all_decisions:
                override_event = OverrideEvent(
                    timestamp=forecast.timestamp,
                    step=t,
                    milp_action=self._decision_to_dict(milp_decision),
                    rl_action=self._decision_to_dict(rl_decision),
                    final_action=self._decision_to_dict(final_decision),
                    source=source,
                    deviation_reason=f"Actual={actual_gen:.1f}MW vs Forecast={forecast_gen:.1f}MW",
                    deviation_magnitude=deviation,
                )
                result.override_events.append(override_event)

            self._total_decisions += 1

        return result

    def _generate_milp_schedule(
        self,
        forecasts: list[GenerationForecast],
        constraints: list[GridConstraint],
        prices: list[MarketPrice],
        start_soc: float,
    ) -> list[DispatchDecision]:
        """Generate baseline MILP schedule.

        Args:
            forecasts: Generation forecasts.
            constraints: Grid constraints.
            prices: Market prices.
            start_soc: Initial SOC.

        Returns:
            List of dispatch decisions.
        """
        try:
            result = self._milp_optimizer.optimize(
                forecasts=forecasts,
                constraints=constraints,
                prices=prices,
                battery_config=self.config.battery_config,
                initial_soc=start_soc,
            )
            # Convert OptimizationDecision to DispatchDecision
            dispatch_decisions = []
            for opt_dec in result.decisions:
                dispatch_decisions.append(
                    DispatchDecision(
                        timestamp=opt_dec.timestamp,
                        timestep=0,  # Will be set properly
                        energy_sold_mw=opt_dec.energy_sold_mw,
                        energy_stored_mw=opt_dec.energy_stored_mw,
                        energy_curtailed_mw=opt_dec.energy_curtailed_mw,
                        battery_soc_mwh=opt_dec.resulting_soc_mwh,
                    )
                )
            return dispatch_decisions
        except Exception:
            # If MILP fails, return empty list (will use naive)
            return []

    def _get_rl_decision(
        self,
        rl_env: CurtailmentEnv,
        actual_gen: float,
        constraint: GridConstraint,
        price: MarketPrice,
        battery: BatteryModel,
        timestep: int,
        timestamp: datetime,
    ) -> DispatchDecision | None:
        """Get decision from RL agent.

        Args:
            rl_env: RL environment.
            actual_gen: Actual generation (MW).
            constraint: Grid constraint.
            price: Market price.
            battery: Battery model.
            timestep: Current timestep.
            timestamp: Current timestamp.

        Returns:
            DispatchDecision or None if agent fails.
        """
        try:
            # Build observation
            obs = self._build_observation(
                actual_gen=actual_gen,
                constraint=constraint,
                price=price,
                battery=battery,
                rl_env=rl_env,
            )

            # Get action from agent
            action = self._rl_agent.select_action(obs, deterministic=True)

            # Convert action to decision
            sell_frac = float(action[0])
            store_frac = float(action[1])

            # Normalize fractions
            total_frac = sell_frac + store_frac
            if total_frac > 1.0:
                sell_frac /= total_frac
                store_frac /= total_frac

            curtail_frac = 1.0 - sell_frac - store_frac

            return DispatchDecision(
                timestamp=timestamp,
                timestep=timestep,
                energy_sold_mw=actual_gen * sell_frac,
                energy_stored_mw=actual_gen * store_frac,
                energy_curtailed_mw=actual_gen * curtail_frac,
                battery_soc_mwh=battery.current_state.soc_mwh,
            )

        except Exception:
            return None

    def _build_observation(
        self,
        actual_gen: float,
        constraint: GridConstraint,
        price: MarketPrice,
        battery: BatteryModel,
        rl_env: CurtailmentEnv,
    ) -> np.ndarray:
        """Build observation vector for RL agent.

        Args:
            actual_gen: Actual generation (MW).
            constraint: Grid constraint.
            price: Market price.
            battery: Battery model.
            rl_env: RL environment (for normalization).

        Returns:
            Observation array.
        """
        config = rl_env.config

        # SOC (normalized 0-1)
        soc = battery.current_state.soc_fraction

        # Generation (normalized)
        gen_norm = actual_gen / config.max_generation_mw
        gen_norm = np.clip(gen_norm, 0.0, 1.0)

        # Grid capacity (normalized)
        grid_norm = constraint.max_export_mw / config.max_grid_capacity_mw
        grid_norm = np.clip(grid_norm, 0.0, 1.0)

        # Price (normalized)
        price_range = config.max_price_per_mwh - config.min_price_per_mwh
        price_norm = (price.effective_price - config.min_price_per_mwh) / price_range
        price_norm = np.clip(price_norm, 0.0, 1.0)

        # Hour (normalized)
        hour_norm = 0.5  # Default if no timestamp

        # Build observation (simplified - without lookahead)
        # Match the expected observation dimension
        obs_dim = rl_env.observation_space.shape[0]
        obs = np.zeros(obs_dim, dtype=np.float32)
        obs[0] = soc
        obs[1] = gen_norm
        # Fill lookahead with same value
        for i in range(2, obs_dim - 3):
            obs[i] = gen_norm
        obs[-3] = grid_norm
        obs[-2] = price_norm
        obs[-1] = hour_norm

        return obs

    def _get_naive_decision(
        self,
        actual_gen: float,
        constraint: GridConstraint,
        price: MarketPrice,  # noqa: ARG002
        battery: BatteryModel,
        timestamp: datetime,
        timestep: int = 0,
    ) -> DispatchDecision:
        """Get decision from naive heuristic.

        Args:
            actual_gen: Actual generation.
            constraint: Grid constraint.
            price: Market price.
            battery: Battery model.
            timestamp: Current timestamp.
            timestep: Current timestep index.

        Returns:
            DispatchDecision from naive controller.
        """
        # Simple heuristic: sell up to grid limit, store excess, curtail rest
        grid_limit = constraint.max_export_mw
        max_charge = battery.get_max_charge_power()

        energy_sold = min(actual_gen, grid_limit)
        remaining = actual_gen - energy_sold

        energy_stored = min(remaining, max_charge)
        energy_curtailed = remaining - energy_stored

        return DispatchDecision(
            timestamp=timestamp,
            timestep=timestep,
            energy_sold_mw=energy_sold,
            energy_stored_mw=energy_stored,
            energy_curtailed_mw=energy_curtailed,
            battery_soc_mwh=battery.current_state.soc_mwh,
        )

    def _is_valid_decision(
        self,
        decision: DispatchDecision | None,
        actual_gen: float,
        constraint: GridConstraint,
        battery: BatteryModel,
    ) -> bool:
        """Check if decision satisfies constraints.

        Args:
            decision: Decision to validate.
            actual_gen: Actual generation.
            constraint: Grid constraint.
            battery: Battery model.

        Returns:
            True if decision is valid.
        """
        if decision is None:
            return False

        # Check grid constraint
        if decision.energy_sold_mw > constraint.max_export_mw * 1.05:
            return False

        # Check battery charge limit
        max_charge = battery.get_max_charge_power()
        if decision.energy_stored_mw > max_charge * 1.05:
            return False

        # Check energy balance (allow some tolerance)
        total = (
            decision.energy_sold_mw
            + decision.energy_stored_mw
            + decision.energy_curtailed_mw
        )
        if abs(total - actual_gen) > actual_gen * 0.1:
            return False

        # Check non-negativity
        return not (
            decision.energy_sold_mw < -0.01
            or decision.energy_stored_mw < -0.01
            or decision.energy_curtailed_mw < -0.01
        )

    def _scale_decision_to_actual(
        self,
        decision: DispatchDecision,
        actual_gen: float,
        constraint: GridConstraint,
        battery: BatteryModel,
    ) -> DispatchDecision:
        """Scale decision to match actual generation.

        Args:
            decision: Original decision.
            actual_gen: Actual generation.
            constraint: Grid constraint.
            battery: Battery model.

        Returns:
            Scaled decision.
        """
        total = (
            decision.energy_sold_mw
            + decision.energy_stored_mw
            + decision.energy_curtailed_mw
        )

        if total <= 0:
            # No energy in original decision, use naive split
            grid_limit = constraint.max_export_mw
            max_charge = battery.get_max_charge_power()

            energy_sold = min(actual_gen, grid_limit)
            remaining = actual_gen - energy_sold
            energy_stored = min(remaining, max_charge)
            energy_curtailed = remaining - energy_stored
        else:
            # Scale proportionally
            scale = actual_gen / total
            energy_sold = min(decision.energy_sold_mw * scale, constraint.max_export_mw)
            remaining = actual_gen - energy_sold
            max_charge = battery.get_max_charge_power()
            energy_stored = min(
                decision.energy_stored_mw * scale, max_charge, remaining
            )
            energy_curtailed = actual_gen - energy_sold - energy_stored

        return DispatchDecision(
            timestamp=decision.timestamp,
            timestep=decision.timestep,
            energy_sold_mw=max(0, energy_sold),
            energy_stored_mw=max(0, energy_stored),
            energy_curtailed_mw=max(0, energy_curtailed),
            battery_soc_mwh=battery.current_state.soc_mwh,
        )

    def _execute_decision(
        self,
        decision: DispatchDecision,
        price: MarketPrice,
        battery: BatteryModel,
        timestamp: datetime,
    ) -> float:
        """Execute decision and compute revenue.

        Args:
            decision: Decision to execute.
            price: Market price.
            battery: Battery model.
            timestamp: Current timestamp.

        Returns:
            Revenue from this timestep.
        """
        # Update battery
        if decision.energy_stored_mw > 0:
            battery.charge(decision.energy_stored_mw, 1.0, timestamp)

        # Compute revenue
        revenue = decision.energy_sold_mw * price.effective_price

        # Subtract degradation cost
        degradation = (
            decision.energy_stored_mw
            * self.config.battery_config.degradation_cost_per_mwh
        )

        return revenue - degradation

    def _can_override(self) -> bool:
        """Check if RL can still override (within limit).

        Returns:
            True if override is allowed.
        """
        if self._total_decisions == 0:
            return True

        current_rate = self._override_count / self._total_decisions
        return current_rate < self.config.max_override_fraction

    def _decision_to_dict(self, decision: DispatchDecision | None) -> dict[str, float]:
        """Convert decision to dictionary.

        Args:
            decision: Decision to convert.

        Returns:
            Dictionary representation.
        """
        if decision is None:
            return {"sold": 0, "stored": 0, "curtailed": 0}

        return {
            "sold": decision.energy_sold_mw,
            "stored": decision.energy_stored_mw,
            "curtailed": decision.energy_curtailed_mw,
        }

    def reset(self) -> None:
        """Reset controller state for new run."""
        self._override_count = 0
        self._total_decisions = 0


def quick_hybrid_run(
    forecasts: list[GenerationForecast],
    constraints: list[GridConstraint],
    prices: list[MarketPrice],
    battery_config: BatteryConfig | None = None,
    enable_rl: bool = True,
    seed: int = 42,
) -> HybridResult:
    """Convenience function for quick hybrid controller run.

    Args:
        forecasts: Generation forecasts.
        constraints: Grid constraints.
        prices: Market prices.
        battery_config: Battery configuration.
        enable_rl: Whether to enable RL override.
        seed: Random seed.

    Returns:
        HybridResult with all metrics.
    """
    config = HybridControllerConfig(
        battery_config=battery_config or BatteryConfig(),
        enable_rl_override=enable_rl,
        seed=seed,
    )

    # Create default heuristic agent if RL enabled
    rl_agent = None
    if enable_rl:
        # Create a simple environment to initialize agent
        env_config = CurtailmentEnvConfig(
            battery_config=config.battery_config,
            episode_length=len(forecasts),
            seed=seed,
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=forecasts,
            grid_constraints=constraints,
            market_prices=prices,
        )
        rl_agent = HeuristicAgent(env)

    controller = HybridController(config=config, rl_agent=rl_agent)
    return controller.run(
        forecasts=forecasts,
        constraints=constraints,
        prices=prices,
    )
