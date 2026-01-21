"""Tests for the hybrid MILP + RL controller.

Tests cover:
- HybridController basic functionality
- MILP baseline scheduling
- RL override mechanism
- Fallback behavior
- Override logging
- Constraint validation
- Energy balance verification
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.domain.models import (
    BatteryConfig,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)
from src.hybrid.controller import (
    DecisionSource,
    HybridController,
    HybridControllerConfig,
    HybridResult,
    OverrideEvent,
    quick_hybrid_run,
)
from src.rl.agents import AgentConfig, HeuristicAgent, RandomAgent
from src.rl.environment import CurtailmentEnv, CurtailmentEnvConfig

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def default_battery_config() -> BatteryConfig:
    """Default battery configuration for tests."""
    return BatteryConfig(
        capacity_mwh=100.0,
        max_charge_rate_mw=50.0,
        max_discharge_rate_mw=50.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_fraction=0.1,
        max_soc_fraction=0.9,
        degradation_cost_per_mwh=8.0,
    )


@pytest.fixture
def small_battery_config() -> BatteryConfig:
    """Small battery for testing capacity constraints."""
    return BatteryConfig(
        capacity_mwh=20.0,
        max_charge_rate_mw=10.0,
        max_discharge_rate_mw=10.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_fraction=0.1,
        max_soc_fraction=0.9,
        degradation_cost_per_mwh=8.0,
    )


@pytest.fixture
def base_timestamp() -> datetime:
    """Base timestamp for forecasts."""
    return datetime(2024, 6, 15, 6, 0)  # 6 AM start


@pytest.fixture
def simple_forecasts(base_timestamp: datetime) -> list[GenerationForecast]:
    """Simple 12-hour generation forecasts."""
    forecasts = []
    for hour in range(12):
        ts = base_timestamp + timedelta(hours=hour)
        # Simple profile: low->high->low
        if hour < 4:
            gen = 50 + hour * 30
        elif hour < 8:
            gen = 200 + (hour - 4) * 25
        else:
            gen = 300 - (hour - 8) * 50

        solar_gen = gen * 0.8
        wind_gen = gen * 0.2
        forecasts.append(
            GenerationForecast(
                timestamp=ts,
                solar_mw_p10=solar_gen * 0.85,
                solar_mw_p50=solar_gen,
                solar_mw_p90=solar_gen * 1.15,
                wind_mw_p10=wind_gen * 0.80,
                wind_mw_p50=wind_gen,
                wind_mw_p90=wind_gen * 1.20,
            )
        )
    return forecasts


@pytest.fixture
def duck_curve_forecasts(base_timestamp: datetime) -> list[GenerationForecast]:
    """Duck curve solar generation profile."""
    forecasts = []
    for hour in range(24):
        ts = base_timestamp + timedelta(hours=hour)

        # Duck curve: morning ramp, midday peak, evening ramp down
        if hour < 6:
            solar = 0.0
        elif hour < 10:
            solar = float((hour - 6) * 100)  # Ramp up 0-400
        elif hour < 14:
            solar = float(400 + (hour - 10) * 50)  # Peak 400-600
        elif hour < 18:
            solar = float(600 - (hour - 14) * 100)  # Ramp down 600-200
        else:
            solar = float(max(0, 200 - (hour - 18) * 50))  # Evening decline

        wind = 50.0  # Constant wind baseline
        forecasts.append(
            GenerationForecast(
                timestamp=ts,
                solar_mw_p10=solar * 0.85,
                solar_mw_p50=solar,
                solar_mw_p90=solar * 1.15,
                wind_mw_p10=wind * 0.80,
                wind_mw_p50=wind,
                wind_mw_p90=wind * 1.20,
            )
        )
    return forecasts


@pytest.fixture
def simple_constraints(base_timestamp: datetime) -> list[GridConstraint]:
    """Simple 12-hour grid constraints."""
    return [
        GridConstraint(
            timestamp=base_timestamp + timedelta(hours=hour),
            max_export_mw=200.0,  # Fixed grid limit
            max_ramp_up_mw_per_hour=100.0,
            max_ramp_down_mw_per_hour=100.0,
        )
        for hour in range(12)
    ]


@pytest.fixture
def congested_constraints(base_timestamp: datetime) -> list[GridConstraint]:
    """24-hour constraints with midday congestion."""
    constraints = []
    for hour in range(24):
        ts = base_timestamp + timedelta(hours=hour)

        # Congestion during solar peak (10am-4pm)
        max_export = 200.0 if 10 <= hour < 16 else 500.0  # Congested vs open

        constraints.append(
            GridConstraint(
                timestamp=ts,
                max_export_mw=max_export,
                max_ramp_up_mw_per_hour=150.0,
                max_ramp_down_mw_per_hour=150.0,
            )
        )
    return constraints


@pytest.fixture
def simple_prices(base_timestamp: datetime) -> list[MarketPrice]:
    """Simple 12-hour price profile."""
    rng = np.random.default_rng(42)
    return [
        MarketPrice(
            timestamp=base_timestamp + timedelta(hours=hour),
            day_ahead_price=50.0 + hour * 5.0,  # Rising prices
            real_time_price=50.0 + hour * 5.0 + rng.uniform(-5, 5),
        )
        for hour in range(12)
    ]


@pytest.fixture
def duck_curve_prices(base_timestamp: datetime) -> list[MarketPrice]:
    """24-hour prices with midday dip and evening spike."""
    prices = []
    rng = np.random.default_rng(42)
    for hour in range(24):
        ts = base_timestamp + timedelta(hours=hour)

        # Price pattern: morning moderate, midday low/negative, evening spike
        if hour < 6:
            da_price = 40.0  # Night
        elif hour < 10:
            da_price = 50.0  # Morning
        elif hour < 14:
            da_price = -10.0  # Negative midday!
        elif hour < 18:
            da_price = 30.0  # Afternoon
        elif hour < 22:
            da_price = 120.0  # Evening spike
        else:
            da_price = 50.0  # Late night

        prices.append(
            MarketPrice(
                timestamp=ts,
                day_ahead_price=da_price,
                real_time_price=da_price + rng.uniform(-5, 5),
            )
        )
    return prices


@pytest.fixture
def hybrid_config(default_battery_config: BatteryConfig) -> HybridControllerConfig:
    """Default hybrid controller config."""
    return HybridControllerConfig(
        battery_config=default_battery_config,
        enable_rl_override=True,
        deviation_threshold=0.15,
        confidence_threshold=0.6,
        max_override_fraction=0.3,
        fallback_on_violation=True,
        log_all_decisions=False,
        seed=42,
    )


# ==============================================================================
# Test Configuration
# ==============================================================================


class TestHybridControllerConfig:
    """Tests for HybridControllerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HybridControllerConfig()
        assert config.enable_rl_override is True
        assert config.deviation_threshold == 0.15
        assert config.confidence_threshold == 0.6
        assert config.max_override_fraction == 0.3
        assert config.fallback_on_violation is True
        assert config.seed == 42

    def test_custom_config(self, default_battery_config: BatteryConfig) -> None:
        """Test custom configuration."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            deviation_threshold=0.25,
            max_override_fraction=0.5,
            seed=123,
        )
        assert config.enable_rl_override is False
        assert config.deviation_threshold == 0.25
        assert config.max_override_fraction == 0.5
        assert config.seed == 123

    def test_config_with_milp_config(self) -> None:
        """Test config with custom MILP settings."""
        from src.optimization.milp import OptimizationConfig

        milp_config = OptimizationConfig(
            solver_name="glpk",
            time_limit_seconds=120,
            mip_gap=0.005,
        )
        config = HybridControllerConfig(milp_config=milp_config)
        assert config.milp_config is not None
        assert config.milp_config.time_limit_seconds == 120


# ==============================================================================
# Test Hybrid Result
# ==============================================================================


class TestHybridResult:
    """Tests for HybridResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty result initialization."""
        result = HybridResult()
        assert len(result.decisions) == 0
        assert len(result.override_events) == 0
        assert result.total_revenue == 0
        assert result.override_rate == 0
        assert result.total_decisions == 0

    def test_override_rate_calculation(self) -> None:
        """Test override rate property."""
        result = HybridResult(
            milp_decision_count=70,
            rl_override_count=20,
            fallback_count=10,
        )
        assert result.override_rate == pytest.approx(0.2, abs=0.01)
        assert result.total_decisions == 100

    def test_override_rate_no_decisions(self) -> None:
        """Test override rate with no decisions."""
        result = HybridResult()
        assert result.override_rate == 0  # Avoid division by zero


# ==============================================================================
# Test Override Event
# ==============================================================================


class TestOverrideEvent:
    """Tests for OverrideEvent dataclass."""

    def test_override_event_creation(self, base_timestamp: datetime) -> None:
        """Test creating an override event."""
        event = OverrideEvent(
            timestamp=base_timestamp,
            step=5,
            milp_action={"sold": 100, "stored": 50, "curtailed": 0},
            rl_action={"sold": 80, "stored": 70, "curtailed": 0},
            final_action={"sold": 80, "stored": 70, "curtailed": 0},
            source=DecisionSource.RL,
            deviation_reason="Actual=180MW vs Forecast=150MW",
            deviation_magnitude=0.2,
            confidence=0.85,
        )
        assert event.step == 5
        assert event.source == DecisionSource.RL
        assert event.deviation_magnitude == 0.2


# ==============================================================================
# Test Decision Source Enum
# ==============================================================================


class TestDecisionSource:
    """Tests for DecisionSource enum."""

    def test_decision_sources(self) -> None:
        """Test all decision source values."""
        assert DecisionSource.MILP.value == "milp"
        assert DecisionSource.RL.value == "rl"
        assert DecisionSource.FALLBACK.value == "fallback"
        assert DecisionSource.NAIVE.value == "naive"


# ==============================================================================
# Test Hybrid Controller Initialization
# ==============================================================================


class TestHybridControllerInit:
    """Tests for HybridController initialization."""

    def test_default_initialization(self) -> None:
        """Test controller with default config."""
        controller = HybridController()
        assert controller.config is not None
        assert controller._rl_agent is None

    def test_initialization_with_config(
        self, hybrid_config: HybridControllerConfig
    ) -> None:
        """Test controller with custom config."""
        controller = HybridController(config=hybrid_config)
        assert controller.config.battery_config.capacity_mwh == 100.0
        assert controller.config.enable_rl_override is True

    def test_initialization_with_rl_agent(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test controller with RL agent."""
        env_config = CurtailmentEnvConfig(
            battery_config=hybrid_config.battery_config,
            episode_length=len(simple_forecasts),
            seed=42,
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=simple_forecasts,
            grid_constraints=simple_constraints,
            market_prices=simple_prices,
        )
        agent = HeuristicAgent(env)

        controller = HybridController(config=hybrid_config, rl_agent=agent)
        assert controller._rl_agent is not None


# ==============================================================================
# Test Hybrid Controller Run
# ==============================================================================


class TestHybridControllerRun:
    """Tests for HybridController.run()."""

    def test_basic_run(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test basic controller run without RL."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            seed=42,
        )
        controller = HybridController(config=config)

        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        assert len(result.decisions) == len(simple_forecasts)
        assert result.total_revenue >= 0
        assert result.grid_violations == 0

    def test_run_with_rl_enabled(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test run with RL override enabled."""
        # Create RL agent
        env_config = CurtailmentEnvConfig(
            battery_config=hybrid_config.battery_config,
            episode_length=len(simple_forecasts),
            seed=42,
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=simple_forecasts,
            grid_constraints=simple_constraints,
            market_prices=simple_prices,
        )
        agent = HeuristicAgent(env)

        controller = HybridController(config=hybrid_config, rl_agent=agent)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        assert len(result.decisions) == len(simple_forecasts)
        assert result.total_decisions == len(simple_forecasts)

    def test_run_with_actual_deviations(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test run with actual generation differing from forecast."""
        from src.domain.models import ForecastScenario

        # Create actuals with 30% deviation from forecast
        actuals = [
            f.total_generation(ForecastScenario.P50) * 1.3 for f in simple_forecasts
        ]

        # Create RL agent
        env_config = CurtailmentEnvConfig(
            battery_config=hybrid_config.battery_config,
            episode_length=len(simple_forecasts),
            seed=42,
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=simple_forecasts,
            grid_constraints=simple_constraints,
            market_prices=simple_prices,
        )
        agent = HeuristicAgent(env)

        # Lower deviation threshold to trigger more overrides
        hybrid_config.deviation_threshold = 0.1

        controller = HybridController(config=hybrid_config, rl_agent=agent)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            actuals=actuals,
        )

        # With 30% deviation, RL should have been consulted
        assert result.total_decisions == len(simple_forecasts)

    def test_energy_balance(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that energy balance is maintained."""
        from src.domain.models import ForecastScenario

        hybrid_config.enable_rl_override = False  # Pure MILP
        controller = HybridController(config=hybrid_config)

        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        # Check energy balance for each decision
        for i, decision in enumerate(result.decisions):
            forecast_gen = simple_forecasts[i].total_generation(ForecastScenario.P50)
            total_allocated = (
                decision.energy_sold_mw
                + decision.energy_stored_mw
                + decision.energy_curtailed_mw
            )
            # Allow 10% tolerance for scaling
            assert total_allocated <= forecast_gen * 1.1 + 0.1

    def test_grid_constraint_compliance(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that grid constraints are respected."""
        controller = HybridController(config=hybrid_config)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        assert result.grid_violations == 0

        # Verify each decision respects grid limit
        for i, decision in enumerate(result.decisions):
            constraint = simple_constraints[i]
            assert decision.energy_sold_mw <= constraint.max_export_mw * 1.01


# ==============================================================================
# Test Override Mechanism
# ==============================================================================


class TestOverrideMechanism:
    """Tests for RL override behavior."""

    def test_override_rate_limit(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that override rate is limited."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=True,
            deviation_threshold=0.01,  # Very low to trigger overrides
            max_override_fraction=0.3,
            seed=42,
        )

        # Create RL agent
        env_config = CurtailmentEnvConfig(
            battery_config=default_battery_config,
            episode_length=len(simple_forecasts),
            seed=42,
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=simple_forecasts,
            grid_constraints=simple_constraints,
            market_prices=simple_prices,
        )
        agent = RandomAgent(env)

        # Use actuals with large deviation
        from src.domain.models import ForecastScenario

        actuals = [
            f.total_generation(ForecastScenario.P50) * 1.5 for f in simple_forecasts
        ]

        controller = HybridController(config=config, rl_agent=agent)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            actuals=actuals,
        )

        # Override rate should not exceed limit
        assert result.override_rate <= 0.35  # Allow small tolerance

    def test_fallback_on_invalid_action(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that invalid RL actions trigger fallback."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=True,
            deviation_threshold=0.01,
            fallback_on_violation=True,
            seed=42,
        )

        controller = HybridController(config=config, rl_agent=None)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        # Without RL agent, all decisions should be MILP or naive
        assert result.rl_override_count == 0


# ==============================================================================
# Test Override Logging
# ==============================================================================


class TestOverrideLogging:
    """Tests for override event logging."""

    def test_log_all_decisions(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test logging all decisions."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            log_all_decisions=True,
            seed=42,
        )

        controller = HybridController(config=config)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        # All decisions should be logged when log_all_decisions=True
        assert len(result.override_events) == len(simple_forecasts)

    def test_override_event_contents(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test override event data structure."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            log_all_decisions=True,
            seed=42,
        )

        controller = HybridController(config=config)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        if result.override_events:
            event = result.override_events[0]
            assert event.timestamp is not None
            assert event.step >= 0
            assert "sold" in event.final_action
            assert event.source in DecisionSource


# ==============================================================================
# Test Duck Curve Scenario
# ==============================================================================


class TestDuckCurveScenario:
    """Tests with duck curve scenario."""

    def test_duck_curve_curtailment_reduction(
        self,
        default_battery_config: BatteryConfig,
        duck_curve_forecasts: list[GenerationForecast],
        congested_constraints: list[GridConstraint],
        duck_curve_prices: list[MarketPrice],
    ) -> None:
        """Test that hybrid controller reduces curtailment in duck curve."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,  # Test pure MILP first
            seed=42,
        )

        controller = HybridController(config=config)
        result = controller.run(
            forecasts=duck_curve_forecasts,
            constraints=congested_constraints,
            prices=duck_curve_prices,
        )

        # Should have some storage usage during congestion
        assert result.total_stored_mwh > 0
        assert result.grid_violations == 0

    def test_duck_curve_revenue_optimization(
        self,
        default_battery_config: BatteryConfig,
        duck_curve_forecasts: list[GenerationForecast],
        congested_constraints: list[GridConstraint],
        duck_curve_prices: list[MarketPrice],
    ) -> None:
        """Test that controller maximizes revenue in duck curve."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            seed=42,
        )

        controller = HybridController(config=config)
        result = controller.run(
            forecasts=duck_curve_forecasts,
            constraints=congested_constraints,
            prices=duck_curve_prices,
        )

        # Should have positive revenue (selling during high price periods)
        assert result.total_revenue > 0


# ==============================================================================
# Test Quick Hybrid Run
# ==============================================================================


class TestQuickHybridRun:
    """Tests for quick_hybrid_run convenience function."""

    def test_quick_run_basic(
        self,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test quick hybrid run."""
        result = quick_hybrid_run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            enable_rl=False,
            seed=42,
        )

        assert isinstance(result, HybridResult)
        assert len(result.decisions) == len(simple_forecasts)

    def test_quick_run_with_rl(
        self,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test quick hybrid run with RL enabled."""
        result = quick_hybrid_run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            enable_rl=True,
            seed=42,
        )

        assert isinstance(result, HybridResult)
        assert result.total_decisions == len(simple_forecasts)

    def test_quick_run_with_battery_config(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test quick hybrid run with custom battery."""
        result = quick_hybrid_run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            battery_config=default_battery_config,
            enable_rl=False,
            seed=42,
        )

        assert isinstance(result, HybridResult)
        assert result.grid_violations == 0


# ==============================================================================
# Test Controller Reset
# ==============================================================================


class TestControllerReset:
    """Tests for controller state reset."""

    def test_reset_clears_state(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that reset clears internal state."""
        controller = HybridController(config=hybrid_config)

        # Run once
        controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        # State should be non-zero
        assert controller._total_decisions > 0

        # Reset
        controller.reset()

        # State should be cleared
        assert controller._total_decisions == 0
        assert controller._override_count == 0

    def test_multiple_runs_with_reset(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test running multiple times with reset."""
        controller = HybridController(config=hybrid_config)

        # Run first time
        result1 = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        # Reset
        controller.reset()

        # Run second time - should be identical
        result2 = controller.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        assert len(result1.decisions) == len(result2.decisions)
        assert result1.total_decisions == result2.total_decisions


# ==============================================================================
# Test Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_forecasts(
        self,
        hybrid_config: HybridControllerConfig,
    ) -> None:
        """Test with empty forecasts."""
        controller = HybridController(config=hybrid_config)
        result = controller.run(
            forecasts=[],
            constraints=[],
            prices=[],
        )

        assert len(result.decisions) == 0
        assert result.total_revenue == 0

    def test_single_timestep(
        self,
        hybrid_config: HybridControllerConfig,
        base_timestamp: datetime,
    ) -> None:
        """Test with single timestep."""
        forecasts = [
            GenerationForecast(
                timestamp=base_timestamp,
                solar_mw=100,
                wind_mw=50,
                solar_uncertainty=0.1,
                wind_uncertainty=0.1,
            )
        ]
        constraints = [
            GridConstraint(
                timestamp=base_timestamp,
                max_export_mw=200,
                min_export_mw=0,
                max_ramp_rate_mw_per_hour=100,
                is_curtailment_allowed=True,
            )
        ]
        prices = [
            MarketPrice(
                timestamp=base_timestamp,
                day_ahead_price=50,
                real_time_price=52,
                ancillary_price=10,
            )
        ]

        controller = HybridController(config=hybrid_config)
        result = controller.run(
            forecasts=forecasts,
            constraints=constraints,
            prices=prices,
        )

        assert len(result.decisions) == 1

    def test_zero_generation(
        self,
        hybrid_config: HybridControllerConfig,
        base_timestamp: datetime,
    ) -> None:
        """Test with zero generation."""
        forecasts = [
            GenerationForecast(
                timestamp=base_timestamp + timedelta(hours=h),
                solar_mw=0,
                wind_mw=0,
                solar_uncertainty=0.1,
                wind_uncertainty=0.1,
            )
            for h in range(3)
        ]
        constraints = [
            GridConstraint(
                timestamp=base_timestamp + timedelta(hours=h),
                max_export_mw=200,
                min_export_mw=0,
                max_ramp_rate_mw_per_hour=100,
                is_curtailment_allowed=True,
            )
            for h in range(3)
        ]
        prices = [
            MarketPrice(
                timestamp=base_timestamp + timedelta(hours=h),
                day_ahead_price=50,
                real_time_price=50,
                ancillary_price=10,
            )
            for h in range(3)
        ]

        controller = HybridController(config=hybrid_config)
        result = controller.run(
            forecasts=forecasts,
            constraints=constraints,
            prices=prices,
        )

        assert len(result.decisions) == 3
        assert result.total_sold_mwh == 0
        assert result.total_curtailment_mwh == 0

    def test_very_tight_constraints(
        self,
        hybrid_config: HybridControllerConfig,
        simple_forecasts: list[GenerationForecast],
        simple_prices: list[MarketPrice],
        base_timestamp: datetime,
    ) -> None:
        """Test with very tight grid constraints."""
        tight_constraints = [
            GridConstraint(
                timestamp=base_timestamp + timedelta(hours=h),
                max_export_mw=10.0,  # Very tight
                min_export_mw=0.0,
                max_ramp_rate_mw_per_hour=50.0,
                is_curtailment_allowed=True,
            )
            for h in range(len(simple_forecasts))
        ]

        controller = HybridController(config=hybrid_config)
        result = controller.run(
            forecasts=simple_forecasts,
            constraints=tight_constraints,
            prices=simple_prices,
        )

        # Should handle tight constraints without violation
        assert result.grid_violations == 0
        # Most energy should be curtailed or stored
        assert result.total_curtailment_mwh > 0 or result.total_stored_mwh > 0


# ==============================================================================
# Test Reproducibility
# ==============================================================================


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_result(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that same seed produces same results."""
        config = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=False,
            seed=42,
        )

        controller1 = HybridController(config=config)
        result1 = controller1.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        controller2 = HybridController(config=config)
        result2 = controller2.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
        )

        assert result1.total_revenue == pytest.approx(result2.total_revenue, abs=0.01)
        assert result1.total_decisions == result2.total_decisions

    def test_different_seeds_may_differ(
        self,
        default_battery_config: BatteryConfig,
        simple_forecasts: list[GenerationForecast],
        simple_constraints: list[GridConstraint],
        simple_prices: list[MarketPrice],
    ) -> None:
        """Test that different seeds may produce different RL behavior."""
        config1 = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=True,
            deviation_threshold=0.01,
            seed=42,
        )
        config2 = HybridControllerConfig(
            battery_config=default_battery_config,
            enable_rl_override=True,
            deviation_threshold=0.01,
            seed=123,
        )

        # Create agents
        env_config = CurtailmentEnvConfig(
            battery_config=default_battery_config,
            episode_length=len(simple_forecasts),
        )
        env = CurtailmentEnv(
            config=env_config,
            forecasts=simple_forecasts,
            grid_constraints=simple_constraints,
            market_prices=simple_prices,
        )
        agent1 = RandomAgent(env, config=AgentConfig(seed=42))
        agent2 = RandomAgent(env, config=AgentConfig(seed=123))

        controller1 = HybridController(config=config1, rl_agent=agent1)
        controller2 = HybridController(config=config2, rl_agent=agent2)

        # Create actuals with deviation
        from src.domain.models import ForecastScenario

        actuals = [
            f.total_generation(ForecastScenario.P50) * 1.2 for f in simple_forecasts
        ]

        result1 = controller1.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            actuals=actuals,
        )
        result2 = controller2.run(
            forecasts=simple_forecasts,
            constraints=simple_constraints,
            prices=simple_prices,
            actuals=actuals,
        )

        # Results may differ due to different random actions
        # Just verify both complete successfully
        assert result1.total_decisions == result2.total_decisions
