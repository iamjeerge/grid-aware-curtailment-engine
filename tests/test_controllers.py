"""Tests for naive heuristic controller."""

from datetime import datetime, timedelta

import pytest

from src.controllers.naive import NaiveController, NaiveWithDischargeController
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)


@pytest.fixture
def battery_config() -> BatteryConfig:
    """Standard battery configuration for tests."""
    return BatteryConfig(
        capacity_mwh=500.0,
        max_power_mw=150.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_fraction=0.1,
        max_soc_fraction=0.9,
        degradation_cost_per_mwh=8.0,
    )


@pytest.fixture
def controller(battery_config: BatteryConfig) -> NaiveController:
    """Initialized naive controller."""
    ctrl = NaiveController(battery_config)
    ctrl.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)
    return ctrl


class TestNaiveController:
    """Tests for basic naive controller."""

    def test_initialization(self, battery_config: BatteryConfig) -> None:
        """Test controller initialization."""
        controller = NaiveController(battery_config)
        controller.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)

        state = controller.battery.current_state
        assert state is not None
        assert state.soc_fraction == pytest.approx(0.5, rel=0.01)

    def test_uninitialized_raises(self, battery_config: BatteryConfig) -> None:
        """Test that uninitialized controller raises error."""
        controller = NaiveController(battery_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            controller.dispatch(
                timestamp=datetime(2025, 7, 15, 12, 0, 0),
                generation_mw=100.0,
                grid_constraint=GridConstraint(
                    timestamp=datetime(2025, 7, 15, 12, 0, 0),
                    max_export_mw=300.0,
                ),
                market_price=MarketPrice(
                    timestamp=datetime(2025, 7, 15, 12, 0, 0),
                    day_ahead_price=50.0,
                ),
            )

    def test_sell_under_grid_limit(self, controller: NaiveController) -> None:
        """Test selling when generation is under grid limit."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=200.0,  # Under 300 MW limit
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=50.0),
        )

        # Should sell all generation
        assert decision.energy_sold_mw == 200.0
        assert decision.energy_stored_mw == 0.0
        assert decision.energy_curtailed_mw == 0.0
        assert decision.revenue_dollars == 200.0 * 50.0

    def test_store_excess_when_grid_limited(self, controller: NaiveController) -> None:
        """Test storing excess when grid is constrained."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=400.0,  # Over 300 MW limit
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=50.0),
        )

        # Should sell 300, store 100
        assert decision.energy_sold_mw == 300.0
        assert decision.energy_stored_mw == 100.0
        assert decision.energy_curtailed_mw == 0.0

    def test_curtail_when_storage_full(self, battery_config: BatteryConfig) -> None:
        """Test curtailment when battery is nearly full."""
        controller = NaiveController(battery_config)
        # Start at 85% SOC (only 5% headroom = 25 MWh)
        controller.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.85)

        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=500.0,  # Way over grid limit
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=50.0),
        )

        # Should sell 300, store limited amount, curtail rest
        assert decision.energy_sold_mw == 300.0
        # Can only store ~26.3 MW (25 MWh headroom / 0.95 efficiency)
        assert decision.energy_stored_mw < 30.0
        assert decision.energy_curtailed_mw > 150.0

    def test_sell_even_at_negative_price(self, controller: NaiveController) -> None:
        """Test that naive controller sells even at negative prices."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=200.0,
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=-25.0),
        )

        # Naive controller doesn't consider negative prices
        assert decision.energy_sold_mw == 200.0
        assert decision.revenue_dollars == 200.0 * -25.0  # Negative revenue!

    def test_degradation_cost_tracking(self, controller: NaiveController) -> None:
        """Test that degradation costs are tracked."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=400.0,  # Will store 100 MW
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=50.0),
        )

        # 100 MWh stored * $8/MWh = $800 degradation
        assert decision.degradation_cost_dollars == 100.0 * 8.0

    def test_never_discharges(self, controller: NaiveController) -> None:
        """Test that basic naive controller never discharges."""
        ts = datetime(2025, 7, 15, 19, 0, 0)  # Evening peak

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=50.0,  # Low generation
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=140.0),  # High price
        )

        # Naive doesn't discharge even at high prices
        assert decision.battery_discharge_mw == 0.0
        assert decision.energy_sold_mw == 50.0


class TestNaiveControllerSimulation:
    """Tests for simulation functionality."""

    def test_run_simulation_basic(self, controller: NaiveController) -> None:
        """Test running a basic simulation."""
        base_time = datetime(2025, 7, 15, 10, 0, 0)

        # Create 4 hours of data
        forecasts = [
            GenerationForecast(
                timestamp=base_time + timedelta(hours=i),
                solar_mw_p10=80.0 + i * 10,
                solar_mw_p50=100.0 + i * 10,
                solar_mw_p90=120.0 + i * 10,
            )
            for i in range(4)
        ]

        grid_constraints = [
            GridConstraint(
                timestamp=base_time + timedelta(hours=i),
                max_export_mw=300.0,
            )
            for i in range(4)
        ]

        market_prices = [
            MarketPrice(
                timestamp=base_time + timedelta(hours=i),
                day_ahead_price=50.0 + i * 10,
            )
            for i in range(4)
        ]

        # Reset controller for fresh simulation
        controller.reset(base_time, initial_soc_fraction=0.5)

        result = controller.run_simulation(
            forecasts=forecasts,
            grid_constraints=grid_constraints,
            market_prices=market_prices,
            scenario=ForecastScenario.P50,
        )

        # Check basic results
        assert result.scenario == ForecastScenario.P50
        assert len(result.decisions) == 4
        assert result.total_generation_mwh == 100 + 110 + 120 + 130
        assert result.total_sold_mwh == result.total_generation_mwh
        assert result.total_curtailed_mwh == 0.0
        assert result.grid_violations_count == 0

    def test_simulation_with_curtailment(self, battery_config: BatteryConfig) -> None:
        """Test simulation that results in curtailment."""
        controller = NaiveController(battery_config)
        base_time = datetime(2025, 7, 15, 10, 0, 0)
        controller.initialize(base_time, initial_soc_fraction=0.85)  # Near full

        # High generation, low grid capacity
        forecasts = [
            GenerationForecast(
                timestamp=base_time + timedelta(hours=i),
                solar_mw_p10=400.0,
                solar_mw_p50=500.0,
                solar_mw_p90=600.0,
            )
            for i in range(4)
        ]

        grid_constraints = [
            GridConstraint(
                timestamp=base_time + timedelta(hours=i),
                max_export_mw=300.0,  # Heavily constrained
            )
            for i in range(4)
        ]

        market_prices = [
            MarketPrice(
                timestamp=base_time + timedelta(hours=i),
                day_ahead_price=50.0,
            )
            for i in range(4)
        ]

        result = controller.run_simulation(
            forecasts=forecasts,
            grid_constraints=grid_constraints,
            market_prices=market_prices,
            scenario=ForecastScenario.P50,
        )

        # Should have significant curtailment
        assert result.total_curtailed_mwh > 0
        assert result.curtailment_rate > 20  # > 20% curtailment (it's in %)

    def test_simulation_mismatched_lengths_raises(
        self, controller: NaiveController
    ) -> None:
        """Test that mismatched input lengths raise error."""
        base_time = datetime(2025, 7, 15, 10, 0, 0)

        forecasts = [
            GenerationForecast(
                timestamp=base_time + timedelta(hours=i),
                solar_mw_p10=100.0,
                solar_mw_p50=100.0,
                solar_mw_p90=100.0,
            )
            for i in range(4)
        ]

        grid_constraints = [
            GridConstraint(
                timestamp=base_time + timedelta(hours=i),
                max_export_mw=300.0,
            )
            for i in range(3)  # Only 3!
        ]

        market_prices = [
            MarketPrice(
                timestamp=base_time + timedelta(hours=i),
                day_ahead_price=50.0,
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="same length"):
            controller.run_simulation(
                forecasts, grid_constraints, market_prices, ForecastScenario.P50
            )


class TestNaiveWithDischargeController:
    """Tests for naive controller with discharge heuristic."""

    def test_discharge_at_high_price(self, battery_config: BatteryConfig) -> None:
        """Test that controller discharges when price is high."""
        controller = NaiveWithDischargeController(
            battery_config,
            discharge_price_threshold=100.0,
        )
        controller.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)

        ts = datetime(2025, 7, 15, 19, 0, 0)  # Evening

        initial_state = controller.battery.current_state
        assert initial_state is not None
        initial_soc = initial_state.soc_mwh

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=100.0,  # Low generation, plenty of grid capacity
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=140.0),  # High!
        )

        # Should discharge to fill grid capacity
        assert decision.battery_discharge_mw > 0
        # Sold = generation + discharge (up to grid limit)
        assert decision.energy_sold_mw == 100.0 + decision.battery_discharge_mw

        # Verify SOC decreased
        final_state = controller.battery.current_state
        assert final_state is not None
        assert final_state.soc_mwh < initial_soc

    def test_no_discharge_at_low_price(self, battery_config: BatteryConfig) -> None:
        """Test that controller doesn't discharge when price is low."""
        controller = NaiveWithDischargeController(
            battery_config,
            discharge_price_threshold=100.0,
        )
        controller.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)

        ts = datetime(2025, 7, 15, 12, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=100.0,
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=50.0),  # Low price
        )

        assert decision.battery_discharge_mw == 0.0

    def test_discharge_degradation_cost(self, battery_config: BatteryConfig) -> None:
        """Test that discharge degradation is tracked."""
        controller = NaiveWithDischargeController(
            battery_config,
            degradation_cost_per_mwh=8.0,
            discharge_price_threshold=100.0,
        )
        controller.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)

        ts = datetime(2025, 7, 15, 19, 0, 0)

        decision = controller.dispatch(
            timestamp=ts,
            generation_mw=100.0,
            grid_constraint=GridConstraint(timestamp=ts, max_export_mw=300.0),
            market_price=MarketPrice(timestamp=ts, day_ahead_price=140.0),
        )

        # Degradation should include discharge cost
        expected_degradation = decision.battery_discharge_mw * 8.0
        assert decision.degradation_cost_dollars == pytest.approx(
            expected_degradation, rel=0.01
        )


class TestDuckCurveScenario:
    """Test duck curve scenario from copilot-instructions.md."""

    def test_duck_curve_naive_baseline(self, battery_config: BatteryConfig) -> None:
        """Test naive controller on duck curve scenario.

        Expected baseline (from instructions):
        - 32% curtailment
        - ~$420k revenue

        This establishes the benchmark for optimization.
        """
        controller = NaiveController(battery_config)
        base_time = datetime(2025, 7, 15, 0, 0, 0)
        controller.initialize(base_time, initial_soc_fraction=0.5)

        # Build duck curve scenario (24 hours)
        forecasts = []
        grid_constraints = []
        market_prices = []

        for hour in range(24):
            ts = base_time + timedelta(hours=hour)

            # Solar generation: peaks at noon
            if 6 <= hour <= 18:
                # Bell curve peaking at 600 MW at noon
                hours_from_noon = abs(hour - 12)
                generation = 600.0 * max(0.0, 1.0 - (hours_from_noon / 6.0) ** 2)
            else:
                generation = 0.0

            forecasts.append(
                GenerationForecast(
                    timestamp=ts,
                    solar_mw_p10=generation * 0.8,
                    solar_mw_p50=generation,
                    solar_mw_p90=generation * 1.2,
                )
            )

            # Grid constrained to 300 MW
            grid_constraints.append(GridConstraint(timestamp=ts, max_export_mw=300.0))

            # Duck curve prices: negative midday, spike evening
            if 10 <= hour <= 14:
                price = -25.0  # Negative midday
            elif 17 <= hour <= 20:
                price = 140.0  # Evening spike
            else:
                price = 50.0  # Normal

            market_prices.append(MarketPrice(timestamp=ts, day_ahead_price=price))

        result = controller.run_simulation(
            forecasts=forecasts,
            grid_constraints=grid_constraints,
            market_prices=market_prices,
            scenario=ForecastScenario.P50,
        )

        # Duck curve should produce significant curtailment
        # Solar peaks at 600 MW but grid limited to 300 MW
        assert result.total_curtailed_mwh > 0, "Should have curtailment"
        assert result.curtailment_rate > 15, "Should have >15% curtailment rate"

        # Net revenue will be lower due to:
        # - Selling during negative prices
        # - Not discharging during price spikes
        # - Degradation from charging
        print("\n=== Duck Curve Naive Baseline ===")
        print(f"Total Generated: {result.total_generation_mwh:.1f} MWh")
        print(f"Total Sold: {result.total_sold_mwh:.1f} MWh")
        print(f"Total Curtailed: {result.total_curtailed_mwh:.1f} MWh")
        print(f"Curtailment Rate: {result.curtailment_rate:.1f}%")
        print(f"Gross Revenue: ${result.total_revenue_dollars:,.0f}")
        print(f"Degradation Cost: ${result.total_degradation_cost_dollars:,.0f}")
        print(f"Net Revenue: ${result.net_revenue:,.0f}")
        print(f"Grid Violations: {result.grid_violations_count}")
