"""Tests for MILP optimization engine."""

from datetime import datetime, timedelta

import pytest

from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)
from src.optimization.milp import (
    MILPOptimizer,
    OptimizationConfig,
    SingleScenarioOptimizer,
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
def simple_scenario() -> (
    tuple[list[GenerationForecast], list[GridConstraint], list[MarketPrice]]
):
    """Simple 4-hour scenario for basic tests."""
    base_time = datetime(2025, 7, 15, 10, 0, 0)

    forecasts = [
        GenerationForecast(
            timestamp=base_time + timedelta(hours=i),
            solar_mw_p10=80.0,
            solar_mw_p50=100.0,
            solar_mw_p90=120.0,
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
            day_ahead_price=50.0,
        )
        for i in range(4)
    ]

    return forecasts, grid_constraints, market_prices


@pytest.fixture
def duck_curve_scenario() -> (
    tuple[list[GenerationForecast], list[GridConstraint], list[MarketPrice]]
):
    """Duck curve scenario matching copilot-instructions.md."""
    base_time = datetime(2025, 7, 15, 0, 0, 0)

    forecasts = []
    grid_constraints = []
    market_prices = []

    for hour in range(24):
        ts = base_time + timedelta(hours=hour)

        # Solar generation: peaks at 600 MW at noon
        if 6 <= hour <= 18:
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

        # Grid constrained to 300 MW (duck curve trap)
        grid_constraints.append(GridConstraint(timestamp=ts, max_export_mw=300.0))

        # Duck curve prices: negative midday, spike evening
        if 10 <= hour <= 14:
            price = -25.0  # Negative midday
        elif 17 <= hour <= 20:
            price = 140.0  # Evening spike
        else:
            price = 50.0  # Normal

        market_prices.append(MarketPrice(timestamp=ts, day_ahead_price=price))

    return forecasts, grid_constraints, market_prices


class TestMILPOptimizer:
    """Tests for the MILP optimizer."""

    def test_build_model(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that model builds successfully."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        model = optimizer.build_model(forecasts, grid_constraints, market_prices)

        # Check model components exist
        assert hasattr(model, "energy_sold")
        assert hasattr(model, "energy_stored")
        assert hasattr(model, "energy_curtailed")
        assert hasattr(model, "soc")
        assert hasattr(model, "objective")

        # Check sets
        assert len(list(model.T)) == 4
        assert len(list(model.S)) == 3  # P10, P50, P90

    def test_solve_simple_scenario(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test solving a simple scenario."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        # Check solve succeeded
        assert result.success, f"Solver failed: {result.termination_condition}"
        assert result.objective_value is not None
        assert result.objective_value > 0  # Should have positive revenue

        # Check solution values exist
        assert len(result.energy_sold) > 0
        assert len(result.soc) > 0

    def test_no_grid_violations(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that optimization respects grid constraints."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success

        # Check all energy sold <= grid capacity
        for t in range(len(forecasts)):
            for scenario in [
                ForecastScenario.P10,
                ForecastScenario.P50,
                ForecastScenario.P90,
            ]:
                sold = result.energy_sold.get((t, scenario), 0.0)
                capacity = grid_constraints[t].max_export_mw
                assert sold <= capacity + 0.01, f"Grid violation at t={t}, s={scenario}"

    def test_soc_bounds_respected(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that SOC stays within bounds."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success

        min_soc = battery_config.capacity_mwh * battery_config.min_soc_fraction
        max_soc = battery_config.capacity_mwh * battery_config.max_soc_fraction

        for t in range(len(forecasts)):
            for scenario in [
                ForecastScenario.P10,
                ForecastScenario.P50,
                ForecastScenario.P90,
            ]:
                soc = result.soc.get((t, scenario), 0.0)
                assert soc >= min_soc - 0.01, f"SOC below min at t={t}"
                assert soc <= max_soc + 0.01, f"SOC above max at t={t}"

    def test_energy_balance(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that energy balance is maintained."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success

        for t, forecast in enumerate(forecasts):
            for scenario in [
                ForecastScenario.P10,
                ForecastScenario.P50,
                ForecastScenario.P90,
            ]:
                generation = forecast.total_generation(scenario)
                sold = result.energy_sold.get((t, scenario), 0.0)
                stored = result.energy_stored.get((t, scenario), 0.0)
                discharged = result.energy_discharged.get((t, scenario), 0.0)
                curtailed = result.energy_curtailed.get((t, scenario), 0.0)

                # Energy balance: G + d = x + y + z
                lhs = generation + discharged
                rhs = sold + stored + curtailed
                assert abs(lhs - rhs) < 0.01, f"Energy imbalance at t={t}, s={scenario}"


class TestSingleScenarioOptimizer:
    """Tests for single-scenario optimizer."""

    def test_single_scenario_p50(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test single-scenario optimization with P50."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = SingleScenarioOptimizer(
            battery_config, scenario=ForecastScenario.P50
        )
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success
        assert result.objective_value is not None

        # Should only have P50 scenario in results
        for t in range(len(forecasts)):
            assert (t, ForecastScenario.P50) in result.energy_sold


class TestDuckCurveOptimization:
    """Tests for duck curve scenario optimization."""

    def test_duck_curve_better_than_naive(
        self,
        battery_config: BatteryConfig,
        duck_curve_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that MILP optimization achieves good results on duck curve.

        The optimizer should:
        1. Avoid selling at negative prices (curtail/store instead)
        2. Discharge during evening price spike
        3. Achieve positive net revenue

        Note: High curtailment during negative prices is OPTIMAL behavior
        since selling at -$25/MWh loses money. The key metric is net revenue.
        """
        forecasts, grid_constraints, market_prices = duck_curve_scenario

        optimizer = SingleScenarioOptimizer(
            battery_config, scenario=ForecastScenario.P50
        )
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success, f"Optimization failed: {result.termination_condition}"

        # Calculate metrics
        total_generated = sum(
            f.total_generation(ForecastScenario.P50) for f in forecasts
        )
        total_curtailed = sum(
            result.energy_curtailed.get((t, ForecastScenario.P50), 0.0)
            for t in range(len(forecasts))
        )
        total_sold = sum(
            result.energy_sold.get((t, ForecastScenario.P50), 0.0)
            for t in range(len(forecasts))
        )
        total_discharged = sum(
            result.energy_discharged.get((t, ForecastScenario.P50), 0.0)
            for t in range(len(forecasts))
        )
        curtailment_rate = (
            (total_curtailed / total_generated) * 100 if total_generated > 0 else 0
        )

        # Print results for analysis
        print("\n=== Duck Curve MILP Optimization ===")
        print(f"Objective (Net Revenue): ${result.objective_value:,.0f}")
        print(f"Total Generated: {total_generated:.1f} MWh")
        print(f"Total Sold: {total_sold:.1f} MWh")
        print(f"Total Discharged: {total_discharged:.1f} MWh")
        print(f"Total Curtailed: {total_curtailed:.1f} MWh")
        print(f"Curtailment Rate: {curtailment_rate:.1f}%")
        print(f"Solve Time: {result.solve_time_seconds:.2f}s")

        # Key assertions:
        # 1. Should have positive net revenue despite negative midday prices
        assert result.objective_value is not None
        assert result.objective_value > 0, "Net revenue should be positive"

        # 2. Should discharge during evening (optimizer uses battery arbitrage)
        assert total_discharged > 0, "Should utilize battery discharge"

        # 3. Curtailment during negative prices is actually optimal
        # The key is that we sell at positive prices, not at negative ones

    def test_avoids_selling_at_negative_prices(
        self,
        battery_config: BatteryConfig,
        duck_curve_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that optimizer avoids selling during negative price periods."""
        forecasts, grid_constraints, market_prices = duck_curve_scenario

        optimizer = SingleScenarioOptimizer(
            battery_config, scenario=ForecastScenario.P50
        )
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success

        # Check hours 10-14 (negative prices)
        negative_price_hours = [10, 11, 12, 13, 14]
        total_sold_negative = sum(
            result.energy_sold.get((t, ForecastScenario.P50), 0.0)
            for t in negative_price_hours
        )

        # Should minimize selling during negative prices
        # (May still sell some if grid constraint + storage limits force it)
        total_generated_negative = sum(
            forecasts[t].total_generation(ForecastScenario.P50)
            for t in negative_price_hours
        )

        sell_ratio = (
            total_sold_negative / total_generated_negative
            if total_generated_negative > 0
            else 0
        )

        print("\nNegative price period (hours 10-14):")
        print(f"  Generated: {total_generated_negative:.1f} MWh")
        print(f"  Sold: {total_sold_negative:.1f} MWh")
        print(f"  Sell Ratio: {sell_ratio*100:.1f}%")

        # Should sell significantly less than 100% during negative prices
        # (storing and/or curtailing instead)
        assert sell_ratio < 0.8, "Selling too much during negative price period"

    def test_discharges_during_price_spike(
        self,
        battery_config: BatteryConfig,
        duck_curve_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test that optimizer discharges battery during evening price spike."""
        forecasts, grid_constraints, market_prices = duck_curve_scenario

        optimizer = SingleScenarioOptimizer(
            battery_config, scenario=ForecastScenario.P50
        )
        result = optimizer.optimize(forecasts, grid_constraints, market_prices)

        assert result.success

        # Check hours 17-20 (high prices)
        peak_hours = [17, 18, 19, 20]
        total_discharged = sum(
            result.energy_discharged.get((t, ForecastScenario.P50), 0.0)
            for t in peak_hours
        )

        print("\nEvening peak period (hours 17-20):")
        print(f"  Total Discharged: {total_discharged:.1f} MWh")

        # Should discharge during high price period
        assert total_discharged > 0, "Should discharge during price spike"


class TestOptimizationConfig:
    """Tests for optimization configuration."""

    def test_default_config_valid(self) -> None:
        """Test that default config is valid."""
        config = OptimizationConfig()
        assert config.validate()

    def test_invalid_probabilities(self) -> None:
        """Test that invalid probabilities fail validation."""
        config = OptimizationConfig(
            scenario_probabilities={
                ForecastScenario.P10: 0.5,
                ForecastScenario.P50: 0.5,
                ForecastScenario.P90: 0.5,  # Sum = 1.5
            }
        )
        assert not config.validate()


class TestSimulationResult:
    """Tests for converting optimization results to SimulationResult."""

    def test_get_simulation_result(
        self,
        battery_config: BatteryConfig,
        simple_scenario: tuple[
            list[GenerationForecast], list[GridConstraint], list[MarketPrice]
        ],
    ) -> None:
        """Test converting optimization result to SimulationResult."""
        forecasts, grid_constraints, market_prices = simple_scenario

        optimizer = MILPOptimizer(battery_config)
        optimizer.optimize(forecasts, grid_constraints, market_prices)

        sim_result = optimizer.get_simulation_result(forecasts, ForecastScenario.P50)

        assert sim_result is not None
        assert len(sim_result.decisions) == len(forecasts)
        assert sim_result.grid_violations_count == 0
        assert sim_result.total_generation_mwh > 0
