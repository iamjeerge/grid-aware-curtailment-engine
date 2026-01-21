"""Monte Carlo stress testing engine.

Runs Monte Carlo simulations to stress test the optimization strategy
across various uncertainty sources:
- Forecast errors (generation uncertainty)
- Price volatility
- Grid outages and constraint changes

This module helps validate robustness and estimate performance distributions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.random import Generator

from src.battery.physics import BatteryModel
from src.controllers.naive import NaiveWithDischargeController
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
)
from src.optimization.milp import MILPOptimizer, OptimizationConfig


class StressScenario(str, Enum):
    """Types of stress scenarios for Monte Carlo simulation."""

    FORECAST_ERROR = "forecast_error"  # Generation deviates from forecast
    PRICE_VOLATILITY = "price_volatility"  # Price spikes and drops
    GRID_OUTAGE = "grid_outage"  # Sudden capacity reductions
    COMBINED = "combined"  # All uncertainties combined


@dataclass
class StressTestConfig:
    """Configuration for Monte Carlo stress testing.

    Attributes:
        n_simulations: Number of Monte Carlo runs.
        seed: Random seed for reproducibility.
        scenarios: Which stress scenarios to simulate.
        forecast_error_std: Std dev of forecast error as fraction of forecast.
        price_volatility_std: Std dev of price deviation as fraction of price.
        grid_outage_probability: Probability of grid outage per timestep.
        grid_outage_severity: Capacity reduction during outage (0.0-1.0).
        use_milp: Whether to test MILP optimizer (vs naive).
    """

    n_simulations: int = 100
    seed: int = 42
    scenarios: list[StressScenario] = field(
        default_factory=lambda: [StressScenario.COMBINED]
    )

    # Forecast error parameters
    forecast_error_std: float = 0.15  # 15% standard deviation
    forecast_error_correlation: float = 0.5  # Temporal correlation

    # Price volatility parameters
    price_volatility_std: float = 0.20  # 20% standard deviation
    price_spike_probability: float = 0.05  # 5% chance of spike
    price_spike_magnitude: float = 2.0  # 2x multiplier for spikes

    # Grid outage parameters
    grid_outage_probability: float = 0.02  # 2% per timestep
    grid_outage_severity: float = 0.5  # 50% capacity reduction
    grid_outage_duration: int = 3  # Hours

    # Controller selection
    use_milp: bool = True
    milp_config: OptimizationConfig | None = None


@dataclass
class SimulationRun:
    """Results from a single Monte Carlo simulation run."""

    run_id: int
    scenario_type: StressScenario

    # Realized values (with uncertainty)
    realized_generation: list[float] = field(default_factory=list)
    realized_prices: list[float] = field(default_factory=list)
    realized_grid_capacity: list[float] = field(default_factory=list)

    # Decision outcomes
    total_sold_mwh: float = 0.0
    total_stored_mwh: float = 0.0
    total_curtailed_mwh: float = 0.0
    total_discharged_mwh: float = 0.0

    # Financial outcomes
    revenue: float = 0.0
    degradation_cost: float = 0.0
    net_profit: float = 0.0

    # Constraint violations
    grid_violations: int = 0
    soc_violations: int = 0

    # Final state
    final_soc: float = 0.0


@dataclass
class StressTestResult:
    """Aggregated results from Monte Carlo stress testing."""

    config: StressTestConfig
    n_runs_completed: int
    total_runtime_seconds: float

    # Individual run results
    runs: list[SimulationRun] = field(default_factory=list)

    # Aggregated statistics
    mean_profit: float = 0.0
    std_profit: float = 0.0
    min_profit: float = 0.0
    max_profit: float = 0.0
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0

    # Curtailment statistics
    mean_curtailment_rate: float = 0.0
    max_curtailment_rate: float = 0.0

    # Violation statistics
    violation_rate: float = 0.0  # Fraction of runs with any violation
    mean_violations_per_run: float = 0.0

    def compute_statistics(self) -> None:
        """Compute aggregated statistics from individual runs."""
        if not self.runs:
            return

        profits = np.array([r.net_profit for r in self.runs])

        self.mean_profit = float(np.mean(profits))
        self.std_profit = float(np.std(profits))
        self.min_profit = float(np.min(profits))
        self.max_profit = float(np.max(profits))
        self.percentile_5 = float(np.percentile(profits, 5))
        self.percentile_25 = float(np.percentile(profits, 25))
        self.percentile_50 = float(np.percentile(profits, 50))
        self.percentile_75 = float(np.percentile(profits, 75))
        self.percentile_95 = float(np.percentile(profits, 95))

        # Curtailment
        total_gen = [sum(r.realized_generation) for r in self.runs]
        curtailment_rates = [
            r.total_curtailed_mwh / max(g, 1e-6)
            for r, g in zip(self.runs, total_gen, strict=True)
        ]
        self.mean_curtailment_rate = float(np.mean(curtailment_rates))
        self.max_curtailment_rate = float(np.max(curtailment_rates))

        # Violations
        violations_per_run = [r.grid_violations + r.soc_violations for r in self.runs]
        self.violation_rate = float(np.mean([v > 0 for v in violations_per_run]))
        self.mean_violations_per_run = float(np.mean(violations_per_run))

    @property
    def profit_at_risk(self) -> float:
        """Profit at risk (5th percentile)."""
        return self.percentile_5

    @property
    def sharpe_ratio(self) -> float:
        """Simple Sharpe-like ratio (mean / std)."""
        if self.std_profit > 0:
            return self.mean_profit / self.std_profit
        return float("inf") if self.mean_profit > 0 else 0.0


class MonteCarloEngine:
    """Monte Carlo simulation engine for stress testing.

    Runs multiple simulations with randomized uncertainty to estimate
    the distribution of outcomes for the optimization strategy.
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        config: StressTestConfig | None = None,
    ) -> None:
        """Initialize the Monte Carlo engine.

        Args:
            battery_config: Battery configuration parameters.
            config: Stress test configuration.
        """
        self.battery_config = battery_config
        self.config = config or StressTestConfig()
        self._rng: Generator = np.random.default_rng(self.config.seed)

    def run_stress_test(
        self,
        base_forecasts: list[GenerationForecast],
        base_grid_constraints: list[GridConstraint],
        base_market_prices: list[MarketPrice],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> StressTestResult:
        """Run Monte Carlo stress test.

        Args:
            base_forecasts: Baseline generation forecasts.
            base_grid_constraints: Baseline grid constraints.
            base_market_prices: Baseline market prices.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            StressTestResult with aggregated statistics.
        """
        import time

        start_time = time.time()

        result = StressTestResult(
            config=self.config,
            n_runs_completed=0,
            total_runtime_seconds=0.0,
        )

        for run_id in range(self.config.n_simulations):
            for scenario in self.config.scenarios:
                sim_run = self._run_single_simulation(
                    run_id=run_id,
                    scenario=scenario,
                    base_forecasts=base_forecasts,
                    base_grid_constraints=base_grid_constraints,
                    base_market_prices=base_market_prices,
                )
                result.runs.append(sim_run)

            result.n_runs_completed = run_id + 1

            if progress_callback:
                progress_callback(run_id + 1, self.config.n_simulations)

        result.total_runtime_seconds = time.time() - start_time
        result.compute_statistics()

        return result

    def _run_single_simulation(
        self,
        run_id: int,
        scenario: StressScenario,
        base_forecasts: list[GenerationForecast],
        base_grid_constraints: list[GridConstraint],
        base_market_prices: list[MarketPrice],
    ) -> SimulationRun:
        """Run a single Monte Carlo simulation.

        Args:
            run_id: Simulation run identifier.
            scenario: Type of stress scenario.
            base_forecasts: Baseline forecasts.
            base_grid_constraints: Baseline grid constraints.
            base_market_prices: Baseline market prices.

        Returns:
            SimulationRun with outcomes.
        """
        # Generate realized values with uncertainty
        realized_gen = self._generate_realized_generation(base_forecasts, scenario)
        realized_prices = self._generate_realized_prices(base_market_prices, scenario)
        realized_capacity = self._generate_realized_capacity(
            base_grid_constraints, scenario
        )

        # Initialize simulation run
        sim_run = SimulationRun(
            run_id=run_id,
            scenario_type=scenario,
            realized_generation=realized_gen,
            realized_prices=realized_prices,
            realized_grid_capacity=realized_capacity,
        )

        # Get optimized decisions using P50 forecast (plan)
        if self.config.use_milp:
            decisions = self._get_milp_decisions(
                base_forecasts, base_grid_constraints, base_market_prices
            )
        else:
            decisions = self._get_naive_decisions(
                base_forecasts, base_grid_constraints, base_market_prices
            )

        # Simulate execution with realized values
        self._simulate_execution(
            sim_run, decisions, realized_gen, realized_prices, realized_capacity
        )

        return sim_run

    def _generate_realized_generation(
        self,
        forecasts: list[GenerationForecast],
        scenario: StressScenario,
    ) -> list[float]:
        """Generate realized generation with forecast error."""
        n = len(forecasts)

        # Get P50 forecast as baseline
        baseline = [f.total_generation(ForecastScenario.P50) for f in forecasts]

        if scenario in [StressScenario.FORECAST_ERROR, StressScenario.COMBINED]:
            # Generate correlated errors
            errors = self._generate_correlated_noise(
                n,
                self.config.forecast_error_std,
                self.config.forecast_error_correlation,
            )

            # Apply multiplicative error
            realized = [
                max(0, b * (1 + e)) for b, e in zip(baseline, errors, strict=True)
            ]
        else:
            realized = baseline

        return realized

    def _generate_realized_prices(
        self,
        prices: list[MarketPrice],
        scenario: StressScenario,
    ) -> list[float]:
        """Generate realized prices with volatility."""
        n = len(prices)
        baseline = [p.effective_price for p in prices]

        if scenario in [StressScenario.PRICE_VOLATILITY, StressScenario.COMBINED]:
            # Generate price shocks
            shocks = self._rng.normal(0, self.config.price_volatility_std, n)

            # Add occasional spikes
            spike_mask = self._rng.random(n) < self.config.price_spike_probability
            spike_direction = self._rng.choice([-1, 1], n)
            spikes = (
                spike_mask * spike_direction * (self.config.price_spike_magnitude - 1)
            )

            realized = [
                b * (1 + s + sp)
                for b, s, sp in zip(baseline, shocks, spikes, strict=True)
            ]
        else:
            realized = baseline

        return realized

    def _generate_realized_capacity(
        self,
        constraints: list[GridConstraint],
        scenario: StressScenario,
    ) -> list[float]:
        """Generate realized grid capacity with outages."""
        n = len(constraints)
        baseline = [c.max_export_mw for c in constraints]

        if scenario in [StressScenario.GRID_OUTAGE, StressScenario.COMBINED]:
            realized = list(baseline)  # Copy

            t = 0
            while t < n:
                if self._rng.random() < self.config.grid_outage_probability:
                    # Outage occurs
                    duration = min(self.config.grid_outage_duration, n - t)
                    severity = self.config.grid_outage_severity

                    for dt in range(duration):
                        realized[t + dt] *= 1 - severity

                    t += duration
                else:
                    t += 1
        else:
            realized = baseline

        return realized

    def _generate_correlated_noise(
        self,
        n: int,
        std: float,
        correlation: float,
    ) -> list[float]:
        """Generate temporally correlated noise using AR(1) process."""
        noise = [self._rng.normal(0, std)]

        for _ in range(1, n):
            innovation = self._rng.normal(0, std * np.sqrt(1 - correlation**2))
            noise.append(correlation * noise[-1] + innovation)

        return noise

    def _get_milp_decisions(
        self,
        forecasts: list[GenerationForecast],
        constraints: list[GridConstraint],
        prices: list[MarketPrice],
    ) -> list[OptimizationDecision]:
        """Get decisions from MILP optimizer."""
        config = self.config.milp_config or OptimizationConfig()
        optimizer = MILPOptimizer(self.battery_config, config)

        # Build and solve for P50 scenario only (deterministic plan)
        optimizer.build_model(
            forecasts, constraints, prices, scenarios=[ForecastScenario.P50]
        )

        result = optimizer.solve()

        if not result.success:
            # Fallback to naive if MILP fails
            return self._get_naive_decisions(forecasts, constraints, prices)

        # Extract decisions
        decisions = []
        for t in range(len(forecasts)):
            key = (t, ForecastScenario.P50)
            generation = forecasts[t].total_generation(ForecastScenario.P50)
            energy_sold = result.energy_sold.get(key, 0.0)
            energy_stored = result.energy_stored.get(key, 0.0)
            energy_discharged = result.energy_discharged.get(key, 0.0)
            energy_curtailed = result.energy_curtailed.get(key, 0.0)
            soc = result.soc.get(key, 0.0)

            price = prices[t].effective_price
            revenue = energy_sold * price
            degradation = (
                energy_stored + energy_discharged
            ) * self.battery_config.degradation_cost_per_mwh

            decisions.append(
                OptimizationDecision(
                    timestamp=forecasts[t].timestamp,
                    scenario=ForecastScenario.P50,
                    generation_mw=generation,
                    energy_sold_mw=energy_sold,
                    energy_stored_mw=energy_stored,
                    energy_curtailed_mw=energy_curtailed,
                    battery_discharge_mw=energy_discharged,
                    resulting_soc_mwh=soc,
                    revenue_dollars=revenue,
                    degradation_cost_dollars=degradation,
                )
            )

        return decisions

    def _get_naive_decisions(
        self,
        forecasts: list[GenerationForecast],
        constraints: list[GridConstraint],
        prices: list[MarketPrice],
    ) -> list[OptimizationDecision]:
        """Get decisions from naive controller."""
        controller = NaiveWithDischargeController(self.battery_config)
        controller.initialize(
            start_time=forecasts[0].timestamp,
            initial_soc_fraction=0.5,
        )

        decisions = []
        for t in range(len(forecasts)):
            generation = forecasts[t].total_generation(ForecastScenario.P50)

            decision = controller.dispatch(
                timestamp=forecasts[t].timestamp,
                generation_mw=generation,
                grid_constraint=constraints[t],
                market_price=prices[t],
                scenario=ForecastScenario.P50,
            )
            decisions.append(decision)

        return decisions

    def _simulate_execution(
        self,
        sim_run: SimulationRun,
        planned_decisions: list[OptimizationDecision],
        realized_gen: list[float],
        realized_prices: list[float],
        realized_capacity: list[float],
    ) -> None:
        """Simulate execution with realized values.

        Adjusts planned decisions based on realized generation and constraints.
        """
        battery = BatteryModel(self.battery_config)
        battery.initialize(
            timestamp=planned_decisions[0].timestamp,
            initial_soc_fraction=0.5,
        )

        for _t, (decision, gen, price, cap) in enumerate(
            zip(
                planned_decisions,
                realized_gen,
                realized_prices,
                realized_capacity,
                strict=True,
            )
        ):
            state = battery.current_state

            # Respect realized grid capacity
            actual_sell = min(decision.energy_sold_mw, cap, gen)

            # Check for grid violation (planned vs actual capacity)
            if decision.energy_sold_mw > cap:
                sim_run.grid_violations += 1

            # Remaining energy after selling
            remaining = gen - actual_sell

            # Store what we can
            max_charge = min(
                remaining,
                self.battery_config.max_power_mw,
                (self.battery_config.capacity_mwh - state.soc_mwh)
                / self.battery_config.charge_efficiency,
            )
            actual_store = max(0, min(decision.energy_stored_mw, max_charge))

            # Curtail the rest
            actual_curtail = max(0, remaining - actual_store)

            # Discharge if needed and possible (for selling during high prices)
            actual_discharge = 0.0
            if price > 0 and actual_sell < cap:
                discharge_headroom = cap - actual_sell
                max_discharge = min(
                    discharge_headroom,
                    self.battery_config.max_power_mw,
                    state.soc_mwh * self.battery_config.discharge_efficiency,
                )
                actual_discharge = min(decision.battery_discharge_mw, max_discharge)
                actual_sell += actual_discharge

            # Update battery
            if actual_store > 0:
                _, charged = battery.charge(actual_store, 1.0, decision.timestamp)
                if charged < actual_store - 0.01:
                    sim_run.soc_violations += 1

            if actual_discharge > 0:
                _, discharged = battery.discharge(
                    actual_discharge, 1.0, decision.timestamp
                )
                if discharged < actual_discharge - 0.01:
                    sim_run.soc_violations += 1

            # Accumulate totals
            sim_run.total_sold_mwh += actual_sell
            sim_run.total_stored_mwh += actual_store
            sim_run.total_curtailed_mwh += actual_curtail
            sim_run.total_discharged_mwh += actual_discharge

            # Financial
            sim_run.revenue += price * actual_sell
            sim_run.degradation_cost += self.battery_config.degradation_cost_per_mwh * (
                actual_store + actual_discharge
            )

        sim_run.net_profit = sim_run.revenue - sim_run.degradation_cost
        sim_run.final_soc = (
            battery.current_state.soc_mwh if battery.current_state else 0.0
        )


def run_quick_stress_test(
    battery_config: BatteryConfig,
    forecasts: list[GenerationForecast],
    grid_constraints: list[GridConstraint],
    market_prices: list[MarketPrice],
    n_simulations: int = 50,
    seed: int = 42,
) -> StressTestResult:
    """Convenience function for quick stress testing.

    Args:
        battery_config: Battery configuration.
        forecasts: Generation forecasts.
        grid_constraints: Grid constraints.
        market_prices: Market prices.
        n_simulations: Number of Monte Carlo runs.
        seed: Random seed.

    Returns:
        StressTestResult with statistics.
    """
    config = StressTestConfig(
        n_simulations=n_simulations,
        seed=seed,
        scenarios=[StressScenario.COMBINED],
    )

    engine = MonteCarloEngine(battery_config, config)
    return engine.run_stress_test(forecasts, grid_constraints, market_prices)
