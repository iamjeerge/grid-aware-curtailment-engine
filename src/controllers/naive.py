"""Naive heuristic controller for baseline benchmarking.

This controller implements simple dispatch logic:
1. Always sell as much as possible to the grid (up to capacity)
2. Charge battery with any excess (up to battery limits)
3. Curtail remaining generation when storage is full

This serves as a benchmark to measure optimization improvements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from src.battery.physics import BatteryModel
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
    SimulationResult,
)


class Controller(Protocol):
    """Protocol for dispatch controllers."""

    def dispatch(
        self,
        timestamp: datetime,
        generation_mw: float,
        grid_constraint: GridConstraint,
        market_price: MarketPrice,
        scenario: ForecastScenario,
    ) -> OptimizationDecision:
        """Make a dispatch decision for a single timestep."""
        ...

    def run_simulation(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
        scenario: ForecastScenario,
    ) -> SimulationResult:
        """Run simulation over multiple timesteps."""
        ...


@dataclass
class DispatchMetrics:
    """Metrics from a single dispatch decision."""

    revenue: float
    degradation_cost: float
    curtailment_mwh: float
    grid_violation: bool
    soc_violation: bool


@dataclass
class NaiveSimulationState:
    """State tracking during simulation."""

    decisions: list[OptimizationDecision] = field(default_factory=list)
    total_revenue: float = 0.0
    total_degradation: float = 0.0
    total_curtailed: float = 0.0
    total_generated: float = 0.0
    total_sold: float = 0.0
    total_stored: float = 0.0
    grid_violations: int = 0


class NaiveController:
    """Naive heuristic controller for baseline benchmarking.

    Strategy:
    1. Sell to grid up to export capacity limit
    2. Store excess in battery (respecting charge rate and SOC limits)
    3. Curtail any remaining generation

    This greedy approach doesn't consider:
    - Future prices (no lookahead)
    - Battery degradation costs
    - Negative pricing (will sell even at loss)
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        degradation_cost_per_mwh: float = 8.0,
    ) -> None:
        """Initialize the naive controller.

        Args:
            battery_config: Battery configuration parameters.
            degradation_cost_per_mwh: Cost per MWh cycled for degradation.
        """
        self.battery_config = battery_config
        self.degradation_cost_per_mwh = degradation_cost_per_mwh
        self.battery = BatteryModel(battery_config)
        self._initialized = False

    def initialize(
        self,
        start_time: datetime,
        initial_soc_fraction: float = 0.5,
    ) -> None:
        """Initialize the controller state.

        Args:
            start_time: Simulation start time.
            initial_soc_fraction: Initial battery SOC as fraction of capacity.
        """
        self.battery.initialize(start_time, initial_soc_fraction)
        self._initialized = True

    def dispatch(
        self,
        timestamp: datetime,
        generation_mw: float,
        grid_constraint: GridConstraint,
        market_price: MarketPrice,
        scenario: ForecastScenario = ForecastScenario.P50,
    ) -> OptimizationDecision:
        """Make a dispatch decision for a single timestep.

        Naive strategy:
        1. Sell up to grid capacity
        2. Store remainder in battery
        3. Curtail what's left

        Args:
            timestamp: Current timestep.
            generation_mw: Available generation (MW).
            grid_constraint: Grid export limits.
            market_price: Current market prices.
            scenario: Forecast scenario for this decision.

        Returns:
            OptimizationDecision with sell/store/curtail amounts.

        Raises:
            RuntimeError: If controller not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        # Step 1: Sell as much as possible to grid
        export_limit = grid_constraint.max_export_mw
        energy_sold = min(generation_mw, export_limit)
        remaining = generation_mw - energy_sold

        # Step 2: Charge battery with excess
        max_charge = self.battery.get_max_charge_power()
        energy_stored = min(remaining, max_charge)

        if energy_stored > 0:
            self.battery.charge(energy_stored, 1.0, timestamp)

        remaining -= energy_stored

        # Step 3: Curtail the rest
        energy_curtailed = max(0.0, remaining)

        # Calculate costs
        price = market_price.effective_price
        revenue = energy_sold * price
        degradation_cost = energy_stored * self.degradation_cost_per_mwh

        # Get resulting SOC
        state = self.battery.current_state
        resulting_soc = state.soc_mwh if state else 0.0

        return OptimizationDecision(
            timestamp=timestamp,
            scenario=scenario,
            generation_mw=generation_mw,
            energy_sold_mw=energy_sold,
            energy_stored_mw=energy_stored,
            energy_curtailed_mw=energy_curtailed,
            battery_discharge_mw=0.0,  # Naive never discharges strategically
            resulting_soc_mwh=resulting_soc,
            revenue_dollars=revenue,
            degradation_cost_dollars=degradation_cost,
        )

    def run_simulation(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
        scenario: ForecastScenario = ForecastScenario.P50,
    ) -> SimulationResult:
        """Run simulation over multiple timesteps.

        Args:
            forecasts: List of generation forecasts per timestep.
            grid_constraints: List of grid constraints per timestep.
            market_prices: List of market prices per timestep.
            scenario: Which forecast scenario to use.

        Returns:
            SimulationResult with aggregated metrics.

        Raises:
            ValueError: If input lists have different lengths.
            RuntimeError: If controller not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        n_steps = len(forecasts)
        if len(grid_constraints) != n_steps or len(market_prices) != n_steps:
            raise ValueError("All input lists must have the same length.")

        state = NaiveSimulationState()

        for i in range(n_steps):
            forecast = forecasts[i]
            constraint = grid_constraints[i]
            price = market_prices[i]

            # Get generation for selected scenario
            generation = forecast.total_generation(scenario)
            state.total_generated += generation

            # Make dispatch decision
            decision = self.dispatch(
                timestamp=forecast.timestamp,
                generation_mw=generation,
                grid_constraint=constraint,
                market_price=price,
                scenario=scenario,
            )
            state.decisions.append(decision)

            # Aggregate metrics
            state.total_revenue += decision.revenue_dollars
            state.total_degradation += decision.degradation_cost_dollars
            state.total_curtailed += decision.energy_curtailed_mw
            state.total_sold += decision.energy_sold_mw
            state.total_stored += decision.energy_stored_mw

            # Check for grid violations
            if decision.energy_sold_mw > constraint.max_export_mw + 0.01:
                state.grid_violations += 1

        # Build result
        return SimulationResult(
            simulation_id=f"naive_{uuid4().hex[:8]}",
            start_time=forecasts[0].timestamp,
            end_time=forecasts[-1].timestamp,
            scenario=scenario,
            decisions=state.decisions,
            total_generation_mwh=state.total_generated,
            total_sold_mwh=state.total_sold,
            total_stored_mwh=state.total_stored,
            total_curtailed_mwh=state.total_curtailed,
            total_revenue_dollars=state.total_revenue,
            total_degradation_cost_dollars=state.total_degradation,
            grid_violations_count=state.grid_violations,
        )

    def reset(self, start_time: datetime, initial_soc_fraction: float = 0.5) -> None:
        """Reset controller state for a new simulation.

        Args:
            start_time: New simulation start time.
            initial_soc_fraction: Initial battery SOC as fraction.
        """
        self.battery = BatteryModel(self.battery_config)
        self.initialize(start_time, initial_soc_fraction)


class NaiveWithDischargeController(NaiveController):
    """Extended naive controller that discharges during high prices.

    Adds a simple heuristic: discharge battery when price exceeds threshold.
    Still doesn't do proper lookahead optimization.
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        degradation_cost_per_mwh: float = 8.0,
        discharge_price_threshold: float = 100.0,
    ) -> None:
        """Initialize controller with discharge threshold.

        Args:
            battery_config: Battery configuration.
            degradation_cost_per_mwh: Degradation cost per MWh.
            discharge_price_threshold: Price above which to discharge battery.
        """
        super().__init__(battery_config, degradation_cost_per_mwh)
        self.discharge_price_threshold = discharge_price_threshold

    def dispatch(
        self,
        timestamp: datetime,
        generation_mw: float,
        grid_constraint: GridConstraint,
        market_price: MarketPrice,
        scenario: ForecastScenario = ForecastScenario.P50,
    ) -> OptimizationDecision:
        """Dispatch with discharge heuristic.

        Strategy:
        1. If price is high, discharge battery to sell more
        2. Sell up to grid capacity (generation + discharge)
        3. Store excess if price is low
        4. Curtail remainder
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        price = market_price.effective_price
        export_limit = grid_constraint.max_export_mw

        battery_discharge = 0.0
        energy_stored = 0.0

        # High price: try to discharge battery
        if price >= self.discharge_price_threshold:
            # How much grid capacity is available beyond generation?
            available_export = max(0.0, export_limit - generation_mw)
            max_discharge = self.battery.get_max_discharge_power()
            battery_discharge = min(available_export, max_discharge)

            if battery_discharge > 0:
                self.battery.discharge(battery_discharge, 1.0, timestamp)

        # Total energy available to sell
        total_available = generation_mw + battery_discharge

        # Sell up to grid limit
        energy_sold = min(total_available, export_limit)

        # Remaining generation after selling (not counting discharge)
        remaining = generation_mw - (energy_sold - battery_discharge)

        # Store excess (only if price is not high)
        if remaining > 0 and price < self.discharge_price_threshold:
            max_charge = self.battery.get_max_charge_power()
            energy_stored = min(remaining, max_charge)
            if energy_stored > 0:
                self.battery.charge(energy_stored, 1.0, timestamp)
            remaining -= energy_stored

        # Curtail the rest
        energy_curtailed = max(0.0, remaining)

        # Calculate financials
        revenue = energy_sold * price
        # Degradation from both charging and discharging
        cycle_cost = (energy_stored + battery_discharge) * self.degradation_cost_per_mwh

        # Get resulting SOC
        state = self.battery.current_state
        resulting_soc = state.soc_mwh if state else 0.0

        return OptimizationDecision(
            timestamp=timestamp,
            scenario=scenario,
            generation_mw=generation_mw,
            energy_sold_mw=energy_sold,
            energy_stored_mw=energy_stored,
            energy_curtailed_mw=energy_curtailed,
            battery_discharge_mw=battery_discharge,
            resulting_soc_mwh=resulting_soc,
            revenue_dollars=revenue,
            degradation_cost_dollars=cycle_cost,
        )
