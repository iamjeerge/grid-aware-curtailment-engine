"""Core domain models for the Grid-Aware Curtailment Engine.

All models use Pydantic with strict validation. Units follow CAISO conventions:
- Power: MW (megawatts)
- Energy: MWh (megawatt-hours)
- Prices: $/MWh
- Time: Hourly timesteps
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Type Aliases with Validation
# =============================================================================

PowerMW = Annotated[float, Field(ge=0, description="Power in megawatts (MW)")]
EnergyMWh = Annotated[float, Field(ge=0, description="Energy in megawatt-hours (MWh)")]
PriceDollarPerMWh = Annotated[
    float, Field(description="Price in $/MWh (can be negative)")
]
Efficiency = Annotated[float, Field(gt=0, le=1, description="Efficiency ratio (0-1)")]
Probability = Annotated[float, Field(ge=0, le=1, description="Probability (0-1)")]


# =============================================================================
# Enums
# =============================================================================


class ForecastScenario(str, Enum):
    """Probabilistic forecast scenarios."""

    P10 = "P10"  # 10th percentile (low generation)
    P50 = "P50"  # 50th percentile (median)
    P90 = "P90"  # 90th percentile (high generation)


class DecisionAction(str, Enum):
    """Possible actions for energy dispatch."""

    SELL = "sell"
    STORE = "store"
    CURTAIL = "curtail"


# =============================================================================
# Core Domain Models
# =============================================================================


class TimeStep(BaseModel):
    """A single timestep in the optimization horizon.

    Represents one hour in CAISO-style hourly dispatch.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    hour_index: Annotated[
        int, Field(ge=0, description="Hour index in horizon (0-23 for day-ahead)")
    ]


class GenerationForecast(BaseModel):
    """Renewable generation forecast with probabilistic bands.

    Contains P10/P50/P90 forecasts for both solar and wind generation.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    solar_mw_p10: PowerMW = 0.0
    solar_mw_p50: PowerMW = 0.0
    solar_mw_p90: PowerMW = 0.0
    wind_mw_p10: PowerMW = 0.0
    wind_mw_p50: PowerMW = 0.0
    wind_mw_p90: PowerMW = 0.0

    def total_generation(self, scenario: ForecastScenario) -> float:
        """Get total generation (solar + wind) for a given scenario."""
        if scenario == ForecastScenario.P10:
            return self.solar_mw_p10 + self.wind_mw_p10
        elif scenario == ForecastScenario.P50:
            return self.solar_mw_p50 + self.wind_mw_p50
        else:  # P90
            return self.solar_mw_p90 + self.wind_mw_p90


class GridConstraint(BaseModel):
    """CAISO-style grid export constraints for a timestep.

    Models export capacity limits, congestion windows, and ramp rate constraints.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    max_export_mw: PowerMW
    max_ramp_up_mw_per_hour: Annotated[
        float, Field(gt=0, description="Maximum ramp up rate (MW/hour)")
    ] = 100.0
    max_ramp_down_mw_per_hour: Annotated[
        float, Field(gt=0, description="Maximum ramp down rate (MW/hour)")
    ] = 100.0
    is_congested: bool = False
    congestion_price_adder: PriceDollarPerMWh = 0.0


class BatteryConfig(BaseModel):
    """Battery Energy Storage System (BESS) configuration.

    Immutable configuration parameters for the battery.
    """

    model_config = ConfigDict(frozen=True)

    capacity_mwh: EnergyMWh = 500.0
    max_power_mw: PowerMW = 150.0
    charge_efficiency: Efficiency = 0.95
    discharge_efficiency: Efficiency = 0.95
    min_soc_fraction: Annotated[float, Field(ge=0, le=1)] = 0.1  # 10% minimum SOC
    max_soc_fraction: Annotated[float, Field(ge=0, le=1)] = 0.9  # 90% maximum SOC
    degradation_cost_per_mwh: Annotated[
        float, Field(ge=0, description="Degradation cost in $/MWh cycled")
    ] = 8.0

    @property
    def usable_capacity_mwh(self) -> float:
        """Usable capacity accounting for SOC limits."""
        return self.capacity_mwh * (self.max_soc_fraction - self.min_soc_fraction)


class BatteryState(BaseModel):
    """Current state of the battery at a given timestep.

    Tracks SOC and cumulative cycling for degradation calculation.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    soc_mwh: EnergyMWh
    soc_fraction: Annotated[
        float, Field(ge=0, le=1, description="SOC as fraction of capacity")
    ]
    cumulative_throughput_mwh: EnergyMWh = 0.0  # For degradation tracking

    def is_within_bounds(self, config: BatteryConfig) -> bool:
        """Check if SOC is within configured bounds."""
        return config.min_soc_fraction <= self.soc_fraction <= config.max_soc_fraction


class MarketPrice(BaseModel):
    """Energy market price for a timestep.

    Includes day-ahead and real-time prices. Prices can be negative during
    oversupply conditions (duck curve).
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    day_ahead_price: PriceDollarPerMWh
    real_time_price: PriceDollarPerMWh | None = (
        None  # May not be available for forecasts
    )
    is_negative_pricing: bool = False

    @property
    def effective_price(self) -> float:
        """Get the effective price (real-time if available, else day-ahead)."""
        return (
            self.real_time_price
            if self.real_time_price is not None
            else self.day_ahead_price
        )


class OptimizationDecision(BaseModel):
    """Optimization decision for a single timestep.

    Records how much energy to sell, store, or curtail.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    scenario: ForecastScenario
    generation_mw: PowerMW
    energy_sold_mw: PowerMW
    energy_stored_mw: PowerMW
    energy_curtailed_mw: PowerMW
    battery_discharge_mw: PowerMW = 0.0
    resulting_soc_mwh: EnergyMWh
    revenue_dollars: float
    degradation_cost_dollars: float = 0.0

    def validate_energy_balance(self, tolerance: float = 0.01) -> bool:
        """Validate that energy balance constraint is satisfied."""
        total_dispatch = (
            self.energy_sold_mw + self.energy_stored_mw + self.energy_curtailed_mw
        )
        return abs(self.generation_mw - total_dispatch) <= tolerance

    @property
    def net_revenue(self) -> float:
        """Net revenue after degradation costs."""
        return self.revenue_dollars - self.degradation_cost_dollars


class SimulationResult(BaseModel):
    """Aggregated results from a complete simulation run.

    Contains all decisions, KPIs, and metadata for analysis.
    """

    model_config = ConfigDict(frozen=True)

    simulation_id: str
    start_time: datetime
    end_time: datetime
    scenario: ForecastScenario
    decisions: list[OptimizationDecision]

    # Aggregated KPIs
    total_generation_mwh: EnergyMWh
    total_sold_mwh: EnergyMWh
    total_stored_mwh: EnergyMWh
    total_curtailed_mwh: EnergyMWh
    total_revenue_dollars: float
    total_degradation_cost_dollars: float
    grid_violations_count: int = 0

    @property
    def curtailment_rate(self) -> float:
        """Percentage of generation that was curtailed."""
        if self.total_generation_mwh == 0:
            return 0.0
        return (self.total_curtailed_mwh / self.total_generation_mwh) * 100

    @property
    def net_revenue(self) -> float:
        """Total revenue minus degradation costs."""
        return self.total_revenue_dollars - self.total_degradation_cost_dollars

    @property
    def has_violations(self) -> bool:
        """Check if any grid violations occurred."""
        return self.grid_violations_count > 0


# =============================================================================
# Scenario Configuration
# =============================================================================


class ScenarioConfig(BaseModel):
    """Configuration for a simulation scenario.

    Groups all parameters needed to run a complete simulation.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    horizon_hours: Annotated[int, Field(gt=0, le=168)] = 24  # Max 1 week
    battery_config: BatteryConfig = Field(default_factory=BatteryConfig)
    scenario_probabilities: dict[ForecastScenario, Probability] = Field(
        default_factory=lambda: {
            ForecastScenario.P10: 0.25,
            ForecastScenario.P50: 0.50,
            ForecastScenario.P90: 0.25,
        }
    )
    random_seed: int | None = None  # For reproducibility

    def validate_probabilities(self) -> bool:
        """Validate that scenario probabilities sum to 1."""
        total = sum(self.scenario_probabilities.values())
        return abs(total - 1.0) < 0.001
