"""Pydantic schemas for API request/response models.

These schemas define the API contract between frontend and backend,
providing validation and serialization for all endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums
# =============================================================================


class ScenarioType(str, Enum):
    """Pre-defined scenario types."""

    DUCK_CURVE = "duck_curve"
    HIGH_VOLATILITY = "high_volatility"
    CONGESTED_GRID = "congested_grid"
    CUSTOM = "custom"


class StrategyType(str, Enum):
    """Available optimization strategies."""

    NAIVE = "naive"
    MILP = "milp"
    RL = "rl"
    HYBRID = "hybrid"


class OptimizationStatus(str, Enum):
    """Status of optimization job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Request Schemas
# =============================================================================


class BatteryConfigRequest(BaseModel):
    """Battery configuration for optimization."""

    model_config = ConfigDict(extra="forbid")

    capacity_mwh: float = Field(
        default=500.0,
        ge=0,
        le=10000,
        description="Battery energy capacity (MWh)",
    )
    max_power_mw: float = Field(
        default=150.0,
        ge=0,
        le=5000,
        description="Battery power rating (MW)",
    )
    charge_efficiency: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Charge efficiency (0-1)",
    )
    discharge_efficiency: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Discharge efficiency (0-1)",
    )
    min_soc_fraction: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Minimum SOC fraction (0-1)",
    )
    max_soc_fraction: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Maximum SOC fraction (0-1)",
    )
    degradation_cost_per_mwh: float = Field(
        default=8.0,
        ge=0,
        le=100,
        description="Degradation cost per MWh cycled ($/MWh)",
    )


class ScenarioConfigRequest(BaseModel):
    """Scenario configuration for optimization."""

    model_config = ConfigDict(extra="forbid")

    scenario_type: ScenarioType = Field(
        default=ScenarioType.DUCK_CURVE,
        description="Pre-defined scenario type",
    )
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Planning horizon (hours)",
    )
    peak_generation_mw: float = Field(
        default=600.0,
        ge=0,
        le=10000,
        description="Peak generation capacity (MW)",
    )
    grid_limit_mw: float = Field(
        default=300.0,
        ge=0,
        le=10000,
        description="Grid export limit (MW)",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for reproducibility",
    )


class OptimizationRequest(BaseModel):
    """Request to run an optimization."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="Demo Optimization",
        min_length=1,
        max_length=200,
        description="Name for this optimization run",
    )
    scenario: ScenarioConfigRequest = Field(
        default_factory=ScenarioConfigRequest,
        description="Scenario configuration",
    )
    battery: BatteryConfigRequest = Field(
        default_factory=BatteryConfigRequest,
        description="Battery configuration",
    )
    strategies: list[StrategyType] = Field(
        default=[StrategyType.NAIVE, StrategyType.MILP],
        min_length=1,
        description="Strategies to run",
    )


# =============================================================================
# Response Schemas
# =============================================================================


class CurtailmentMetricsResponse(BaseModel):
    """Curtailment metrics in response."""

    total_generation_mwh: float
    total_curtailed_mwh: float
    total_sold_mwh: float
    total_stored_mwh: float
    curtailment_rate: float


class RevenueMetricsResponse(BaseModel):
    """Revenue metrics in response."""

    gross_revenue: float
    degradation_cost: float
    net_profit: float
    average_price: float


class BatteryMetricsResponse(BaseModel):
    """Battery metrics in response."""

    total_charged_mwh: float
    total_discharged_mwh: float
    cycles: float
    utilization_rate: float
    avg_soc: float


class GridComplianceResponse(BaseModel):
    """Grid compliance metrics in response."""

    violation_count: int
    total_violation_mwh: float
    max_violation_mw: float
    compliance_rate: float


class PerformanceSummaryResponse(BaseModel):
    """Complete performance summary for a strategy."""

    strategy_name: str
    curtailment: CurtailmentMetricsResponse
    revenue: RevenueMetricsResponse
    battery: BatteryMetricsResponse
    grid_compliance: GridComplianceResponse


class DecisionResponse(BaseModel):
    """Single optimization decision."""

    timestep: int
    timestamp: datetime
    generation_mw: float
    energy_sold_mw: float
    energy_stored_mw: float
    energy_curtailed_mw: float
    battery_discharge_mw: float
    soc_mwh: float
    price: float
    grid_limit_mw: float


class StrategyResultResponse(BaseModel):
    """Results for a single strategy."""

    strategy: StrategyType
    summary: PerformanceSummaryResponse
    decisions: list[DecisionResponse]
    solve_time_seconds: float | None = None


class ComparisonResponse(BaseModel):
    """Strategy comparison results."""

    best_strategy: str
    curtailment_reduction_pct: float
    revenue_uplift_pct: float
    revenue_uplift_dollars: float


class OptimizationResultResponse(BaseModel):
    """Complete optimization result."""

    id: UUID
    name: str
    status: OptimizationStatus
    created_at: datetime
    completed_at: datetime | None = None
    scenario: ScenarioConfigRequest
    battery: BatteryConfigRequest
    results: dict[str, StrategyResultResponse] = Field(default_factory=dict)
    comparison: ComparisonResponse | None = None
    error_message: str | None = None


class OptimizationListResponse(BaseModel):
    """List of optimizations."""

    items: list[OptimizationResultResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Health & Info Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    timestamp: datetime


class SystemInfoResponse(BaseModel):
    """System information response."""

    version: str
    python_version: str
    available_strategies: list[StrategyType]
    available_scenarios: list[ScenarioType]
    solver_available: bool


# =============================================================================
# Demo Data Schemas
# =============================================================================


class DemoScenarioResponse(BaseModel):
    """Pre-configured demo scenario."""

    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    config: ScenarioConfigRequest
    battery: BatteryConfigRequest


# =============================================================================
# Industry & Business Metrics
# =============================================================================


class FinancialMetricsResponse(BaseModel):
    """Financial metrics for business analysis."""

    total_revenue: float = Field(description="Total revenue from all optimizations ($)")
    total_cost: float = Field(description="Total operational costs ($)")
    net_profit: float = Field(description="Net profit (revenue - cost) ($)")
    roi_percentage: float = Field(description="Return on Investment (%)")
    average_profit_per_mwh: float = Field(
        description="Average profit per MWh sold ($/MWh)"
    )
    revenue_uplift_vs_naive: float = Field(
        description="Revenue improvement vs naive strategy (%)"
    )
    total_degradation_cost: float = Field(
        description="Battery degradation costs incurred ($)"
    )


class GridReliabilityMetricsResponse(BaseModel):
    """Grid reliability and compliance metrics."""

    total_violations: int = Field(description="Total grid constraint violations")
    total_violation_mwh: float = Field(description="Total violated energy (MWh)")
    compliance_rate: float = Field(description="Grid compliance rate (%)")
    max_violation_mw: float = Field(description="Maximum violation magnitude (MW)")
    ramp_rate_violations: int = Field(description="Number of ramp rate violations")
    export_capacity_utilization: float = Field(
        description="Average grid export capacity utilization (%)"
    )


class CurtailmentReductionMetricsResponse(BaseModel):
    """Curtailment and renewable optimization metrics."""

    total_generation_mwh: float = Field(description="Total renewable generation (MWh)")
    total_curtailed_mwh: float = Field(description="Total curtailed energy (MWh)")
    curtailment_rate_baseline: float = Field(
        description="Baseline curtailment rate without optimization (%)"
    )
    curtailment_rate_optimized: float = Field(
        description="Optimized curtailment rate (%)"
    )
    curtailment_reduction_pct: float = Field(
        description="Curtailment reduction achieved (%)"
    )
    avoided_curtailment_mwh: float = Field(
        description="Energy saved from curtailment reduction (MWh)"
    )
    avoided_curtailment_value: float = Field(
        description="Economic value of avoided curtailment ($)"
    )


class BatteryHealthMetricsResponse(BaseModel):
    """Battery health and efficiency metrics."""

    total_cycles_equivalent: float = Field(
        description="Total equivalent full charge/discharge cycles"
    )
    remaining_useful_life_pct: float = Field(
        description="Estimated remaining useful life (%)"
    )
    round_trip_efficiency_actual: float = Field(
        description="Actual round-trip efficiency achieved (%)"
    )
    energy_arbitrage_captured: float = Field(
        description="Value of price arbitrage ($)"
    )
    peak_shaving_contribution: float = Field(
        description="Contribution to grid peak shaving (MWh)"
    )


class EnvironmentalMetricsResponse(BaseModel):
    """Environmental and sustainability metrics."""

    co2_avoided_metric_tons: float = Field(
        description="CO2 emissions avoided due to curtailment reduction (metric tons)"
    )
    equivalent_household_days: float = Field(
        description="Equivalent clean energy for household-days"
    )
    grid_renewable_penetration_improvement: float = Field(
        description="Improvement in grid renewable penetration (%)"
    )


class IndustryDashboardResponse(BaseModel):
    """Comprehensive industry dashboard with all KPIs."""

    total_optimizations_run: int
    financial_metrics: FinancialMetricsResponse
    grid_reliability: GridReliabilityMetricsResponse
    curtailment_reduction: CurtailmentReductionMetricsResponse
    battery_health: BatteryHealthMetricsResponse
    environmental: EnvironmentalMetricsResponse
    summary: str = Field(description="Executive summary of key insights")


class OptimizationSummaryItemResponse(BaseModel):
    """Summary of a single optimization for listing."""

    id: UUID
    name: str
    scenario_type: str
    strategy: str
    created_at: datetime
    net_profit: float
    curtailment_reduction: float
    compliance_rate: float
    grid_violations: int
