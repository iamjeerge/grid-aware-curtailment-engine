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
