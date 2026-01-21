"""Real-world assumption documentation and validation.

This module provides a structured framework for documenting all assumptions
made in the optimization model, along with their evidence, limitations,
and validation status.

The goal is to ensure transparency and credibility by explicitly stating
what the model assumes and where those assumptions may break down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssumptionCategory(str, Enum):
    """Categories of assumptions in the optimization model."""

    GRID = "grid"
    BATTERY = "battery"
    MARKET = "market"
    FORECAST = "forecast"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"


class ValidationSeverity(str, Enum):
    """Severity levels for assumption validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Assumption:
    """A documented assumption in the optimization model.

    Attributes:
        id: Unique identifier for the assumption.
        category: Category of the assumption.
        title: Short descriptive title.
        description: Detailed description of what is assumed.
        rationale: Why this assumption is made.
        evidence: Supporting evidence or references.
        limitations: Known limitations or edge cases.
        impact_if_violated: What happens if assumption is wrong.
        validation_method: How to validate this assumption.
        default_value: Default value if applicable.
        valid_range: Valid range of values if applicable.
        unit: Unit of measurement if applicable.
    """

    id: str
    category: AssumptionCategory
    title: str
    description: str
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    impact_if_violated: str = ""
    validation_method: str = ""
    default_value: Any = None
    valid_range: tuple[float, float] | None = None
    unit: str = ""


@dataclass
class ValidationResult:
    """Result of validating a single assumption.

    Attributes:
        assumption_id: ID of the assumption being validated.
        is_valid: Whether the assumption holds.
        severity: Severity level if invalid.
        message: Detailed message about the validation result.
        actual_value: Actual value observed (if applicable).
        expected_range: Expected range (if applicable).
    """

    assumption_id: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    actual_value: Any = None
    expected_range: tuple[float, float] | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for all assumptions.

    Attributes:
        results: List of validation results.
        total_assumptions: Total number of assumptions checked.
        valid_count: Number of valid assumptions.
        warning_count: Number of assumptions with warnings.
        error_count: Number of assumptions with errors.
        critical_count: Number of critical issues.
    """

    results: list[ValidationResult] = field(default_factory=list)
    total_assumptions: int = 0
    valid_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    critical_count: int = 0

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        self.total_assumptions += 1

        if result.is_valid:
            self.valid_count += 1
        elif result.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        elif result.severity == ValidationSeverity.ERROR:
            self.error_count += 1
        elif result.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1

    @property
    def is_valid(self) -> bool:
        """Check if all assumptions are valid (no errors or critical)."""
        return self.error_count == 0 and self.critical_count == 0

    @property
    def summary(self) -> str:
        """Generate a summary of the validation report."""
        return (
            f"Validation Report: {self.valid_count}/{self.total_assumptions} valid, "
            f"{self.warning_count} warnings, {self.error_count} errors, "
            f"{self.critical_count} critical"
        )


class AssumptionRegistry:
    """Central registry of all documented assumptions.

    This class provides a single source of truth for all assumptions
    made in the optimization model, organized by category.
    """

    def __init__(self) -> None:
        """Initialize the assumption registry."""
        self._assumptions: dict[str, Assumption] = {}
        self._register_all_assumptions()

    def _register_all_assumptions(self) -> None:
        """Register all documented assumptions."""
        # Grid assumptions
        self._register_grid_assumptions()
        # Battery assumptions
        self._register_battery_assumptions()
        # Market assumptions
        self._register_market_assumptions()
        # Forecast assumptions
        self._register_forecast_assumptions()
        # Regulatory assumptions
        self._register_regulatory_assumptions()
        # Operational assumptions
        self._register_operational_assumptions()

    def _register_grid_assumptions(self) -> None:
        """Register grid-related assumptions."""
        self.register(
            Assumption(
                id="GRID-001",
                category=AssumptionCategory.GRID,
                title="Static Grid Export Capacity",
                description=(
                    "Grid export capacity remains constant within each planning "
                    "horizon (typically 24 hours). Capacity changes occur only "
                    "at predefined intervals based on historical congestion patterns."
                ),
                rationale=(
                    "Grid operators typically publish day-ahead capacity forecasts. "
                    "Real-time changes are modeled as stress scenarios."
                ),
                evidence=[
                    "CAISO publishes day-ahead congestion forecasts",
                    "Most ISOs provide hourly transmission limits",
                    "Operational changes are typically announced 1+ hours ahead",
                ],
                limitations=[
                    "Does not capture real-time emergency curtailments",
                    "Transmission outages may cause sudden capacity drops",
                    "Severe weather can force rapid grid reconfiguration",
                ],
                impact_if_violated=(
                    "Unexpected capacity reductions could force immediate curtailment, "
                    "revenue loss from stranded generation, potential grid penalties"
                ),
                validation_method="Compare planned vs actual grid capacity from historical data",
                default_value=300,
                valid_range=(50, 1000),
                unit="MW",
            )
        )

        self.register(
            Assumption(
                id="GRID-002",
                category=AssumptionCategory.GRID,
                title="Ramp Rate Constraints",
                description=(
                    "Generation ramp rates are limited to prevent rapid changes "
                    "that could destabilize the grid. Default: 50 MW/hour up, "
                    "100 MW/hour down."
                ),
                rationale=(
                    "Grid operators impose ramp rate limits to maintain frequency "
                    "stability. Solar/wind have inherent variability limits."
                ),
                evidence=[
                    "CAISO ramp rate requirements: typically 10-20% of capacity/hour",
                    "FERC Order 764 requires 15-minute scheduling",
                    "Modern solar inverters can ramp 100%/minute if unrestricted",
                ],
                limitations=[
                    "Actual ramp capability depends on inverter technology",
                    "Cloud transients can cause faster ramps than modeled",
                    "Grid may request emergency ramps outside normal limits",
                ],
                impact_if_violated=(
                    "Grid penalties for excessive ramping, potential curtailment "
                    "orders, interconnection agreement violations"
                ),
                validation_method="Check actual ramp rates against configured limits",
                default_value=50,
                valid_range=(10, 200),
                unit="MW/hour",
            )
        )

        self.register(
            Assumption(
                id="GRID-003",
                category=AssumptionCategory.GRID,
                title="Congestion Window Predictability",
                description=(
                    "Grid congestion windows follow predictable patterns based on "
                    "historical data. Peak congestion typically occurs during "
                    "midday solar peak (10am-2pm) and evening demand peak (5pm-8pm)."
                ),
                rationale=(
                    "Historical congestion data shows consistent temporal patterns "
                    "aligned with solar generation and demand curves."
                ),
                evidence=[
                    "Duck Curve pattern is well-documented in California",
                    "CAISO congestion data shows consistent daily patterns",
                    "Seasonal variations are predictable (summer AC, winter heating)",
                ],
                limitations=[
                    "Unusual weather can shift congestion patterns",
                    "Major outages disrupt normal patterns",
                    "New generation/load additions change baselines",
                ],
                impact_if_violated=(
                    "Suboptimal battery dispatch timing, missed arbitrage "
                    "opportunities, increased curtailment during unexpected congestion"
                ),
                validation_method="Compare predicted vs actual congestion windows",
            )
        )

        self.register(
            Assumption(
                id="GRID-004",
                category=AssumptionCategory.GRID,
                title="Grid Connection Reliability",
                description=(
                    "Grid connection is available 99%+ of the time. Outages are "
                    "rare and typically short-duration."
                ),
                rationale=(
                    "Modern grid infrastructure has high reliability. Planned "
                    "outages are scheduled in advance."
                ),
                evidence=[
                    "US grid reliability: ~99.97% availability",
                    "Most outages are weather-related and predictable",
                    "Planned maintenance is scheduled months ahead",
                ],
                limitations=[
                    "Extreme weather events (wildfires, hurricanes)",
                    "Cascading failures can cause extended outages",
                    "Cyber attacks are an emerging risk",
                ],
                impact_if_violated=(
                    "Complete revenue loss during outage, potential equipment damage "
                    "from sudden disconnection, battery cycling for backup"
                ),
                validation_method="Track actual grid availability vs assumed 99%",
            )
        )

    def _register_battery_assumptions(self) -> None:
        """Register battery-related assumptions."""
        self.register(
            Assumption(
                id="BATT-001",
                category=AssumptionCategory.BATTERY,
                title="Round-Trip Efficiency",
                description=(
                    "Battery round-trip efficiency is 90% (95% charge × 95% discharge). "
                    "This accounts for inverter losses, thermal losses, and "
                    "battery chemistry inefficiencies."
                ),
                rationale=(
                    "Modern lithium-ion batteries achieve 90-95% round-trip efficiency. "
                    "We use 90% as a conservative estimate."
                ),
                evidence=[
                    "Tesla Megapack: 92% round-trip efficiency",
                    "LFP batteries: 90-95% typical",
                    "Inverter losses: 2-3% each direction",
                ],
                limitations=[
                    "Efficiency degrades with temperature extremes",
                    "High C-rates reduce efficiency",
                    "Aging batteries lose efficiency over time",
                ],
                impact_if_violated=(
                    "Overestimate arbitrage value if efficiency is lower, "
                    "energy accounting errors, financial model inaccuracies"
                ),
                validation_method="Measure actual energy in vs out over cycles",
                default_value=0.90,
                valid_range=(0.80, 0.98),
                unit="fraction",
            )
        )

        self.register(
            Assumption(
                id="BATT-002",
                category=AssumptionCategory.BATTERY,
                title="Degradation Cost Model",
                description=(
                    "Battery degradation is modeled as a linear cost per MWh cycled. "
                    "Default: $8/MWh throughput, representing capacity fade over "
                    "the battery's 10-year economic life."
                ),
                rationale=(
                    "Degradation is approximately linear with throughput for "
                    "moderate cycling. Cost derived from replacement economics."
                ),
                evidence=[
                    "NREL battery cost projections: $150-200/kWh installed",
                    "10-year life at 1 cycle/day = 3,650 cycles",
                    "20% capacity fade = $8-12/MWh throughput cost",
                ],
                limitations=[
                    "Deep cycling accelerates degradation non-linearly",
                    "Calendar aging adds to cycle aging",
                    "Temperature extremes accelerate degradation",
                    "High C-rates cause additional stress",
                ],
                impact_if_violated=(
                    "Underestimate true cost of aggressive cycling, "
                    "premature battery replacement, lower actual ROI"
                ),
                validation_method="Track capacity fade vs cycles, compare to model",
                default_value=8.0,
                valid_range=(2.0, 20.0),
                unit="$/MWh",
            )
        )

        self.register(
            Assumption(
                id="BATT-003",
                category=AssumptionCategory.BATTERY,
                title="SOC Operating Range",
                description=(
                    "Battery operates between 10% and 90% SOC to preserve "
                    "longevity. Full charge (100%) and deep discharge (0%) "
                    "are avoided except in emergencies."
                ),
                rationale=(
                    "Lithium-ion batteries degrade faster at extreme SOC levels. "
                    "Operating in the middle range extends life significantly."
                ),
                evidence=[
                    "Literature shows 2-3x life extension with 20-80% range",
                    "Tesla recommends 20-90% for daily use",
                    "Most grid batteries use 10-90% default range",
                ],
                limitations=[
                    "Reduces effective capacity by 20%",
                    "May miss peak arbitrage if SOC limits are hit",
                    "Emergency situations may require full range",
                ],
                impact_if_violated=(
                    "Accelerated degradation if operated outside range, "
                    "warranty voidance, capacity fade faster than modeled"
                ),
                validation_method="Monitor actual SOC range in operations",
                default_value=(0.10, 0.90),
                valid_range=(0.0, 1.0),
                unit="fraction",
            )
        )

        self.register(
            Assumption(
                id="BATT-004",
                category=AssumptionCategory.BATTERY,
                title="Power Rating Consistency",
                description=(
                    "Battery can charge and discharge at rated power (150 MW) "
                    "regardless of SOC level. C-rate is effectively 0.3C for "
                    "a 500 MWh system."
                ),
                rationale=(
                    "Modern grid batteries are designed for consistent power "
                    "delivery. 0.3C is a conservative, sustainable rate."
                ),
                evidence=[
                    "Grid-scale batteries typically rated for 0.25-1C continuous",
                    "Thermal management allows sustained power at 0.3C",
                    "Higher C-rates available but reduce efficiency",
                ],
                limitations=[
                    "Power may derate at extreme SOC (near 0% or 100%)",
                    "High temperatures cause power derating",
                    "Aging may reduce power capability",
                ],
                impact_if_violated=(
                    "Cannot meet dispatch commitments, grid penalties, "
                    "optimization assumes power that isn't available"
                ),
                validation_method="Test actual power delivery across SOC range",
                default_value=150,
                valid_range=(50, 500),
                unit="MW",
            )
        )

    def _register_market_assumptions(self) -> None:
        """Register market-related assumptions."""
        self.register(
            Assumption(
                id="MKT-001",
                category=AssumptionCategory.MARKET,
                title="Price Taker Assumption",
                description=(
                    "The facility is a price taker - its dispatch decisions "
                    "do not affect market prices. This is valid for facilities "
                    "small relative to the market."
                ),
                rationale=(
                    "Most renewable facilities are small relative to total "
                    "market volume. Price impact is negligible."
                ),
                evidence=[
                    "CAISO market volume: ~30,000 MW",
                    "Single 500 MW facility = ~1.7% of market",
                    "Price impact is diluted across all generators",
                ],
                limitations=[
                    "Large facilities may have measurable price impact",
                    "During congestion, local impact may be significant",
                    "Coordinated dispatch across facilities affects prices",
                ],
                impact_if_violated=(
                    "Optimization may over-dispatch at peak prices, "
                    "actual prices lower than forecast, reduced revenue"
                ),
                validation_method="Compare dispatch impact on local prices",
            )
        )

        self.register(
            Assumption(
                id="MKT-002",
                category=AssumptionCategory.MARKET,
                title="Day-Ahead Price Forecast Accuracy",
                description=(
                    "Day-ahead prices can be forecast with reasonable accuracy "
                    "(±20% MAPE typical). Extreme prices are harder to predict."
                ),
                rationale=(
                    "Historical analysis shows day-ahead prices are somewhat "
                    "predictable based on load forecasts and generation mix."
                ),
                evidence=[
                    "Academic studies show 10-30% MAPE for DA forecasts",
                    "Load forecasting is mature (2-5% error)",
                    "Generation mix is largely known day-ahead",
                ],
                limitations=[
                    "Extreme weather causes forecast errors",
                    "Outages cause price spikes hard to predict",
                    "Negative prices during high renewable periods",
                ],
                impact_if_violated=(
                    "Suboptimal battery dispatch, missed arbitrage windows, "
                    "selling at low prices instead of storing"
                ),
                validation_method="Track forecast vs actual price MAPE",
                default_value=0.20,
                valid_range=(0.05, 0.50),
                unit="fraction (MAPE)",
            )
        )

        self.register(
            Assumption(
                id="MKT-003",
                category=AssumptionCategory.MARKET,
                title="Negative Price Frequency",
                description=(
                    "Negative prices occur 5-15% of daytime hours during high "
                    "solar periods. This creates storage arbitrage opportunity."
                ),
                rationale=(
                    "Increasing solar penetration causes midday oversupply. "
                    "California regularly experiences negative prices."
                ),
                evidence=[
                    "CAISO: 10-20% of spring midday hours have negative prices",
                    "Germany: up to 30% negative hours in some months",
                    "Trend is increasing with renewable additions",
                ],
                limitations=[
                    "Frequency varies significantly by season",
                    "Market rule changes may reduce negative prices",
                    "Storage additions will arbitrage away extremes",
                ],
                impact_if_violated=(
                    "If negative prices are rarer, storage value is lower; "
                    "if more common, opportunity is underestimated"
                ),
                validation_method="Track actual negative price frequency",
                default_value=0.10,
                valid_range=(0.0, 0.50),
                unit="fraction of hours",
            )
        )

        self.register(
            Assumption(
                id="MKT-004",
                category=AssumptionCategory.MARKET,
                title="Settlement and Payment Terms",
                description=(
                    "Revenue is realized at the settlement price. We assume "
                    "no payment delays, credit risk, or market rule changes."
                ),
                rationale=(
                    "ISO markets have established settlement processes. "
                    "Credit requirements minimize counterparty risk."
                ),
                evidence=[
                    "CAISO settles monthly with 60-day reconciliation",
                    "Credit requirements enforced by ISO",
                    "Market rules are relatively stable year-to-year",
                ],
                limitations=[
                    "Settlement disputes can delay payment",
                    "Market rule changes can affect revenue",
                    "Force majeure events may suspend markets",
                ],
                impact_if_violated=(
                    "Cash flow timing differs from model, "
                    "working capital requirements higher than expected"
                ),
                validation_method="Track actual vs modeled settlement timing",
            )
        )

    def _register_forecast_assumptions(self) -> None:
        """Register forecast-related assumptions."""
        self.register(
            Assumption(
                id="FCST-001",
                category=AssumptionCategory.FORECAST,
                title="Generation Forecast Uncertainty Bands",
                description=(
                    "Generation forecasts use P10/P50/P90 bands representing "
                    "10th, 50th, and 90th percentile outcomes. The P50 is "
                    "the expected value used for deterministic planning."
                ),
                rationale=(
                    "Probabilistic forecasting captures uncertainty. "
                    "P10/P90 represent reasonable worst/best cases."
                ),
                evidence=[
                    "Industry standard for renewable forecasting",
                    "P10/P90 typically ±15-25% of P50 for solar",
                    "Wind uncertainty is typically higher (±20-35%)",
                ],
                limitations=[
                    "Actual distribution may not be symmetric",
                    "Extreme events fall outside P10/P90",
                    "Forecast skill degrades with horizon",
                ],
                impact_if_violated=(
                    "Risk quantification is wrong, reserves may be "
                    "insufficient, stress tests underestimate tail risks"
                ),
                validation_method="Verify actual outcomes fall within bands",
            )
        )

        self.register(
            Assumption(
                id="FCST-002",
                category=AssumptionCategory.FORECAST,
                title="Forecast Horizon",
                description=(
                    "Optimization uses a 24-hour rolling horizon with "
                    "hourly resolution. Forecasts are updated every hour."
                ),
                rationale=(
                    "Day-ahead planning aligns with market structure. "
                    "Hourly resolution balances accuracy and computation."
                ),
                evidence=[
                    "Day-ahead markets use 24-hour horizon",
                    "Forecast skill is reasonable up to 24-48 hours",
                    "Hourly resolution captures diurnal patterns",
                ],
                limitations=[
                    "Misses sub-hourly variability (clouds, gusts)",
                    "Cannot optimize for real-time market 5-min intervals",
                    "Longer horizons needed for weekly storage planning",
                ],
                impact_if_violated=(
                    "May miss intra-hour opportunities, "
                    "optimization doesn't align with actual market intervals"
                ),
                validation_method="Assess value of shorter/longer horizons",
                default_value=24,
                valid_range=(1, 168),
                unit="hours",
            )
        )

        self.register(
            Assumption(
                id="FCST-003",
                category=AssumptionCategory.FORECAST,
                title="Weather Data Quality",
                description=(
                    "Weather data (irradiance, wind speed) is available "
                    "with sufficient accuracy for forecast generation. "
                    "Assumes access to NWP model outputs or on-site sensors."
                ),
                rationale=(
                    "Modern NWP models provide reasonable forecasts. "
                    "On-site sensors can improve local accuracy."
                ),
                evidence=[
                    "NOAA HRRR model: 3km resolution, hourly updates",
                    "Commercial forecasts achieve 5-10% error (solar)",
                    "On-site pyranometers can detect clouds early",
                ],
                limitations=[
                    "NWP models struggle with local effects",
                    "Cloud timing errors are common",
                    "Data outages may degrade forecast quality",
                ],
                impact_if_violated=(
                    "Poor forecasts lead to suboptimal dispatch, "
                    "more curtailment or grid violations than expected"
                ),
                validation_method="Compare NWP/sensor data to actuals",
            )
        )

    def _register_regulatory_assumptions(self) -> None:
        """Register regulatory-related assumptions."""
        self.register(
            Assumption(
                id="REG-001",
                category=AssumptionCategory.REGULATORY,
                title="Interconnection Agreement Stability",
                description=(
                    "Interconnection agreement terms (capacity, ramp rates, "
                    "curtailment rules) remain stable over the analysis period."
                ),
                rationale=(
                    "Interconnection agreements are long-term contracts. "
                    "Changes require regulatory approval."
                ),
                evidence=[
                    "Typical IA term: 20+ years",
                    "Changes require FERC/state approval",
                    "Force majeure clauses provide some protection",
                ],
                limitations=[
                    "Grid upgrades may change interconnection terms",
                    "Regulatory reform can override contracts",
                    "Emergency orders can supersede agreements",
                ],
                impact_if_violated=(
                    "May need to renegotiate terms, potential for "
                    "reduced capacity allocation or new constraints"
                ),
                validation_method="Monitor regulatory proceedings",
            )
        )

        self.register(
            Assumption(
                id="REG-002",
                category=AssumptionCategory.REGULATORY,
                title="No Curtailment Compensation",
                description=(
                    "Curtailed energy receives no compensation. Generator "
                    "bears full cost of any curtailment required."
                ),
                rationale=(
                    "Economic curtailment is typically uncompensated. "
                    "Only reliability curtailments may receive payment."
                ),
                evidence=[
                    "CAISO: economic curtailment is not compensated",
                    "ERCOT: similar rules for economic dispatch",
                    "Some markets provide limited compensation",
                ],
                limitations=[
                    "Rules vary by market and may change",
                    "Reliability curtailment may be compensated",
                    "PPAs may have curtailment clauses",
                ],
                impact_if_violated=(
                    "If compensation exists, model undervalues some scenarios; "
                    "optimization may over-prioritize curtailment avoidance"
                ),
                validation_method="Review market rules and PPA terms",
            )
        )

        self.register(
            Assumption(
                id="REG-003",
                category=AssumptionCategory.REGULATORY,
                title="Tax and Incentive Stability",
                description=(
                    "Tax credits (ITC/PTC) and other incentives remain "
                    "as currently legislated. No retroactive changes."
                ),
                rationale=(
                    "IRA provides 10-year visibility on clean energy credits. "
                    "Retroactive changes are rare in tax policy."
                ),
                evidence=[
                    "IRA: ITC/PTC extended through 2032+",
                    "Technology-neutral credits starting 2025",
                    "Safe harbor provisions protect existing projects",
                ],
                limitations=[
                    "Political changes could affect implementation",
                    "IRS guidance may interpret rules differently",
                    "State incentives vary and change more often",
                ],
                impact_if_violated=(
                    "Project economics may change significantly, "
                    "ROI calculations would need revision"
                ),
                validation_method="Monitor legislative and regulatory changes",
            )
        )

        self.register(
            Assumption(
                id="REG-004",
                category=AssumptionCategory.REGULATORY,
                title="Environmental Compliance",
                description=(
                    "Facility meets all environmental requirements. "
                    "No additional constraints from environmental rules."
                ),
                rationale=(
                    "Solar/wind/battery have minimal environmental impact. "
                    "Permits are assumed to be in place."
                ),
                evidence=[
                    "Renewable projects have streamlined permitting",
                    "Battery systems have minimal emissions",
                    "Land use permits typically long-term",
                ],
                limitations=[
                    "Endangered species can affect operations",
                    "Water rights (for cleaning) may be restricted",
                    "End-of-life battery disposal regulations evolving",
                ],
                impact_if_violated=(
                    "May face operational restrictions, permit issues, "
                    "or additional compliance costs"
                ),
                validation_method="Review permits and environmental assessments",
            )
        )

    def _register_operational_assumptions(self) -> None:
        """Register operational assumptions."""
        self.register(
            Assumption(
                id="OPS-001",
                category=AssumptionCategory.OPERATIONAL,
                title="Perfect Dispatch Execution",
                description=(
                    "Dispatch instructions are executed exactly as optimized. "
                    "No delays, errors, or communication failures."
                ),
                rationale=(
                    "Modern SCADA systems provide reliable control. "
                    "Automation reduces human error."
                ),
                evidence=[
                    "Grid-scale systems use automated dispatch",
                    "Communication latency typically <1 second",
                    "Redundant systems ensure reliability",
                ],
                limitations=[
                    "Equipment failures can prevent dispatch",
                    "Communication outages do occur",
                    "Human override may deviate from optimal",
                ],
                impact_if_violated=(
                    "Actual dispatch differs from plan, "
                    "suboptimal outcomes, potential grid violations"
                ),
                validation_method="Compare optimized vs actual dispatch",
            )
        )

        self.register(
            Assumption(
                id="OPS-002",
                category=AssumptionCategory.OPERATIONAL,
                title="Equipment Availability",
                description=(
                    "All equipment (inverters, battery modules, transformers) "
                    "is available 98%+ of the time. Maintenance is scheduled."
                ),
                rationale=(
                    "Modern equipment has high reliability. "
                    "Preventive maintenance minimizes failures."
                ),
                evidence=[
                    "Inverter availability: 98-99% typical",
                    "Battery module failures: <1% annually",
                    "Scheduled maintenance: 1-2% downtime",
                ],
                limitations=[
                    "Infant mortality in new equipment",
                    "Extreme weather can damage equipment",
                    "Supply chain issues may extend repairs",
                ],
                impact_if_violated=(
                    "Reduced capacity during outages, "
                    "lost revenue, potential grid penalties"
                ),
                validation_method="Track actual equipment availability",
                default_value=0.98,
                valid_range=(0.90, 1.00),
                unit="fraction",
            )
        )

        self.register(
            Assumption(
                id="OPS-003",
                category=AssumptionCategory.OPERATIONAL,
                title="No Co-located Load",
                description=(
                    "The facility is purely generation/storage with no "
                    "significant on-site load. All energy is for grid export."
                ),
                rationale=(
                    "Most utility-scale projects are dedicated generation. "
                    "Auxiliary load is negligible (<1% of capacity)."
                ),
                evidence=[
                    "Typical auxiliary load: 1-3% of capacity",
                    "Battery HVAC: 2-5 kW per MW capacity",
                    "Control systems: negligible consumption",
                ],
                limitations=[
                    "Behind-the-meter projects have different economics",
                    "Co-located data centers/mining change dynamics",
                    "Future hydrogen production may add load",
                ],
                impact_if_violated=(
                    "Energy balance changes, may need to model "
                    "on-site load explicitly, different optimization"
                ),
                validation_method="Measure actual auxiliary consumption",
            )
        )

    def register(self, assumption: Assumption) -> None:
        """Register an assumption in the registry.

        Args:
            assumption: The assumption to register.
        """
        self._assumptions[assumption.id] = assumption

    def get(self, assumption_id: str) -> Assumption | None:
        """Get an assumption by ID.

        Args:
            assumption_id: The ID of the assumption.

        Returns:
            The assumption if found, None otherwise.
        """
        return self._assumptions.get(assumption_id)

    def get_by_category(self, category: AssumptionCategory) -> list[Assumption]:
        """Get all assumptions in a category.

        Args:
            category: The category to filter by.

        Returns:
            List of assumptions in the category.
        """
        return [a for a in self._assumptions.values() if a.category == category]

    def get_all(self) -> list[Assumption]:
        """Get all registered assumptions.

        Returns:
            List of all assumptions.
        """
        return list(self._assumptions.values())

    def to_markdown(self) -> str:
        """Export all assumptions as markdown documentation.

        Returns:
            Markdown-formatted documentation of all assumptions.
        """
        lines = ["# Model Assumptions Documentation\n"]
        lines.append("This document describes all assumptions made in the ")
        lines.append("Grid-Aware Curtailment Optimization Engine.\n\n")

        for category in AssumptionCategory:
            assumptions = self.get_by_category(category)
            if not assumptions:
                continue

            lines.append(f"## {category.value.title()} Assumptions\n\n")

            for a in assumptions:
                lines.append(f"### {a.id}: {a.title}\n\n")
                lines.append(f"**Description:** {a.description}\n\n")

                if a.rationale:
                    lines.append(f"**Rationale:** {a.rationale}\n\n")

                if a.evidence:
                    lines.append("**Evidence:**\n")
                    for e in a.evidence:
                        lines.append(f"- {e}\n")
                    lines.append("\n")

                if a.limitations:
                    lines.append("**Limitations:**\n")
                    for lim in a.limitations:
                        lines.append(f"- {lim}\n")
                    lines.append("\n")

                if a.impact_if_violated:
                    lines.append(f"**Impact if Violated:** {a.impact_if_violated}\n\n")

                if a.default_value is not None:
                    unit = f" {a.unit}" if a.unit else ""
                    lines.append(f"**Default Value:** {a.default_value}{unit}\n\n")

                if a.valid_range:
                    unit = f" {a.unit}" if a.unit else ""
                    lines.append(
                        f"**Valid Range:** {a.valid_range[0]} - "
                        f"{a.valid_range[1]}{unit}\n\n"
                    )

                lines.append("---\n\n")

        return "".join(lines)


# Module-level registry singleton
_registry: AssumptionRegistry | None = None


def get_assumption_registry() -> AssumptionRegistry:
    """Get the global assumption registry.

    Returns:
        The singleton AssumptionRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = AssumptionRegistry()
    return _registry


def get_all_assumptions() -> list[Assumption]:
    """Get all registered assumptions.

    Returns:
        List of all assumptions in the registry.
    """
    return get_assumption_registry().get_all()


def validate_assumptions(
    grid_capacity_mw: float | None = None,
    battery_efficiency: float | None = None,
    battery_power_mw: float | None = None,
    degradation_cost: float | None = None,
    forecast_horizon_hours: int | None = None,
) -> ValidationReport:
    """Validate assumptions against provided values.

    Args:
        grid_capacity_mw: Actual grid capacity in MW.
        battery_efficiency: Actual round-trip efficiency.
        battery_power_mw: Actual battery power rating in MW.
        degradation_cost: Actual degradation cost in $/MWh.
        forecast_horizon_hours: Planning horizon in hours.

    Returns:
        ValidationReport with results for each assumption checked.
    """
    registry = get_assumption_registry()
    report = ValidationReport()

    # Validate grid capacity
    if grid_capacity_mw is not None:
        assumption = registry.get("GRID-001")
        if assumption and assumption.valid_range:
            is_valid = (
                assumption.valid_range[0]
                <= grid_capacity_mw
                <= assumption.valid_range[1]
            )
            report.add_result(
                ValidationResult(
                    assumption_id="GRID-001",
                    is_valid=is_valid,
                    severity=(
                        ValidationSeverity.WARNING
                        if not is_valid
                        else ValidationSeverity.INFO
                    ),
                    message=(
                        f"Grid capacity {grid_capacity_mw} MW is "
                        f"{'within' if is_valid else 'outside'} valid range"
                    ),
                    actual_value=grid_capacity_mw,
                    expected_range=assumption.valid_range,
                )
            )

    # Validate battery efficiency
    if battery_efficiency is not None:
        assumption = registry.get("BATT-001")
        if assumption and assumption.valid_range:
            is_valid = (
                assumption.valid_range[0]
                <= battery_efficiency
                <= assumption.valid_range[1]
            )
            report.add_result(
                ValidationResult(
                    assumption_id="BATT-001",
                    is_valid=is_valid,
                    severity=(
                        ValidationSeverity.ERROR
                        if not is_valid
                        else ValidationSeverity.INFO
                    ),
                    message=(
                        f"Battery efficiency {battery_efficiency:.2%} is "
                        f"{'within' if is_valid else 'outside'} valid range"
                    ),
                    actual_value=battery_efficiency,
                    expected_range=assumption.valid_range,
                )
            )

    # Validate battery power
    if battery_power_mw is not None:
        assumption = registry.get("BATT-004")
        if assumption and assumption.valid_range:
            is_valid = (
                assumption.valid_range[0]
                <= battery_power_mw
                <= assumption.valid_range[1]
            )
            report.add_result(
                ValidationResult(
                    assumption_id="BATT-004",
                    is_valid=is_valid,
                    severity=(
                        ValidationSeverity.WARNING
                        if not is_valid
                        else ValidationSeverity.INFO
                    ),
                    message=(
                        f"Battery power {battery_power_mw} MW is "
                        f"{'within' if is_valid else 'outside'} valid range"
                    ),
                    actual_value=battery_power_mw,
                    expected_range=assumption.valid_range,
                )
            )

    # Validate degradation cost
    if degradation_cost is not None:
        assumption = registry.get("BATT-002")
        if assumption and assumption.valid_range:
            is_valid = (
                assumption.valid_range[0]
                <= degradation_cost
                <= assumption.valid_range[1]
            )
            report.add_result(
                ValidationResult(
                    assumption_id="BATT-002",
                    is_valid=is_valid,
                    severity=(
                        ValidationSeverity.WARNING
                        if not is_valid
                        else ValidationSeverity.INFO
                    ),
                    message=(
                        f"Degradation cost ${degradation_cost}/MWh is "
                        f"{'within' if is_valid else 'outside'} valid range"
                    ),
                    actual_value=degradation_cost,
                    expected_range=assumption.valid_range,
                )
            )

    # Validate forecast horizon
    if forecast_horizon_hours is not None:
        assumption = registry.get("FCST-002")
        if assumption and assumption.valid_range:
            is_valid = (
                assumption.valid_range[0]
                <= forecast_horizon_hours
                <= assumption.valid_range[1]
            )
            report.add_result(
                ValidationResult(
                    assumption_id="FCST-002",
                    is_valid=is_valid,
                    severity=(
                        ValidationSeverity.WARNING
                        if not is_valid
                        else ValidationSeverity.INFO
                    ),
                    message=(
                        f"Forecast horizon {forecast_horizon_hours} hours is "
                        f"{'within' if is_valid else 'outside'} valid range"
                    ),
                    actual_value=forecast_horizon_hours,
                    expected_range=assumption.valid_range,
                )
            )

    return report
