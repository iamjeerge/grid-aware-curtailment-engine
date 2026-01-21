"""Performance analyzer for optimization results.

This module provides tools for analyzing optimization results, comparing
strategies, and generating performance reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.domain.models import BatteryConfig, ForecastScenario
from src.metrics.kpi import (
    BatteryMetrics,
    CurtailmentMetrics,
    GridComplianceMetrics,
    PerformanceSummary,
    RevenueMetrics,
    StrategyComparison,
)

if TYPE_CHECKING:
    from src.domain.models import (
        GenerationForecast,
        GridConstraint,
        MarketPrice,
        OptimizationDecision,
    )
    from src.hybrid.controller import HybridResult


@dataclass
class AnalyzerConfig:
    """Configuration for performance analyzer.

    Attributes:
        battery_config: Battery configuration for metrics.
        include_baseline: Whether to compute baseline comparison.
        baseline_name: Name for baseline strategy.
        degradation_cost_per_mwh: Degradation cost for battery cycling.
        violation_penalty_per_mw: Penalty cost per MW of grid violation.
    """

    battery_config: BatteryConfig = field(default_factory=BatteryConfig)
    include_baseline: bool = True
    baseline_name: str = "naive"
    degradation_cost_per_mwh: float = 8.0
    violation_penalty_per_mw: float = 100.0


class PerformanceAnalyzer:
    """Analyzer for computing and comparing optimization performance.

    Analyzes dispatch decisions to compute comprehensive KPIs including
    curtailment reduction, revenue uplift, battery utilization, and
    grid compliance scores.

    Example:
        ```python
        analyzer = PerformanceAnalyzer(config)

        # Analyze a single strategy
        summary = analyzer.analyze_decisions(
            decisions=milp_decisions,
            forecasts=forecasts,
            prices=prices,
            constraints=constraints,
            strategy_name="MILP",
        )

        # Compare multiple strategies
        comparison = analyzer.compare_strategies(
            {"naive": naive_decisions, "milp": milp_decisions, "rl": rl_decisions},
            forecasts, prices, constraints,
        )
        print(f"Best: {comparison.best_strategy}")
        ```
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize the performance analyzer.

        Args:
            config: Analyzer configuration.
        """
        self.config = config or AnalyzerConfig()

    def analyze_decisions(
        self,
        decisions: list[OptimizationDecision],
        forecasts: list[GenerationForecast],  # noqa: ARG002
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        strategy_name: str = "unknown",
        scenario_name: str = "default",
        baseline_decisions: list[OptimizationDecision] | None = None,
    ) -> PerformanceSummary:
        """Analyze optimization decisions and compute KPIs.

        Args:
            decisions: List of optimization decisions.
            forecasts: Generation forecasts (used for interface compatibility).
            prices: Market prices.
            constraints: Grid constraints.
            strategy_name: Name of the strategy.
            scenario_name: Name of the scenario.
            baseline_decisions: Baseline decisions for comparison.

        Returns:
            PerformanceSummary with all computed metrics.
        """
        if not decisions:
            return PerformanceSummary(
                strategy_name=strategy_name,
                scenario_name=scenario_name,
            )

        # Extract data from decisions
        sold = [d.energy_sold_mw for d in decisions]
        stored = [d.energy_stored_mw for d in decisions]
        curtailed = [d.energy_curtailed_mw for d in decisions]
        discharged = [d.battery_discharge_mw for d in decisions]
        soc_values = [d.resulting_soc_mwh for d in decisions]

        # Get generation, prices, and constraints
        generations = [d.generation_mw for d in decisions]
        price_values = [p.effective_price for p in prices[: len(decisions)]]
        capacity_limits = [c.max_export_mw for c in constraints[: len(decisions)]]
        ramp_limits = [c.max_ramp_up_mw_per_hour for c in constraints[: len(decisions)]]

        # Baseline data for comparison
        baseline_curtailed = None
        baseline_revenue = None
        if baseline_decisions is not None:
            baseline_curtailed = [d.energy_curtailed_mw for d in baseline_decisions]
            baseline_sold = [d.energy_sold_mw for d in baseline_decisions]
            baseline_discharged = [d.battery_discharge_mw for d in baseline_decisions]
            baseline_revenue = sum(
                (s + d) * p
                for s, d, p in zip(
                    baseline_sold, baseline_discharged, price_values, strict=False
                )
            )

        # Compute curtailment metrics
        curtailment_metrics = CurtailmentMetrics.from_dispatch(
            generations=generations,
            curtailed=curtailed,
            sold=sold,
            stored=stored,
            baseline_curtailed=baseline_curtailed,
        )

        # Compute degradation cost
        throughput = sum(stored) + sum(discharged)
        degradation_cost = (
            throughput * self.config.degradation_cost_per_mwh / 2
        )  # Round-trip

        # Compute penalty cost for violations
        penalty_cost = 0.0
        for s, cap in zip(sold, capacity_limits, strict=False):
            if s > cap:
                penalty_cost += (s - cap) * self.config.violation_penalty_per_mw

        # Compute revenue metrics
        revenue_metrics = RevenueMetrics.from_dispatch(
            sold=sold,
            discharged=discharged,
            prices=price_values,
            degradation_cost=degradation_cost,
            penalty_cost=penalty_cost,
            baseline_revenue=baseline_revenue,
        )

        # Compute battery metrics
        battery_metrics = BatteryMetrics.from_dispatch(
            charged=stored,
            discharged=discharged,
            soc_values=soc_values,
            prices=price_values,
            battery_capacity_mwh=self.config.battery_config.capacity_mwh,
            degradation_cost_per_mwh=self.config.degradation_cost_per_mwh,
            charge_efficiency=self.config.battery_config.charge_efficiency,
        )

        # Compute grid compliance metrics
        compliance_metrics = GridComplianceMetrics.from_dispatch(
            sold=sold,
            capacity_limits=capacity_limits,
            ramp_limits=ramp_limits,
        )

        return PerformanceSummary(
            timestamp=datetime.now(),
            horizon_hours=len(decisions),
            curtailment=curtailment_metrics,
            revenue=revenue_metrics,
            battery=battery_metrics,
            grid_compliance=compliance_metrics,
            strategy_name=strategy_name,
            scenario_name=scenario_name,
        )

    def analyze_hybrid_result(
        self,
        result: HybridResult,
        forecasts: list[GenerationForecast],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        strategy_name: str = "hybrid",
        scenario_name: str = "default",
        baseline_result: HybridResult | None = None,
    ) -> PerformanceSummary:
        """Analyze a HybridResult from the hybrid controller.

        Args:
            result: Hybrid controller result.
            forecasts: Generation forecasts.
            prices: Market prices.
            constraints: Grid constraints.
            strategy_name: Name of the strategy.
            scenario_name: Name of the scenario.
            baseline_result: Baseline result for comparison.

        Returns:
            PerformanceSummary with all computed metrics.
        """
        if not result.decisions:
            return PerformanceSummary(
                strategy_name=strategy_name,
                scenario_name=scenario_name,
            )

        # Extract data from dispatch decisions
        sold = [d.energy_sold_mw for d in result.decisions]
        stored = [d.energy_stored_mw for d in result.decisions]
        curtailed = [d.energy_curtailed_mw for d in result.decisions]
        soc_values = [d.battery_soc_mwh for d in result.decisions]

        # Discharged is implicit - computed from SOC changes
        discharged = []
        for i, d in enumerate(result.decisions):
            if i == 0:
                discharged.append(0.0)
            else:
                prev_soc = result.decisions[i - 1].battery_soc_mwh
                curr_soc = d.battery_soc_mwh
                charge = (
                    d.energy_stored_mw * self.config.battery_config.charge_efficiency
                )
                # discharge = (prev_soc + charge - curr_soc) / efficiency
                delta = prev_soc + charge - curr_soc
                discharge = max(
                    0, delta / self.config.battery_config.discharge_efficiency
                )
                discharged.append(discharge)

        # Get generation from forecasts (P50)
        generations = [
            f.total_generation(ForecastScenario.P50)
            for f in forecasts[: len(result.decisions)]
        ]
        price_values = [p.effective_price for p in prices[: len(result.decisions)]]
        capacity_limits = [
            c.max_export_mw for c in constraints[: len(result.decisions)]
        ]
        ramp_limits = [
            c.max_ramp_up_mw_per_hour for c in constraints[: len(result.decisions)]
        ]

        # Baseline data
        baseline_curtailed = None
        baseline_revenue = None
        if baseline_result is not None and baseline_result.decisions:
            baseline_curtailed = [
                d.energy_curtailed_mw for d in baseline_result.decisions
            ]
            baseline_sold = [d.energy_sold_mw for d in baseline_result.decisions]
            baseline_revenue = sum(
                s * p for s, p in zip(baseline_sold, price_values, strict=False)
            )

        # Compute all metrics
        curtailment_metrics = CurtailmentMetrics.from_dispatch(
            generations=generations,
            curtailed=curtailed,
            sold=sold,
            stored=stored,
            baseline_curtailed=baseline_curtailed,
        )

        # Degradation and penalty costs
        throughput = sum(stored) + sum(discharged)
        degradation_cost = throughput * self.config.degradation_cost_per_mwh / 2
        penalty_cost = sum(
            max(0, s - cap) * self.config.violation_penalty_per_mw
            for s, cap in zip(sold, capacity_limits, strict=False)
        )

        revenue_metrics = RevenueMetrics.from_dispatch(
            sold=sold,
            discharged=discharged,
            prices=price_values,
            degradation_cost=degradation_cost,
            penalty_cost=penalty_cost,
            baseline_revenue=baseline_revenue,
        )

        battery_metrics = BatteryMetrics.from_dispatch(
            charged=stored,
            discharged=discharged,
            soc_values=soc_values,
            prices=price_values,
            battery_capacity_mwh=self.config.battery_config.capacity_mwh,
            degradation_cost_per_mwh=self.config.degradation_cost_per_mwh,
            charge_efficiency=self.config.battery_config.charge_efficiency,
        )

        compliance_metrics = GridComplianceMetrics.from_dispatch(
            sold=sold,
            capacity_limits=capacity_limits,
            ramp_limits=ramp_limits,
        )

        return PerformanceSummary(
            timestamp=datetime.now(),
            horizon_hours=len(result.decisions),
            curtailment=curtailment_metrics,
            revenue=revenue_metrics,
            battery=battery_metrics,
            grid_compliance=compliance_metrics,
            strategy_name=strategy_name,
            scenario_name=scenario_name,
        )

    def compare_strategies(
        self,
        strategy_decisions: dict[str, list[OptimizationDecision]],
        forecasts: list[GenerationForecast],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        scenario_name: str = "default",
        baseline_strategy: str | None = None,
    ) -> StrategyComparison:
        """Compare multiple optimization strategies.

        Args:
            strategy_decisions: Dictionary mapping strategy name to decisions.
            forecasts: Generation forecasts.
            prices: Market prices.
            constraints: Grid constraints.
            scenario_name: Name of the scenario.
            baseline_strategy: Name of baseline strategy for comparison.

        Returns:
            StrategyComparison with all strategies analyzed.
        """
        comparison = StrategyComparison()

        # Get baseline decisions if specified
        baseline_decisions = None
        if baseline_strategy and baseline_strategy in strategy_decisions:
            baseline_decisions = strategy_decisions[baseline_strategy]

        for name, decisions in strategy_decisions.items():
            summary = self.analyze_decisions(
                decisions=decisions,
                forecasts=forecasts,
                prices=prices,
                constraints=constraints,
                strategy_name=name,
                scenario_name=scenario_name,
                baseline_decisions=(
                    baseline_decisions if name != baseline_strategy else None
                ),
            )
            comparison.strategies[name] = summary

        return comparison


def compute_kpis_from_decisions(
    decisions: list[OptimizationDecision],
    prices: list[MarketPrice],
    constraints: list[GridConstraint],
    battery_config: BatteryConfig | None = None,
) -> dict[str, float]:
    """Convenience function to compute key KPIs from decisions.

    Args:
        decisions: Optimization decisions.
        prices: Market prices.
        constraints: Grid constraints.
        battery_config: Battery configuration.

    Returns:
        Dictionary of key KPI values.
    """
    if not decisions:
        return {
            "curtailment_rate": 0.0,
            "revenue": 0.0,
            "battery_cycles": 0.0,
            "compliance_rate": 1.0,
        }

    battery_config = battery_config or BatteryConfig()
    analyzer = PerformanceAnalyzer(AnalyzerConfig(battery_config=battery_config))

    # Create placeholder forecasts
    from src.domain.models import GenerationForecast

    forecasts = [
        GenerationForecast(
            timestamp=d.timestamp,
            solar_mw_p50=d.generation_mw * 0.8,
            wind_mw_p50=d.generation_mw * 0.2,
        )
        for d in decisions
    ]

    summary = analyzer.analyze_decisions(
        decisions=decisions,
        forecasts=forecasts,
        prices=prices,
        constraints=constraints,
    )

    return {
        "curtailment_rate": summary.curtailment.curtailment_rate,
        "curtailment_mwh": summary.curtailment.total_curtailed_mwh,
        "revenue": summary.revenue.total_revenue,
        "net_profit": summary.revenue.net_profit,
        "battery_cycles": summary.battery.total_cycles,
        "battery_utilization": summary.battery.utilization_rate,
        "compliance_rate": summary.grid_compliance.compliance_rate,
        "violations": summary.grid_compliance.violation_count,
        "overall_score": summary.overall_score,
    }


def quick_kpi_summary(
    sold: list[float],
    stored: list[float],
    curtailed: list[float],
    prices: list[float],
    constraints: list[float],
    battery_capacity: float = 500.0,
) -> dict[str, float]:
    """Quick KPI calculation from raw data.

    Args:
        sold: Sold energy per timestep (MW).
        stored: Stored energy per timestep (MW).
        curtailed: Curtailed energy per timestep (MW).
        prices: Prices per timestep ($/MWh).
        constraints: Export limits per timestep (MW).
        battery_capacity: Battery capacity (MWh).

    Returns:
        Dictionary of key KPI values.
    """
    # Total generation
    total_gen = sum(sold) + sum(stored) + sum(curtailed)

    # Curtailment rate
    curt_rate = sum(curtailed) / max(total_gen, 0.001)

    # Revenue
    revenue = sum(s * p for s, p in zip(sold, prices, strict=False))

    # Battery cycles
    cycles = sum(stored) / max(battery_capacity, 0.001)

    # Compliance
    violations = sum(1 for s, c in zip(sold, constraints, strict=False) if s > c * 1.01)
    compliance_rate = (len(sold) - violations) / max(len(sold), 1)

    return {
        "total_generation_mwh": total_gen,
        "total_sold_mwh": sum(sold),
        "total_stored_mwh": sum(stored),
        "total_curtailed_mwh": sum(curtailed),
        "curtailment_rate": curt_rate,
        "revenue": revenue,
        "battery_cycles": cycles,
        "compliance_rate": compliance_rate,
        "violation_count": violations,
    }
