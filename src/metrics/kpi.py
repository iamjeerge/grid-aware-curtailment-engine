"""Performance metrics and business KPIs for curtailment optimization.

This module provides comprehensive KPI calculations for evaluating the
performance of curtailment optimization strategies.

Key Metrics:
- Curtailment avoided (%)
- Revenue uplift (%)
- Battery utilization
- Degradation cost
- Grid compliance score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class CurtailmentMetrics:
    """Metrics related to curtailment performance.

    Attributes:
        total_generation_mwh: Total energy generated.
        total_curtailed_mwh: Total energy curtailed.
        total_sold_mwh: Total energy sold to grid.
        total_stored_mwh: Total energy stored in battery.
        curtailment_rate: Fraction of generation curtailed.
        curtailment_avoided_mwh: Energy saved from curtailment vs baseline.
        curtailment_avoided_pct: Percentage reduction vs baseline.
    """

    total_generation_mwh: float = 0.0
    total_curtailed_mwh: float = 0.0
    total_sold_mwh: float = 0.0
    total_stored_mwh: float = 0.0
    curtailment_rate: float = 0.0
    curtailment_avoided_mwh: float = 0.0
    curtailment_avoided_pct: float = 0.0

    @classmethod
    def from_dispatch(
        cls,
        generations: list[float],
        curtailed: list[float],
        sold: list[float],
        stored: list[float],
        baseline_curtailed: list[float] | None = None,
    ) -> CurtailmentMetrics:
        """Calculate curtailment metrics from dispatch data.

        Args:
            generations: Generation values per timestep (MW).
            curtailed: Curtailed energy per timestep (MW).
            sold: Sold energy per timestep (MW).
            stored: Stored energy per timestep (MW).
            baseline_curtailed: Baseline curtailment for comparison.

        Returns:
            CurtailmentMetrics with calculated values.
        """
        total_gen = sum(generations)
        total_curt = sum(curtailed)
        total_sold = sum(sold)
        total_stored = sum(stored)

        # Curtailment rate
        curt_rate = total_curt / max(total_gen, 0.001)

        # Curtailment avoided
        avoided_mwh = 0.0
        avoided_pct = 0.0
        if baseline_curtailed is not None:
            baseline_total = sum(baseline_curtailed)
            avoided_mwh = baseline_total - total_curt
            avoided_pct = avoided_mwh / max(baseline_total, 0.001) * 100

        return cls(
            total_generation_mwh=total_gen,
            total_curtailed_mwh=total_curt,
            total_sold_mwh=total_sold,
            total_stored_mwh=total_stored,
            curtailment_rate=curt_rate,
            curtailment_avoided_mwh=avoided_mwh,
            curtailment_avoided_pct=avoided_pct,
        )


@dataclass
class RevenueMetrics:
    """Metrics related to revenue performance.

    Attributes:
        total_revenue: Total revenue from energy sales.
        revenue_from_sales: Revenue from direct grid sales.
        revenue_from_discharge: Revenue from battery discharge.
        total_costs: Total costs (degradation, penalties).
        degradation_cost: Battery degradation costs.
        penalty_cost: Costs from violations/penalties.
        net_profit: Total revenue minus costs.
        revenue_uplift_pct: Percentage improvement vs baseline.
        average_price_captured: Average price achieved ($/MWh).
    """

    total_revenue: float = 0.0
    revenue_from_sales: float = 0.0
    revenue_from_discharge: float = 0.0
    total_costs: float = 0.0
    degradation_cost: float = 0.0
    penalty_cost: float = 0.0
    net_profit: float = 0.0
    revenue_uplift_pct: float = 0.0
    average_price_captured: float = 0.0

    @classmethod
    def from_dispatch(
        cls,
        sold: list[float],
        discharged: list[float],
        prices: list[float],
        degradation_cost: float = 0.0,
        penalty_cost: float = 0.0,
        baseline_revenue: float | None = None,
    ) -> RevenueMetrics:
        """Calculate revenue metrics from dispatch data.

        Args:
            sold: Sold energy per timestep (MW).
            discharged: Discharged energy per timestep (MW).
            prices: Market prices per timestep ($/MWh).
            degradation_cost: Total degradation cost.
            penalty_cost: Total penalty cost.
            baseline_revenue: Baseline revenue for comparison.

        Returns:
            RevenueMetrics with calculated values.
        """
        # Calculate revenue from sales
        revenue_sales = sum(s * p for s, p in zip(sold, prices, strict=False))

        # Calculate revenue from discharge
        revenue_discharge = sum(d * p for d, p in zip(discharged, prices, strict=False))

        total_revenue = revenue_sales + revenue_discharge
        total_costs = degradation_cost + penalty_cost
        net_profit = total_revenue - total_costs

        # Revenue uplift
        uplift_pct = 0.0
        if baseline_revenue is not None and baseline_revenue > 0:
            uplift_pct = (net_profit - baseline_revenue) / baseline_revenue * 100

        # Average price captured
        total_energy = sum(sold) + sum(discharged)
        avg_price = total_revenue / max(total_energy, 0.001)

        return cls(
            total_revenue=total_revenue,
            revenue_from_sales=revenue_sales,
            revenue_from_discharge=revenue_discharge,
            total_costs=total_costs,
            degradation_cost=degradation_cost,
            penalty_cost=penalty_cost,
            net_profit=net_profit,
            revenue_uplift_pct=uplift_pct,
            average_price_captured=avg_price,
        )


@dataclass
class BatteryMetrics:
    """Metrics related to battery utilization and health.

    Attributes:
        total_charged_mwh: Total energy charged.
        total_discharged_mwh: Total energy discharged.
        total_cycles: Equivalent full cycles.
        utilization_rate: Fraction of capacity utilized.
        average_soc: Mean state of charge.
        min_soc: Minimum SOC reached.
        max_soc: Maximum SOC reached.
        soc_range: Range of SOC values.
        throughput_mwh: Total energy throughput.
        degradation_cost: Total degradation cost.
        charge_efficiency_realized: Actual charging efficiency.
        arbitrage_value: Value captured from price arbitrage.
    """

    total_charged_mwh: float = 0.0
    total_discharged_mwh: float = 0.0
    total_cycles: float = 0.0
    utilization_rate: float = 0.0
    average_soc: float = 0.0
    min_soc: float = 0.0
    max_soc: float = 0.0
    soc_range: float = 0.0
    throughput_mwh: float = 0.0
    degradation_cost: float = 0.0
    charge_efficiency_realized: float = 0.0
    arbitrage_value: float = 0.0

    @classmethod
    def from_dispatch(
        cls,
        charged: list[float],
        discharged: list[float],
        soc_values: list[float],
        prices: list[float],
        battery_capacity_mwh: float = 500.0,
        degradation_cost_per_mwh: float = 8.0,
        charge_efficiency: float = 0.95,
    ) -> BatteryMetrics:
        """Calculate battery metrics from dispatch data.

        Args:
            charged: Charged energy per timestep (MW).
            discharged: Discharged energy per timestep (MW).
            soc_values: SOC at each timestep (MWh).
            prices: Market prices per timestep ($/MWh).
            battery_capacity_mwh: Battery capacity.
            degradation_cost_per_mwh: Degradation cost per MWh cycled.
            charge_efficiency: Charging efficiency.

        Returns:
            BatteryMetrics with calculated values.
        """
        total_charged = sum(charged)
        total_discharged = sum(discharged)

        # Equivalent full cycles (using discharge as the measure)
        cycles = total_discharged / max(battery_capacity_mwh, 0.001)

        # Throughput
        throughput = total_charged + total_discharged

        # SOC statistics
        if soc_values:
            avg_soc = np.mean(soc_values)
            min_soc = min(soc_values)
            max_soc = max(soc_values)
            soc_range = max_soc - min_soc
        else:
            avg_soc = min_soc = max_soc = soc_range = 0.0

        # Utilization rate (fraction of capacity used on average)
        utilization = avg_soc / max(battery_capacity_mwh, 0.001)

        # Degradation cost
        deg_cost = (
            throughput * degradation_cost_per_mwh / 2
        )  # Divide by 2 for round-trip

        # Realized charge efficiency
        if total_charged > 0:
            realized_eff = total_discharged / total_charged
        else:
            realized_eff = charge_efficiency

        # Arbitrage value - simplified calculation
        # Compare prices when charging vs discharging
        charge_prices = [p for c, p in zip(charged, prices, strict=False) if c > 0]
        discharge_prices = [
            p for d, p in zip(discharged, prices, strict=False) if d > 0
        ]

        if charge_prices and discharge_prices:
            avg_charge_price = np.mean(charge_prices)
            avg_discharge_price = np.mean(discharge_prices)
            arbitrage = (avg_discharge_price - avg_charge_price) * total_discharged
        else:
            arbitrage = 0.0

        return cls(
            total_charged_mwh=total_charged,
            total_discharged_mwh=total_discharged,
            total_cycles=cycles,
            utilization_rate=utilization,
            average_soc=float(avg_soc),
            min_soc=float(min_soc),
            max_soc=float(max_soc),
            soc_range=float(soc_range),
            throughput_mwh=throughput,
            degradation_cost=deg_cost,
            charge_efficiency_realized=realized_eff,
            arbitrage_value=arbitrage,
        )


@dataclass
class GridComplianceMetrics:
    """Metrics related to grid constraint compliance.

    Attributes:
        total_timesteps: Total timesteps analyzed.
        compliant_timesteps: Timesteps without violations.
        violation_count: Number of constraint violations.
        compliance_rate: Fraction of timesteps compliant.
        max_violation_mw: Maximum violation magnitude.
        total_violation_mwh: Total energy in violations.
        ramp_violations: Number of ramp rate violations.
        capacity_violations: Number of capacity violations.
    """

    total_timesteps: int = 0
    compliant_timesteps: int = 0
    violation_count: int = 0
    compliance_rate: float = 1.0
    max_violation_mw: float = 0.0
    total_violation_mwh: float = 0.0
    ramp_violations: int = 0
    capacity_violations: int = 0

    @classmethod
    def from_dispatch(
        cls,
        sold: list[float],
        capacity_limits: list[float],
        ramp_limits: list[float] | None = None,
    ) -> GridComplianceMetrics:
        """Calculate grid compliance metrics from dispatch data.

        Args:
            sold: Sold energy per timestep (MW).
            capacity_limits: Export capacity limits per timestep (MW).
            ramp_limits: Ramp rate limits per timestep (MW/hr).

        Returns:
            GridComplianceMetrics with calculated values.
        """
        n_steps = len(sold)

        capacity_violations = 0
        ramp_violations = 0
        max_violation = 0.0
        total_violation = 0.0

        for t, (s, cap) in enumerate(zip(sold, capacity_limits, strict=False)):
            # Check capacity violation
            if s > cap * 1.01:  # 1% tolerance
                capacity_violations += 1
                violation = s - cap
                max_violation = max(max_violation, violation)
                total_violation += violation

            # Check ramp violation
            if ramp_limits is not None and t > 0:
                ramp = abs(sold[t] - sold[t - 1])
                if ramp > ramp_limits[t] * 1.01:
                    ramp_violations += 1

        total_violations = capacity_violations + ramp_violations
        compliant = n_steps - total_violations

        return cls(
            total_timesteps=n_steps,
            compliant_timesteps=compliant,
            violation_count=total_violations,
            compliance_rate=compliant / max(n_steps, 1),
            max_violation_mw=max_violation,
            total_violation_mwh=total_violation,
            ramp_violations=ramp_violations,
            capacity_violations=capacity_violations,
        )


@dataclass
class PerformanceSummary:
    """Comprehensive performance summary combining all metrics.

    Attributes:
        timestamp: When the analysis was performed.
        horizon_hours: Planning horizon duration.
        curtailment: Curtailment-related metrics.
        revenue: Revenue-related metrics.
        battery: Battery-related metrics.
        grid_compliance: Grid compliance metrics.
        strategy_name: Name of the optimization strategy.
        scenario_name: Name of the scenario analyzed.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    horizon_hours: int = 24
    curtailment: CurtailmentMetrics = field(default_factory=CurtailmentMetrics)
    revenue: RevenueMetrics = field(default_factory=RevenueMetrics)
    battery: BatteryMetrics = field(default_factory=BatteryMetrics)
    grid_compliance: GridComplianceMetrics = field(
        default_factory=GridComplianceMetrics
    )
    strategy_name: str = "unknown"
    scenario_name: str = "default"

    @property
    def overall_score(self) -> float:
        """Calculate an overall performance score (0-100).

        Weights:
        - 30% Grid compliance
        - 25% Curtailment reduction
        - 25% Revenue performance
        - 20% Battery efficiency
        """
        # Grid compliance (0-30)
        compliance_score = self.grid_compliance.compliance_rate * 30

        # Curtailment score (0-25) - lower curtailment is better
        curt_score = (1 - self.curtailment.curtailment_rate) * 25

        # Revenue score (0-25) - normalized, positive uplift is good
        # Cap uplift at 50% for scoring purposes
        revenue_uplift = min(max(self.revenue.revenue_uplift_pct, -50), 50)
        revenue_score = (revenue_uplift + 50) / 100 * 25

        # Battery score (0-20) - based on utilization and efficiency
        batt_util_score = min(self.battery.utilization_rate, 1.0) * 10
        batt_eff_score = self.battery.charge_efficiency_realized * 10

        return (
            compliance_score
            + curt_score
            + revenue_score
            + batt_util_score
            + batt_eff_score
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "horizon_hours": self.horizon_hours,
            "strategy_name": self.strategy_name,
            "scenario_name": self.scenario_name,
            "overall_score": self.overall_score,
            "curtailment": {
                "total_generation_mwh": self.curtailment.total_generation_mwh,
                "total_curtailed_mwh": self.curtailment.total_curtailed_mwh,
                "curtailment_rate": self.curtailment.curtailment_rate,
                "curtailment_avoided_pct": self.curtailment.curtailment_avoided_pct,
            },
            "revenue": {
                "total_revenue": self.revenue.total_revenue,
                "net_profit": self.revenue.net_profit,
                "revenue_uplift_pct": self.revenue.revenue_uplift_pct,
                "average_price_captured": self.revenue.average_price_captured,
            },
            "battery": {
                "total_cycles": self.battery.total_cycles,
                "utilization_rate": self.battery.utilization_rate,
                "degradation_cost": self.battery.degradation_cost,
                "arbitrage_value": self.battery.arbitrage_value,
            },
            "grid_compliance": {
                "compliance_rate": self.grid_compliance.compliance_rate,
                "violation_count": self.grid_compliance.violation_count,
                "max_violation_mw": self.grid_compliance.max_violation_mw,
            },
        }


@dataclass
class StrategyComparison:
    """Comparison between multiple optimization strategies.

    Attributes:
        strategies: Dictionary mapping strategy name to PerformanceSummary.
        best_strategy: Name of the best performing strategy.
        ranking: Ordered list of strategy names by score.
    """

    strategies: dict[str, PerformanceSummary] = field(default_factory=dict)

    @property
    def best_strategy(self) -> str:
        """Get the name of the best performing strategy."""
        if not self.strategies:
            return "none"
        return max(
            self.strategies.keys(), key=lambda k: self.strategies[k].overall_score
        )

    @property
    def ranking(self) -> list[str]:
        """Get strategies ranked by score."""
        return sorted(
            self.strategies.keys(),
            key=lambda k: self.strategies[k].overall_score,
            reverse=True,
        )

    def get_comparison_table(self) -> list[dict]:
        """Get comparison data as a list of dicts."""
        result = []
        for name in self.ranking:
            summary = self.strategies[name]
            result.append(
                {
                    "strategy": name,
                    "score": summary.overall_score,
                    "revenue": summary.revenue.net_profit,
                    "curtailment_rate": summary.curtailment.curtailment_rate * 100,
                    "compliance_rate": summary.grid_compliance.compliance_rate * 100,
                    "battery_cycles": summary.battery.total_cycles,
                }
            )
        return result

    def get_delta(
        self,
        strategy1: str,
        strategy2: str,
    ) -> dict[str, float]:
        """Calculate delta between two strategies.

        Args:
            strategy1: First strategy name.
            strategy2: Second strategy name (baseline).

        Returns:
            Dictionary of metric deltas (strategy1 - strategy2).
        """
        if strategy1 not in self.strategies or strategy2 not in self.strategies:
            return {}

        s1 = self.strategies[strategy1]
        s2 = self.strategies[strategy2]

        return {
            "score_delta": s1.overall_score - s2.overall_score,
            "revenue_delta": s1.revenue.net_profit - s2.revenue.net_profit,
            "curtailment_delta_pct": (
                s1.curtailment.curtailment_rate - s2.curtailment.curtailment_rate
            )
            * 100,
            "compliance_delta_pct": (
                s1.grid_compliance.compliance_rate - s2.grid_compliance.compliance_rate
            )
            * 100,
        }
