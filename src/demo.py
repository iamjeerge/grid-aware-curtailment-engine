"""Demo module for showcasing the Grid-Aware Curtailment Engine.

This module provides ready-to-run demo scenarios that showcase
the key capabilities of the optimization engine.

Usage:
    python -m src.demo

Or in Python:
    from src.demo import run_duck_curve_demo
    results = run_duck_curve_demo()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.battery.physics import BatteryModel
from src.controllers.naive import NaiveController
from src.domain.models import (
    BatteryConfig,
    BatteryState,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
)
from src.generators import (
    CongestionPattern,
    GenerationGenerator,
    GridConstraintGenerator,
    MarketPriceGenerator,
    PricePattern,
)
from src.metrics import (
    AnalyzerConfig,
    PerformanceAnalyzer,
    PerformanceSummary,
    StrategyComparison,
)
from src.optimization.milp import MILPOptimizer, OptimizationConfig
from src.reporting import ReportConfig, generate_executive_report
from src.visualization import (
    create_comparison_dashboard,
    create_dispatch_timeline,
)


@dataclass
class DemoConfig:
    """Configuration for demo scenarios.

    Attributes:
        horizon_hours: Planning horizon in hours.
        peak_generation_mw: Peak solar generation capacity.
        grid_limit_mw: Grid export capacity limit.
        battery_capacity_mwh: Battery energy capacity.
        battery_power_mw: Battery power rating.
        seed: Random seed for reproducibility.
    """

    horizon_hours: int = 24
    peak_generation_mw: float = 600.0
    grid_limit_mw: float = 300.0
    battery_capacity_mwh: float = 500.0
    battery_power_mw: float = 150.0
    seed: int = 42


@dataclass
class DemoResults:
    """Results from running a demo scenario.

    Attributes:
        naive_summary: Performance summary for naive strategy.
        milp_summary: Performance summary for MILP strategy.
        comparison: Strategy comparison object.
        naive_decisions: List of naive controller decisions.
        milp_decisions: List of MILP optimizer decisions.
        forecasts: Generation forecasts used.
        prices: Market prices used.
        constraints: Grid constraints used.
    """

    naive_summary: PerformanceSummary
    milp_summary: PerformanceSummary
    comparison: StrategyComparison
    naive_decisions: list[OptimizationDecision] = field(default_factory=list)
    milp_decisions: list[OptimizationDecision] = field(default_factory=list)
    forecasts: list[GenerationForecast] = field(default_factory=list)
    prices: list[MarketPrice] = field(default_factory=list)
    constraints: list[GridConstraint] = field(default_factory=list)

    @property
    def curtailment_reduction(self) -> float:
        """Calculate curtailment reduction percentage."""
        naive_rate = self.naive_summary.curtailment.curtailment_rate
        milp_rate = self.milp_summary.curtailment.curtailment_rate
        if naive_rate > 0:
            return (naive_rate - milp_rate) / naive_rate
        return 0.0

    @property
    def revenue_uplift(self) -> float:
        """Calculate revenue uplift in dollars."""
        return (
            self.milp_summary.revenue.net_profit - self.naive_summary.revenue.net_profit
        )

    @property
    def revenue_uplift_pct(self) -> float:
        """Calculate revenue uplift percentage."""
        naive = self.naive_summary.revenue.net_profit
        if naive > 0:
            return self.revenue_uplift / naive
        return 0.0

    @property
    def violations(self) -> int:
        """Get number of grid violations in MILP solution."""
        return self.milp_summary.grid_compliance.violation_count

    def print_summary(self) -> None:
        """Print a summary of demo results to console."""
        print("\n" + "=" * 60)
        print("🦆 Duck Curve Optimization Demo Results")
        print("=" * 60)

        print("\n📊 Naive Strategy:")
        print(
            f"   • Curtailment Rate: "
            f"{self.naive_summary.curtailment.curtailment_rate:.1%}"
        )
        print(f"   • Net Profit: ${self.naive_summary.revenue.net_profit:,.0f}")
        print(
            f"   • Grid Violations: "
            f"{self.naive_summary.grid_compliance.violation_count}"
        )

        print("\n⚡ MILP Optimized Strategy:")
        print(
            f"   • Curtailment Rate: "
            f"{self.milp_summary.curtailment.curtailment_rate:.1%}"
        )
        print(f"   • Net Profit: ${self.milp_summary.revenue.net_profit:,.0f}")
        print(
            f"   • Grid Violations: "
            f"{self.milp_summary.grid_compliance.violation_count}"
        )

        print("\n✅ Improvement:")
        print(f"   • Curtailment Reduced: {self.curtailment_reduction:.1%}")
        print(
            f"   • Revenue Uplift: {self.revenue_uplift_pct:.1%} "
            f"(+${self.revenue_uplift:,.0f})"
        )
        print(f"   • Zero Violations: {'✓' if self.violations == 0 else '✗'}")
        print("=" * 60 + "\n")

    def plot_dashboard(self, save_path: str | Path | None = None) -> Any:
        """Generate and optionally save visualization dashboard."""
        if self.milp_decisions:
            fig = create_dispatch_timeline(
                decisions=self.milp_decisions,
                prices=self.prices,
                constraints=self.constraints,
                title="MILP Optimized Dispatch",
            )
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Dashboard saved to {save_path}")
            return fig
        return None

    def plot_comparison(self, save_path: str | Path | None = None) -> Any:
        """Generate strategy comparison visualization."""
        fig = create_comparison_dashboard(self.comparison)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comparison saved to {save_path}")
        return fig

    def generate_report(self, output_path: str | Path) -> Path:
        """Generate executive PDF report."""
        config = ReportConfig(
            title="Duck Curve Optimization Analysis",
            author="Grid-Aware Curtailment Engine",
            company="Demo Scenario",
        )
        return generate_executive_report(
            comparison=self.comparison,
            summary=self.milp_summary,
            output_path=output_path,
            config=config,
        )


def generate_duck_curve_data(
    config: DemoConfig,
) -> tuple[list[GenerationForecast], list[MarketPrice], list[GridConstraint]]:
    """Generate synthetic data for the duck curve scenario.

    Args:
        config: Demo configuration.

    Returns:
        Tuple of (forecasts, prices, constraints).
    """
    _ = np.random.default_rng(config.seed)  # For reproducibility
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate solar forecasts with duck curve pattern
    gen_generator = GenerationGenerator(
        solar_capacity_mw=config.peak_generation_mw,
        wind_capacity_mw=0.0,  # Solar only for duck curve demo
        seed=config.seed,
    )

    forecasts: list[GenerationForecast] = []
    for hour in range(config.horizon_hours):
        timestamp = base_time + timedelta(hours=hour)
        forecast = gen_generator.generate_forecast(timestamp)
        forecasts.append(forecast)

    # Generate prices with duck curve pattern (negative midday, evening spike)
    price_gen = MarketPriceGenerator(
        base_price_multiplier=1.0,
        volatility=0.1,
        seed=config.seed,
    )

    prices: list[MarketPrice] = []
    for hour in range(config.horizon_hours):
        timestamp = base_time + timedelta(hours=hour)
        price = price_gen.generate_price(
            timestamp=timestamp,
            pattern=PricePattern.DUCK_CURVE,
            include_real_time=True,
        )
        prices.append(price)

    # Generate grid constraints with midday congestion
    grid_gen = GridConstraintGenerator(
        base_export_capacity_mw=config.grid_limit_mw,
        min_export_capacity_mw=config.grid_limit_mw * 0.5,
        max_ramp_rate_mw_per_hour=150.0,
        seed=config.seed,
    )

    constraints: list[GridConstraint] = []
    for hour in range(config.horizon_hours):
        timestamp = base_time + timedelta(hours=hour)
        constraint = grid_gen.generate_constraint(
            timestamp=timestamp,
            pattern=CongestionPattern.MIDDAY,
        )
        constraints.append(constraint)

    return forecasts, prices, constraints


def run_naive_strategy(
    forecasts: list[GenerationForecast],
    prices: list[MarketPrice],
    constraints: list[GridConstraint],
    battery_config: BatteryConfig,
) -> tuple[list[OptimizationDecision], PerformanceSummary]:
    """Run the naive heuristic strategy.

    Args:
        forecasts: Generation forecasts.
        prices: Market prices.
        constraints: Grid constraints.
        battery_config: Battery configuration.

    Returns:
        Tuple of (decisions, performance_summary).
    """
    controller = NaiveController(battery_config)
    battery = BatteryModel(battery_config)

    # Initialize analyzer for metrics
    analyzer_config = AnalyzerConfig(battery_config=battery_config)
    analyzer = PerformanceAnalyzer(analyzer_config)

    decisions: list[OptimizationDecision] = []
    state = BatteryState(
        soc_mwh=battery_config.capacity_mwh * 0.5,
        capacity_mwh=battery_config.capacity_mwh,
        max_charge_mw=battery_config.max_power_mw,
        max_discharge_mw=battery_config.max_power_mw,
    )

    for forecast, price, constraint in zip(forecasts, prices, constraints, strict=True):
        # Use P50 scenario for single-point forecasts
        decision = controller.dispatch(
            timestamp=forecast.timestamp,
            generation_mw=forecast.p50_mw,
            grid_constraint=constraint,
            market_price=price,
            scenario=ForecastScenario.P50,
        )
        decisions.append(decision)

        # Update battery state
        state = battery.apply_decision(state, decision)

    # Analyze performance
    summary = analyzer.analyze_decisions(
        decisions=decisions,
        forecasts=forecasts,
        prices=prices,
        constraints=constraints,
        strategy_name="Naive",
    )

    return decisions, summary


def run_milp_strategy(
    forecasts: list[GenerationForecast],
    prices: list[MarketPrice],
    constraints: list[GridConstraint],
    battery_config: BatteryConfig,
) -> tuple[list[OptimizationDecision], PerformanceSummary]:
    """Run the MILP optimization strategy.

    Args:
        forecasts: Generation forecasts.
        prices: Market prices.
        constraints: Grid constraints.
        battery_config: Battery configuration.

    Returns:
        Tuple of (decisions, performance_summary).
    """
    opt_config = OptimizationConfig(
        solver_name="glpk",
        time_limit_seconds=60.0,
        mip_gap=0.01,
    )
    optimizer = MILPOptimizer(battery_config, opt_config)

    # Initialize analyzer for metrics
    analyzer_config = AnalyzerConfig(battery_config=battery_config)
    analyzer = PerformanceAnalyzer(analyzer_config)

    # Run optimization
    result = optimizer.optimize(
        forecasts=forecasts,
        grid_constraints=constraints,
        market_prices=prices,
    )

    # Convert result to decisions list
    decisions = optimizer.get_decisions(result, ForecastScenario.P50)

    # Analyze performance
    summary = analyzer.analyze_decisions(
        decisions=decisions,
        forecasts=forecasts,
        prices=prices,
        constraints=constraints,
        strategy_name="MILP",
    )

    return decisions, summary


def run_duck_curve_demo(config: DemoConfig | None = None) -> DemoResults:
    """Run the complete duck curve optimization demo.

    This is the main entry point for demonstrating the optimization
    engine's capabilities on the canonical duck curve scenario.

    Args:
        config: Optional demo configuration. Uses defaults if not provided.

    Returns:
        DemoResults with all metrics and decisions.

    Example:
        >>> results = run_duck_curve_demo()
        >>> results.print_summary()
        >>> results.plot_dashboard("output/dashboard.png")
    """
    if config is None:
        config = DemoConfig()

    print("\n🦆 Running Duck Curve Optimization Demo...")
    print(f"   • Horizon: {config.horizon_hours} hours")
    print(f"   • Peak Generation: {config.peak_generation_mw} MW")
    print(f"   • Grid Limit: {config.grid_limit_mw} MW")
    print(
        f"   • Battery: {config.battery_capacity_mwh} MWh / "
        f"{config.battery_power_mw} MW"
    )

    # Generate scenario data
    print("\n📊 Generating scenario data...")
    forecasts, prices, constraints = generate_duck_curve_data(config)

    # Configure battery
    battery_config = BatteryConfig(
        capacity_mwh=config.battery_capacity_mwh,
        max_power_mw=config.battery_power_mw,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_fraction=0.10,
        max_soc_fraction=0.90,
        degradation_cost_per_mwh=8.0,
    )

    # Run naive strategy
    print("⚙️  Running naive heuristic...")
    naive_decisions, naive_summary = run_naive_strategy(
        forecasts, prices, constraints, battery_config
    )

    # Run MILP strategy
    print("⚡ Running MILP optimization...")
    milp_decisions, milp_summary = run_milp_strategy(
        forecasts, prices, constraints, battery_config
    )

    # Create comparison
    comparison = StrategyComparison.from_summaries(
        summaries=[naive_summary, milp_summary],
    )

    results = DemoResults(
        naive_summary=naive_summary,
        milp_summary=milp_summary,
        comparison=comparison,
        naive_decisions=naive_decisions,
        milp_decisions=milp_decisions,
        forecasts=forecasts,
        prices=prices,
        constraints=constraints,
    )

    print("✅ Demo complete!")
    return results


def main() -> None:
    """Main entry point for running demos from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Grid-Aware Curtailment Engine Demo")
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Planning horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--peak-gen",
        type=float,
        default=600.0,
        help="Peak generation capacity in MW (default: 600)",
    )
    parser.add_argument(
        "--grid-limit",
        type=float,
        default=300.0,
        help="Grid export limit in MW (default: 300)",
    )
    parser.add_argument(
        "--battery-mwh",
        type=float,
        default=500.0,
        help="Battery capacity in MWh (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports and charts",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate PDF report",
    )

    args = parser.parse_args()

    config = DemoConfig(
        horizon_hours=args.horizon,
        peak_generation_mw=args.peak_gen,
        grid_limit_mw=args.grid_limit,
        battery_capacity_mwh=args.battery_mwh,
    )

    results = run_duck_curve_demo(config)
    results.print_summary()

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results.plot_dashboard(output_dir / "dashboard.png")
        results.plot_comparison(output_dir / "comparison.png")

        if args.report:
            report_path = results.generate_report(output_dir / "report.pdf")
            print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
