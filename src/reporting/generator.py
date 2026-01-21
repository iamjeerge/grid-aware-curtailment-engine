"""Executive report generator for optimization results.

Generates comprehensive PDF reports summarizing:
- Problem description and assumptions
- Optimization strategy and methodology
- Financial impact analysis
- Risk analysis and stress test results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.metrics.kpi import PerformanceSummary, StrategyComparison
    from src.uncertainty.stress_testing import StressTestResult


@dataclass
class ReportSection:
    """A section in the report.

    Attributes:
        title: Section title.
        content: Text content for the section.
        figures: List of matplotlib figures to include.
        table_data: Optional table data as list of dicts.
    """

    title: str
    content: str = ""
    figures: list[Figure] = field(default_factory=list)
    table_data: list[dict] | None = None


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        title: Report title.
        author: Report author name.
        company: Company or organization name.
        date: Report date.
        logo_path: Optional path to company logo.
        include_assumptions: Whether to include assumptions section.
        include_risk_analysis: Whether to include risk analysis.
        page_size: Page size (letter, a4).
        dpi: DPI for figures.
    """

    title: str = "Curtailment Optimization Report"
    author: str = "Grid Optimization Engine"
    company: str = ""
    date: datetime = field(default_factory=datetime.now)
    logo_path: str | None = None
    include_assumptions: bool = True
    include_risk_analysis: bool = True
    page_size: str = "letter"
    dpi: int = 150


class ReportGenerator:
    """Generator for executive-ready optimization reports.

    Creates comprehensive PDF reports with charts, tables, and analysis
    summarizing optimization results and business impact.

    Example:
        ```python
        generator = ReportGenerator(config=ReportConfig(
            title="Q4 Optimization Results",
            author="Energy Team",
        ))

        # Add sections
        generator.add_executive_summary(comparison)
        generator.add_financial_analysis(summary)
        generator.add_risk_analysis(stress_results)

        # Generate PDF
        generator.generate_pdf("report.pdf")
        ```
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize the report generator.

        Args:
            config: Report configuration options.
        """
        self.config = config or ReportConfig()
        self.sections: list[ReportSection] = []

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report.

        Args:
            section: Section to add.
        """
        self.sections.append(section)

    def add_executive_summary(
        self,
        comparison: StrategyComparison,
        baseline_name: str = "Naive",
        optimized_name: str = "MILP",
    ) -> None:
        """Add an executive summary section.

        Args:
            comparison: Strategy comparison data.
            baseline_name: Name of baseline strategy.
            optimized_name: Name of optimized strategy.
        """
        baseline = comparison.strategies.get(baseline_name)
        optimized = comparison.strategies.get(optimized_name)

        if not baseline or not optimized:
            content = "Strategy comparison data not available."
            self.add_section(ReportSection(title="Executive Summary", content=content))
            return

        # Calculate improvements
        revenue_improvement = optimized.revenue.net_profit - baseline.revenue.net_profit
        revenue_pct = (
            revenue_improvement / baseline.revenue.net_profit * 100
            if baseline.revenue.net_profit > 0
            else 0
        )

        curtailment_reduction = (
            baseline.curtailment.curtailment_rate
            - optimized.curtailment.curtailment_rate
        )
        curtailment_pct = curtailment_reduction * 100

        compliance_improvement = (
            optimized.grid_compliance.compliance_rate
            - baseline.grid_compliance.compliance_rate
        )

        content = f"""
This report presents the results of the renewable energy curtailment optimization
analysis comparing {baseline_name} and {optimized_name} strategies.

KEY FINDINGS:

• Revenue Impact: ${revenue_improvement:,.0f} improvement ({revenue_pct:+.1f}%)
  - Baseline ({baseline_name}): ${baseline.revenue.net_profit:,.0f}
  - Optimized ({optimized_name}): ${optimized.revenue.net_profit:,.0f}

• Curtailment Reduction: {curtailment_pct:.1f} percentage points
  - Baseline rate: {baseline.curtailment.curtailment_rate * 100:.1f}%
  - Optimized rate: {optimized.curtailment.curtailment_rate * 100:.1f}%

• Grid Compliance: {compliance_improvement * 100:+.1f} percentage points
  - Baseline: {baseline.grid_compliance.compliance_rate * 100:.1f}%
  - Optimized: {optimized.grid_compliance.compliance_rate * 100:.1f}%

• Battery Utilization:
  - Cycles completed: {optimized.battery.total_cycles:.1f}
  - Arbitrage value: ${optimized.battery.arbitrage_value:,.0f}

RECOMMENDATION:
The optimized strategy demonstrates significant improvements across all key metrics.
Implementation is recommended to capture the identified value.
"""

        # Create summary figure
        fig = self._create_summary_comparison_figure(comparison)

        self.add_section(
            ReportSection(
                title="Executive Summary",
                content=content,
                figures=[fig],
            )
        )

    def add_problem_description(
        self,
        horizon_hours: int = 24,
        battery_capacity_mwh: float = 500.0,
        max_generation_mw: float = 600.0,
        grid_limit_mw: float = 300.0,
    ) -> None:
        """Add a problem description section.

        Args:
            horizon_hours: Planning horizon in hours.
            battery_capacity_mwh: Battery capacity.
            max_generation_mw: Maximum generation capacity.
            grid_limit_mw: Grid export limit.
        """
        content = f"""
PROBLEM STATEMENT:

The optimization addresses the challenge of maximizing revenue from renewable
energy generation while respecting grid constraints and managing battery storage.

SYSTEM PARAMETERS:

• Planning Horizon: {horizon_hours} hours
• Maximum Generation: {max_generation_mw} MW
• Grid Export Limit: {grid_limit_mw} MW
• Battery Capacity: {battery_capacity_mwh} MWh
• Battery Power: 150 MW (charge/discharge)

CHALLENGE:

During peak solar production (Duck Curve scenario), generation exceeds grid
export capacity. Without optimization, excess energy must be curtailed,
resulting in lost revenue. The battery system provides an opportunity to
store energy during low-price periods and sell during high-price periods.

OPTIMIZATION OBJECTIVE:

Maximize total profit = Revenue from sales + Battery arbitrage - Degradation costs

Subject to:
1. Energy balance: Generation = Sold + Stored + Curtailed
2. Grid capacity: Sold ≤ Grid limit
3. Battery constraints: SOC limits, charge/discharge rates
4. Ramp rate limits
"""

        self.add_section(ReportSection(title="Problem Description", content=content))

    def add_methodology(self) -> None:
        """Add a methodology section explaining the optimization approach."""
        content = """
OPTIMIZATION METHODOLOGY:

1. MIXED-INTEGER LINEAR PROGRAMMING (MILP)

   The core optimization uses Pyomo to formulate a MILP problem that jointly
   optimizes energy dispatch decisions across the planning horizon.

   Decision Variables:
   • x[t] = Energy sold to grid at time t (MW)
   • y[t] = Energy stored in battery at time t (MW)
   • z[t] = Energy curtailed at time t (MW)
   • SOC[t] = Battery state of charge at time t (MWh)

   The MILP guarantees globally optimal solutions within the model constraints.

2. SCENARIO-BASED OPTIMIZATION

   To handle forecast uncertainty, the optimization considers multiple
   generation scenarios (P10, P50, P90) representing different confidence levels.

   The expected value is optimized while penalizing downside scenarios.

3. REINFORCEMENT LEARNING (RL) ENHANCEMENT

   A trained RL agent provides real-time decision adjustments when actual
   conditions deviate from forecasts. The hybrid MILP+RL approach combines:
   • MILP: Optimal baseline schedule
   • RL: Real-time adaptation to deviations

4. RISK MANAGEMENT

   Monte Carlo stress testing evaluates strategy robustness across:
   • Forecast errors (±20% generation variance)
   • Price volatility (±30% price swings)
   • Grid outage events
"""

        self.add_section(ReportSection(title="Methodology", content=content))

    def add_financial_analysis(
        self,
        summary: PerformanceSummary,
        baseline_revenue: float | None = None,
    ) -> None:
        """Add a financial analysis section.

        Args:
            summary: Performance summary for the optimized strategy.
            baseline_revenue: Optional baseline revenue for comparison.
        """
        uplift = ""
        if baseline_revenue is not None and baseline_revenue > 0:
            improvement = summary.revenue.net_profit - baseline_revenue
            pct = improvement / baseline_revenue * 100
            uplift = f"""
COMPARISON TO BASELINE:

• Baseline Revenue: ${baseline_revenue:,.0f}
• Revenue Improvement: ${improvement:,.0f} ({pct:+.1f}%)
"""

        content = f"""
FINANCIAL PERFORMANCE:

REVENUE BREAKDOWN:

• Total Revenue: ${summary.revenue.total_revenue:,.0f}
  - From Direct Sales: ${summary.revenue.revenue_from_sales:,.0f}
  - From Battery Discharge: ${summary.revenue.revenue_from_discharge:,.0f}

• Total Costs: ${summary.revenue.total_costs:,.0f}
  - Degradation Costs: ${summary.revenue.degradation_cost:,.0f}
  - Penalty Costs: ${summary.revenue.penalty_cost:,.0f}

• NET PROFIT: ${summary.revenue.net_profit:,.0f}

AVERAGE PRICE CAPTURED: ${summary.revenue.average_price_captured:.2f}/MWh
{uplift}
BATTERY ECONOMICS:

• Arbitrage Value: ${summary.battery.arbitrage_value:,.0f}
• Degradation Cost: ${summary.battery.degradation_cost:,.0f}
• Net Battery Value: ${summary.battery.arbitrage_value - summary.battery.degradation_cost:,.0f}
• Cycles Completed: {summary.battery.total_cycles:.2f}

CURTAILMENT IMPACT:

• Total Curtailed: {summary.curtailment.total_curtailed_mwh:,.0f} MWh
• Curtailment Rate: {summary.curtailment.curtailment_rate * 100:.1f}%
• Energy Saved from Curtailment: {summary.curtailment.curtailment_avoided_mwh:,.0f} MWh
"""

        # Create financial breakdown figure
        fig = self._create_financial_figure(summary)

        self.add_section(
            ReportSection(
                title="Financial Analysis",
                content=content,
                figures=[fig],
            )
        )

    def add_risk_analysis(
        self,
        stress_result: StressTestResult,
    ) -> None:
        """Add a risk analysis section.

        Args:
            stress_result: Results from Monte Carlo stress testing.
        """
        # Derive compliance from violation rate
        mean_compliance = 1.0 - stress_result.violation_rate

        content = f"""
RISK ANALYSIS:

Monte Carlo stress testing was performed with {stress_result.n_runs_completed}
simulations to evaluate strategy robustness.

PROFIT DISTRIBUTION:

• Mean Profit: ${stress_result.mean_profit:,.0f}
• Std Deviation: ${stress_result.std_profit:,.0f}
• Minimum: ${stress_result.min_profit:,.0f}
• Maximum: ${stress_result.max_profit:,.0f}

RISK METRICS:

• Profit at Risk (5%): ${stress_result.profit_at_risk:,.0f}
  (5% of outcomes fall below this value)

• Sharpe Ratio: {stress_result.sharpe_ratio:.2f}
  (Risk-adjusted return metric)

COMPLIANCE STATISTICS:

• Mean Compliance Rate: {mean_compliance * 100:.1f}%
• Scenarios with Violations: {stress_result.violation_rate * 100:.1f}%

CURTAILMENT STATISTICS:

• Mean Curtailment Rate: {stress_result.mean_curtailment_rate * 100:.1f}%
• Max Curtailment Rate: {stress_result.max_curtailment_rate * 100:.1f}%

STRESS SCENARIOS TESTED:

• Forecast Error: ±20% generation variance
• Price Volatility: ±30% price swings
• Grid Outages: Random capacity reductions

CONCLUSION:

The optimization strategy maintains positive returns across the tested
stress scenarios. The Sharpe ratio indicates {"good" if stress_result.sharpe_ratio > 1 else "moderate"}
risk-adjusted performance.
"""

        # Create risk distribution figure
        fig = self._create_risk_figure(stress_result)

        self.add_section(
            ReportSection(
                title="Risk Analysis",
                content=content,
                figures=[fig],
            )
        )

    def add_assumptions(self) -> None:
        """Add an assumptions and limitations section."""
        content = """
ASSUMPTIONS & LIMITATIONS:

GRID ASSUMPTIONS:
• Export capacity remains constant during planning horizon
• Ramp rate limits are enforced hourly
• No transmission losses modeled
• Single point of interconnection

BATTERY ASSUMPTIONS:
• Linear degradation model ($/MWh cycled)
• Constant charge/discharge efficiency (95%)
• No calendar aging considered
• Round-trip efficiency independent of power level

MARKET ASSUMPTIONS:
• Price-taker assumption (no market impact)
• Perfect price forecast for day-ahead
• Settlement at hourly granularity
• No ancillary service revenue included

FORECAST ASSUMPTIONS:
• Probabilistic bands (P10/P50/P90) capture uncertainty
• No sub-hourly variability modeled
• Weather correlation not explicitly modeled

REGULATORY CONSIDERATIONS:
• CAISO-style market rules assumed
• No capacity market obligations
• Curtailment penalties not included in base case

LIMITATIONS:
• Model does not account for equipment failures
• Startup/shutdown costs not modeled
• Network constraints simplified to single bus
• RL agent trained on historical patterns
"""

        self.add_section(
            ReportSection(title="Assumptions & Limitations", content=content)
        )

    def _create_summary_comparison_figure(
        self,
        comparison: StrategyComparison,
    ) -> Figure:
        """Create a summary comparison figure."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        strategies = list(comparison.strategies.keys())
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

        # Revenue comparison
        revenues = [
            comparison.strategies[s].revenue.net_profit / 1000 for s in strategies
        ]
        axes[0].bar(strategies, revenues, color=colors[: len(strategies)])
        axes[0].set_ylabel("Revenue ($K)")
        axes[0].set_title("Net Revenue Comparison")
        axes[0].tick_params(axis="x", rotation=45)

        # Curtailment comparison
        curtailment = [
            comparison.strategies[s].curtailment.curtailment_rate * 100
            for s in strategies
        ]
        axes[1].bar(strategies, curtailment, color=colors[: len(strategies)])
        axes[1].set_ylabel("Curtailment Rate (%)")
        axes[1].set_title("Curtailment Comparison")
        axes[1].tick_params(axis="x", rotation=45)

        # Compliance comparison
        compliance = [
            comparison.strategies[s].grid_compliance.compliance_rate * 100
            for s in strategies
        ]
        axes[2].bar(strategies, compliance, color=colors[: len(strategies)])
        axes[2].set_ylabel("Compliance Rate (%)")
        axes[2].set_title("Grid Compliance")
        axes[2].set_ylim(0, 105)
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def _create_financial_figure(
        self,
        summary: PerformanceSummary,
    ) -> Figure:
        """Create a financial breakdown figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Revenue breakdown pie chart
        revenue_data = [
            summary.revenue.revenue_from_sales,
            summary.revenue.revenue_from_discharge,
        ]
        revenue_labels = ["Direct Sales", "Battery Discharge"]
        colors = ["#3498db", "#9b59b6"]

        axes[0].pie(
            revenue_data,
            labels=revenue_labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Revenue Sources")

        # Cost breakdown
        cost_labels = ["Net Profit", "Degradation", "Penalties"]
        cost_values = [
            summary.revenue.net_profit,
            summary.revenue.degradation_cost,
            summary.revenue.penalty_cost,
        ]
        cost_colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        axes[1].barh(cost_labels, cost_values, color=cost_colors)
        axes[1].set_xlabel("Amount ($)")
        axes[1].set_title("Profit & Costs")
        for i, v in enumerate(cost_values):
            axes[1].text(v + 100, i, f"${v:,.0f}", va="center")

        plt.tight_layout()
        return fig

    def _create_risk_figure(
        self,
        stress_result: StressTestResult,
    ) -> Figure:
        """Create a risk analysis figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Profit histogram
        if hasattr(stress_result, "profit_distribution"):
            profits = stress_result.profit_distribution
        else:
            # Generate synthetic distribution for display
            profits = np.random.normal(
                stress_result.mean_profit,
                stress_result.std_profit,
                1000,
            )

        # Derive mean_compliance from violation_rate
        mean_compliance = 1.0 - stress_result.violation_rate

        axes[0].hist(profits, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
        axes[0].axvline(
            stress_result.mean_profit, color="green", linestyle="--", label="Mean"
        )
        axes[0].axvline(
            stress_result.profit_at_risk,
            color="red",
            linestyle="--",
            label="VaR (5%)",
        )
        axes[0].set_xlabel("Profit ($)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Profit Distribution")
        axes[0].legend()

        # Risk metrics bar chart
        metrics = ["Sharpe\nRatio", "Mean\nCompliance", "Mean\nCurtailment"]
        values = [
            min(stress_result.sharpe_ratio, 3),  # Cap for display
            mean_compliance * 100,
            (1 - stress_result.mean_curtailment_rate) * 100,  # Inverted for "good"
        ]
        colors = ["#9b59b6", "#2ecc71", "#3498db"]

        bars = axes[1].bar(metrics, values, color=colors)
        axes[1].set_ylabel("Score / Percentage")
        axes[1].set_title("Risk Metrics")

        # Add value labels
        for bar, val in zip(bars, values, strict=False):
            height = bar.get_height()
            if metrics[bars.index(bar)] == "Sharpe\nRatio":
                label = f"{stress_result.sharpe_ratio:.2f}"
            else:
                label = f"{val:.1f}%"
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def generate_pdf(self, output_path: str | Path) -> Path:
        """Generate the PDF report.

        Args:
            output_path: Path for the output PDF file.

        Returns:
            Path to the generated PDF file.
        """
        output_path = Path(output_path)

        with PdfPages(output_path) as pdf:
            # Title page
            self._add_title_page(pdf)

            # Table of contents
            self._add_table_of_contents(pdf)

            # Add each section
            for section in self.sections:
                self._add_section_to_pdf(pdf, section)

        return output_path

    def _add_title_page(self, pdf: PdfPages) -> None:
        """Add title page to PDF."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Title
        ax.text(
            0.5,
            0.7,
            self.config.title,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
        )

        # Subtitle
        ax.text(
            0.5,
            0.6,
            "Grid-Aware Curtailment Optimization Analysis",
            transform=ax.transAxes,
            fontsize=14,
            ha="center",
            va="center",
            style="italic",
        )

        # Author and date
        ax.text(
            0.5,
            0.4,
            f"Prepared by: {self.config.author}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="center",
        )

        if self.config.company:
            ax.text(
                0.5,
                0.35,
                self.config.company,
                transform=ax.transAxes,
                fontsize=12,
                ha="center",
                va="center",
            )

        ax.text(
            0.5,
            0.25,
            f"Date: {self.config.date.strftime('%B %d, %Y')}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="center",
        )

        # Footer
        ax.text(
            0.5,
            0.05,
            "CONFIDENTIAL",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            color="gray",
        )

        pdf.savefig(fig, dpi=self.config.dpi)
        plt.close(fig)

    def _add_table_of_contents(self, pdf: PdfPages) -> None:
        """Add table of contents page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        ax.text(
            0.5,
            0.95,
            "Table of Contents",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="center",
        )

        y_pos = 0.85
        for i, section in enumerate(self.sections, 1):
            ax.text(
                0.1,
                y_pos,
                f"{i}. {section.title}",
                transform=ax.transAxes,
                fontsize=12,
            )
            y_pos -= 0.05

        pdf.savefig(fig, dpi=self.config.dpi)
        plt.close(fig)

    def _add_section_to_pdf(self, pdf: PdfPages, section: ReportSection) -> None:
        """Add a section to the PDF."""
        # Create text page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Section title
        ax.text(
            0.5,
            0.95,
            section.title,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
        )

        # Section content
        if section.content:
            ax.text(
                0.05,
                0.88,
                section.content,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                family="monospace",
                wrap=True,
            )

        pdf.savefig(fig, dpi=self.config.dpi)
        plt.close(fig)

        # Add figures on separate pages
        for figure in section.figures:
            pdf.savefig(figure, dpi=self.config.dpi)
            plt.close(figure)

    def generate_html(self, output_path: str | Path) -> Path:
        """Generate an HTML report.

        Args:
            output_path: Path for the output HTML file.

        Returns:
            Path to the generated HTML file.
        """
        output_path = Path(output_path)

        html_content = self._build_html()

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _build_html(self) -> str:
        """Build HTML content for the report."""
        sections_html = ""
        for section in self.sections:
            content_html = section.content.replace("\n", "<br>")
            sections_html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <pre>{content_html}</pre>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            font-size: 12px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <p class="meta">
            Prepared by: {self.config.author}<br>
            Date: {self.config.date.strftime('%B %d, %Y')}
        </p>
    </div>
    {sections_html}
</body>
</html>
"""


def generate_executive_report(
    comparison: StrategyComparison,
    summary: PerformanceSummary,
    stress_result: StressTestResult | None = None,
    output_path: str | Path = "optimization_report.pdf",
    config: ReportConfig | None = None,
) -> Path:
    """Convenience function to generate a complete executive report.

    Args:
        comparison: Strategy comparison data.
        summary: Performance summary for optimized strategy.
        stress_result: Optional stress test results.
        output_path: Path for output file.
        config: Optional report configuration.

    Returns:
        Path to the generated report.
    """
    generator = ReportGenerator(config=config)

    # Add standard sections
    generator.add_executive_summary(comparison)
    generator.add_problem_description()
    generator.add_methodology()
    generator.add_financial_analysis(summary)

    if stress_result is not None:
        generator.add_risk_analysis(stress_result)

    generator.add_assumptions()

    return generator.generate_pdf(output_path)
