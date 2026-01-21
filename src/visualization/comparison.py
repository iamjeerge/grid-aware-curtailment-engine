"""Scenario comparison dashboard for strategy analysis.

Compares optimization strategies:
- Naive vs MILP vs RL
- Revenue deltas
- Curtailment reduction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.metrics.kpi import StrategyComparison


@dataclass
class ComparisonPlotConfig:
    """Configuration for comparison plots.

    Attributes:
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure resolution.
        colors: Color scheme for different strategies.
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        bar_width: Width of bars in bar charts.
    """

    figsize: tuple[float, float] = (14, 10)
    dpi: int = 100
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "naive": "#e74c3c",
            "milp": "#3498db",
            "rl": "#9b59b6",
            "hybrid": "#2ecc71",
            "optimized": "#2ecc71",
            "baseline": "#95a5a6",
        }
    )
    title_fontsize: int = 14
    label_fontsize: int = 12
    bar_width: float = 0.25


class ComparisonDashboard:
    """Dashboard for comparing optimization strategies.

    Creates visualizations comparing performance metrics across
    different optimization strategies (Naive, MILP, RL, Hybrid).

    Example:
        ```python
        dashboard = ComparisonDashboard()

        # Compare from StrategyComparison object
        fig = dashboard.plot_comparison(
            comparison=strategy_comparison,
            title="Strategy Comparison",
        )
        fig.savefig("comparison.png")

        # Create radar chart
        fig = dashboard.plot_radar_comparison(comparison)
        ```
    """

    def __init__(self, config: ComparisonPlotConfig | None = None) -> None:
        """Initialize the comparison dashboard.

        Args:
            config: Plot configuration options.
        """
        self.config = config or ComparisonPlotConfig()

    def _get_color(self, strategy_name: str) -> str:
        """Get color for a strategy name."""
        name_lower = strategy_name.lower()
        for key, color in self.config.colors.items():
            if key in name_lower:
                return color
        # Default color if no match
        return "#34495e"

    def plot_revenue_comparison(
        self,
        comparison: StrategyComparison,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot revenue comparison across strategies.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        strategies = list(comparison.strategies.keys())
        revenues = [comparison.strategies[s].revenue.net_profit for s in strategies]
        colors = [self._get_color(s) for s in strategies]

        bars = ax.bar(strategies, revenues, color=colors, alpha=0.8, edgecolor="black")

        # Add value labels
        for bar, rev in zip(bars, revenues, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"${rev:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Strategy", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Net Revenue ($)", fontsize=self.config.label_fontsize)
        ax.set_title("Revenue Comparison", fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def plot_curtailment_comparison(
        self,
        comparison: StrategyComparison,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot curtailment rate comparison across strategies.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        strategies = list(comparison.strategies.keys())
        curtailment_rates = [
            comparison.strategies[s].curtailment.curtailment_rate * 100
            for s in strategies
        ]
        colors = [self._get_color(s) for s in strategies]

        bars = ax.bar(
            strategies, curtailment_rates, color=colors, alpha=0.8, edgecolor="black"
        )

        # Add value labels
        for bar, rate in zip(bars, curtailment_rates, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Strategy", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Curtailment Rate (%)", fontsize=self.config.label_fontsize)
        ax.set_title("Curtailment Comparison", fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(curtailment_rates) * 1.2 if curtailment_rates else 100)

        return ax

    def plot_compliance_comparison(
        self,
        comparison: StrategyComparison,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot grid compliance comparison across strategies.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        strategies = list(comparison.strategies.keys())
        compliance_rates = [
            comparison.strategies[s].grid_compliance.compliance_rate * 100
            for s in strategies
        ]
        colors = [self._get_color(s) for s in strategies]

        bars = ax.bar(
            strategies, compliance_rates, color=colors, alpha=0.8, edgecolor="black"
        )

        # Add value labels
        for bar, rate in zip(bars, compliance_rates, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Strategy", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Compliance Rate (%)", fontsize=self.config.label_fontsize)
        ax.set_title("Grid Compliance Comparison", fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Target")

        return ax

    def plot_overall_scores(
        self,
        comparison: StrategyComparison,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot overall performance scores across strategies.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        strategies = list(comparison.strategies.keys())
        scores = [comparison.strategies[s].overall_score for s in strategies]
        colors = [self._get_color(s) for s in strategies]

        bars = ax.barh(strategies, scores, color=colors, alpha=0.8, edgecolor="black")

        # Add value labels
        for bar, score in zip(bars, scores, strict=False):
            width = bar.get_width()
            ax.annotate(
                f"{score:.1f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Overall Score", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Strategy", fontsize=self.config.label_fontsize)
        ax.set_title("Overall Performance Score", fontsize=self.config.title_fontsize)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, 100)

        return ax

    def plot_radar_comparison(
        self,
        comparison: StrategyComparison,
        save_path: str | Path | None = None,
    ) -> Figure:
        """Create a radar chart comparing strategies across metrics.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            save_path: Path to save the figure (optional).

        Returns:
            Matplotlib Figure object.
        """
        strategies = list(comparison.strategies.keys())
        categories = [
            "Revenue",
            "Low Curtailment",
            "Compliance",
            "Battery Util",
            "Efficiency",
        ]
        n_categories = len(categories)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

        # Calculate angles for radar chart
        angles = [n / n_categories * 2 * np.pi for n in range(n_categories)]
        angles += angles[:1]  # Complete the loop

        for strategy in strategies:
            summary = comparison.strategies[strategy]

            # Normalize metrics to 0-100 scale
            # Revenue: normalize to max revenue in comparison
            max_revenue = max(
                s.revenue.net_profit for s in comparison.strategies.values()
            )
            revenue_score = (
                summary.revenue.net_profit / max_revenue * 100 if max_revenue > 0 else 0
            )

            # Curtailment: invert (lower is better)
            curtailment_score = (1 - summary.curtailment.curtailment_rate) * 100

            # Compliance
            compliance_score = summary.grid_compliance.compliance_rate * 100

            # Battery utilization
            battery_score = min(summary.battery.utilization_rate * 100, 100)

            # Efficiency
            efficiency_score = summary.battery.charge_efficiency_realized * 100

            values = [
                revenue_score,
                curtailment_score,
                compliance_score,
                battery_score,
                efficiency_score,
            ]
            values += values[:1]  # Complete the loop

            color = self._get_color(strategy)
            ax.plot(angles, values, "o-", linewidth=2, label=strategy, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=self.config.label_fontsize)
        ax.set_ylim(0, 100)
        ax.set_title(
            "Strategy Comparison Radar",
            fontsize=self.config.title_fontsize,
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def plot_metrics_heatmap(
        self,
        comparison: StrategyComparison,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Create a heatmap of normalized metrics across strategies.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        strategies = list(comparison.strategies.keys())
        metrics = ["Revenue", "Curtailment", "Compliance", "Battery", "Score"]

        # Build data matrix
        data = []
        for strategy in strategies:
            summary = comparison.strategies[strategy]
            row = [
                summary.revenue.net_profit / 1000,  # In thousands
                summary.curtailment.curtailment_rate * 100,
                summary.grid_compliance.compliance_rate * 100,
                summary.battery.utilization_rate * 100,
                summary.overall_score,
            ]
            data.append(row)

        data = np.array(data)

        # Normalize each column to 0-1
        normalized = np.zeros_like(data)
        for col in range(data.shape[1]):
            col_data = data[:, col]
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                normalized[:, col] = (col_data - col_min) / (col_max - col_min)
            else:
                normalized[:, col] = 0.5

        # For curtailment, lower is better
        normalized[:, 1] = 1 - normalized[:, 1]

        ax.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)

        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(metrics)):
                if j == 0:
                    text = f"${data[i, j]:.0f}k"
                elif j in [1, 2, 3]:
                    text = f"{data[i, j]:.1f}%"
                else:
                    text = f"{data[i, j]:.1f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if 0.3 < normalized[i, j] < 0.7 else "white",
                )

        ax.set_title(
            "Strategy Metrics Heatmap (Green = Better)",
            fontsize=self.config.title_fontsize,
        )

        return ax

    def plot_full_comparison(
        self,
        comparison: StrategyComparison,
        title: str = "Strategy Comparison Dashboard",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Create a comprehensive comparison dashboard.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            title: Main title for the dashboard.
            save_path: Path to save the figure (optional).

        Returns:
            Matplotlib Figure object.
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Revenue comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_revenue_comparison(comparison, ax=ax1)

        # Curtailment comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_curtailment_comparison(comparison, ax=ax2)

        # Compliance comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_compliance_comparison(comparison, ax=ax3)

        # Overall scores
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_overall_scores(comparison, ax=ax4)

        # Heatmap
        ax5 = fig.add_subplot(gs[1, 1:])
        self.plot_metrics_heatmap(comparison, ax=ax5)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 4, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig

    def plot_improvement_waterfall(
        self,
        comparison: StrategyComparison,
        baseline: str = "naive",
        target: str = "milp",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Create a waterfall chart showing improvement from baseline to target.

        Args:
            comparison: StrategyComparison object with strategy summaries.
            baseline: Name of baseline strategy.
            target: Name of target strategy.
            ax: Matplotlib axes to plot on (creates new if None).

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        if baseline not in comparison.strategies or target not in comparison.strategies:
            ax.text(0.5, 0.5, "Strategies not found", ha="center", va="center")
            return ax

        base = comparison.strategies[baseline]
        targ = comparison.strategies[target]

        # Calculate deltas
        deltas = {
            "Base Revenue": base.revenue.net_profit,
            "Reduced\nCurtailment": (
                (
                    base.curtailment.total_curtailed_mwh
                    - targ.curtailment.total_curtailed_mwh
                )
                * 50  # Assume $50/MWh average value
            ),
            "Battery\nArbitrage": targ.battery.arbitrage_value
            - base.battery.arbitrage_value,
            "Degradation\nSavings": base.battery.degradation_cost
            - targ.battery.degradation_cost,
        }

        # Build waterfall data
        labels = list(deltas.keys()) + [f"{target.upper()}\nRevenue"]
        values = list(deltas.values())

        # Calculate positions
        cumsum = np.cumsum([0] + values)
        starts = cumsum[:-1]
        ends = cumsum[1:]

        colors = []
        for v in values:
            if v >= 0:
                colors.append("#2ecc71")
            else:
                colors.append("#e74c3c")
        colors.append("#3498db")  # Final bar

        # Plot bars
        for i, (_label, start, end, color) in enumerate(
            zip(labels[:-1], starts, ends, colors, strict=False)
        ):
            ax.bar(
                i,
                end - start,
                bottom=start,
                color=color,
                edgecolor="black",
                alpha=0.8,
            )
            # Add value label
            mid = (start + end) / 2
            ax.annotate(
                f"${values[i]:+,.0f}",
                xy=(i, mid),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Final bar
        final_value = targ.revenue.net_profit
        ax.bar(
            len(labels) - 1,
            final_value,
            color=colors[-1],
            edgecolor="black",
            alpha=0.8,
        )
        ax.annotate(
            f"${final_value:,.0f}",
            xy=(len(labels) - 1, final_value / 2),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Revenue ($)", fontsize=self.config.label_fontsize)
        ax.set_title(
            f"Revenue Waterfall: {baseline.upper()} → {target.upper()}",
            fontsize=self.config.title_fontsize,
        )
        ax.grid(True, alpha=0.3, axis="y")

        return ax


def create_comparison_dashboard(
    comparison: StrategyComparison,
    title: str = "Strategy Comparison",
    save_path: str | Path | None = None,
) -> Figure:
    """Convenience function to create a comparison dashboard.

    Args:
        comparison: StrategyComparison object.
        title: Title for the dashboard.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    dashboard = ComparisonDashboard()
    return dashboard.plot_full_comparison(
        comparison=comparison,
        title=title,
        save_path=save_path,
    )
