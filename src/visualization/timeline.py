"""Decision timeline visualizer for optimization results.

Generates plots for:
- Generation vs sold vs curtailed
- Battery SOC over time
- Price signals
- Grid limits and violations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.domain.models import (
        GridConstraint,
        MarketPrice,
        OptimizationDecision,
    )
    from src.hybrid.controller import DispatchDecision


@dataclass
class TimelinePlotConfig:
    """Configuration for timeline plots.

    Attributes:
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure resolution.
        style: Matplotlib style to use.
        colors: Color scheme for different series.
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        legend_fontsize: Font size for legend.
        grid_alpha: Alpha value for grid lines.
        save_format: Format for saving figures.
    """

    figsize: tuple[float, float] = (14, 10)
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid"
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "generation": "#2ecc71",
            "sold": "#3498db",
            "stored": "#9b59b6",
            "curtailed": "#e74c3c",
            "discharged": "#f39c12",
            "soc": "#1abc9c",
            "price": "#34495e",
            "grid_limit": "#c0392b",
            "ramp_limit": "#d35400",
        }
    )
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    grid_alpha: float = 0.3
    save_format: str = "png"


class TimelineVisualizer:
    """Visualizer for optimization decision timelines.

    Creates comprehensive visualizations showing how energy is dispatched
    over time, including generation, sales, storage, and curtailment.

    Example:
        ```python
        visualizer = TimelineVisualizer()

        # Plot from optimization decisions
        fig = visualizer.plot_dispatch_timeline(
            decisions=decisions,
            prices=prices,
            constraints=constraints,
            title="24-Hour Optimization Results",
        )
        fig.savefig("dispatch_timeline.png")

        # Plot comprehensive dashboard
        fig = visualizer.plot_full_dashboard(
            decisions=decisions,
            forecasts=forecasts,
            prices=prices,
            constraints=constraints,
        )
        ```
    """

    def __init__(self, config: TimelinePlotConfig | None = None) -> None:
        """Initialize the timeline visualizer.

        Args:
            config: Plot configuration options.
        """
        self.config = config or TimelinePlotConfig()

    def plot_energy_dispatch(
        self,
        decisions: list[OptimizationDecision],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot energy dispatch breakdown (generation, sold, stored, curtailed).

        Args:
            decisions: List of optimization decisions.
            ax: Matplotlib axes to plot on (creates new if None).
            show_legend: Whether to show the legend.

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        hours = list(range(len(decisions)))
        generation = [d.generation_mw for d in decisions]
        sold = [d.energy_sold_mw for d in decisions]
        stored = [d.energy_stored_mw for d in decisions]
        curtailed = [d.energy_curtailed_mw for d in decisions]

        colors = self.config.colors

        # Stacked area chart
        ax.fill_between(
            hours,
            0,
            sold,
            alpha=0.7,
            color=colors["sold"],
            label="Sold to Grid",
        )
        ax.fill_between(
            hours,
            sold,
            [s + st for s, st in zip(sold, stored, strict=False)],
            alpha=0.7,
            color=colors["stored"],
            label="Stored in Battery",
        )
        ax.fill_between(
            hours,
            [s + st for s, st in zip(sold, stored, strict=False)],
            [s + st + c for s, st, c in zip(sold, stored, curtailed, strict=False)],
            alpha=0.7,
            color=colors["curtailed"],
            label="Curtailed",
        )

        # Generation line
        ax.plot(
            hours,
            generation,
            color=colors["generation"],
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
            label="Total Generation",
        )

        ax.set_xlabel("Hour", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Power (MW)", fontsize=self.config.label_fontsize)
        ax.set_title("Energy Dispatch Breakdown", fontsize=self.config.title_fontsize)
        ax.set_xlim(0, len(decisions) - 1)
        ax.set_ylim(0, max(generation) * 1.1 if generation else 100)
        ax.grid(True, alpha=self.config.grid_alpha)

        if show_legend:
            ax.legend(loc="upper right", fontsize=self.config.legend_fontsize)

        return ax

    def plot_battery_soc(
        self,
        decisions: list[OptimizationDecision],
        battery_capacity_mwh: float = 500.0,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot battery state of charge over time.

        Args:
            decisions: List of optimization decisions.
            battery_capacity_mwh: Battery capacity for reference line.
            ax: Matplotlib axes to plot on (creates new if None).
            show_legend: Whether to show the legend.

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        hours = list(range(len(decisions)))
        soc = [d.resulting_soc_mwh for d in decisions]

        colors = self.config.colors

        # SOC line
        ax.fill_between(hours, 0, soc, alpha=0.5, color=colors["soc"])
        ax.plot(
            hours,
            soc,
            color=colors["soc"],
            linewidth=2,
            marker="s",
            markersize=4,
            label="State of Charge",
        )

        # Capacity reference line
        ax.axhline(
            y=battery_capacity_mwh,
            color=colors["grid_limit"],
            linestyle="--",
            linewidth=1.5,
            label=f"Capacity ({battery_capacity_mwh:.0f} MWh)",
        )

        # Zero reference
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Hour", fontsize=self.config.label_fontsize)
        ax.set_ylabel("SOC (MWh)", fontsize=self.config.label_fontsize)
        ax.set_title("Battery State of Charge", fontsize=self.config.title_fontsize)
        ax.set_xlim(0, len(decisions) - 1)
        ax.set_ylim(0, battery_capacity_mwh * 1.1)
        ax.grid(True, alpha=self.config.grid_alpha)

        if show_legend:
            ax.legend(loc="upper right", fontsize=self.config.legend_fontsize)

        return ax

    def plot_prices(
        self,
        prices: list[MarketPrice],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot market prices over time.

        Args:
            prices: List of market prices.
            ax: Matplotlib axes to plot on (creates new if None).
            show_legend: Whether to show the legend.

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        hours = list(range(len(prices)))
        price_values = [p.effective_price for p in prices]

        colors = self.config.colors

        # Color based on positive/negative
        positive_prices = [max(0, p) for p in price_values]
        negative_prices = [min(0, p) for p in price_values]

        ax.bar(
            hours,
            positive_prices,
            color=colors["price"],
            alpha=0.7,
            label="Positive Price",
        )
        ax.bar(
            hours,
            negative_prices,
            color=colors["curtailed"],
            alpha=0.7,
            label="Negative Price",
        )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Hour", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Price ($/MWh)", fontsize=self.config.label_fontsize)
        ax.set_title("Market Prices", fontsize=self.config.title_fontsize)
        ax.set_xlim(-0.5, len(prices) - 0.5)
        ax.grid(True, alpha=self.config.grid_alpha, axis="y")

        if show_legend and any(p < 0 for p in price_values):
            ax.legend(loc="upper right", fontsize=self.config.legend_fontsize)

        return ax

    def plot_grid_constraints(
        self,
        decisions: list[OptimizationDecision],
        constraints: list[GridConstraint],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot grid constraints and actual export.

        Args:
            decisions: List of optimization decisions.
            constraints: List of grid constraints.
            ax: Matplotlib axes to plot on (creates new if None).
            show_legend: Whether to show the legend.

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        hours = list(range(len(decisions)))
        sold = [d.energy_sold_mw for d in decisions]
        limits = [c.max_export_mw for c in constraints[: len(decisions)]]

        colors = self.config.colors

        # Grid limits
        ax.fill_between(
            hours,
            limits,
            max(limits) * 1.2,
            alpha=0.2,
            color=colors["grid_limit"],
            label="Constrained Zone",
        )
        ax.step(
            hours,
            limits,
            where="mid",
            color=colors["grid_limit"],
            linewidth=2,
            linestyle="--",
            label="Export Limit",
        )

        # Actual exports
        ax.bar(
            hours,
            sold,
            color=colors["sold"],
            alpha=0.7,
            label="Actual Export",
        )

        # Mark violations
        violations = [
            (h, s)
            for h, (s, lim) in enumerate(zip(sold, limits, strict=False))
            if s > lim
        ]
        if violations:
            v_hours, v_sold = zip(*violations, strict=False)
            ax.scatter(
                v_hours,
                v_sold,
                color=colors["curtailed"],
                s=100,
                marker="x",
                linewidths=3,
                label="Violations",
                zorder=5,
            )

        ax.set_xlabel("Hour", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Power (MW)", fontsize=self.config.label_fontsize)
        ax.set_title("Grid Export vs Limits", fontsize=self.config.title_fontsize)
        ax.set_xlim(-0.5, len(decisions) - 0.5)
        ax.grid(True, alpha=self.config.grid_alpha)

        if show_legend:
            ax.legend(loc="upper right", fontsize=self.config.legend_fontsize)

        return ax

    def plot_revenue_breakdown(
        self,
        decisions: list[OptimizationDecision],
        prices: list[MarketPrice],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot cumulative revenue over time.

        Args:
            decisions: List of optimization decisions.
            prices: List of market prices.
            ax: Matplotlib axes to plot on (creates new if None).
            show_legend: Whether to show the legend.

        Returns:
            Matplotlib axes with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        hours = list(range(len(decisions)))

        # Calculate hourly and cumulative revenue
        hourly_revenue = [
            d.energy_sold_mw * prices[i].effective_price
            for i, d in enumerate(decisions)
        ]
        cumulative_revenue = np.cumsum(hourly_revenue)

        colors = self.config.colors

        # Hourly revenue bars
        positive_rev = [max(0, r) for r in hourly_revenue]
        negative_rev = [min(0, r) for r in hourly_revenue]

        ax.bar(hours, positive_rev, color=colors["sold"], alpha=0.5, label="Positive")
        ax.bar(
            hours, negative_rev, color=colors["curtailed"], alpha=0.5, label="Negative"
        )

        # Cumulative line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            hours,
            cumulative_revenue,
            color=colors["generation"],
            linewidth=2,
            marker="o",
            markersize=3,
            label="Cumulative Revenue",
        )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Hour", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Hourly Revenue ($)", fontsize=self.config.label_fontsize)
        ax2.set_ylabel("Cumulative Revenue ($)", fontsize=self.config.label_fontsize)
        ax.set_title("Revenue Analysis", fontsize=self.config.title_fontsize)
        ax.set_xlim(-0.5, len(decisions) - 0.5)
        ax.grid(True, alpha=self.config.grid_alpha)

        if show_legend:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper left",
                fontsize=self.config.legend_fontsize,
            )

        return ax

    def plot_full_dashboard(
        self,
        decisions: list[OptimizationDecision],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        battery_capacity_mwh: float = 500.0,
        title: str = "Optimization Dashboard",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Create a comprehensive dashboard with all timeline plots.

        Args:
            decisions: List of optimization decisions.
            prices: List of market prices.
            constraints: List of grid constraints.
            battery_capacity_mwh: Battery capacity for SOC plot.
            title: Main title for the dashboard.
            save_path: Path to save the figure (optional).

        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(
            4,
            1,
            figsize=self.config.figsize,
            height_ratios=[2, 1, 1, 1],
        )

        # Energy dispatch (main plot)
        self.plot_energy_dispatch(decisions, ax=axes[0])

        # Battery SOC
        self.plot_battery_soc(
            decisions, battery_capacity_mwh=battery_capacity_mwh, ax=axes[1]
        )

        # Market prices
        self.plot_prices(prices[: len(decisions)], ax=axes[2])

        # Grid constraints
        self.plot_grid_constraints(decisions, constraints, ax=axes[3])

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path,
                dpi=self.config.dpi,
                format=self.config.save_format,
                bbox_inches="tight",
            )

        return fig

    def plot_dispatch_decisions(
        self,
        decisions: list[DispatchDecision],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        battery_capacity_mwh: float = 500.0,
        title: str = "Hybrid Controller Dispatch",
        save_path: str | Path | None = None,
    ) -> Figure:
        """Create dashboard from hybrid controller DispatchDecision objects.

        Args:
            decisions: List of dispatch decisions from hybrid controller.
            prices: List of market prices.
            constraints: List of grid constraints.
            battery_capacity_mwh: Battery capacity for SOC plot.
            title: Main title for the dashboard.
            save_path: Path to save the figure (optional).

        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(
            4,
            1,
            figsize=self.config.figsize,
            height_ratios=[2, 1, 1, 1],
        )

        hours = list(range(len(decisions)))
        colors = self.config.colors

        # Extract data
        generation = [
            d.energy_sold_mw + d.energy_stored_mw + d.energy_curtailed_mw
            for d in decisions
        ]
        sold = [d.energy_sold_mw for d in decisions]
        stored = [d.energy_stored_mw for d in decisions]
        curtailed = [d.energy_curtailed_mw for d in decisions]
        soc = [d.battery_soc_mwh for d in decisions]

        # Plot 1: Energy dispatch
        ax = axes[0]
        ax.fill_between(hours, 0, sold, alpha=0.7, color=colors["sold"], label="Sold")
        ax.fill_between(
            hours,
            sold,
            [s + st for s, st in zip(sold, stored, strict=False)],
            alpha=0.7,
            color=colors["stored"],
            label="Stored",
        )
        ax.fill_between(
            hours,
            [s + st for s, st in zip(sold, stored, strict=False)],
            [s + st + c for s, st, c in zip(sold, stored, curtailed, strict=False)],
            alpha=0.7,
            color=colors["curtailed"],
            label="Curtailed",
        )
        ax.plot(
            hours,
            generation,
            color=colors["generation"],
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
            label="Generation",
        )

        # Mark decision sources
        milp_hours = [i for i, d in enumerate(decisions) if d.decision_source == "MILP"]
        rl_hours = [i for i, d in enumerate(decisions) if d.decision_source == "RL"]
        if milp_hours:
            ax.scatter(
                milp_hours,
                [generation[i] for i in milp_hours],
                marker="^",
                color="blue",
                s=50,
                label="MILP",
                zorder=5,
            )
        if rl_hours:
            ax.scatter(
                rl_hours,
                [generation[i] for i in rl_hours],
                marker="v",
                color="orange",
                s=50,
                label="RL Override",
                zorder=5,
            )

        ax.set_ylabel("Power (MW)")
        ax.set_title("Energy Dispatch with Decision Source")
        ax.legend(loc="upper right", fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=self.config.grid_alpha)

        # Plot 2: Battery SOC
        ax = axes[1]
        ax.fill_between(hours, 0, soc, alpha=0.5, color=colors["soc"])
        ax.plot(hours, soc, color=colors["soc"], linewidth=2, marker="s", markersize=3)
        ax.axhline(
            y=battery_capacity_mwh,
            color=colors["grid_limit"],
            linestyle="--",
            alpha=0.7,
        )
        ax.set_ylabel("SOC (MWh)")
        ax.set_title("Battery State of Charge")
        ax.grid(True, alpha=self.config.grid_alpha)

        # Plot 3: Prices
        ax = axes[2]
        price_values = [p.effective_price for p in prices[: len(decisions)]]
        ax.bar(
            hours,
            [max(0, p) for p in price_values],
            color=colors["price"],
            alpha=0.7,
        )
        ax.bar(
            hours,
            [min(0, p) for p in price_values],
            color=colors["curtailed"],
            alpha=0.7,
        )
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Price ($/MWh)")
        ax.set_title("Market Prices")
        ax.grid(True, alpha=self.config.grid_alpha)

        # Plot 4: Grid constraints
        ax = axes[3]
        limits = [c.max_export_mw for c in constraints[: len(decisions)]]
        ax.step(hours, limits, where="mid", color=colors["grid_limit"], linewidth=2)
        ax.bar(hours, sold, color=colors["sold"], alpha=0.7)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (MW)")
        ax.set_title("Grid Export vs Limits")
        ax.grid(True, alpha=self.config.grid_alpha)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig


def create_dispatch_timeline(
    decisions: list[OptimizationDecision],
    prices: list[MarketPrice],
    constraints: list[GridConstraint],
    title: str = "Optimization Results",
    save_path: str | Path | None = None,
) -> Figure:
    """Convenience function to create a dispatch timeline visualization.

    Args:
        decisions: List of optimization decisions.
        prices: List of market prices.
        constraints: List of grid constraints.
        title: Title for the visualization.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    visualizer = TimelineVisualizer()
    return visualizer.plot_full_dashboard(
        decisions=decisions,
        prices=prices,
        constraints=constraints,
        title=title,
        save_path=save_path,
    )
