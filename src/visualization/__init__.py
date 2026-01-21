"""Matplotlib plots, reports, and dashboards.

This module provides visualization tools for the curtailment engine:

- TimelineVisualizer: Plots for optimization decisions over time
- ComparisonDashboard: Strategy comparison charts and dashboards
"""

from src.visualization.comparison import (
    ComparisonDashboard,
    ComparisonPlotConfig,
    create_comparison_dashboard,
)
from src.visualization.timeline import (
    TimelinePlotConfig,
    TimelineVisualizer,
    create_dispatch_timeline,
)

__all__ = [
    "ComparisonDashboard",
    "ComparisonPlotConfig",
    "create_comparison_dashboard",
    "TimelinePlotConfig",
    "TimelineVisualizer",
    "create_dispatch_timeline",
]
