"""Metrics module for performance analysis and KPI computation.

This module provides comprehensive tools for computing and analyzing
optimization KPIs including curtailment reduction, revenue uplift,
battery utilization, and grid compliance.
"""

from src.metrics.analyzer import (
    AnalyzerConfig,
    PerformanceAnalyzer,
    compute_kpis_from_decisions,
    quick_kpi_summary,
)
from src.metrics.kpi import (
    BatteryMetrics,
    CurtailmentMetrics,
    GridComplianceMetrics,
    PerformanceSummary,
    RevenueMetrics,
    StrategyComparison,
)

__all__ = [
    # KPI dataclasses
    "CurtailmentMetrics",
    "RevenueMetrics",
    "BatteryMetrics",
    "GridComplianceMetrics",
    "PerformanceSummary",
    "StrategyComparison",
    # Analyzer
    "AnalyzerConfig",
    "PerformanceAnalyzer",
    # Convenience functions
    "compute_kpis_from_decisions",
    "quick_kpi_summary",
]
