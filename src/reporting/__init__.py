"""Reporting and export functionality for optimization results.

This module provides executive-ready report generation including:
- PDF report generation with charts and tables
- Summary statistics and KPI formatting
- Risk analysis documentation
"""

from src.reporting.generator import (
    ReportConfig,
    ReportGenerator,
    ReportSection,
    generate_executive_report,
)

__all__ = [
    "ReportConfig",
    "ReportGenerator",
    "ReportSection",
    "generate_executive_report",
]
