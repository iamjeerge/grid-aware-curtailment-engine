"""Uncertainty and risk awareness module.

Phase 5: Extends optimization to handle uncertainty through:
- Scenario-based optimization with probability weighting
- CVaR (Conditional Value-at-Risk) for risk-aware optimization
- Monte Carlo simulation for stress testing
"""

from src.uncertainty.risk import CVaROptimizer, RiskAwareConfig, RiskMetrics
from src.uncertainty.stress_testing import (
    MonteCarloEngine,
    StressScenario,
    StressTestConfig,
    StressTestResult,
)

__all__ = [
    "CVaROptimizer",
    "RiskAwareConfig",
    "RiskMetrics",
    "MonteCarloEngine",
    "StressScenario",
    "StressTestConfig",
    "StressTestResult",
]
