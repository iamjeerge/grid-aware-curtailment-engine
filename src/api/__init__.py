"""FastAPI endpoints for the optimization engine.

This module provides the REST API for the curtailment engine,
allowing web applications to interact with the optimization system.
"""

from src.api.main import app, create_app
from src.api.schemas import (
    BatteryConfigRequest,
    OptimizationRequest,
    OptimizationResultResponse,
    ScenarioConfigRequest,
    ScenarioType,
    StrategyType,
)

__all__ = [
    "app",
    "create_app",
    "BatteryConfigRequest",
    "OptimizationRequest",
    "OptimizationResultResponse",
    "ScenarioConfigRequest",
    "ScenarioType",
    "StrategyType",
]
