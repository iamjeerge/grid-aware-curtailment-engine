"""FastAPI application for the Grid-Aware Curtailment Engine.

This module provides the main FastAPI application with all routes,
middleware, and configuration.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import demo_router
from src.api.routes import router as optimization_router
from src.api.schemas import (
    HealthResponse,
    ScenarioType,
    StrategyType,
    SystemInfoResponse,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    print("🚀 Starting Grid-Aware Curtailment Engine API...")
    yield
    # Shutdown
    print("👋 Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Grid-Aware Curtailment Engine",
    description="""
Production-grade optimization system for renewable curtailment and storage.

## Features

- **MILP Optimization**: Pyomo-based mixed-integer linear programming
- **Multiple Strategies**: Naive, MILP, RL, and Hybrid controllers
- **CAISO-style Modeling**: Grid constraints, congestion, price patterns
- **Battery Storage**: SOC tracking, degradation costs, efficiency losses

## Key Endpoints

- `POST /api/v1/optimizations/`: Run a new optimization
- `GET /api/v1/demos/scenarios`: List pre-configured demo scenarios
- `POST /api/v1/demos/run/{scenario_id}`: Run a demo scenario

## Demo Scenarios

1. **Duck Curve Challenge**: Solar peak with negative midday prices
2. **High Volatility Market**: Extreme price swings for arbitrage
3. **Congested Grid**: Evening congestion with limited export
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(optimization_router)
app.include_router(demo_router)


# =============================================================================
# Root & Health Endpoints
# =============================================================================


@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Grid-Aware Curtailment Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(),
    )


@app.get("/system", response_model=SystemInfoResponse)
async def system_info() -> SystemInfoResponse:
    """Get system information."""
    # Check if GLPK solver is available
    try:
        from pyomo.opt import SolverFactory

        solver = SolverFactory("glpk")
        solver_available = solver.available()
    except Exception:
        solver_available = False

    return SystemInfoResponse(
        version="0.1.0",
        python_version=sys.version,
        available_strategies=list(StrategyType),
        available_scenarios=list(ScenarioType),
        solver_available=solver_available,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
