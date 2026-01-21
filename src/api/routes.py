"""FastAPI router for optimization endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    BatteryConfigRequest,
    DemoScenarioResponse,
    OptimizationListResponse,
    OptimizationRequest,
    OptimizationResultResponse,
    ScenarioConfigRequest,
    ScenarioType,
)
from src.api.services import optimization_service

router = APIRouter(prefix="/api/v1/optimizations", tags=["optimizations"])


@router.post("/", response_model=OptimizationResultResponse)
async def create_optimization(
    request: OptimizationRequest,
) -> OptimizationResultResponse:
    """Run a new optimization.

    This endpoint accepts scenario and battery configurations,
    runs the specified optimization strategies, and returns results.
    """
    try:
        result = optimization_service.run_optimization(
            name=request.name,
            scenario_config=request.scenario,
            battery_config=request.battery,
            strategies=request.strategies,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/", response_model=OptimizationListResponse)
async def list_optimizations(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
) -> OptimizationListResponse:
    """List all optimizations with pagination."""
    items, total = optimization_service.list_optimizations(page, page_size)
    return OptimizationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{optimization_id}", response_model=OptimizationResultResponse)
async def get_optimization(optimization_id: UUID) -> OptimizationResultResponse:
    """Get optimization result by ID."""
    result = optimization_service.get_optimization(optimization_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return result


@router.delete("/{optimization_id}")
async def delete_optimization(optimization_id: UUID) -> dict:
    """Delete optimization by ID."""
    if optimization_service.delete_optimization(optimization_id):
        return {"message": "Optimization deleted successfully"}
    raise HTTPException(status_code=404, detail="Optimization not found")


# =============================================================================
# Demo Endpoints
# =============================================================================

demo_router = APIRouter(prefix="/api/v1/demos", tags=["demos"])


@demo_router.get("/scenarios", response_model=list[DemoScenarioResponse])
async def list_demo_scenarios() -> list[DemoScenarioResponse]:
    """List pre-configured demo scenarios."""
    return [
        DemoScenarioResponse(
            id="duck_curve",
            name="Duck Curve Challenge",
            description=(
                "Classic California duck curve with solar peak at noon, "
                "negative midday prices, and evening price spike. "
                "Grid constrained to 300 MW during peak solar."
            ),
            scenario_type=ScenarioType.DUCK_CURVE,
            config=ScenarioConfigRequest(
                scenario_type=ScenarioType.DUCK_CURVE,
                horizon_hours=24,
                peak_generation_mw=600.0,
                grid_limit_mw=300.0,
                seed=42,
            ),
            battery=BatteryConfigRequest(
                capacity_mwh=500.0,
                max_power_mw=150.0,
            ),
        ),
        DemoScenarioResponse(
            id="high_volatility",
            name="High Volatility Market",
            description=(
                "Extreme price swings with potential for significant "
                "arbitrage opportunities. Tests battery dispatch under "
                "uncertain conditions."
            ),
            scenario_type=ScenarioType.HIGH_VOLATILITY,
            config=ScenarioConfigRequest(
                scenario_type=ScenarioType.HIGH_VOLATILITY,
                horizon_hours=24,
                peak_generation_mw=500.0,
                grid_limit_mw=400.0,
                seed=123,
            ),
            battery=BatteryConfigRequest(
                capacity_mwh=400.0,
                max_power_mw=200.0,
            ),
        ),
        DemoScenarioResponse(
            id="congested_grid",
            name="Congested Grid",
            description=(
                "Evening congestion scenario with limited export capacity "
                "during peak demand hours. Requires strategic battery charging."
            ),
            scenario_type=ScenarioType.CONGESTED_GRID,
            config=ScenarioConfigRequest(
                scenario_type=ScenarioType.CONGESTED_GRID,
                horizon_hours=24,
                peak_generation_mw=400.0,
                grid_limit_mw=200.0,
                seed=456,
            ),
            battery=BatteryConfigRequest(
                capacity_mwh=300.0,
                max_power_mw=100.0,
            ),
        ),
    ]


@demo_router.post("/run/{scenario_id}", response_model=OptimizationResultResponse)
async def run_demo_scenario(scenario_id: str) -> OptimizationResultResponse:
    """Run a pre-configured demo scenario."""
    scenarios = await list_demo_scenarios()
    scenario = next((s for s in scenarios if s.id == scenario_id), None)

    if scenario is None:
        raise HTTPException(status_code=404, detail="Demo scenario not found")

    try:
        result = optimization_service.run_optimization(
            name=f"Demo: {scenario.name}",
            scenario_config=scenario.config,
            battery_config=scenario.battery,
            strategies=["naive", "milp"],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
