# Grid-Aware Curtailment Engine - Copilot Instructions

## Project Overview

Production-grade optimization system that minimizes renewable curtailment and maximizes revenue by jointly optimizing generation, grid constraints, battery storage, and market prices under uncertainty. Models CAISO-style grid congestion and price volatility.

## Tech Stack & Setup

- **Python 3.11** with Poetry for dependency management
- **Backend**: FastAPI for API endpoints, Pydantic for validation
- **Optimization**: Pyomo for MILP formulation
- **Data**: NumPy, Pandas, SciPy for numerical computing
- **RL**: Gymnasium (not legacy `gym`), PPO/DQN agents
- **Visualization**: Matplotlib for plots and reports
- **Code quality**: Black, Ruff, MyPy (strict typing required)
- **Testing**: Pytest with fixtures for reproducible scenarios

```bash
poetry install          # Install dependencies
poetry run pytest       # Run tests
poetry run black .      # Format code
poetry run ruff check . # Lint
poetry run mypy .       # Type check
```

## Architecture & Module Boundaries

```
src/
├── domain/          # Pydantic models (immutable where applicable)
├── generators/      # Synthetic data: weather, grid constraints, prices
├── battery/         # BESS physics, SOC tracking, degradation costs
├── optimization/    # MILP formulation (Pyomo), solver interface
├── rl/              # Gymnasium environment, agent training
├── hybrid/          # MILP + RL controller with override logging
├── metrics/         # KPI computation, performance analysis
├── visualization/   # Matplotlib plots, reports, dashboards
└── api/             # FastAPI endpoints (future)
```

## Units Convention (CAISO-style)

- **Power**: MW (megawatts)
- **Energy**: MWh (megawatt-hours)
- **Prices**: $/MWh
- **Time**: Hourly timesteps, typically 24h horizon

## Domain Models (in `domain/`)

All models use **Pydantic** with strict validation:

- `TimeStep`, `GenerationForecast` (P10/P50/P90 probabilistic bands)
- `GridConstraint` (CAISO-style: export capacity, ramp limits, congestion windows)
- `BatteryState` (SOC, charge/discharge efficiency, power limits, degradation)
- `MarketPrice` (day-ahead, real-time, negative pricing events)
- `OptimizationDecision` (sell, store, curtail quantities per timestep)
- `SimulationResult` (aggregated outcomes for analysis)

## MILP Formulation Reference

### Indices

- `t ∈ T`: hourly time steps
- `s ∈ S`: forecast scenarios (P10, P50, P90)

### Parameters

- `G[t,s]`: generation forecast (MW)
- `P[t]`: market price ($/MWh)
- `C[t]`: grid export capacity (MW)
- `η_c, η_d`: charge/discharge efficiency (default: 0.95)
- `E_max`: battery capacity (MWh)
- `P_max`: battery power limit (MW)
- `λ`: degradation cost ($/MWh cycled, default: $8)

### Decision Variables (Pyomo naming)

```python
model.energy_sold[t,s]      # x: energy sold to grid (MW)
model.energy_stored[t,s]    # y: energy charged to battery (MW)
model.energy_curtailed[t,s] # z: curtailed energy (MW)
model.soc[t,s]              # SOC: state of charge (MWh)
```

### Constraints (must enforce all)

1. **Energy Balance**: `G[t,s] = x[t,s] + y[t,s] + z[t,s]`
2. **Grid Capacity**: `x[t,s] <= C[t]`
3. **SOC Dynamics**: `SOC[t,s] = SOC[t-1,s] + η_c * y[t,s] - d[t,s] / η_d`
4. **SOC Bounds**: `0 <= SOC[t,s] <= E_max`
5. **Charge Rate**: `0 <= y[t,s] <= P_max`

### Objective

Maximize expected profit: `Σ_s π_s Σ_t (P[t] * x[t,s] - λ * y[t,s])`

## Key Patterns

### Battery Default Configuration

```python
BATTERY_DEFAULTS = {
    "capacity_mwh": 500,
    "max_power_mw": 150,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "degradation_cost_per_mwh": 8
}
```

### Scenario Generation

Use `numpy.random.Generator` with explicit seeds for reproducibility. All generators accept a `seed` parameter.

### RL Environment (Gymnasium)

```python
# State: [SOC, forecast, grid_capacity, price]
# Action: continuous [sell_fraction, store_fraction] (curtail = remainder)
# Reward: revenue - degradation_penalty - violation_penalty
```

## Demo Scenario: Duck Curve Trap

Reference scenario for testing optimization quality:

- Solar peaks 600 MW at noon, grid limited to 300 MW
- Negative prices midday (-$25/MWh), evening spike $140/MWh
- **Naive result**: 32% curtailment, $420k revenue, grid violations
- **Optimized target**: <10% curtailment, >$650k revenue, zero violations

## Testing Conventions

- Use `@pytest.fixture` for standard scenarios (sunny day, congested grid, price spike)
- Test constraint violations explicitly (grid capacity, SOC bounds, ramp rates)
- Compare optimization results against naive heuristic baseline
- All simulations must be reproducible via seed parameters

## Success Criteria (Must Pass)

- **Zero grid violations** in all scenarios
- **Curtailment reduced >20%** vs naive baseline
- **Revenue uplift demonstrated** with statistical significance
- **Fully reproducible** simulations (same seed → same results)

## Metrics to Track

- Curtailment avoided (%), Revenue uplift (%), Battery utilization
- Degradation cost, Grid compliance score (violations must = 0)
