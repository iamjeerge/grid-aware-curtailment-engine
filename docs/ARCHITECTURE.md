# Architecture Documentation

## System Overview

The Grid-Aware Curtailment Engine is a modular optimization system designed to minimize renewable energy curtailment and maximize revenue from energy sales and battery arbitrage.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    (CLI / Python API / Future Web API)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ORCHESTRATION LAYER                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Demo Module   │  │ Report Generator│  │  Validation     │             │
│  │   (src/demo)    │  │ (src/reporting) │  │  (src/validation)│             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECISION ENGINE                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hybrid Controller                               │   │
│  │                      (src/hybrid/)                                   │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐          │   │
│  │  │    MILP Optimizer       │  │    RL Agent             │          │   │
│  │  │    (src/optimization/)  │  │    (src/rl/)            │          │   │
│  │  │                         │  │                         │          │   │
│  │  │  • Pyomo Model          │  │  • Gymnasium Env        │          │   │
│  │  │  • GLPK Solver          │  │  • PPO/DQN Policy       │          │   │
│  │  │  • Global Optimal       │  │  • Real-time Adapt      │          │   │
│  │  └─────────────────────────┘  └─────────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Naive Controller (Baseline)                       │   │
│  │                    (src/controllers/)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHYSICAL MODELS                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Battery Energy Storage System                     │   │
│  │                    (src/battery/)                                    │   │
│  │                                                                      │   │
│  │  • State of Charge Tracking    • Charge/Discharge Efficiency        │   │
│  │  • Degradation Cost Model      • Power/Energy Limits                │   │
│  │  • SOC Range Constraints       • Cycle Counting                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA GENERATION                                    │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   Solar/Wind      │  │   Grid Constraint │  │   Market Price    │       │
│  │   Generator       │  │   Generator       │  │   Generator       │       │
│  │   (generators/)   │  │   (generators/)   │  │   (generators/)   │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYTICS & OUTPUT                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   KPI Calculator  │  │   Visualization   │  │   Stress Testing  │       │
│  │   (src/metrics/)  │  │   (visualization/)│  │   (uncertainty/)  │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOMAIN LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Pydantic Models (src/domain/)                     │   │
│  │                                                                      │   │
│  │  TimeStep │ GenerationForecast │ GridConstraint │ MarketPrice       │   │
│  │  BatteryState │ OptimizationDecision │ SimulationResult             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
domain/ ─────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    ▼                                                                         │
generators/ ───────────────────┬──────────────────┬──────────────────────────┤
    │                          │                  │                          │
    ▼                          ▼                  ▼                          │
battery/ ──────────────►  controllers/     optimization/      rl/            │
    │                          │                  │             │            │
    │                          ▼                  ▼             ▼            │
    └──────────────────► hybrid/ ◄────────────────┴─────────────┘            │
                               │                                              │
                               ▼                                              │
                          metrics/ ───────► visualization/ ───► reporting/   │
                               │                                              │
                               ▼                                              │
                        uncertainty/ ────────────────────────────────────────┤
                               │                                              │
                               ▼                                              │
                        validation/ ──────────────────────────────────────────┘
```

## Key Design Principles

### 1. Immutable Domain Models

All domain models in `src/domain/` are Pydantic models with frozen=True where appropriate. This ensures:
- Thread safety
- Predictable state
- Easy serialization
- Clear data flow

### 2. Dependency Injection

Controllers and optimizers accept configuration objects rather than hard-coded values:

```python
# Good: Configuration is explicit and testable
optimizer = MILPOptimizer(battery_config, opt_config)

# Avoid: Hard-coded values
optimizer = MILPOptimizer()  # Uses hidden defaults
```

### 3. Strategy Pattern

Multiple optimization strategies share a common interface:

```python
class Controller(Protocol):
    def decide(
        self,
        forecast: GenerationForecast,
        price: MarketPrice,
        constraint: GridConstraint,
        battery_state: BatteryState,
    ) -> OptimizationDecision:
        ...
```

### 4. Reproducibility

All random processes accept seed parameters:

```python
generator = SolarGenerator(SolarConfig(seed=42))
results1 = generator.generate(24)
results2 = generator.generate(24)  # Identical to results1
```

## Data Flow

### Optimization Pipeline

```
1. INPUT GENERATION
   ┌─────────────────┐
   │ Generate        │
   │ • Forecasts     │──────┐
   │ • Prices        │      │
   │ • Constraints   │      │
   └─────────────────┘      │
                            ▼
2. OPTIMIZATION        ┌─────────────────┐
                       │ MILP Optimizer  │
                       │                 │
                       │ max Σ(price×    │
                       │      sold -     │
                       │      degradation)│
                       │                 │
                       │ s.t. constraints│
                       └────────┬────────┘
                                │
                            ▼
3. DECISION OUTPUT     ┌─────────────────┐
                       │ For each hour:  │
                       │ • sell_mw       │
                       │ • store_mw      │
                       │ • curtail_mw    │
                       │ • soc_mwh       │
                       └────────┬────────┘
                                │
                            ▼
4. KPI CALCULATION     ┌─────────────────┐
                       │ Compute:        │
                       │ • Revenue       │
                       │ • Curtailment   │
                       │ • Compliance    │
                       │ • Battery util  │
                       └────────┬────────┘
                                │
                            ▼
5. REPORTING           ┌─────────────────┐
                       │ Generate:       │
                       │ • Visualizations│
                       │ • PDF Reports   │
                       │ • Comparisons   │
                       └─────────────────┘
```

## MILP Formulation

### Sets
- T = {0, 1, ..., H-1}: Time steps (hours)
- S = {P10, P50, P90}: Forecast scenarios

### Parameters
- G[t,s]: Generation forecast (MW)
- P[t]: Day-ahead price ($/MWh)
- C[t]: Grid export capacity (MW)
- η_c: Charge efficiency (0.95)
- η_d: Discharge efficiency (0.95)
- E_max: Battery capacity (MWh)
- P_max: Battery power limit (MW)
- λ: Degradation cost ($/MWh)

### Decision Variables
- x[t,s]: Energy sold to grid (MW)
- y[t,s]: Energy charged to battery (MW)
- z[t,s]: Energy curtailed (MW)
- d[t,s]: Energy discharged from battery (MW)
- SOC[t,s]: State of charge (MWh)

### Objective
```
max Σ_s π_s × Σ_t [ P[t] × (x[t,s] + d[t,s]) - λ × (y[t,s] + d[t,s]) ]
```

### Constraints
```
1. Energy Balance:    G[t,s] + d[t,s]/η_d = x[t,s] + y[t,s]×η_c + z[t,s]
2. Grid Capacity:     x[t,s] + d[t,s]/η_d ≤ C[t]
3. SOC Dynamics:      SOC[t,s] = SOC[t-1,s] + y[t,s]×η_c - d[t,s]
4. SOC Lower Bound:   SOC[t,s] ≥ SOC_min
5. SOC Upper Bound:   SOC[t,s] ≤ SOC_max
6. Charge Limit:      y[t,s] ≤ P_max
7. Discharge Limit:   d[t,s] ≤ P_max
8. Non-negativity:    x, y, z, d, SOC ≥ 0
```

## Testing Strategy

### Unit Tests
- Each module has corresponding test file
- Tests use fixtures for reproducible scenarios
- Mocking for external dependencies

### Integration Tests
- End-to-end optimization pipeline
- Strategy comparison validation
- Report generation verification

### Property-Based Tests
- Energy conservation invariants
- SOC bounds never violated
- Grid capacity respected

## Extension Points

### Adding New Strategies

1. Implement the Controller protocol
2. Add configuration dataclass
3. Register in hybrid controller
4. Add tests

### Adding New Generators

1. Create generator class with `generate(hours)` method
2. Add configuration dataclass
3. Ensure seed parameter for reproducibility

### Adding New Metrics

1. Add to `src/metrics/kpi.py`
2. Update `PerformanceSummary` dataclass
3. Add visualization support
