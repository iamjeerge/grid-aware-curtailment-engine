# Complete Feature Guide

This document provides in-depth explanations of every tool and feature in the Grid-Aware Curtailment Engine.

---

## 📑 Table of Contents

1. [Optimization Algorithms](#optimization-algorithms)
2. [Analytics & Monitoring](#analytics--monitoring)
3. [Scenario Management](#scenario-management)
4. [Reporting & Export](#reporting--export)
5. [Advanced Features](#advanced-features)

---

## Optimization Algorithms

### 1. Naive Controller

**What it does**: Simple heuristic baseline for comparison

**Algorithm**:
```
For each hour t:
  IF generation > grid_capacity:
    Curtail excess, charge battery if not full
  ELSE:
    Sell to grid
  IF price < 0:
    Hold (don't sell)
```

**When to use**:
- Establishing baseline performance
- Understanding how much value optimization provides
- Quick decisions with no setup time

**Advantages**:
- ✅ Simple, deterministic
- ✅ No optimization time
- ✅ Good baseline comparison

**Disadvantages**:
- ❌ Doesn't plan ahead
- ❌ Misses arbitrage opportunities
- ❌ Often curtails excessively

**Example**:
```python
from src.controllers import NaiveController
from src.domain import TimeStep

controller = NaiveController()

# Hourly decisions
for t in range(24):
    state = TimeStep(
        hour=t,
        generation_mw=generation_forecast[t],
        grid_capacity_mw=grid_capacity[t],
        price_per_mwh=prices[t],
        battery_soc=battery.current_soc,
    )
    
    decision = controller.decide(state)
    print(f"Hour {t}: Sell={decision.sell_mw}, Store={decision.store_mw}, Curtail={decision.curtail_mw}")
```

---

### 2. MILP Optimizer

**What it does**: Finds mathematically optimal dispatch decisions

**Algorithm**:
1. Formulates problem as Mixed-Integer Linear Program (MILP)
2. Uses GLPK solver to find optimal solution
3. Respects all constraints exactly
4. Returns hourly dispatch decisions for full 24-hour horizon

**When to use**:
- Planning decisions (day-ahead market)
- Need guaranteed optimality
- Forecast accuracy is high
- Horizon ≤ 24 hours

**Mathematical Formulation**:

**Objective** (Maximize expected profit):
$$\text{max} \sum_{t=1}^{24} \left( P_t \cdot x_t - \lambda \cdot y_t \right)$$

Where:
- $P_t$ = Market price at hour $t$ ($/MWh)
- $x_t$ = Energy sold (MWh)
- $y_t$ = Energy charged (MWh)
- $\lambda$ = Degradation cost ($/MWh)

**Subject to**:

1. Energy Balance:
$$G_t = x_t + y_t + z_t$$
- $G_t$ = Generation (MWh)
- $z_t$ = Curtailed energy (MWh)

2. Grid Capacity:
$$x_t \leq C_t$$
- $C_t$ = Export capacity (MWh)

3. SOC Dynamics:
$$S_t = S_{t-1} + 0.95 \cdot y_t - \frac{d_t}{0.95}$$
- $S_t$ = State of charge (MWh)
- 0.95 = Round-trip efficiency

4. SOC Bounds:
$$0.1 \cdot E_{max} \leq S_t \leq 0.9 \cdot E_{max}$$
- $E_{max}$ = Battery capacity (MWh)

5. Power Limits:
$$0 \leq x_t, y_t, d_t \leq P_{max}$$
- $P_{max}$ = Max charge/discharge power (MW)

**Advantages**:
- ✅ Mathematically optimal (best possible)
- ✅ Deterministic and auditable
- ✅ Respects all constraints exactly
- ✅ Fast solution time (<1 second)

**Disadvantages**:
- ❌ Requires perfect 24-hour forecast
- ❌ Can't adapt to deviations in real-time
- ❌ Computationally expensive for longer horizons

**Example**:
```python
from src.optimization import MILPOptimizer
import numpy as np

optimizer = MILPOptimizer(
    time_horizon_hours=24,
    solver="glpk",
    time_limit_seconds=60
)

# 24-hour forecast
forecast = {
    "generation": [100, 120, 150, 300, 500, 600, 550, 400, 200, 100, 80, 60],
    "grid_capacity": [300, 300, 300, 300, 300, 300, 300, 300, 400, 400, 400, 400],
    "prices": [50, 45, 40, -25, -20, -25, 30, 80, 140, 135, 130, 100],
}

decision = optimizer.optimize(
    generation_forecast=forecast["generation"],
    grid_capacity=forecast["grid_capacity"],
    prices=forecast["prices"],
    battery_capacity_mwh=500,
    battery_max_power_mw=150,
    initial_soc=250,  # 50% charged
)

print(f"Optimal revenue: ${decision.total_revenue:,.0f}")
print(f"Curtailment: {decision.total_curtailment:.1%}")
print(f"Grid violations: {decision.violation_count}")
```

---

### 3. RL Agent

**What it does**: Learns to make decisions from experience

**Architecture**:
- **Observation Space**: [SOC%, Generation, Grid Capacity, Price]
- **Action Space**: [Sell Fraction, Store Fraction] (curtail = remainder)
- **Algorithm**: PPO (Proximal Policy Optimization) or DQN
- **Training**: Gymnasium environment with diverse scenarios

**When to use**:
- Real-time adaptation needed
- Forecast deviations expected
- Need to learn from past patterns
- Fast inference critical

**How it Works**:

1. **State Observation**:
```python
state = {
    "soc_pct": battery.soc / battery.capacity,  # 0-100%
    "generation_mw": current_generation,  # MW
    "grid_capacity_mw": available_capacity,  # MW
    "price_per_mwh": current_price,  # $/MWh
}
```

2. **Decision Making**:
```python
action = agent.act(state)  # Returns [sell_frac, store_frac]
```

3. **Reward Function**:
$$\text{Reward} = \text{Revenue} - \text{Degradation Cost} - \text{Violation Penalty}$$

**Advantages**:
- ✅ Adapts to forecast deviations
- ✅ Fast inference (milliseconds)
- ✅ Learns complex patterns
- ✅ No explicit forecasts needed

**Disadvantages**:
- ❌ May not be globally optimal
- ❌ Requires training data and computation
- ❌ Less interpretable ("black box")
- ❌ Needs validation on new scenarios

**Training Example**:
```python
from src.rl import RLEnvironment, PPOAgent

# Create Gymnasium environment
env = RLEnvironment(
    battery_capacity_mwh=500,
    max_power_mw=150,
    scenarios=training_scenarios,
)

# Train PPO agent
agent = PPOAgent(
    env=env,
    num_episodes=1000,
    learning_rate=3e-4,
    batch_size=128,
)

agent.train()
agent.save("models/ppo_agent.pkl")

# Deploy
deployed_agent = PPOAgent.load("models/ppo_agent.pkl")
action = deployed_agent.act(state)
```

---

### 4. Hybrid Controller

**What it does**: Combines MILP and RL for robustness and optimality

**Architecture**:
```
MILP Optimizer → Base Plan (Hour 1)
      ↓
   Monitor Reality vs Plan
      ↓
   Deviation > Threshold?
   ├─ YES → RL Override (Real-time adapt)
   └─ NO → Stick with MILP Plan
      ↓
   Log Override for Retraining
```

**When to use**:
- Production deployment
- Need both optimality AND robustness
- Forecast accuracy varies
- Can't afford violations

**Configuration**:
```python
from src.hybrid import HybridController, OverridePolicy

controller = HybridController(
    milp_optimizer=MILPOptimizer(),
    rl_agent=RLAgent(),
    
    # Override triggers
    override_policy=OverridePolicy(
        deviation_threshold_mw=50,  # If > 50 MW off forecast
        violation_threshold=0.1,  # If < 10% margin to grid limit
        soc_warning_level=0.15,  # If SOC drift > ±15%
    ),
    
    # Logging
    log_overrides=True,
    override_log_path="logs/overrides.csv",
)

# Usage
decision = controller.decide(
    forecast_24h=forecast,
    current_state=state,
)

if decision.override_applied:
    print(f"RL Override: {decision.reason}")
    print(f"MILP decision: Sell={decision.milp_decision.sell_mw} MW")
    print(f"RL decision: Sell={decision.rl_decision.sell_mw} MW")
```

---

## Analytics & Monitoring

### 1. KPI Dashboard

**What it tracks**:

| KPI | Formula | Interpretation |
|-----|---------|-----------------|
| **Curtailment Rate** | Curtailed / Total Generation | % of solar wasted, lower is better |
| **Profit per MWh** | Net Profit / MWh Sold | $/MWh, higher is better |
| **Battery Cycles** | Total Cycled / Capacity | # full cycles, < 4000 for longevity |
| **Grid Compliance** | Hours without violation / 24 | 100% ideal |
| **Degradation Cost** | MWh Cycled × $8 | Battery wear expense |
| **Revenue Uplift** | (MILP - Naive) / Naive | % improvement vs baseline |

**Access Methods**:

```python
# Python API
from src.metrics import KPICalculator

calc = KPICalculator()
kpis = calc.compute_kpis(results)

print(f"Curtailment Rate: {kpis.curtailment_rate:.1%}")
print(f"Profit/MWh: ${kpis.profit_per_mwh:.2f}")
print(f"Revenue Uplift: {kpis.revenue_uplift:.1%}")

# REST API
GET /api/v1/optimizations/{id}/metrics/kpi

# Web UI
Dashboard → KPIs tab
```

---

### 2. Industry Dashboard

**Aggregates metrics across all optimizations**:

```python
GET /api/v1/dashboard/industry

{
  "total_optimizations_run": 42,
  "financial_metrics": {
    "total_revenue": 15000000,
    "total_cost": 6250000,
    "net_profit": 8750000,
    "roi_percentage": 87.5,
    "average_profit_per_mwh": 58.2,
    "revenue_uplift_vs_naive": 73.1,
    "total_degradation_cost": 400000
  },
  "grid_reliability": {
    "total_violations": 3,
    "total_violation_mwh": 45,
    "compliance_rate": 99.4,
    "max_violation_mw": 50,
    "ramp_rate_violations": 0,
    "export_capacity_utilization": 75.2
  },
  "curtailment_reduction": {
    "total_generation_mwh": 250000,
    "total_curtailed_mwh": 12500,
    "curtailment_rate_baseline": 5.0,
    "curtailment_rate_optimized": 2.1,
    "curtailment_reduction_pct": 58.0,
    "avoided_curtailment_mwh": 2500,
    "avoided_curtailment_value": 150000
  },
  "battery_health": {
    "total_cycles_equivalent": 1520,
    "remaining_useful_life_pct": 62.0,
    "round_trip_efficiency_actual": 91.0,
    "energy_arbitrage_captured": 3500000,
    "peak_shaving_contribution": 37500
  },
  "environmental": {
    "co2_avoided_metric_tons": 15000,
    "equivalent_household_days": 37500,
    "grid_renewable_penetration_improvement": 58.0
  },
  "summary": "Across 42 optimizations: $8.75M net profit, 58.0% curtailment reduction, 99.4% grid compliance, 15000 MT CO2 avoided."
}
```

---

### 3. Stress Testing (Monte Carlo)

**What it does**: Quantifies risk through simulations

**Method**:
1. Take base scenario
2. Generate 100+ variants with random deviations
3. Run optimization on each
4. Analyze distribution of outcomes

**Metrics Computed**:
- Revenue percentiles (5th, 25th, 50th, 75th, 95th)
- Curtailment distribution
- Violation probability
- Confidence intervals

**Example**:
```python
from src.uncertainty import StressTest

stress_test = StressTest(
    num_scenarios=200,
    generation_volatility=0.15,  # 15% random deviation
    price_volatility=0.25,  # 25% random deviation
    grid_capacity_volatility=0.10,  # 10% random deviation
)

results = stress_test.run(
    base_scenario=my_scenario,
    optimizer=MILPOptimizer(),
)

print(f"Revenue 5th percentile (worst case): ${results.revenue_p5:,.0f}")
print(f"Revenue 95th percentile (best case): ${results.revenue_p95:,.0f}")
print(f"Revenue median: ${results.revenue_p50:,.0f}")
print(f"Violation probability: {results.violation_probability:.1%}")

# Visualize
results.plot_distribution()
results.plot_curtailment_heatmap()
```

---

## Scenario Management

### 1. Pre-configured Scenarios

#### Duck Curve (Default)
```python
from src.demo import DuckCurveScenario

scenario = DuckCurveScenario(
    peak_generation_mw=600,  # Peak solar
    grid_limit_mw=300,  # Grid constraint
    battery_capacity_mwh=500,
    negative_price_hours=[10, 11, 12, 13],
    negative_price_value=-25,  # $/MWh
    evening_price_spike_mwh=140,
)
```

**Characteristics**:
- Solar peaks at noon (600 MW)
- Grid can only export 300 MW
- Negative prices midday force curtailment without battery
- High prices evening (6-8 PM) reward battery discharge
- **Naive result**: 32% curtailment
- **Optimized target**: <10% curtailment

---

#### Price Arbitrage
```python
from src.demo import PriceArbitrageScenario

scenario = PriceArbitrageScenario(
    base_generation_mw=300,
    generation_volatility=0.2,
    price_range=(-50, 200),  # Wide price swings
    price_volatility=0.4,
    battery_capacity_mwh=500,
)
```

**Characteristics**:
- Steady generation with volatility
- Extreme price swings create arbitrage opportunity
- Battery can charge cheap, discharge expensive
- **Success metric**: Profit per MWh

---

#### Grid Emergency
```python
from src.demo import GridEmergencyScenario

scenario = GridEmergencyScenario(
    normal_capacity_mw=400,
    emergency_capacity_mw=150,  # 60% reduction
    emergency_hours=[14, 15, 16],
    generation_profile="sunny",
    battery_capacity_mwh=500,
)
```

**Characteristics**:
- Sudden grid constraint reduction
- Tests ability to avoid violations
- Battery must discharge proactively
- **Success metric**: Zero violations

---

### 2. Custom Scenario Builder

```python
from src.domain import Scenario

scenario = Scenario(
    name="Custom Peak Shaving",
    description="Peak demand reduction test",
    
    # Hourly profiles (24 hours)
    generation_profile={
        "values": [100, 120, 150, 300, 500, 600, 550, 400, 200, 100, 80, 60] * 2,
        "uncertainty_p10": 0.9,  # 10% below
        "uncertainty_p90": 1.1,  # 10% above
    },
    
    grid_capacity={
        "values": [300] * 12 + [400] * 12,  # Increase in afternoon
    },
    
    prices={
        "values": [50, 45, 40, -25, -20, -25, 30, 80, 140, 135, 130, 100] * 2,
    },
    
    battery={
        "capacity_mwh": 500,
        "max_power_mw": 150,
        "initial_soc_pct": 50,
    },
)

results = scenario.run_all_strategies()
```

---

## Reporting & Export

### 1. PDF Reports

```python
from src.reporting import ReportGenerator

generator = ReportGenerator(
    style="professional",  # or "academic"
    include_confidential=False,
)

report = generator.generate(
    results=optimization_results,
    output_path="reports/duck_curve_analysis.pdf",
    include_sections=[
        "executive_summary",
        "scenario_description",
        "strategy_comparison_table",
        "strategy_comparison_charts",
        "kpi_analysis",
        "risk_analysis",
        "assumptions",
        "appendix",
    ],
)
```

**Report Structure**:

1. **Executive Summary** (1 page)
   - Key metrics at a glance
   - Main findings
   - Recommendations

2. **Scenario Description** (1-2 pages)
   - Problem statement
   - Assumptions
   - Input parameters

3. **Strategy Comparison** (2-3 pages)
   - Table: All strategies side-by-side
   - Charts: Dispatch decisions, SOC trajectories
   - Key differences highlighted

4. **KPI Analysis** (1-2 pages)
   - Metric definitions
   - Results with context
   - Performance vs targets

5. **Risk Analysis** (1-2 pages)
   - Stress test results
   - Sensitivity analysis
   - Robustness metrics

6. **Assumptions** (1 page)
   - All assumptions listed
   - Validation results
   - Sensitivity to key assumptions

7. **Appendix** (2-5 pages)
   - Hourly dispatch tables
   - Financial details
   - Technical formulas

---

### 2. Data Export

```python
# CSV Export
results.to_csv("results/hourly_dispatch.csv")

# JSON Export
results.to_json("results/optimization_results.json")

# Excel Export (with charts)
results.to_excel("results/analysis.xlsx", include_charts=True)

# Pickle (Python serialization)
results.save_pickle("results/results.pkl")
```

---

## Advanced Features

### 1. Uncertainty Propagation

```python
from src.uncertainty import PropagateUncertainty

# Propagate P10/P50/P90 generation scenarios
propagator = PropagateUncertainty()

scenarios = propagator.generate_scenarios(
    base_generation=base_forecast,
    p10_factor=0.9,
    p90_factor=1.1,
    num_samples=100,
)

# Run optimization on each
results_by_scenario = {}
for scenario in scenarios:
    results_by_scenario[scenario.name] = optimizer.optimize(scenario)
```

---

### 2. Sensitivity Analysis

```python
from src.uncertainty import SensitivityAnalysis

analysis = SensitivityAnalysis()

# Vary key parameters, measure impact
sensitivities = analysis.compute(
    base_scenario=scenario,
    parameters={
        "battery_efficiency": [0.85, 0.90, 0.95],
        "degradation_cost_per_mwh": [4, 8, 12],
        "grid_capacity_mw": [250, 300, 350],
    },
    metrics=["revenue", "curtailment", "violations"],
)

print(sensitivities.to_dataframe())
#           Parameter              Value  Revenue  Curtailment  Violations
# 0  battery_efficiency              0.85   620000        10.2%            2
# 1  battery_efficiency              0.90   650000         8.5%            1
# 2  battery_efficiency              0.95   680000         6.2%            0
# ...
```

---

### 3. Assumption Validation

```python
from src.validation import ValidateAssumptions

validator = ValidateAssumptions()

# Check if model assumptions hold
report = validator.validate(
    scenario=my_scenario,
    checks=[
        "generation_forecast_accuracy",
        "battery_efficiency_realistic",
        "grid_capacity_steady",
        "price_taker_assumption",
        "degradation_model_valid",
    ],
)

print(report.summary)
# "WARNING: Price volatility exceeds forecast uncertainty band."
# "INFO: Battery efficiency within industry standard (95%)."
# ...

# Show which assumptions might be violated
for check in report.failed_checks:
    print(f"⚠️ {check.name}: {check.reason}")
    print(f"   Recommendation: {check.mitigation}")
```

---

### 4. Real-time Override Logging

```python
# Access override logs in Hybrid Controller
from src.hybrid import OverrideLog

logs = OverrideLog.load("logs/overrides.csv")

for entry in logs:
    print(f"Hour {entry.hour}: MILP={entry.milp_decision}, RL={entry.rl_decision}")
    print(f"  Reason: {entry.override_reason}")
    print(f"  Deviation: {entry.generation_deviation_mw} MW")

# Analyze override patterns
analysis = logs.analyze()
print(f"Override frequency: {analysis.override_frequency:.1%}")
print(f"Most common reason: {analysis.most_common_reason}")
```

---

## Integration Examples

### With Solar Farm Management System

```python
from src.api import OptimizationAPI

api = OptimizationAPI(base_url="http://localhost:8080/api/v1")

# Real-time optimization
response = api.create_optimization(
    scenario={
        "generation_forecast_mw": get_latest_forecast(),
        "grid_capacity_mw": get_grid_status(),
        "prices": get_market_prices(),
        "battery_soc_pct": get_battery_soc(),
    },
    strategies=["hybrid"],  # Use production-ready hybrid
)

# Get decision
result = api.get_optimization(response.optimization_id)
decision = result.results["hybrid"].summary.current_hour_decision

# Execute
execute_dispatch(
    sell_mw=decision.sell_mw,
    store_mw=decision.store_mw,
    curtail_mw=decision.curtail_mw,
)
```

---

### With Data Lake

```python
from src.reporting import DataLakeExporter

exporter = DataLakeExporter(
    s3_bucket="energy-optimization-data",
    prefix="curtailment-engine/",
)

# Automatically push results
exporter.export(
    results=optimization_results,
    timestamp=datetime.now(),
    tags={"scenario": "duck_curve", "version": "v2.1"},
)

# Query later
historical_results = exporter.query(
    start_date="2024-01-01",
    end_date="2024-01-31",
    scenario="duck_curve",
)
```

