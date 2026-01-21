# Grid-Aware Renewable Curtailment & Storage Optimization Engine


<p align="center">
  <img src="banner.png" alt="Banner" width="800"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 301 passing](https://img.shields.io/badge/tests-301%20passing-brightgreen.svg)]()

**Production-grade optimization system that minimizes renewable energy curtailment and maximizes revenue by jointly optimizing generation dispatch, grid constraints, battery storage, and market prices under uncertainty.**

<p align="center">
  <img src="docs/images/duck_curve_solution.png" alt="Duck Curve Solution" width="800"/>
</p>

---

## 🎯 The Problem: Duck Curve Curtailment

During peak solar production, renewable generation often exceeds grid export capacity, forcing operators to **curtail (waste) clean energy**. The California "Duck Curve" exemplifies this challenge:

```
                    ☀️ Peak Solar (600 MW)
                         ___
                       /     \
                      /       \
Grid Limit -------- /-----300 MW------\--------
(300 MW)           /                   \
                  /                     \
                 /                       \
        Morning                          Evening
                                        ⚡ Price Spike
```

**Without optimization:**
- 32% of midday solar is curtailed
- Revenue loss from negative pricing periods
- Grid violations from ramping constraints

**With this engine:**
- <10% curtailment through intelligent battery dispatch
- 73% revenue uplift via price arbitrage
- Zero grid violations guaranteed

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/iamjeerge/grid-aware-curtailment-engine.git
cd grid-aware-curtailment-engine

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

### 30-Second Demo

```python
from src.demo import run_duck_curve_demo

# Run the flagship demo scenario
results = run_duck_curve_demo()

# View key metrics
print(f"Curtailment Reduced: {results['curtailment_reduction']:.1%}")
print(f"Revenue Uplift: ${results['revenue_uplift']:,.0f}")
print(f"Grid Violations: {results['violations']}")
```

**Expected Output:**
```
🦆 Duck Curve Optimization Demo
================================
Scenario: 600 MW solar peak, 300 MW grid limit, 500 MWh battery

Naive Strategy Results:
  • Curtailment: 32.1%
  • Revenue: $420,000
  • Grid Violations: 5

MILP Optimized Results:
  • Curtailment: 8.2%
  • Revenue: $727,000
  • Grid Violations: 0

Improvement:
  ✅ Curtailment Reduced: 74.4%
  ✅ Revenue Uplift: 73.1% (+$307,000)
  ✅ Zero Grid Violations
```

---

## 📊 Key Features

### 1. Multi-Strategy Optimization

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Naive** | Sell immediately, charge when curtailed | Baseline comparison |
| **MILP** | Mixed-Integer Linear Programming | Optimal day-ahead scheduling |
| **RL** | Reinforcement Learning (PPO/DQN) | Real-time adaptation |
| **Hybrid** | MILP + RL with override logic | Production deployment |

### 2. Uncertainty Handling

- **Probabilistic Forecasts**: P10/P50/P90 generation scenarios
- **Monte Carlo Stress Testing**: 100+ simulations for risk quantification
- **Scenario-Based Optimization**: Hedge against forecast errors

### 3. Battery Physics & Economics

- Round-trip efficiency modeling (90% default)
- Degradation cost tracking ($8/MWh throughput)
- SOC constraints and ramp rate limits
- Arbitrage value computation

### 4. CAISO-Style Grid Modeling

- Dynamic export capacity limits
- Congestion window detection
- Ramp rate constraints
- Emergency curtailment events

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Weather    │  │    Grid      │  │   Market     │              │
│  │  Generator   │  │ Constraints  │  │   Prices     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│                      OPTIMIZATION LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Hybrid Controller                          │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐          │   │
│  │  │   MILP Optimizer    │◄──►│    RL Agent         │          │   │
│  │  │   (Pyomo/GLPK)      │    │    (Gymnasium)      │          │   │
│  │  │                     │    │                     │          │   │
│  │  │  • Day-ahead plan   │    │  • Real-time adjust │          │   │
│  │  │  • Global optimal   │    │  • Handle deviations│          │   │
│  │  └─────────────────────┘    └─────────────────────┘          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│                       BATTERY MODEL                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  • SOC Tracking (10-90% range)                               │   │
│  │  • Charge/Discharge Efficiency (95%/95%)                     │   │
│  │  • Degradation Cost Model ($8/MWh cycled)                    │   │
│  │  • Power Limits (150 MW charge/discharge)                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│                       OUTPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Decisions   │  │   Metrics    │  │   Reports    │              │
│  │  (per hour)  │  │   & KPIs     │  │   (PDF)      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/
├── domain/           # Pydantic models (immutable data structures)
├── generators/       # Synthetic data: weather, grid, prices
├── battery/          # BESS physics, SOC tracking, degradation
├── controllers/      # Naive heuristic baseline controller
├── optimization/     # MILP formulation (Pyomo), solver interface
├── rl/               # Gymnasium environment, PPO/DQN agents
├── hybrid/           # MILP + RL combined controller
├── uncertainty/      # Monte Carlo stress testing, risk analysis
├── metrics/          # KPI computation, performance analysis
├── visualization/    # Matplotlib plots, dashboards
├── reporting/        # PDF report generation
└── validation/       # Assumption documentation & auditing
```

---

## 📈 Demo Scenarios

### Scenario 1: Duck Curve Trap (Default)

The flagship scenario demonstrating the core value proposition:

```python
from src.demo import DuckCurveScenario

scenario = DuckCurveScenario(
    peak_generation_mw=600,
    grid_limit_mw=300,
    battery_capacity_mwh=500,
    negative_price_hours=[10, 11, 12, 13],  # -$25/MWh midday
    evening_price_spike=140,  # $/MWh at 6-8 PM
)

results = scenario.run()
results.plot_dashboard()
results.generate_report("duck_curve_analysis.pdf")
```

### Scenario 2: Price Volatility Arbitrage

Maximize battery value during extreme price swings:

```python
from src.demo import PriceArbitrageScenario

scenario = PriceArbitrageScenario(
    price_range=(-50, 200),  # $/MWh
    volatility=0.4,
)
```

### Scenario 3: Grid Emergency Response

Handle sudden capacity reductions:

```python
from src.demo import GridEmergencyScenario

scenario = GridEmergencyScenario(
    normal_capacity_mw=400,
    emergency_capacity_mw=150,
    emergency_hours=[14, 15, 16],
)
```

---

## 🧪 Testing

```bash
# Run all tests (301 tests)
poetry run pytest tests/ -v

# Run specific module tests
poetry run pytest tests/test_optimization.py -v
poetry run pytest tests/test_battery.py -v
poetry run pytest tests/test_rl.py -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html
```

### Test Coverage by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| Domain Models | 25 | 100% |
| Generators | 23 | 95% |
| Battery | 18 | 98% |
| Controllers | 16 | 95% |
| Optimization | 30 | 92% |
| RL Environment | 22 | 90% |
| Hybrid Controller | 20 | 88% |
| Uncertainty | 32 | 94% |
| Metrics | 33 | 96% |
| Visualization | 31 | 85% |
| Reporting | 22 | 90% |
| Validation | 40 | 100% |

---

## 📋 Configuration

### Battery Defaults (CAISO-style)

```python
BATTERY_DEFAULTS = {
    "capacity_mwh": 500,
    "max_power_mw": 150,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "min_soc": 0.10,
    "max_soc": 0.90,
    "degradation_cost_per_mwh": 8.0,
}
```

### Optimization Parameters

```python
OPTIMIZATION_DEFAULTS = {
    "horizon_hours": 24,
    "solver": "glpk",
    "time_limit_seconds": 60,
    "mip_gap": 0.01,
}
```

---

## 📖 Documentation

### Key Assumptions

This model makes explicit assumptions documented in `src/validation/assumptions.py`:

| Category | Key Assumptions |
|----------|-----------------|
| **Grid** | Static export capacity within planning horizon |
| **Battery** | 90% round-trip efficiency, $8/MWh degradation |
| **Market** | Price-taker (facility doesn't affect prices) |
| **Forecast** | P10/P50/P90 uncertainty bands valid |

Run assumption audit:

```python
from src.validation import get_all_assumptions, validate_assumptions

# List all assumptions
for assumption in get_all_assumptions():
    print(f"{assumption.id}: {assumption.title}")

# Validate against actual parameters
report = validate_assumptions(
    grid_capacity_mw=300,
    battery_efficiency=0.90,
)
print(report.summary)
```

---

## 🔧 Development

### Code Quality

```bash
# Format code
poetry run black .

# Lint
poetry run ruff check . --fix

# Type check
poetry run mypy src/
```

### Adding New Features

1. Define domain models in `src/domain/`
2. Implement business logic in appropriate module
3. Add tests in `tests/test_<module>.py`
4. Update documentation

---

## 📊 Sample Results

### Revenue Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│              Revenue Breakdown (24-hour horizon)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Energy Sales      ████████████████████████████  $650,000   │
│  Battery Arbitrage ████████                      $77,000    │
│  Degradation Cost  ██                           -$16,000    │
│  ─────────────────────────────────────────────────────────  │
│  Net Profit                                      $711,000   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Risk Analysis (Monte Carlo)

```
┌─────────────────────────────────────────────────────────────┐
│              Profit Distribution (100 simulations)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Mean:     $711,000                                         │
│  Std Dev:  $45,000                                          │
│  VaR (5%): $635,000                                         │
│  Sharpe:   15.8                                             │
│                                                              │
│  Histogram:                                                  │
│       ▂▃▅▇█████▇▅▃▂                                         │
│     $600k    $700k    $800k                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CAISO** for market structure inspiration
- **NREL** for battery cost data
- **Pyomo** for optimization framework
- **Gymnasium** for RL environment standards

---

## 📞 Contact

**Gururaj Jeerge** - [@iamjeerge](https://github.com/iamjeerge)

Project Link: [https://github.com/iamjeerge/grid-aware-curtailment-engine](https://github.com/iamjeerge/grid-aware-curtailment-engine)
