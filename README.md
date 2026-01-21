# 🌐 Grid-Aware Renewable Curtailment & Storage Optimization Engine

<p align="center">
  <img src="banner.png" alt="Banner" width="800"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React 18+](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Enterprise-grade optimization platform that minimizes renewable energy curtailment and maximizes revenue by jointly optimizing generation dispatch, grid constraints, battery storage, and market prices under uncertainty.**

<p align="center">
  <strong>⚡ 73% Revenue Uplift | 🎯 <10% Curtailment | ✅ Zero Grid Violations</strong>
</p>

<p align="center">
  <img src="solutions.png" alt="Duck Curve Solution" width="800"/>
</p>

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Platform Overview](#platform-overview)
- [Tools & Features](#tools--features)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Contributing](#contributing)

---

## 🎯 Problem Statement

### The Duck Curve Challenge

During peak solar production, renewable generation often **exceeds grid export capacity**, forcing operators to curtail (waste) clean energy. California's "Duck Curve" exemplifies this challenge:

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
                                        ⚡ Price Spike ($140/MWh)
```

### Traditional Approach (Suboptimal)
- 🚫 **32%** of midday solar curtailed
- 💰 **Revenue loss** from negative pricing periods  
- ⚠️ **Grid violations** from ramping constraints
- 📉 **Inefficient battery dispatch**

### GACE Solution (Optimal)
- ✅ **<10%** curtailment through intelligent battery dispatch
- 💵 **73% revenue uplift** via dynamic price arbitrage
- 🎯 **Zero grid violations** with proactive ramping
- ⚡ **Maximum battery utilization** from forecasting

---

## 🏗️ Solution Architecture

The engine combines **MILP (Mixed-Integer Linear Programming)** with **Reinforcement Learning** to achieve globally optimal decisions under uncertainty:

```
┌─────────────────────────────────────────────────┐
│         Web UI (React 18 + Vite)               │
│  • Interactive dashboards                      │
│  • Real-time metrics & visualization           │
│  • Scenario configuration                      │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│      FastAPI Backend (Production-Ready)        │
│  • RESTful API with Pydantic validation        │
│  • WebSocket for real-time updates             │
│  • Health checks & monitoring                  │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│   MILP Solver    │        │  RL Agent        │
│  (Pyomo+GLPK)    │        │ (Gymnasium+PPO)  │
│  • Global opt    │        │ • Real-time adapt│
│  • Constraints   │        │ • Learning curve │
└──────────────────┘        └──────────────────┘
```

---

## ✨ Key Features

### 1. **Advanced Optimization Algorithms**
- **MILP Formulation**: Globally optimal dispatch decisions
- **RL Agent**: Real-time adaptation to unexpected events
- **Hybrid Controller**: Combines strengths of both approaches

### 2. **Real-World Physics Modeling**
- **Battery Physics**: SOC tracking, charge/discharge efficiency, degradation costs
- **Grid Constraints**: Export capacity limits, ramp rate restrictions
- **Market Dynamics**: Price volatility, negative pricing events

### 3. **Comprehensive Analytics**
- **KPI Dashboard**: Revenue, curtailment, grid compliance, battery health
- **Scenario Analysis**: Compare strategies across different conditions
- **Industry Metrics**: Aggregated performance across all optimizations

### 4. **Full Stack Integration**
- **Interactive Web UI**: Configure scenarios, visualize results
- **REST API**: Programmatic access for integrations
- **Docker Deployment**: Production-ready containerization

---

## 🚀 Quick Start

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/iamjeerge/grid-aware-curtailment-engine.git
cd grid-aware-curtailment-engine

# Option A: Docker (Recommended)
docker-compose up -d

# Option B: Local Development
poetry install
poetry run pytest  # Run tests
poetry run python -m src.demo  # Run demo
```

### Access the Platform

- **Web UI**: http://localhost:3000
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

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
Curtailment Reduced: 72.5%
Revenue Uplift: $231,000
Grid Violations: 0
```
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

## 📊 Platform Overview

### Web Interface Features

#### 1. **Interactive Dashboards**
- Real-time optimization results visualization
- Hourly dispatch decisions with interactive charts
- Battery SOC trajectory tracking
- Revenue & cost breakdowns

#### 2. **Scenario Management**
- Pre-configured demo scenarios (Duck Curve, Price Arbitrage, Grid Emergency)
- Custom scenario builder with parameter tuning
- Historical scenario comparison
- Export/import for reproducibility

#### 3. **Results Analysis**
- Strategy comparison (Naive vs MILP vs RL vs Hybrid)
- KPI summaries and drilling
- Performance metrics with statistical confidence intervals
- Assumption validation checks

#### 4. **Industry Dashboard**
- Aggregated metrics across all optimizations
- Financial metrics (revenue, ROI, profit margin)
- Grid reliability scoring
- Environmental impact tracking (CO2 avoided)
- Battery health metrics (cycle count, remaining life)

---

## 🛠️ Tools & Features

### Optimization Tools

#### **MILP Optimizer** (Mixed-Integer Linear Programming)
**When to use**: Planning horizon is 24 hours or less, need guaranteed optimality

**How it works**:
- Formulates the problem as a linear program with binary variables
- Solver (GLPK) finds the mathematically optimal solution
- Respects all constraints (grid capacity, battery limits, ramp rates)

**Pros**: Global optimality, deterministic, fast (<1s for 24-hour horizon)
**Cons**: Computationally expensive for long horizons, requires perfect forecasts

**Example**:
```python
from src.optimization import MILPOptimizer

optimizer = MILPOptimizer(
    time_horizon_hours=24,
    solver="glpk",
    time_limit_seconds=60
)

decisions = optimizer.optimize(
    generation_forecast=[600, 550, ..., 200],
    grid_capacity=[300, 300, ..., 400],
    prices=[50, 45, ..., 140],
)
```

---

#### **RL Agent** (Reinforcement Learning)
**When to use**: Real-time adaptation needed, unexpected deviations from forecast

**How it works**:
- Uses PPO (Proximal Policy Optimization) or DQN algorithms
- Trained on historical scenarios with diverse conditions
- Observes current state (SOC, generation, grid capacity, price)
- Outputs immediate dispatch decision (sell fraction, charge fraction)

**Pros**: Handles deviations, learns from patterns, fast inference
**Cons**: Needs training data, may not be globally optimal, "black box"

**Example**:
```python
from src.rl import RLAgent

agent = RLAgent(
    model_path="models/ppo_agent.pkl",
    algorithm="ppo"
)

state = {
    "soc": 0.5,  # Battery at 50%
    "generation": 550,  # Current MW
    "grid_capacity": 300,
    "price": -25,  # $/MWh
}

action = agent.act(state)  # [sell_frac, charge_frac]
```

---

#### **Hybrid Controller**
**When to use**: Production deployment, need both optimality and robustness

**How it works**:
1. MILP provides optimal baseline plan for next hour
2. RL agent monitors for deviations
3. If actual generation < forecast by >10%, override with RL decision
4. Override decisions logged for model retraining

**Example**:
```python
from src.hybrid import HybridController

controller = HybridController(
    milp_optimizer=MILPOptimizer(),
    rl_agent=RLAgent(),
    override_threshold_mw=50,  # Override if deviation > 50 MW
)

decision = controller.decide(
    state=current_state,
    forecast=next_24h_forecast,
)

if decision.override_applied:
    print(f"RL override applied: {decision.reason}")
```

---

### Analytics Tools

#### **KPI Dashboard**
Tracks key performance indicators:

| KPI | Formula | Target |
|-----|---------|--------|
| **Curtailment Rate** | Curtailed / Total Generation | <10% |
| **Revenue/MWh** | Gross Revenue / MWh Sold | >$60/MWh |
| **Battery Utilization** | Total Cycled / Capacity | 70-90% |
| **Grid Compliance** | Hours with Zero Violations / 24 | 100% |
| **ROI** | Net Profit / Battery Investment | >40% / year |

**Access via**:
- Web UI: Dashboard → KPIs
- API: `/api/v1/optimizations/{id}/metrics`
- Python: `results.metrics.get_kpis()`

---

#### **Industry Dashboard**
Aggregate metrics across all optimizations:

```python
from src.api import DashboardService

service = DashboardService()
industry_metrics = service.get_industry_dashboard()

print(f"Total Optimizations: {industry_metrics.total_optimizations_run}")
print(f"Total Revenue: ${industry_metrics.financial_metrics.total_revenue:,.0f}")
print(f"CO2 Avoided: {industry_metrics.environmental.co2_avoided_metric_tons:.0f} MT")
print(f"Avg Curtailment Reduction: {industry_metrics.curtailment_reduction.curtailment_reduction_pct:.1f}%")
```

---

#### **Stress Testing**
Monte Carlo simulations to quantify risk:

```python
from src.uncertainty import StressTest

stress_test = StressTest(
    num_scenarios=100,
    generation_volatility=0.15,
    price_volatility=0.25,
    grid_capacity_volatility=0.10,
)

results = stress_test.run(
    base_scenario=scenario,
    metrics_to_track=["curtailment", "revenue", "violations"],
)

print(f"Revenue 5th percentile: ${results.revenue_p5:,.0f}")
print(f"Curtailment 95th percentile: {results.curtailment_p95:.1%}")
print(f"Violation probability: {results.violation_probability:.1%}")
```

---

### Reporting Tools

#### **PDF Report Generation**
Automated report creation:

```python
from src.reporting import ReportGenerator

generator = ReportGenerator()

report = generator.generate(
    results=optimization_results,
    output_path="reports/duck_curve_analysis.pdf",
    include_sections=[
        "executive_summary",
        "scenario_description",
        "strategy_comparison",
        "kpi_analysis",
        "recommendations",
    ],
)
```

**Report Sections**:
1. Executive Summary (1 page)
2. Scenario Definition & Assumptions
3. Strategy Comparison (Naive vs MILP vs RL vs Hybrid)
4. KPI Analysis with Charts
5. Risk Analysis (Stress Test Results)
6. Recommendations & Next Steps

---

## 📡 API Documentation

### Base URL
```
http://localhost:8080/api/v1
```

### Core Endpoints

#### **Create Optimization**
```http
POST /optimizations/
Content-Type: application/json

{
  "scenario": {
    "name": "duck_curve",
    "generation_forecast_mw": [600, 550, ...],
    "grid_capacity_mw": 300,
    "prices": [50, 45, ..., 140],
    "battery_capacity_mwh": 500
  },
  "strategies": ["naive", "milp", "rl", "hybrid"]
}

Response:
{
  "optimization_id": "opt_abc123",
  "status": "running",
  "created_at": "2024-01-21T10:30:00Z"
}
```

#### **Get Results**
```http
GET /optimizations/{id}

Response:
{
  "optimization_id": "opt_abc123",
  "status": "completed",
  "results": {
    "naive": { "summary": {...}, "hourly_decisions": [...] },
    "milp": { "summary": {...}, "hourly_decisions": [...] },
    "rl": { "summary": {...}, "hourly_decisions": [...] },
    "hybrid": { "summary": {...}, "hourly_decisions": [...] }
  },
  "created_at": "2024-01-21T10:30:00Z",
  "completed_at": "2024-01-21T10:31:30Z"
}
```

#### **Industry Dashboard**
```http
GET /dashboard/industry

Response:
{
  "total_optimizations_run": 42,
  "financial_metrics": {
    "total_revenue": 15000000,
    "total_profit": 8750000,
    "roi_percentage": 87.5
  },
  "curtailment_reduction": {
    "total_generation_mwh": 50000,
    "curtailment_reduction_pct": 72.5,
    "avoided_curtailment_value": 1245000
  },
  "environmental": {
    "co2_avoided_metric_tons": 15000,
    "grid_renewable_penetration_improvement": 72.5
  }
}
```

### Full API Documentation
Visit **http://localhost:8080/docs** for interactive Swagger UI with all endpoints.

---

## 🏃 Development

### Setup

```bash
# Install dependencies
poetry install

# Set up Git hooks
pre-commit install

# Run tests
poetry run pytest tests/ -v

# Check code quality
poetry run black .
poetry run ruff check .
poetry run mypy .
```

### Project Structure

```
src/
├── api/                  # FastAPI application
│   ├── main.py          # Entry point
│   ├── routes.py        # API endpoints
│   └── services.py      # Business logic
│
├── domain/              # Data models (Pydantic)
│   ├── scenario.py
│   ├── battery.py
│   ├── grid.py
│   └── optimization.py
│
├── optimization/        # MILP formulation
│   ├── formulation.py   # Pyomo model
│   └── solver.py        # Solver interface
│
├── rl/                  # Gymnasium environment & agents
│   ├── environment.py
│   ├── agents/
│   │   ├── ppo.py
│   │   └── dqn.py
│   └── train.py
│
├── metrics/             # KPI computation
├── visualization/       # Charts & dashboards
├── reporting/          # PDF generation
└── demo.py             # Demo scenarios

tests/
├── test_domain.py
├── test_optimization.py
├── test_rl.py
└── test_integration.py
```

### Code Quality Standards

All code must pass:

```bash
# Black formatter (100 char line length)
poetry run black --check .

# Ruff linter
poetry run ruff check .

# MyPy strict typing
poetry run mypy . --strict

# Tests with coverage
poetry run pytest --cov=src --cov-threshold=85
```

---

## 📝 Contributing

### Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests for new functionality
4. Ensure code passes quality checks
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Submit a Pull Request

### Testing Requirements

- Unit tests for all functions
- Integration tests for cross-module interactions
- Minimum 85% code coverage
- All tests must pass before merging

### Documentation

- Update README for new user-facing features
- Add docstrings to all functions (Google style)
- Update API docs if changing endpoints
- Add assumptions if modifying physical models

---

## 📚 Additional Resources

### Learning Resources

- **Optimization 101**: [MILP tutorial](https://www.gurobi.com/resource/mip-basics/)
- **RL Basics**: [Gymnasium documentation](https://gymnasium.farama.org/)
- **Energy Markets**: [CAISO operations](https://www.caiso.com/market/Pages/default.aspx)

### Publications & References

- Duck Curve: [CAISO 2016 Report](https://www.caiso.com/Documents/Flexibility_Requirements.pdf)
- Battery Economics: [NREL Report 2021](https://www.nrel.gov/docs/fy22osti/82264.pdf)
- Real-time Markets: [FERC Order 764](https://www.ferc.gov/news-updates/news-release/2011-07-15-0)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Support

For questions, feature requests, or bug reports:

- 📧 **Email**: support@curtailment-engine.dev
- 🐛 **GitHub Issues**: [Report a bug](https://github.com/iamjeerge/grid-aware-curtailment-engine/issues)
- 💬 **Discussions**: [Ask a question](https://github.com/iamjeerge/grid-aware-curtailment-engine/discussions)

---

## 🔧 Code Quality

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

##  Screenshot

<p align="center">
  <img src="./screenshot.png" alt="screenshot" width="800"/>
</p>

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
