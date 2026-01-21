# Documentation Index

Welcome to the Grid-Aware Curtailment Engine documentation. This is your complete reference guide.

---

## 📚 Documentation Structure

```
docs/
├── INDEX.md (you are here)
├── ARCHITECTURE.md          # System design & module dependencies
├── FEATURES.md              # In-depth feature explanations
├── USER_GUIDE.md            # Web interface walkthrough
├── API.md                   # REST API reference
└── ../README.md             # Main project README
```

---

## 🚀 Quick Links by Role

### For First-Time Users
1. **Start here**: [Main README](../README.md#-quick-start)
2. **Understand the problem**: [Duck Curve Problem](../README.md#-problem-statement)
3. **Try a demo**: Run `docker-compose up` and visit http://localhost:3000
4. **Learn the UI**: [User Guide - Getting Started](USER_GUIDE.md#-getting-started)
5. **Explore features**: [Features Guide - Optimization Tools](FEATURES.md#optimization-algorithms)

### For Developers
1. **Architecture overview**: [Architecture Document](ARCHITECTURE.md)
2. **Code structure**: [Module Structure](../README.md#module-structure)
3. **API documentation**: [API Reference](API.md)
4. **Development setup**: [Development Section](../README.md#-development)
5. **Testing**: [Testing Guide](../README.md#-testing)

### For Data Scientists / Researchers
1. **MILP formulation**: [MILP Optimizer Details](FEATURES.md#-milp-optimizer)
2. **RL environment**: [RL Agent Details](FEATURES.md#-rl-agent)
3. **Battery physics**: [Hybrid Controller Section](FEATURES.md#-hybrid-controller)
4. **Stress testing**: [Uncertainty Handling](FEATURES.md#advanced-features)
5. **Research papers**: See [Additional Resources](../README.md#-additional-resources)

### For Platform Operators
1. **Deployment**: [Docker Deployment](../README.md#-quick-start)
2. **Configuration**: [Configuration Section](../README.md#-configuration)
3. **Monitoring**: Check API health at `/health` endpoint
4. **Industry dashboard**: [Dashboard Guide](USER_GUIDE.md#-dashboard-features)
5. **PDF reports**: [Reporting Tools](FEATURES.md#reporting--export)

### For Integration Partners
1. **API overview**: [API Reference](API.md)
2. **Integration examples**: [API Integration Examples](API.md#integration-examples)
3. **Data export**: [Data Export Options](FEATURES.md#data-export)
4. **Webhook support**: Check WebSocket section in [API.md](API.md#websocket-streaming-optional)

---

## 📖 Documentation Sections

### Main README
**What it covers**: Project overview, quick start, high-level architecture, tools overview, key features

**When to read it**: 
- Getting started (5 min read)
- Understanding project scope
- High-level feature overview
- Installation instructions

**Key sections**:
- Problem Statement
- Solution Architecture
- Quick Start
- Platform Overview
- Tools & Features

---

### Architecture Document
**What it covers**: Detailed system design, module dependencies, data flow, design patterns

**When to read it**:
- Contributing code
- Understanding module interactions
- Debugging issues
- Planning extensions

**Key sections**:
- High-Level Architecture (visual)
- Module Dependencies (visual)
- Module Structure (detailed)
- Data Flow Diagrams
- Key Patterns & Conventions

---

### Features Document
**What it covers**: In-depth explanation of every tool and feature

**When to read it**:
- Understanding algorithm details
- Comparing strategies
- Learning metric definitions
- Exploring advanced features

**Key sections**:
- Optimization Algorithms (MILP, RL, Hybrid)
- Analytics & Monitoring (KPIs, Dashboards, Stress Testing)
- Scenario Management (Pre-configured, Custom Builder)
- Reporting & Export
- Advanced Features (Uncertainty, Sensitivity, Validation)
- Integration Examples

---

### User Guide
**What it covers**: Complete walkthrough of web interface

**When to read it**:
- Using the platform for the first time
- Running optimizations
- Analyzing results
- Managing scenarios
- Exporting data

**Key sections**:
- Getting Started (Login, Dashboard)
- Dashboard Features (Metrics, Recent, KPIs)
- Running Optimizations (Step-by-step guide)
- Results Analysis (Charts, Comparisons)
- Data Import/Export
- Settings & Preferences
- Example Workflows

---

### API Documentation
**What it covers**: Complete REST API reference with examples

**When to read it**:
- Building integrations
- Programmatic access
- Scripting optimizations
- Real-time monitoring
- Data pipeline setup

**Key sections**:
- Base URL & Authentication
- Core Endpoints (CRUD operations)
- Dashboard & Analytics Endpoints
- Error Handling
- Rate Limiting
- WebSocket Streaming
- Integration Examples (Python, JavaScript, cURL)
- Pagination & Filtering

---

## 🎯 Common Tasks & Where to Find Help

### Task: Understand the core algorithm
→ [MILP Optimizer in Features.md](FEATURES.md#-milp-optimizer)

### Task: Deploy to production
→ [Docker Deployment in README.md](../README.md#-quick-start)

### Task: Run an optimization
→ [Running Optimizations in User Guide](USER_GUIDE.md#-running-optimizations)

### Task: Compare strategies
→ [Strategy Comparison in User Guide](USER_GUIDE.md#section-2-strategy-comparison)

### Task: Access results programmatically
→ [API Reference](API.md)

### Task: Generate PDF report
→ [PDF Reports in Features.md](FEATURES.md#-pdf-reports)

### Task: Stress test a scenario
→ [Stress Testing in Features.md](FEATURES.md#advanced-features)

### Task: Extend the platform
→ [Architecture Document](ARCHITECTURE.md)

### Task: Integrate with external system
→ [Integration Examples in API.md](API.md#integration-examples)

### Task: Understand constraints
→ [MILP Formulation in Features.md](FEATURES.md#mathematical-formulation)

### Task: Learn battery physics
→ [Hybrid Controller in Features.md](FEATURES.md#-hybrid-controller)

---

## 📊 Key Concepts at a Glance

### The Problem
- **Duck Curve**: Solar peaks at noon, grid can't export excess → Curtailment (waste)
- **Suboptimal Today**: Naive strategy curtails 32%, misses price arbitrage
- **Optimal Target**: Smart battery dispatch can reduce curtailment to <10%, earn 73% more

### The Solution
- **MILP Optimizer**: Mathematically optimal day-ahead plan
- **RL Agent**: Real-time adaptation to forecast errors
- **Hybrid Controller**: Combined MILP + RL for production robustness

### Key Metrics
- **Curtailment Rate**: % of generation wasted (target: <10%)
- **Revenue Uplift**: % more profitable vs naive (target: >50%)
- **Grid Compliance**: % of hours within capacity (target: 100%)
- **Battery Cycles**: Equivalent full cycles (target: <4000 over 10 years)
- **ROI**: Return on battery investment (target: >40% annually)

### Technologies
- **Optimization**: Pyomo (modeling) + GLPK (solver)
- **RL**: Gymnasium environment + PPO/DQN agents
- **API**: FastAPI (production-grade)
- **Frontend**: React 18 + Vite + TypeScript
- **Deployment**: Docker + Docker Compose

---

## 🔍 For Each Component

### Battery Energy Storage System (BESS)
- **Where to learn**: [FEATURES.md - Battery Physics](FEATURES.md#-hybrid-controller)
- **Default config**: 500 MWh capacity, 150 MW power
- **Physics**: 95% charge/discharge efficiency, $8/MWh degradation cost
- **Constraints**: SOC between 10-90%, power limits

### MILP Optimization
- **Where to learn**: [FEATURES.md - MILP Optimizer](FEATURES.md#-milp-optimizer)
- **Math**: Mixed-Integer Linear Program (full formulation shown)
- **Solver**: GLPK (open source, fast)
- **Best for**: Day-ahead planning, guaranteed optimality
- **Worst for**: Long horizons (>24h), forecast uncertainty

### RL Agent
- **Where to learn**: [FEATURES.md - RL Agent](FEATURES.md#-rl-agent)
- **Algorithm**: PPO or DQN
- **Environment**: Gymnasium (OpenAI standard)
- **Best for**: Real-time adaptation, learning patterns
- **Worst for**: Guaranteed optimality, interpretability

### Grid Constraints
- **Where to learn**: [FEATURES.md - Grid Constraints](FEATURES.md#caiso-style-grid-modeling)
- **Constraint types**: Export capacity, ramp rates, congestion windows
- **CAISO-style**: Dynamic limits, emergency procedures
- **Validation**: Assume <10% margin for safety

### Market Prices
- **Where to learn**: [FEATURES.md - Market Dynamics](FEATURES.md#-hybrid-controller)
- **Price range**: Typically -$50 to +$200/MWh
- **Assumption**: Price-taker (facility doesn't affect prices)
- **Value**: Battery value comes from arbitrage between low/high

---

## 🛠️ Troubleshooting Guide

### "I can't connect to the API"
- Check: Is backend running? `docker ps | grep backend`
- Check: Is it healthy? `curl http://localhost:8080/health`
- Check: Are ports configured correctly? (default: 8080)

### "Optimization is slow"
- Try: Set timeout lower in Advanced Options
- Try: Use MIP gap tolerance 5% instead of 1%
- Try: Reduce Monte Carlo simulations from 100 to 50

### "Results don't match expected"
- Check: Are inputs (generation, prices, grid capacity) correct?
- Try: Run Duck Curve demo first to validate setup
- Try: Enable Assumption Validation to check model validity

### "RL agent not learning"
- Check: Is there enough training data? (needs diverse scenarios)
- Try: Retrain from scratch: `poetry run python src/rl/train.py`
- Try: Use pre-trained model from model zoo

### "Grid violations detected"
- Check: Are grid capacity values realistic?
- Try: Enable Stress Testing to quantify violation probability
- Try: Add safety margin (reduce grid capacity by 5%)

---

## 📈 Learning Path

**Beginner (30 minutes)**:
1. Read [README Problem Statement](../README.md#-problem-statement)
2. Run Duck Curve demo (see [User Guide](USER_GUIDE.md#section-1-executive-summary))
3. Examine results comparison table
4. Read [KEY FEATURES](../README.md#-key-features)

**Intermediate (2 hours)**:
1. Read [FEATURES.md - Optimization Algorithms](FEATURES.md#optimization-algorithms)
2. Understand each strategy's pros/cons
3. Explore API via Swagger UI at `/docs`
4. Try custom scenario in web UI
5. Read [USER GUIDE - Results Analysis](USER_GUIDE.md#-results-analysis)

**Advanced (1 day)**:
1. Deep dive [MILP Formulation](FEATURES.md#mathematical-formulation)
2. Study RL environment in source: `src/rl/environment.py`
3. Review battery physics: `src/battery/`
4. Run full API integration example
5. Run stress tests and sensitivity analysis

**Expert (ongoing)**:
1. Contribute to optimization algorithms
2. Extend RL training pipeline
3. Integrate with your systems
4. Customize for your use case
5. Contribute improvements back to community

---

## 🤝 Getting Help

### Documentation Questions
- Check relevant section in docs/
- Search for keyword in docs/
- See [Troubleshooting Guide](#-troubleshooting-guide) above

### Technical Issues
- File GitHub issue with:
  - Step to reproduce
  - Expected vs actual behavior
  - System info (OS, Python version, etc.)

### Feature Requests
- Open GitHub discussion
- Describe use case
- Suggest implementation approach

### Integration Help
- Review [API Integration Examples](API.md#integration-examples)
- Check [Architecture Document](ARCHITECTURE.md) for design patterns
- Contact support@curtailment-engine.dev

---

## 📊 Statistics

- **Total Documentation**: ~50 pages
- **Code Examples**: 100+ snippets
- **API Endpoints**: 15+ endpoints
- **Features**: 12+ major tools
- **Demo Scenarios**: 3 pre-configured
- **Components**: 8+ frontend components with help

---

## 🎓 Educational Value

This platform is designed to be educational. Each feature includes:
- Problem motivation
- Algorithm explanation
- Mathematical formulation (where applicable)
- Practical examples
- Pros/cons analysis
- Real-world considerations

Great for:
- University research projects
- Energy engineering education
- Optimization algorithm learning
- Full-stack development case study
- Open-source contribution practice

---

## 📝 Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1 | 2024-01-21 | Comprehensive documentation suite |
| 2.0 | 2024-01-15 | API documentation complete |
| 1.5 | 2024-01-10 | Features guide added |
| 1.0 | 2024-01-01 | Initial documentation |

---

## ✅ Documentation Checklist

- ✅ Main README with quick start
- ✅ Architecture document
- ✅ Features guide with algorithm details
- ✅ User guide for web interface
- ✅ Complete API reference
- ✅ Troubleshooting guide
- ✅ Integration examples
- ✅ Learning paths
- ✅ Interactive help tooltips
- ✅ Code examples for each feature

---

## 🚀 Next Steps

**To get started**:
1. Visit [README Quick Start](../README.md#-quick-start)
2. Launch with `docker-compose up`
3. Open http://localhost:3000
4. Read [User Guide - Getting Started](USER_GUIDE.md#-getting-started)

**To understand the science**:
1. Read [FEATURES.md - Optimization Algorithms](FEATURES.md#optimization-algorithms)
2. Study [MILP Formulation](FEATURES.md#mathematical-formulation)
3. Explore [Hybrid Controller Logic](FEATURES.md#-hybrid-controller)

**To integrate with your system**:
1. Review [API Reference](API.md)
2. Check [Integration Examples](API.md#integration-examples)
3. Follow [Your Use Case Guide](#-common-tasks--where-to-find-help)

---

**Happy optimizing! 🎯⚡📊**

