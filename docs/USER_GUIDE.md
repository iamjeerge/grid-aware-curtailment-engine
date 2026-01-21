# Web Interface User Guide

Complete walkthrough of the Grid-Aware Curtailment Engine web platform.

---

## 🏠 Getting Started

### Login & Dashboard

1. Navigate to **http://localhost:3000**
2. You're on the **Main Dashboard** - overview of all optimizations

**Main Dashboard shows**:
- Total optimizations run
- Recent scenarios
- Key metrics summary
- Quick links to features

---

## 📊 Dashboard Features

### 1. Industry Metrics Overview

**Location**: Dashboard → Industry Metrics (top card)

Shows aggregate statistics across ALL optimizations:

```
Financial Summary
├─ Total Revenue: $15.2M
├─ Total Cost: $6.1M
├─ Net Profit: $9.1M
└─ ROI: 91%

Grid Reliability
├─ Compliance Rate: 99.4%
├─ Total Violations: 3
└─ Max Violation: 50 MW

Curtailment Reduction
├─ Baseline Rate: 5.0%
├─ Optimized Rate: 2.1%
├─ Reduction: 58%
└─ Value Saved: $150K

Environmental Impact
├─ CO2 Avoided: 15,000 MT
└─ Household Equivalent: 37,500 days
```

**Use case**: Understand overall portfolio performance

---

### 2. Recent Optimizations

**Location**: Dashboard → Recent Scenarios (main card)

Shows last 10 optimizations with:
- Scenario name
- Strategy used
- Date run
- Key results (revenue, curtailment)
- Status

**Actions**:
- Click any row to view detailed results
- Export results as PDF
- Compare with other scenarios
- Re-run with new parameters

---

## 🔧 Running Optimizations

### Step 1: Select Scenario

**Location**: Top navigation → "New Optimization"

Three options:

#### Option A: Pre-configured Demo (Recommended first-time)

```
Select Scenario Type
├─ Duck Curve Trap ⭐ (RECOMMENDED)
│  └─ Solar peak vs grid limit, ideal for demo
├─ Price Arbitrage
│  └─ Extreme price swings for battery value
└─ Grid Emergency
   └─ Sudden capacity reduction, test robustness
```

**Each shows**:
- Scenario description
- Key characteristics
- Expected results
- "Launch" button

---

#### Option B: Custom Scenario

**Location**: "Create Custom Scenario"

Fill in parameters:

```
Generation Profile
├─ Peak generation MW: 600
├─ Time of peak: 12:00 (noon)
└─ Profile shape: "sunny" / "cloudy" / "custom"

Grid Constraints
├─ Export capacity MW: 300
├─ Peak capacity time: varies / constant
└─ Emergency hours: none / custom

Market Prices
├─ Price range: $40-$140 per MWh
├─ Negative price hours: 10-13
└─ Evening spike: $140 per MWh

Battery Configuration
├─ Capacity MWh: 500
├─ Max Power MW: 150
└─ Starting SOC%: 50%
```

**Tips**:
- Hover for parameter explanations
- Suggested ranges shown in grey
- Click "Validate" to check feasibility
- Click "Advanced" for more options

---

### Step 2: Select Strategies

**Location**: "Choose Optimization Strategies"

```
☑ Naive
   └─ Simple baseline (always recommended for comparison)

☑ MILP Optimizer
   └─ Mathematically optimal (for planning)

☑ RL Agent
   └─ Adaptive learning (for real-time)

☑ Hybrid Controller
   └─ Combined MILP + RL (production ready)
```

**Recommendations by use case**:

| Use Case | Strategies |
|----------|-----------|
| First-time learning | Naive + MILP |
| Production deployment | Hybrid only |
| Research/analysis | All four |
| Risk assessment | MILP + stress test |

---

### Step 3: Advanced Options

**Location**: "Advanced Settings" (optional)

```
Analysis Options
├─ Include Stress Testing
│  ├─ Number of Monte Carlo simulations: 100
│  ├─ Generation volatility: 15%
│  ├─ Price volatility: 25%
│  └─ Grid capacity volatility: 10%
├─ Include Sensitivity Analysis
│  └─ Vary battery efficiency, degradation cost, etc.
└─ Include Assumption Validation
   └─ Check if model assumptions hold

Solver Options
├─ Optimization timeout: 60 seconds
├─ MILP gap tolerance: 1%
└─ RL inference batch size: 32

Reporting
├─ Generate PDF report: YES
└─ Export data as CSV/Excel: YES
```

---

### Step 4: Run Optimization

**Location**: "Run Optimization" button

**Status page shows**:
- Real-time solver progress
- Estimated time remaining
- Current step
  - "Building MILP model..."
  - "Solving optimization..."
  - "Training RL agent..." (if selected)
  - "Running stress tests..." (if selected)
  - "Generating report..."

**Time estimate**:
- Naive: ~1 second
- MILP: ~5-10 seconds
- RL: ~3-5 seconds (if model cached)
- Stress test (100 runs): ~30-60 seconds
- Full analysis: 2-3 minutes

---

## 📈 Results Analysis

### View Results

**Location**: After optimization completes, automatically shows Results page

---

### Section 1: Executive Summary

```
DUCK CURVE OPTIMIZATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scenario: Duck Curve Trap
Peak Generation: 600 MW | Grid Limit: 300 MW | Battery: 500 MWh

KEY FINDINGS
────────────
Revenue Uplift: 73.1% (+$307K)
Curtailment Reduction: 74.4% (from 32% to 8%)
Grid Violations: 0 (vs 5 in baseline)
Battery Utilization: 82%

RECOMMENDATION: Deploy Hybrid Controller for production
```

**Actions**:
- Print summary
- Share (copy link)
- Export as PDF

---

### Section 2: Strategy Comparison

**Shows 4 strategies side-by-side**:

```
                    NAIVE    MILP    RL    HYBRID
Revenue            $420K    $727K   $680K  $720K
Curtailment        32.1%    8.2%    12.1%  9.5%
Grid Violations    5        0       2      0
Battery Cycles     0.8      1.2     0.9    1.1
Profit/MWh        $35       $61     $57    $60
Computation Time   <1s      8s      2s     10s
```

**Insights**:
- MILP most profitable (globally optimal)
- RL more adaptive (better for real-time)
- Hybrid balances both (recommended)
- Naive shows value of optimization

**Click row to drill down**:
- Hourly dispatch decisions
- Battery SOC trajectory
- Revenue breakdown
- Violation timeline

---

### Section 3: Interactive Charts

#### Chart 1: Hourly Dispatch Decisions

```
[Stacked area chart showing]
- Generation (solid line, top)
- Grid Export (blue area)
- Battery Charging (green area)
- Battery Discharging (yellow area)
- Curtailment (red area)
```

**Interactive**:
- Hover for exact values
- Click legend to show/hide strategies
- Zoom to time range
- Download as PNG

**Key patterns**:
- **9-13h**: Charge during negative prices
- **14-16h**: Curtailment if needed
- **17-19h**: Discharge during price spike
- **20-24h**: Charge during low prices

---

#### Chart 2: Battery State of Charge (SOC)

```
[Line chart with confidence band showing]
- MILP SOC trajectory (blue line)
- RL SOC trajectory (green line)
- Min/Max SOC bounds (grey shaded)
- Current SOC (red dot)
```

**Key observations**:
- MILP: Smooth trajectory, predictable
- RL: More dynamic, reactive
- Both stay within bounds (10-90%)
- SOC reflects generation and prices

---

#### Chart 3: Revenue Attribution

```
[Waterfall chart showing]
Generation → Grid Sales → Battery Discharge Revenue
         ↓
      Degradation Cost → Battery Charge Cost
         ↓
      Net Profit
```

**Breakdown**:
- Gross from grid sales: $480K
- Bonus from discharge: $247K
- Less: Degradation costs: -$12K
- Net profit: $727K

---

### Section 4: KPI Analysis

**Key Performance Indicators with targets**:

```
CURTAILMENT REDUCTION
Target: <10% | Achieved: 8.2% | Status: ✅ PASS
└─ Explanation: Battery charged during oversupply, discharged later

REVENUE OPTIMIZATION
Target: >$60/MWh | Achieved: $61/MWh | Status: ✅ PASS
└─ Explanation: Arbitrage profit from price swings

GRID COMPLIANCE
Target: 100% | Achieved: 100% (0/24 hours violated) | Status: ✅ PASS
└─ Explanation: Export limits respected at all times

BATTERY HEALTH
Target: Longevity >8 years | Cycles: 1.2 | Status: ✅ PASS
└─ Explanation: Sustainable cycling rate, degradation cost: $9.6K
```

---

### Section 5: Risk Analysis (if Stress Test enabled)

```
MONTE CARLO RESULTS (100 simulations)
════════════════════════════════════════

Revenue Distribution
  5th percentile:  $580K  (worst 5% of scenarios)
 25th percentile:  $650K
 50th percentile:  $720K  (median)
 75th percentile:  $800K
 95th percentile:  $850K  (best 5% of scenarios)

Value at Risk (95% confidence): $580K minimum expected

Curtailment Distribution
  Mean: 9.2%
  Std Dev: 2.1%
  Best case: 3.2%
  Worst case: 18.5%
  
Violation Probability: 2.3% (2% of scenarios exceeded grid limits)
```

**Interpretation**:
- Likely range: $650K - $800K
- Unlikely to earn less than $580K (5% tail)
- Unlikely to have grid violations (2.3% probability)

---

## 🔄 Scenario Comparison

**Location**: Dashboard → "Compare Scenarios"

### Compare Multiple Runs

```
Select Scenarios:
☑ Duck Curve (Jan 21, 2024)
☑ Duck Curve (Jan 22, 2024)
☑ Price Arbitrage (Jan 21, 2024)
☑ Grid Emergency (Jan 21, 2024)

Metrics to Compare:
☑ Revenue
☑ Curtailment
☑ Grid Violations
☑ Battery Cycles
☑ Profit/MWh

Visualization: [Tables] [Charts] [Statistical]
```

### Results

```
COMPARISON TABLE
┌──────────────────┬──────────┬──────────┬──────────┐
│ Scenario         │ Revenue  │ Curtal   │ Profit   │
├──────────────────┼──────────┼──────────┼──────────┤
│ Duck Curve 1/21  │ $727K    │ 8.2%     │ $61/MWh  │
│ Duck Curve 1/22  │ $698K    │ 9.1%     │ $58/MWh  │
│ Price Arb        │ $850K    │ 2.1%     │ $71/MWh  │
│ Grid Emerg       │ $580K    │ 15.2%    │ $48/MWh  │
└──────────────────┴──────────┴──────────┴──────────┘

INSIGHTS
────────
• Price Arbitrage most profitable (expected with price swings)
• Duck Curve scenarios consistent day-to-day
• Grid Emergency tests robustness (lower profit, acceptable)
```

---

## 📥 Importing Historical Data

**Location**: Top navigation → "Import Data"

### Upload CSV File

```
Format:
Hour, Generation_MW, GridCapacity_MW, Price_$/MWh, Actual_Generation_MW

Example:
0,100,300,50,98
1,120,300,45,122
2,150,300,40,148
...
```

**Features**:
- Validate format before import
- Check for data gaps
- Preview first 10 rows
- Create scenario from historical data
- Option: Compare optimization vs actual

---

## 📤 Exporting Results

**Location**: Results page → "Export" menu

### Options

```
Export As
├─ PDF Report
│  └─ Complete analysis with charts
├─ Excel Workbook
│  ├─ Summary sheet
│  ├─ Hourly decisions
│  ├─ KPI metrics
│  └─ Charts
├─ CSV (Hourly Data)
│  └─ For external analysis
└─ JSON
   └─ For API/database integration
```

**Example Excel export structure**:
```
Sheet 1: Executive Summary
Sheet 2: Hourly Dispatch (MILP)
Sheet 3: Hourly Dispatch (RL)
Sheet 4: KPIs
Sheet 5: Charts (embedded)
```

---

## ⚙️ Settings & Preferences

**Location**: Top navigation → "Settings" (gear icon)

### User Settings

```
Display Preferences
├─ Dark Mode: OFF / ON
├─ Number Format: $1,234.56 (US) / $1.234,56 (EU)
└─ Currency: USD / EUR / AUD

Units Preferences
├─ Power: MW / kW / GW
├─ Energy: MWh / kWh / GWh
├─ Price: $/MWh / €/MWh
└─ CO2: MT / tons / kg

Chart Preferences
├─ Color scheme: Default / Colorblind / High Contrast
├─ Default chart type: Area / Line / Stacked
└─ Auto-refresh dashboard: OFF / 1hr / 5min
```

### Default Optimization Settings

```
Default Strategies
├─ Always include Naive: ON
├─ Always include MILP: ON
├─ Always include RL: ON
├─ Always include Hybrid: ON

Default Battery
├─ Capacity MWh: 500
├─ Max Power MW: 150
├─ Initial SOC%: 50%

Default Analysis
├─ Include stress test: OFF
├─ Number of MC sims: 100
├─ Include sensitivity analysis: OFF
```

---

## 🆘 Help & Support

### Help Panel

**Location**: Top right → "?" icon

Contextual help for current page:
- Feature explanation
- Tips & tricks
- Common mistakes
- Links to full documentation

---

### API Documentation

**Location**: Top navigation → "API Docs"

Or visit directly: **http://localhost:8080/docs**

Interactive Swagger UI with:
- All endpoint documentation
- Try-it-out console
- Request/response examples
- Error codes & troubleshooting

---

## 📊 Example Workflows

### Workflow 1: Learning the Platform (15 minutes)

1. **Home Page** → See overview (2 min)
2. **Duck Curve Scenario** → Run demo with all strategies (8 min)
3. **Results Analysis** → Understand comparison (3 min)
4. **Read Features.md** → Deep dive into algorithms (2 min)

---

### Workflow 2: Production Deployment (1 hour)

1. **Custom Scenario** → Configure actual farm (10 min)
2. **Advanced Options** → Enable stress testing (5 min)
3. **Run Optimization** → Generate analysis (15 min)
4. **PDF Export** → Create stakeholder report (5 min)
5. **Set Up Monitoring** → Configure alerts (25 min)

---

### Workflow 3: Risk Assessment (30 minutes)

1. **Custom Scenario** → Define baseline (5 min)
2. **Advanced Options** → Enable all analysis (3 min)
3. **Run Full Analysis** → MILP + Stress Test (15 min)
4. **View Risk Analysis** → Review distributions (5 min)
5. **Export Results** → Excel for further analysis (2 min)

