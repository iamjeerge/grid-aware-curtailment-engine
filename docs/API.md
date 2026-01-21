# API Reference

Complete REST API documentation for the Grid-Aware Curtailment Engine.

---

## 📡 Base URL

```
http://localhost:8080/api/v1
```

## 🔐 Authentication

Currently no authentication required. Production deployments should add API key authentication.

---

## Core Endpoints

### Optimization Management

#### Create Optimization

**Request**:
```http
POST /optimizations/
Content-Type: application/json

{
  "scenario": {
    "name": "duck_curve",
    "description": "Test duck curve scenario",
    "generation_forecast_mw": [100, 120, 150, 300, 500, 600, 550, 400, 200, 100, 80, 60],
    "grid_capacity_mw": 300,
    "prices_per_mwh": [50, 45, 40, -25, -20, -25, 30, 80, 140, 135, 130, 100],
    "battery_capacity_mwh": 500,
    "battery_max_power_mw": 150,
    "battery_initial_soc_pct": 50
  },
  "strategies": ["naive", "milp", "rl", "hybrid"],
  "options": {
    "include_stress_test": true,
    "num_monte_carlo_simulations": 100,
    "include_sensitivity_analysis": false,
    "include_assumption_validation": true,
    "optimization_timeout_seconds": 60
  }
}
```

**Response** (202 Accepted):
```json
{
  "optimization_id": "opt_abc123def456",
  "status": "queued",
  "created_at": "2024-01-21T10:30:00Z",
  "estimated_completion_time": "2024-01-21T10:32:30Z"
}
```

**Status codes**:
- `202 Accepted` - Optimization queued
- `400 Bad Request` - Invalid parameters
- `422 Unprocessable Entity` - Validation failed

---

#### Get Optimization Status

**Request**:
```http
GET /optimizations/{optimization_id}
```

**Response** (200 OK):
```json
{
  "optimization_id": "opt_abc123def456",
  "status": "running",
  "progress_percent": 45,
  "current_step": "Running RL agent training...",
  "created_at": "2024-01-21T10:30:00Z",
  "started_at": "2024-01-21T10:30:15Z",
  "estimated_completion_time": "2024-01-21T10:32:30Z"
}
```

**Status values**:
- `queued` - Waiting to start
- `running` - Actively optimizing
- `completed` - Finished successfully
- `failed` - Error occurred
- `cancelled` - User cancelled

---

#### Get Optimization Results

**Request**:
```http
GET /optimizations/{optimization_id}/results
```

**Response** (200 OK):
```json
{
  "optimization_id": "opt_abc123def456",
  "status": "completed",
  "scenario": {
    "name": "duck_curve",
    "description": "...",
    "generation_forecast_mw": [...],
    "grid_capacity_mw": 300,
    "prices_per_mwh": [...]
  },
  "results": {
    "naive": {
      "strategy": "naive",
      "summary": {
        "revenue": {
          "gross_revenue": 420000,
          "degradation_cost": 6400,
          "net_profit": 413600
        },
        "curtailment": {
          "total_generation_mwh": 2400,
          "total_curtailed_mwh": 770,
          "curtailment_rate_pct": 32.1,
          "total_sold_mwh": 1200,
          "total_stored_mwh": 430
        },
        "grid_compliance": {
          "violation_count": 5,
          "total_violation_mwh": 85,
          "max_violation_mw": 42,
          "hours_compliant": 19
        },
        "battery": {
          "cycles": 0.8,
          "average_soc_pct": 45,
          "min_soc_pct": 15,
          "max_soc_pct": 88
        }
      },
      "hourly_decisions": [
        {
          "hour": 0,
          "generation_mw": 100,
          "sell_mw": 100,
          "store_mw": 0,
          "curtail_mw": 0,
          "soc_pct": 50,
          "revenue": 5000,
          "grid_violation": false
        },
        ...
      ]
    },
    "milp": {
      "strategy": "milp",
      "summary": {...},
      "hourly_decisions": [...]
    },
    "rl": {
      "strategy": "rl",
      "summary": {...},
      "hourly_decisions": [...]
    },
    "hybrid": {
      "strategy": "hybrid",
      "summary": {...},
      "hourly_decisions": [...]
    }
  },
  "analysis": {
    "stress_test": {
      "num_simulations": 100,
      "revenue_p5": 580000,
      "revenue_p25": 650000,
      "revenue_p50": 720000,
      "revenue_p75": 800000,
      "revenue_p95": 850000,
      "curtailment_mean_pct": 9.2,
      "curtailment_std_pct": 2.1,
      "violation_probability_pct": 2.3
    },
    "sensitivity_analysis": {
      "battery_efficiency_impact": [...],
      "degradation_cost_impact": [...],
      "grid_capacity_impact": [...]
    },
    "assumptions": {
      "validated": true,
      "checks": [...]
    }
  },
  "completed_at": "2024-01-21T10:31:45Z",
  "total_computation_time_seconds": 105
}
```

---

#### List Optimizations

**Request**:
```http
GET /optimizations/?limit=10&offset=0&sort=created_at&order=desc
```

**Query Parameters**:
- `limit` (int, 1-100, default: 10) - Number of results
- `offset` (int, default: 0) - Pagination offset
- `sort` (string) - Sort field: `created_at`, `status`, `revenue`
- `order` (string) - Sort order: `asc`, `desc`
- `strategy` (string) - Filter by strategy: `naive`, `milp`, `rl`, `hybrid`
- `status` (string) - Filter by status: `queued`, `running`, `completed`, `failed`

**Response** (200 OK):
```json
{
  "items": [
    {
      "optimization_id": "opt_abc123def456",
      "scenario_name": "duck_curve",
      "status": "completed",
      "created_at": "2024-01-21T10:30:00Z",
      "completed_at": "2024-01-21T10:31:45Z"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

---

#### Delete Optimization

**Request**:
```http
DELETE /optimizations/{optimization_id}
```

**Response** (204 No Content)

---

### Dashboard & Analytics

#### Get Industry Dashboard

**Request**:
```http
GET /dashboard/industry
```

**Response** (200 OK):
```json
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
  "summary": "Across 42 optimizations: ..."
}
```

---

#### Get KPI Metrics

**Request**:
```http
GET /optimizations/{optimization_id}/metrics/kpis
```

**Response** (200 OK):
```json
{
  "optimization_id": "opt_abc123def456",
  "kpis": {
    "curtailment_rate_pct": 8.2,
    "curtailment_rate_target_pct": 10.0,
    "curtailment_rate_status": "PASS",
    
    "profit_per_mwh": 61.0,
    "profit_per_mwh_target": 60.0,
    "profit_per_mwh_status": "PASS",
    
    "battery_utilization_pct": 82,
    "battery_utilization_target_pct": "70-90",
    "battery_utilization_status": "PASS",
    
    "grid_compliance_pct": 100,
    "grid_compliance_target_pct": 100,
    "grid_compliance_status": "PASS",
    
    "roi_percentage": 87.5,
    "roi_target_percentage": 40,
    "roi_status": "PASS"
  },
  "computed_at": "2024-01-21T10:31:45Z"
}
```

---

### Demo Scenarios

#### List Pre-configured Scenarios

**Request**:
```http
GET /demos/scenarios
```

**Response** (200 OK):
```json
{
  "scenarios": [
    {
      "id": "duck_curve",
      "name": "Duck Curve Trap",
      "description": "Solar peak vs grid limit with price arbitrage",
      "parameters": {
        "peak_generation_mw": 600,
        "grid_limit_mw": 300,
        "battery_capacity_mwh": 500
      },
      "expected_results": {
        "naive_curtailment_pct": 32.1,
        "optimized_curtailment_pct": 8.2,
        "revenue_uplift_pct": 73.1
      }
    },
    {
      "id": "price_arbitrage",
      "name": "Price Arbitrage",
      "description": "Extreme price swings for battery value"
    },
    {
      "id": "grid_emergency",
      "name": "Grid Emergency",
      "description": "Sudden capacity reduction test"
    }
  ]
}
```

---

#### Run Demo Scenario

**Request**:
```http
POST /demos/run/{scenario_id}
```

**Response** (202 Accepted):
```json
{
  "optimization_id": "opt_abc123def456",
  "status": "queued",
  "scenario_id": "duck_curve"
}
```

---

### System

#### Health Check

**Request**:
```http
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-21T10:30:00Z",
  "database": "connected",
  "solver": "glpk available",
  "cache": "redis connected"
}
```

---

#### Get System Info

**Request**:
```http
GET /info
```

**Response** (200 OK):
```json
{
  "name": "Grid-Aware Curtailment Engine",
  "version": "2.1.0",
  "api_version": "v1",
  "python_version": "3.11.0",
  "components": {
    "optimization": "Pyomo 6.6.1",
    "solver": "GLPK 5.0",
    "rl": "Gymnasium 0.29.0",
    "api": "FastAPI 0.104.1"
  }
}
```

---

## Error Handling

All errors return JSON with consistent format:

```json
{
  "error": "Invalid request",
  "detail": "Scenario name is required",
  "status_code": 400,
  "timestamp": "2024-01-21T10:30:00Z",
  "request_id": "req_xyz789"
}
```

### Common Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 400 | Bad Request | Missing required parameter |
| 404 | Not Found | Optimization ID doesn't exist |
| 422 | Validation Error | Invalid grid capacity value |
| 500 | Internal Server Error | Solver crashed |
| 503 | Service Unavailable | Solver not installed |

---

## Rate Limiting

Currently no rate limiting. Production deployments should implement:
- Max 10 optimizations per hour per IP
- Max 100 total concurrent optimizations

---

## WebSocket Streaming (Optional)

Connect to stream optimization progress:

```javascript
const ws = new WebSocket(
  'ws://localhost:8080/api/v1/optimizations/opt_abc123/stream'
);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress_percent}%`);
  console.log(`Current step: ${data.current_step}`);
};
```

**Message format**:
```json
{
  "type": "progress",
  "progress_percent": 45,
  "current_step": "Running RL agent training...",
  "timestamp": "2024-01-21T10:31:00Z"
}
```

When completed:
```json
{
  "type": "completed",
  "optimization_id": "opt_abc123def456",
  "timestamp": "2024-01-21T10:31:45Z"
}
```

---

## Integration Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"

# Create optimization
response = requests.post(
    f"{BASE_URL}/optimizations/",
    json={
        "scenario": {
            "name": "custom",
            "generation_forecast_mw": [100, 200, ...],
            "grid_capacity_mw": 300,
            "prices_per_mwh": [50, 60, ...],
            "battery_capacity_mwh": 500,
            "battery_max_power_mw": 150,
            "battery_initial_soc_pct": 50
        },
        "strategies": ["milp", "hybrid"]
    }
)

opt_id = response.json()["optimization_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"{BASE_URL}/optimizations/{opt_id}")
    if status.json()["status"] == "completed":
        break
    time.sleep(2)

# Get results
results = requests.get(f"{BASE_URL}/optimizations/{opt_id}/results").json()
print(f"Revenue: ${results['results']['milp']['summary']['revenue']['net_profit']}")
```

---

### JavaScript/Node.js Client

```javascript
const BASE_URL = "http://localhost:8080/api/v1";

// Create optimization
const response = await fetch(`${BASE_URL}/optimizations/`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    scenario: {
      name: "duck_curve",
      generation_forecast_mw: [100, 200, ...],
      grid_capacity_mw: 300,
      prices_per_mwh: [50, 60, ...],
      battery_capacity_mwh: 500,
      battery_max_power_mw: 150,
      battery_initial_soc_pct: 50
    },
    strategies: ["milp", "hybrid"]
  })
});

const { optimization_id } = await response.json();

// Poll for completion
const pollInterval = setInterval(async () => {
  const status = await fetch(
    `${BASE_URL}/optimizations/${optimization_id}`
  ).then(r => r.json());
  
  if (status.status === "completed") {
    clearInterval(pollInterval);
    
    // Get results
    const results = await fetch(
      `${BASE_URL}/optimizations/${optimization_id}/results`
    ).then(r => r.json());
    
    console.log(
      `Revenue: $${results.results.milp.summary.revenue.net_profit}`
    );
  }
}, 2000);
```

---

### cURL Examples

**Create optimization**:
```bash
curl -X POST http://localhost:8080/api/v1/optimizations/ \
  -H "Content-Type: application/json" \
  -d @scenario.json
```

**Get results**:
```bash
curl http://localhost:8080/api/v1/optimizations/opt_abc123def456/results
```

**Get industry dashboard**:
```bash
curl http://localhost:8080/api/v1/dashboard/industry
```

---

## Pagination

List endpoints support cursor-based pagination:

```
GET /optimizations/?limit=10&offset=20
```

Returns:
```json
{
  "items": [...],
  "total": 42,
  "limit": 10,
  "offset": 20,
  "has_next": true,
  "next_offset": 30
}
```

---

## Filtering

List endpoints support filtering:

```
GET /optimizations/?strategy=milp&status=completed
```

---

## Sorting

```
GET /optimizations/?sort=created_at&order=desc
```

Sortable fields: `created_at`, `status`, `revenue`

