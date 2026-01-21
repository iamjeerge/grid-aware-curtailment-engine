/**
 * TypeScript types for the Grid-Aware Curtailment Engine API.
 */

// =============================================================================
// Enums
// =============================================================================

export type ScenarioType = 'duck_curve' | 'high_volatility' | 'congested_grid' | 'custom';
export type StrategyType = 'naive' | 'milp' | 'rl' | 'hybrid';
export type OptimizationStatus = 'pending' | 'running' | 'completed' | 'failed';

// =============================================================================
// Request Types
// =============================================================================

export interface BatteryConfig {
  capacity_mwh: number;
  max_power_mw: number;
  charge_efficiency: number;
  discharge_efficiency: number;
  min_soc_fraction: number;
  max_soc_fraction: number;
  degradation_cost_per_mwh: number;
}

export interface ScenarioConfig {
  scenario_type: ScenarioType;
  horizon_hours: number;
  peak_generation_mw: number;
  grid_limit_mw: number;
  seed: number | null;
}

export interface OptimizationRequest {
  name: string;
  scenario: ScenarioConfig;
  battery: BatteryConfig;
  strategies: StrategyType[];
}

// =============================================================================
// Response Types
// =============================================================================

export interface CurtailmentMetrics {
  total_generation_mwh: number;
  total_curtailed_mwh: number;
  total_sold_mwh: number;
  total_stored_mwh: number;
  curtailment_rate: number;
}

export interface RevenueMetrics {
  gross_revenue: number;
  degradation_cost: number;
  net_profit: number;
  average_price: number;
}

export interface BatteryMetrics {
  total_charged_mwh: number;
  total_discharged_mwh: number;
  cycles: number;
  utilization_rate: number;
  avg_soc: number;
}

export interface GridComplianceMetrics {
  violation_count: number;
  total_violation_mwh: number;
  max_violation_mw: number;
  compliance_rate: number;
}

export interface PerformanceSummary {
  strategy_name: string;
  curtailment: CurtailmentMetrics;
  revenue: RevenueMetrics;
  battery: BatteryMetrics;
  grid_compliance: GridComplianceMetrics;
}

export interface Decision {
  timestep: number;
  timestamp: string;
  generation_mw: number;
  energy_sold_mw: number;
  energy_stored_mw: number;
  energy_curtailed_mw: number;
  battery_discharge_mw: number;
  soc_mwh: number;
  price: number;
  grid_limit_mw: number;
}

export interface StrategyResult {
  strategy: StrategyType;
  summary: PerformanceSummary;
  decisions: Decision[];
  solve_time_seconds: number | null;
}

export interface Comparison {
  best_strategy: string;
  curtailment_reduction_pct: number;
  revenue_uplift_pct: number;
  revenue_uplift_dollars: number;
}

export interface OptimizationResult {
  id: string;
  name: string;
  status: OptimizationStatus;
  created_at: string;
  completed_at: string | null;
  scenario: ScenarioConfig;
  battery: BatteryConfig;
  results: Record<string, StrategyResult>;
  comparison: Comparison | null;
  error_message: string | null;
}

export interface OptimizationListResponse {
  items: OptimizationResult[];
  total: number;
  page: number;
  page_size: number;
}

// =============================================================================
// Demo Types
// =============================================================================

export interface DemoScenario {
  id: string;
  name: string;
  description: string;
  scenario_type: ScenarioType;
  config: ScenarioConfig;
  battery: BatteryConfig;
}

// =============================================================================
// Health & System Types
// =============================================================================

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface SystemInfo {
  version: string;
  python_version: string;
  available_strategies: StrategyType[];
  available_scenarios: ScenarioType[];
  solver_available: boolean;
}
