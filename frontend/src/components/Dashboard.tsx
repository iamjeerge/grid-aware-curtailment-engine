/**
 * Dashboard component showing optimization results.
 */

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import type { OptimizationResult, Decision } from '../types';
import { Battery, Zap, TrendingUp, AlertTriangle, Clock, DollarSign } from 'lucide-react';

interface DashboardProps {
  result: OptimizationResult;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color,
}: {
  title: string;
  value: string;
  subtitle?: string;
  icon: typeof Battery;
  color: string;
}) {
  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
      <div className="flex items-center">
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-grid-muted">{title}</p>
          <p className="text-2xl font-semibold text-grid-text">{value}</p>
          {subtitle && <p className="text-sm text-grid-muted">{subtitle}</p>}
        </div>
      </div>
    </div>
  );
}

function DispatchChart({ decisions }: { decisions: Decision[] }) {
  const chartData = decisions.map((d, i) => ({
    hour: i,
    generation: d.generation_mw,
    sold: d.energy_sold_mw,
    stored: d.energy_stored_mw,
    curtailed: d.energy_curtailed_mw,
    discharged: d.battery_discharge_mw,
    price: d.price,
    gridLimit: d.grid_limit_mw,
  }));

  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
      <h3 className="text-lg font-semibold text-grid-text mb-4">Dispatch Timeline</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="hour" label={{ value: 'Hour', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'MW', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Area
            type="monotone"
            dataKey="sold"
            stackId="1"
            stroke="#10b981"
            fill="#10b981"
            name="Sold to Grid"
          />
          <Area
            type="monotone"
            dataKey="stored"
            stackId="1"
            stroke="#3b82f6"
            fill="#3b82f6"
            name="Stored"
          />
          <Area
            type="monotone"
            dataKey="curtailed"
            stackId="1"
            stroke="#ef4444"
            fill="#ef4444"
            name="Curtailed"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function PriceChart({ decisions }: { decisions: Decision[] }) {
  const chartData = decisions.map((d, i) => ({
    hour: i,
    price: d.price,
    gridLimit: d.grid_limit_mw,
  }));

  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
      <h3 className="text-lg font-semibold text-grid-text mb-4">Market Prices</h3>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="hour" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Area
            type="monotone"
            dataKey="price"
            stroke="#8b5cf6"
            fill="#8b5cf6"
            fillOpacity={0.3}
            name="Price ($/MWh)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function ComparisonChart({ result }: { result: OptimizationResult }) {
  const strategies = Object.keys(result.results);
  const chartData = strategies.map((strategy) => {
    const r = result.results[strategy];
    return {
      name: strategy.toUpperCase(),
      curtailmentRate: r.summary.curtailment.curtailment_rate * 100,
      netProfit: r.summary.revenue.net_profit / 1000,
      violations: r.summary.grid_compliance.violation_count,
    };
  });

  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
      <h3 className="text-lg font-semibold text-grid-text mb-4">Strategy Comparison</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis yAxisId="left" orientation="left" stroke="#10b981" />
          <YAxis yAxisId="right" orientation="right" stroke="#3b82f6" />
          <Tooltip />
          <Legend />
          <Bar
            yAxisId="left"
            dataKey="netProfit"
            fill="#10b981"
            name="Net Profit ($k)"
          />
          <Bar
            yAxisId="right"
            dataKey="curtailmentRate"
            fill="#ef4444"
            name="Curtailment Rate (%)"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Dashboard({ result }: DashboardProps) {
  const milpResult = result.results['milp'];
  const naiveResult = result.results['naive'];
  const comparison = result.comparison;

  // Use MILP results for display, or naive if MILP not available
  const mainResult = milpResult || naiveResult;
  if (!mainResult) {
    return <div>No results available</div>;
  }

  const summary = mainResult.summary;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
        <h2 className="text-2xl font-bold text-grid-text">{result.name}</h2>
        <p className="text-grid-muted mt-1">
          {result.scenario.horizon_hours} hour horizon • {result.scenario.scenario_type.replace('_', ' ')}
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Net Profit"
          value={formatCurrency(summary.revenue.net_profit)}
          subtitle={comparison ? `${formatPercent(comparison.revenue_uplift_pct)} vs baseline` : undefined}
          icon={DollarSign}
          color="bg-green-500"
        />
        <MetricCard
          title="Curtailment Rate"
          value={formatPercent(summary.curtailment.curtailment_rate)}
          subtitle={comparison ? `${formatPercent(comparison.curtailment_reduction_pct)} reduction` : undefined}
          icon={Zap}
          color="bg-yellow-500"
        />
        <MetricCard
          title="Battery Cycles"
          value={summary.battery.cycles.toFixed(1)}
          subtitle={`${formatPercent(summary.battery.utilization_rate)} utilization`}
          icon={Battery}
          color="bg-blue-500"
        />
        <MetricCard
          title="Grid Violations"
          value={summary.grid_compliance.violation_count.toString()}
          subtitle={`${formatPercent(summary.grid_compliance.compliance_rate)} compliance`}
          icon={summary.grid_compliance.violation_count === 0 ? TrendingUp : AlertTriangle}
          color={summary.grid_compliance.violation_count === 0 ? 'bg-green-500' : 'bg-red-500'}
        />
      </div>

      {/* Improvement Summary */}
      {comparison && (
        <div className="bg-grid-panel border border-energy-orange/30 rounded-lg shadow-glow p-6">
          <h3 className="text-lg font-semibold text-grid-text mb-3">
            🎯 Optimization Impact
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-grid-muted">Revenue Uplift</p>
              <p className="text-2xl font-bold text-success">
                +{formatCurrency(comparison.revenue_uplift_dollars)}
              </p>
            </div>
            <div>
              <p className="text-sm text-grid-muted">Curtailment Reduced</p>
              <p className="text-2xl font-bold text-energy-orange">
                {formatPercent(comparison.curtailment_reduction_pct)}
              </p>
            </div>
            <div>
              <p className="text-sm text-grid-muted">Best Strategy</p>
              <p className="text-2xl font-bold text-energy-orangeSoft">
                {comparison.best_strategy.toUpperCase()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DispatchChart decisions={mainResult.decisions} />
        <PriceChart decisions={mainResult.decisions} />
      </div>

      {/* Strategy Comparison Chart */}
      {Object.keys(result.results).length > 1 && (
        <ComparisonChart result={result} />
      )}

      {/* Solve Time */}
      {mainResult.solve_time_seconds && (
        <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-4 flex items-center">
          <Clock className="w-5 h-5 text-grid-muted mr-2" />
          <span className="text-sm text-grid-muted">
            Optimization completed in {mainResult.solve_time_seconds.toFixed(2)} seconds
          </span>
        </div>
      )}
    </div>
  );
}
