/**
 * Scenario selector component for choosing and configuring scenarios.
 */

import { useState } from 'react';
import type { DemoScenario, BatteryConfig, ScenarioConfig, StrategyType } from '../types';
import { Play, Settings, Zap, Battery as BatteryIcon, Sun, Cloud } from 'lucide-react';

interface ScenarioSelectorProps {
  scenarios: DemoScenario[];
  onRunDemo: (scenarioId: string) => void;
  onRunCustom: (
    name: string,
    scenario: ScenarioConfig,
    battery: BatteryConfig,
    strategies: StrategyType[]
  ) => void;
  isLoading: boolean;
}

function ScenarioCard({
  scenario,
  onRun,
  isLoading,
}: {
  scenario: DemoScenario;
  onRun: () => void;
  isLoading: boolean;
}) {
  const icons: Record<string, typeof Sun> = {
    duck_curve: Sun,
    high_volatility: Zap,
    congested_grid: Cloud,
  };
  const Icon = icons[scenario.scenario_type] || Settings;

  const colors: Record<string, string> = {
    duck_curve: 'from-yellow-400 to-orange-500',
    high_volatility: 'from-purple-400 to-pink-500',
    congested_grid: 'from-blue-400 to-indigo-500',
  };
  const gradient = colors[scenario.scenario_type] || 'from-gray-400 to-gray-500';

  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel overflow-hidden">
      <div className={`bg-gradient-to-r ${gradient} p-4`}>
        <Icon className="w-10 h-10 text-white" />
      </div>
      <div className="p-6">
        <h3 className="text-lg font-semibold text-grid-text">{scenario.name}</h3>
        <p className="text-sm text-grid-muted mt-2">{scenario.description}</p>
        
        <div className="mt-4 space-y-2 text-sm text-grid-muted">
          <div className="flex justify-between">
            <span>Horizon</span>
            <span className="font-medium">{scenario.config.horizon_hours} hours</span>
          </div>
          <div className="flex justify-between">
            <span>Peak Generation</span>
            <span className="font-medium">{scenario.config.peak_generation_mw} MW</span>
          </div>
          <div className="flex justify-between">
            <span>Grid Limit</span>
            <span className="font-medium">{scenario.config.grid_limit_mw} MW</span>
          </div>
          <div className="flex justify-between">
            <span>Battery</span>
            <span className="font-medium">
              {scenario.battery.capacity_mwh} MWh / {scenario.battery.max_power_mw} MW
            </span>
          </div>
        </div>

        <button
          onClick={onRun}
          disabled={isLoading}
          className={`mt-6 w-full py-2 px-4 rounded-lg flex items-center justify-center
            ${isLoading 
              ? 'bg-gradient-to-r from-gray-500 to-gray-600 cursor-not-allowed opacity-60' 
              : `bg-gradient-to-r ${gradient} hover:opacity-90`
            } text-white font-medium transition-opacity`}
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Run Demo
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function CustomScenarioForm({
  onSubmit,
  isLoading,
}: {
  onSubmit: (
    name: string,
    scenario: ScenarioConfig,
    battery: BatteryConfig,
    strategies: StrategyType[]
  ) => void;
  isLoading: boolean;
}) {
  const [name, setName] = useState('Custom Optimization');
  const [horizonHours, setHorizonHours] = useState(24);
  const [peakGen, setPeakGen] = useState(600);
  const [gridLimit, setGridLimit] = useState(300);
  const [batteryMwh, setBatteryMwh] = useState(500);
  const [batteryMw, setBatteryMw] = useState(150);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const scenario: ScenarioConfig = {
      scenario_type: 'custom',
      horizon_hours: horizonHours,
      peak_generation_mw: peakGen,
      grid_limit_mw: gridLimit,
      seed: 42,
    };

    const battery: BatteryConfig = {
      capacity_mwh: batteryMwh,
      max_power_mw: batteryMw,
      charge_efficiency: 0.95,
      discharge_efficiency: 0.95,
      min_soc_fraction: 0.1,
      max_soc_fraction: 0.9,
      degradation_cost_per_mwh: 8,
    };

    onSubmit(name, scenario, battery, ['naive', 'milp']);
  };

  return (
    <div className="bg-grid-panel border border-grid-border rounded-lg shadow-panel p-6">
      <div className="flex items-center mb-4">
        <Settings className="w-6 h-6 text-energy-orange mr-2" />
        <h3 className="text-lg font-semibold text-grid-text">Custom Scenario</h3>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-grid-text">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
              focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-grid-text">
              Horizon (hours)
            </label>
            <input
              type="number"
              min={1}
              max={168}
              value={horizonHours}
              onChange={(e) => setHorizonHours(Number(e.target.value))}
              className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
                focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-grid-text">
              Peak Generation (MW)
            </label>
            <input
              type="number"
              min={0}
              max={10000}
              value={peakGen}
              onChange={(e) => setPeakGen(Number(e.target.value))}
              className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
                focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-grid-text">
              Grid Limit (MW)
            </label>
            <input
              type="number"
              min={0}
              max={10000}
              value={gridLimit}
              onChange={(e) => setGridLimit(Number(e.target.value))}
              className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
                focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-grid-text">
              Battery Capacity (MWh)
            </label>
            <input
              type="number"
              min={0}
              max={10000}
              value={batteryMwh}
              onChange={(e) => setBatteryMwh(Number(e.target.value))}
              className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
                focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-grid-text">
            Battery Power (MW)
          </label>
          <input
            type="number"
            min={0}
            max={5000}
            value={batteryMw}
            onChange={(e) => setBatteryMw(Number(e.target.value))}
            className="mt-1 block w-full bg-grid-dark border border-grid-border rounded-md text-grid-text
              focus:border-energy-orange focus:ring-1 focus:ring-energy-orange focus:outline-none px-3 py-2"
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className={`w-full py-2 px-4 rounded-lg flex items-center justify-center
            ${isLoading
              ? 'bg-energy-orange/40 cursor-not-allowed opacity-60'
              : 'bg-energy-orange hover:bg-energy-orangeSoft'
            } text-white font-medium transition-colors`}
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
              Running...
            </>
          ) : (
            <>
              <BatteryIcon className="w-5 h-5 mr-2" />
              Run Optimization
            </>
          )}
        </button>
      </form>
    </div>
  );
}

export default function ScenarioSelector({
  scenarios,
  onRunDemo,
  onRunCustom,
  isLoading,
}: ScenarioSelectorProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-grid-text mb-4">
          Pre-configured Scenarios
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {scenarios.map((scenario) => (
            <ScenarioCard
              key={scenario.id}
              scenario={scenario}
              onRun={() => onRunDemo(scenario.id)}
              isLoading={isLoading}
            />
          ))}
        </div>
      </div>

      <div className="border-t border-grid-border pt-6">
        <h2 className="text-xl font-semibold text-grid-text mb-4">
          Custom Configuration
        </h2>
        <div className="max-w-md">
          <CustomScenarioForm onSubmit={onRunCustom} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
}
