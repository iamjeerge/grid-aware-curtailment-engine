/**
 * Main application component for the Grid-Aware Curtailment Engine.
 */

import { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';
import Header from './components/Header';
import ScenarioSelector from './components/ScenarioSelector';
import Dashboard from './components/Dashboard';
import {
  listDemoScenarios,
  runDemoScenario,
  createOptimization,
  getHealth,
} from './api';
import type { OptimizationResult, BatteryConfig, ScenarioConfig, StrategyType } from './types';
import { ArrowLeft } from 'lucide-react';

const queryClient = new QueryClient();

function AppContent() {
  const [currentResult, setCurrentResult] = useState<OptimizationResult | null>(null);
  const [isApiConnected, setIsApiConnected] = useState(false);

  // Check API health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await getHealth();
        setIsApiConnected(true);
      } catch {
        setIsApiConnected(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch demo scenarios
  const { data: scenarios = [], isLoading: scenariosLoading } = useQuery({
    queryKey: ['demoScenarios'],
    queryFn: listDemoScenarios,
    enabled: isApiConnected,
  });

  // Run demo mutation
  const runDemoMutation = useMutation({
    mutationFn: runDemoScenario,
    onSuccess: (result) => {
      setCurrentResult(result);
    },
  });

  // Run custom optimization mutation
  const runCustomMutation = useMutation({
    mutationFn: ({
      name,
      scenario,
      battery,
      strategies,
    }: {
      name: string;
      scenario: ScenarioConfig;
      battery: BatteryConfig;
      strategies: StrategyType[];
    }) =>
      createOptimization({
        name,
        scenario,
        battery,
        strategies,
      }),
    onSuccess: (result) => {
      setCurrentResult(result);
    },
  });

  const isLoading = runDemoMutation.isPending || runCustomMutation.isPending;

  const handleRunDemo = (scenarioId: string) => {
    runDemoMutation.mutate(scenarioId);
  };

  const handleRunCustom = (
    name: string,
    scenario: ScenarioConfig,
    battery: BatteryConfig,
    strategies: StrategyType[]
  ) => {
    runCustomMutation.mutate({ name, scenario, battery, strategies });
  };

  const handleBack = () => {
    setCurrentResult(null);
  };

  return (
    <div className="min-h-screen bg-grid-gradient">
      <Header isApiConnected={isApiConnected} />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {!isApiConnected ? (
          <div className="bg-yellow-900/20 border-l-4 border-yellow-600 p-4 rounded">
            <div className="flex">
              <div className="ml-3">
                <p className="text-sm text-yellow-400">
                  <strong>API not connected.</strong> Please start the backend server:
                </p>
                <pre className="mt-2 text-xs bg-yellow-900/40 text-yellow-300 p-2 rounded">
                  cd grid-aware-curtailment-engine{'\n'}
                  poetry run uvicorn src.api.main:app --reload
                </pre>
              </div>
            </div>
          </div>
        ) : currentResult ? (
          <div>
            <button
              onClick={handleBack}
              className="mb-6 flex items-center text-energy-orange hover:text-energy-orangeSoft transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-1" />
              Back to Scenarios
            </button>
            <Dashboard result={currentResult} />
          </div>
        ) : scenariosLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-energy-orange" />
          </div>
        ) : (
          <ScenarioSelector
            scenarios={scenarios}
            onRunDemo={handleRunDemo}
            onRunCustom={handleRunCustom}
            isLoading={isLoading}
          />
        )}

        {/* Error display */}
        {(runDemoMutation.error || runCustomMutation.error) && (
          <div className="mt-4 bg-danger/10 border-l-4 border-danger p-4 rounded">
            <p className="text-sm text-danger">
              Error: {(runDemoMutation.error || runCustomMutation.error)?.message || 'Unknown error'}
            </p>
          </div>
        )}
      </main>

      <footer className="bg-grid-dark border-t border-grid-border text-grid-muted py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          Grid-Aware Curtailment Engine &copy; 2025 • Production-grade renewable energy optimization
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
