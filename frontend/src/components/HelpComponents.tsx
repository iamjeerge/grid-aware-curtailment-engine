import React, { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';

interface HelpTooltipProps {
  title: string;
  description: string;
  learnMore?: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  children?: React.ReactNode;
}

/**
 * HelpTooltip Component
 * 
 * Displays helpful information for features, metrics, and parameters.
 * 
 * Usage:
 * <HelpTooltip
 *   title="Curtailment Rate"
 *   description="Percentage of total solar generation that was curtailed (wasted) due to grid limits."
 *   learnMore="Learn more in our documentation"
 * />
 */
export const HelpTooltip: React.FC<HelpTooltipProps> = ({
  title,
  description,
  learnMore,
  position = 'top',
  children,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="text-blue-500 hover:text-blue-700 ml-1"
        aria-label={`Help: ${title}`}
      >
        <HelpCircle size={16} />
      </button>

      {isOpen && (
        <div
          className={`absolute z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-4 w-64 ${
            position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
          }`}
        >
          <div className="flex justify-between items-start mb-2">
            <h4 className="font-bold text-gray-900">{title}</h4>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X size={16} />
            </button>
          </div>
          <p className="text-sm text-gray-600 mb-3">{description}</p>
          {learnMore && (
            <a href="#" className="text-blue-500 text-sm hover:underline">
              {learnMore} →
            </a>
          )}
        </div>
      )}

      {children}
    </div>
  );
};

interface InfoBoxProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  variant?: 'info' | 'success' | 'warning' | 'error';
}

/**
 * InfoBox Component
 * 
 * Displays informational messages with optional icons and color variants.
 * 
 * Usage:
 * <InfoBox
 *   title="Duck Curve Scenario"
 *   description="Solar generation peaks at noon, grid capacity is limited to 300 MW. Battery can store excess energy and discharge during evening price spike."
 *   variant="info"
 * />
 */
export const InfoBox: React.FC<InfoBoxProps> = ({
  title,
  description,
  icon,
  variant = 'info',
}) => {
  const colorMap = {
    info: 'bg-blue-50 border-blue-200 text-blue-900',
    success: 'bg-green-50 border-green-200 text-green-900',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-900',
    error: 'bg-red-50 border-red-200 text-red-900',
  };

  return (
    <div className={`border rounded-lg p-4 ${colorMap[variant]}`}>
      <div className="flex gap-3">
        {icon && <div className="flex-shrink-0">{icon}</div>}
        <div>
          <h3 className="font-bold">{title}</h3>
          <p className="text-sm mt-1">{description}</p>
        </div>
      </div>
    </div>
  );
};

interface MetricExplanationProps {
  metricName: string;
  value: number | string;
  unit?: string;
  target?: number | string;
  explanation: string;
  formula?: string;
  interpretation?: string;
}

/**
 * MetricExplanation Component
 * 
 * Displays a metric with detailed explanation, formula, and interpretation.
 * 
 * Usage:
 * <MetricExplanation
 *   metricName="Revenue Uplift"
 *   value={73.1}
 *   unit="%"
 *   target=">50%"
 *   explanation="How much better MILP performs vs naive baseline"
 *   formula="(MILP Revenue - Naive Revenue) / Naive Revenue"
 *   interpretation="73% means MILP earns 73% more than naive strategy"
 * />
 */
export const MetricExplanation: React.FC<MetricExplanationProps> = ({
  metricName,
  value,
  unit,
  target,
  explanation,
  formula,
  interpretation,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div
        className="cursor-pointer flex justify-between items-start"
        onClick={() => setShowDetails(!showDetails)}
      >
        <div>
          <h4 className="font-bold text-gray-900">{metricName}</h4>
          <p className="text-2xl font-bold text-blue-600 mt-1">
            {value}{unit}
          </p>
        </div>
        {target && (
          <div className="text-right">
            <p className="text-xs text-gray-500">Target</p>
            <p className="text-sm font-semibold text-gray-900">{target}</p>
          </div>
        )}
      </div>

      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
          <div>
            <p className="text-sm text-gray-600">{explanation}</p>
          </div>
          {formula && (
            <div>
              <p className="text-xs font-semibold text-gray-500">FORMULA</p>
              <p className="text-sm font-mono bg-gray-50 p-2 rounded mt-1">
                {formula}
              </p>
            </div>
          )}
          {interpretation && (
            <div>
              <p className="text-xs font-semibold text-gray-500">WHAT IT MEANS</p>
              <p className="text-sm text-gray-600 mt-1">{interpretation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface FeatureCardProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  learnMoreUrl?: string;
}

/**
 * FeatureCard Component
 * 
 * Displays a feature or capability with icon and description.
 * 
 * Usage:
 * <FeatureCard
 *   title="MILP Optimizer"
 *   description="Mathematically optimal dispatch decisions using mixed-integer linear programming"
 *   learnMoreUrl="/docs/features#milp"
 * />
 */
export const FeatureCard: React.FC<FeatureCardProps> = ({
  title,
  description,
  icon,
  learnMoreUrl,
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
      {icon && <div className="mb-3">{icon}</div>}
      <h3 className="font-bold text-lg text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm mb-4">{description}</p>
      {learnMoreUrl && (
        <a
          href={learnMoreUrl}
          className="text-blue-500 text-sm font-semibold hover:underline"
        >
          Learn more →
        </a>
      )}
    </div>
  );
};

/**
 * FEATURE REFERENCE TOOLTIPS
 * 
 * Common tooltips used throughout the application.
 * Import and use in your components:
 * 
 * <HelpTooltip {...TOOLTIPS.CURTAILMENT_RATE} />
 */
export const TOOLTIPS = {
  CURTAILMENT_RATE: {
    title: 'Curtailment Rate',
    description: 'Percentage of total solar generation that must be curtailed (wasted) due to grid capacity limits. Lower is better. Target: <10%.',
    learnMore: 'Read about curtailment',
  },
  
  REVENUE_UPLIFT: {
    title: 'Revenue Uplift',
    description: 'How much more revenue the optimized strategy earns compared to the naive baseline. Formula: (Optimized Revenue - Naive Revenue) / Naive Revenue × 100%',
    learnMore: 'How revenue is calculated',
  },
  
  GRID_COMPLIANCE: {
    title: 'Grid Compliance',
    description: 'Percentage of hours where the facility stayed within grid export capacity limits. Must be 100% in production. Any violation risks penalties.',
    learnMore: 'Grid constraint details',
  },
  
  BATTERY_CYCLES: {
    title: 'Battery Cycles',
    description: 'Equivalent full charge/discharge cycles completed. High cycle count reduces battery lifespan. Modern batteries support ~4,000 cycles over 10 years.',
    learnMore: 'Battery degradation model',
  },
  
  MILP_OPTIMIZER: {
    title: 'MILP Optimizer',
    description: 'Finds the mathematically optimal dispatch decision for a 24-hour planning horizon. Uses Pyomo + GLPK solver. Guarantees global optimality but requires accurate forecasts.',
    learnMore: 'MILP algorithm details',
  },
  
  RL_AGENT: {
    title: 'RL Agent',
    description: 'Reinforcement Learning agent (PPO/DQN) that learns to make real-time decisions. Adapts to forecast deviations and unexpected events. No planning horizon required.',
    learnMore: 'RL agent training',
  },
  
  HYBRID_CONTROLLER: {
    title: 'Hybrid Controller',
    description: 'Combines MILP (planning) + RL (real-time adaptation). MILP creates optimal baseline plan, RL overrides if deviations detected. Recommended for production.',
    learnMore: 'Hybrid control logic',
  },
  
  STRESS_TEST: {
    title: 'Stress Testing',
    description: 'Monte Carlo simulation with 100+ scenarios. Varies generation, prices, and grid capacity randomly. Quantifies risk and confidence intervals.',
    learnMore: 'Uncertainty analysis',
  },

  DUCK_CURVE: {
    title: 'Duck Curve Scenario',
    description: 'Solar generation peaks at 600 MW (noon), but grid can only export 300 MW. Negative prices midday (-$25), high prices evening ($140). Tests battery arbitrage value.',
    learnMore: 'About duck curve',
  },

  INDUSTRY_DASHBOARD: {
    title: 'Industry Dashboard',
    description: 'Aggregates metrics across ALL optimizations ever run. Shows portfolio-level financial performance, environmental impact, and grid contribution.',
    learnMore: 'Dashboard metrics guide',
  },
};

/**
 * FEATURE CARDS FOR HOME PAGE
 */
export const FEATURE_CARDS = [
  {
    title: 'Advanced Optimization',
    description: 'MILP solver for mathematically optimal dispatch with guaranteed constraint satisfaction.',
    learnMore: 'Explore MILP',
  },
  {
    title: 'Real-Time Learning',
    description: 'RL agent adapts to forecast errors and unexpected grid conditions in real time.',
    learnMore: 'Explore RL',
  },
  {
    title: 'Risk Quantification',
    description: 'Monte Carlo stress testing quantifies profit ranges and violation probabilities.',
    learnMore: 'Explore Risk Analysis',
  },
  {
    title: 'Full Stack Analytics',
    description: 'Dashboard, reports, and API for comprehensive decision support and integration.',
    learnMore: 'Explore Analytics',
  },
];
