"""Risk-aware optimization with CVaR.

Implements Conditional Value-at-Risk (CVaR) optimization for risk-aware
decision making under uncertainty. CVaR focuses on tail risk by optimizing
the expected value in the worst α% of scenarios.

CVaR Formulation:
    CVaR_α = E[X | X ≤ VaR_α]

    Where VaR_α is the Value-at-Risk at confidence level α (e.g., 95%)

The risk-aware objective becomes:
    maximize: (1-λ) * E[profit] + λ * CVaR_α(profit)

Where λ ∈ [0,1] controls the risk aversion level.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from src.domain.models import (
    BatteryConfig,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
)


@dataclass
class RiskAwareConfig:
    """Configuration for risk-aware optimization with CVaR.

    Attributes:
        solver_name: Optimization solver to use.
        time_limit_seconds: Maximum solve time.
        confidence_level: CVaR confidence level (e.g., 0.95 for 95% CVaR).
        risk_weight: Weight on CVaR vs expected value (0=risk-neutral, 1=max risk-averse).
        scenario_probabilities: Probability weights for each scenario.
        initial_soc_fraction: Initial battery state of charge as fraction of capacity.
        curtailment_penalty: Optional penalty for curtailment ($/MWh).
    """

    solver_name: str = "glpk"
    time_limit_seconds: float = 120.0
    mip_gap: float = 0.01

    # CVaR parameters
    confidence_level: float = 0.95  # α for CVaR (e.g., 95% confidence)
    risk_weight: float = 0.3  # λ: weight on CVaR vs expected value

    # Extended scenarios for better tail estimation
    scenario_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "P05": 0.10,  # Extreme low generation
            "P10": 0.15,
            "P25": 0.15,
            "P50": 0.20,
            "P75": 0.15,
            "P90": 0.15,
            "P95": 0.10,  # Extreme high generation
        }
    )

    initial_soc_fraction: float = 0.5
    curtailment_penalty: float = 0.0

    def validate(self) -> bool:
        """Validate configuration."""
        prob_sum = sum(self.scenario_probabilities.values())
        valid_probs = abs(prob_sum - 1.0) < 0.001
        valid_confidence = 0.5 <= self.confidence_level <= 0.99
        valid_risk_weight = 0.0 <= self.risk_weight <= 1.0
        return valid_probs and valid_confidence and valid_risk_weight


@dataclass
class RiskMetrics:
    """Risk metrics from optimization result."""

    expected_profit: float  # E[profit]
    var_profit: float  # VaR at confidence level
    cvar_profit: float  # CVaR at confidence level
    worst_case_profit: float  # Minimum across scenarios
    best_case_profit: float  # Maximum across scenarios
    profit_std_dev: float  # Standard deviation

    # Profit by scenario
    scenario_profits: dict[str, float] = field(default_factory=dict)

    @property
    def risk_adjusted_profit(self) -> float:
        """Risk-adjusted expected profit using Sharpe-like ratio."""
        if self.profit_std_dev > 0:
            return self.expected_profit / self.profit_std_dev
        return float("inf") if self.expected_profit > 0 else 0.0

    @property
    def downside_exposure(self) -> float:
        """Downside risk as percentage of expected profit."""
        if self.expected_profit > 0:
            return (self.expected_profit - self.cvar_profit) / self.expected_profit
        return 0.0


@dataclass
class CVaRResult:
    """Results from CVaR optimization."""

    success: bool
    solver_status: str
    termination_condition: str
    objective_value: float | None
    solve_time_seconds: float

    # Risk metrics
    risk_metrics: RiskMetrics | None = None

    # Solution values by timestep and scenario
    energy_sold: dict[tuple[int, str], float] = field(default_factory=dict)
    energy_stored: dict[tuple[int, str], float] = field(default_factory=dict)
    energy_discharged: dict[tuple[int, str], float] = field(default_factory=dict)
    energy_curtailed: dict[tuple[int, str], float] = field(default_factory=dict)
    soc: dict[tuple[int, str], float] = field(default_factory=dict)


class CVaROptimizer:
    """Risk-aware optimizer using Conditional Value-at-Risk.

    Extends the basic MILP formulation to optimize a risk-adjusted objective
    that balances expected profit with tail risk (CVaR).

    The formulation uses the Rockafellar-Uryasev linearization of CVaR:
        CVaR_α ≈ max_η { η - (1/(1-α)) * Σ_s π_s * [η - profit_s]⁺ }

    Which can be linearized with auxiliary variables:
        CVaR_α = η - (1/(1-α)) * Σ_s π_s * u_s
        where: u_s ≥ η - profit_s, u_s ≥ 0
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        config: RiskAwareConfig | None = None,
    ) -> None:
        """Initialize the CVaR optimizer.

        Args:
            battery_config: Battery configuration parameters.
            config: Risk-aware optimization configuration.
        """
        self.battery_config = battery_config
        self.config = config or RiskAwareConfig()
        self._model: pyo.ConcreteModel | None = None
        self._result: CVaRResult | None = None

    def build_model(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
    ) -> pyo.ConcreteModel:
        """Build the Pyomo optimization model with CVaR objective.

        Args:
            forecasts: Generation forecasts for each timestep.
            grid_constraints: Grid constraints for each timestep.
            market_prices: Market prices for each timestep.

        Returns:
            Pyomo ConcreteModel ready for solving.
        """
        n_timesteps = len(forecasts)
        model = pyo.ConcreteModel(name="CVaROptimization")

        scenarios = list(self.config.scenario_probabilities.keys())

        # =============================================================
        # Sets
        # =============================================================
        model.T = pyo.Set(initialize=range(n_timesteps), doc="Time periods")
        model.S = pyo.Set(initialize=scenarios, doc="Scenarios")

        # =============================================================
        # Parameters
        # =============================================================

        # Generation forecasts - interpolate for extended scenarios
        def generation_init(_m: Any, t: int, s: str) -> float:
            return self._interpolate_generation(forecasts[t], s)

        model.generation = pyo.Param(
            model.T, model.S, initialize=generation_init, doc="Generation forecast (MW)"
        )

        # Grid capacity
        def grid_capacity_init(_m: Any, t: int) -> float:
            return grid_constraints[t].max_export_mw

        model.grid_capacity = pyo.Param(
            model.T, initialize=grid_capacity_init, doc="Grid export capacity (MW)"
        )

        # Market prices
        def price_init(_m: Any, t: int) -> float:
            return market_prices[t].effective_price

        model.price = pyo.Param(
            model.T, initialize=price_init, doc="Market price ($/MWh)"
        )

        # Scenario probabilities
        def prob_init(_m: Any, s: str) -> float:
            return self.config.scenario_probabilities[s]

        model.probability = pyo.Param(
            model.S, initialize=prob_init, doc="Scenario probability"
        )

        # Battery parameters
        model.battery_capacity = pyo.Param(
            initialize=self.battery_config.capacity_mwh, doc="Battery capacity (MWh)"
        )
        model.max_charge_power = pyo.Param(
            initialize=self.battery_config.max_power_mw, doc="Max charge power (MW)"
        )
        model.max_discharge_power = pyo.Param(
            initialize=self.battery_config.max_power_mw, doc="Max discharge power (MW)"
        )
        model.charge_efficiency = pyo.Param(
            initialize=self.battery_config.charge_efficiency, doc="Charge efficiency"
        )
        model.discharge_efficiency = pyo.Param(
            initialize=self.battery_config.discharge_efficiency,
            doc="Discharge efficiency",
        )
        model.degradation_cost = pyo.Param(
            initialize=self.battery_config.degradation_cost_per_mwh,
            doc="Degradation cost ($/MWh)",
        )
        model.initial_soc = pyo.Param(
            initialize=self.config.initial_soc_fraction
            * self.battery_config.capacity_mwh,
            doc="Initial SOC (MWh)",
        )

        # CVaR parameters
        model.alpha = pyo.Param(
            initialize=self.config.confidence_level, doc="CVaR confidence level"
        )
        model.risk_weight = pyo.Param(
            initialize=self.config.risk_weight, doc="Weight on CVaR vs expected value"
        )

        # =============================================================
        # Decision Variables
        # =============================================================

        # Energy dispatch variables
        model.energy_sold = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy sold to grid (MW)",
        )
        model.energy_stored = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy charged to battery (MW)",
        )
        model.energy_discharged = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy discharged from battery (MW)",
        )
        model.energy_curtailed = pyo.Var(
            model.T, model.S, domain=pyo.NonNegativeReals, doc="Curtailed energy (MW)"
        )
        model.soc = pyo.Var(
            model.T, model.S, domain=pyo.NonNegativeReals, doc="State of charge (MWh)"
        )

        # Scenario profit variable
        model.scenario_profit = pyo.Var(
            model.S, domain=pyo.Reals, doc="Profit in each scenario ($)"
        )

        # CVaR auxiliary variables (Rockafellar-Uryasev formulation)
        model.var_threshold = pyo.Var(domain=pyo.Reals, doc="VaR threshold (η)")
        model.cvar_shortfall = pyo.Var(
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Shortfall below VaR threshold (u_s)",
        )

        # =============================================================
        # Constraints
        # =============================================================

        # Energy balance: generation + discharge = sold + stored + curtailed
        def energy_balance_rule(m: pyo.ConcreteModel, t: int, s: str) -> pyo.Expression:
            return (
                m.generation[t, s] + m.energy_discharged[t, s]
                == m.energy_sold[t, s]
                + m.energy_stored[t, s]
                + m.energy_curtailed[t, s]
            )

        model.energy_balance = pyo.Constraint(
            model.T, model.S, rule=energy_balance_rule, doc="Energy balance constraint"
        )

        # Grid capacity limit
        def grid_capacity_rule(m: pyo.ConcreteModel, t: int, s: str) -> pyo.Expression:
            return m.energy_sold[t, s] <= m.grid_capacity[t]

        model.grid_limit = pyo.Constraint(
            model.T, model.S, rule=grid_capacity_rule, doc="Grid capacity constraint"
        )

        # SOC dynamics
        def soc_dynamics_rule(m: pyo.ConcreteModel, t: int, s: str) -> pyo.Expression:
            prev_soc = m.initial_soc if t == 0 else m.soc[t - 1, s]
            charge_contribution = m.charge_efficiency * m.energy_stored[t, s]
            discharge_contribution = m.energy_discharged[t, s] / m.discharge_efficiency
            return (
                m.soc[t, s] == prev_soc + charge_contribution - discharge_contribution
            )

        model.soc_dynamics = pyo.Constraint(
            model.T, model.S, rule=soc_dynamics_rule, doc="SOC dynamics constraint"
        )

        # SOC bounds
        def soc_upper_bound_rule(
            m: pyo.ConcreteModel, t: int, s: str
        ) -> pyo.Expression:
            return m.soc[t, s] <= m.battery_capacity

        model.soc_upper = pyo.Constraint(
            model.T, model.S, rule=soc_upper_bound_rule, doc="SOC upper bound"
        )

        # Charge rate limit
        def charge_rate_rule(m: pyo.ConcreteModel, t: int, s: str) -> pyo.Expression:
            return m.energy_stored[t, s] <= m.max_charge_power

        model.charge_limit = pyo.Constraint(
            model.T, model.S, rule=charge_rate_rule, doc="Charge rate limit"
        )

        # Discharge rate limit
        def discharge_rate_rule(m: pyo.ConcreteModel, t: int, s: str) -> pyo.Expression:
            return m.energy_discharged[t, s] <= m.max_discharge_power

        model.discharge_limit = pyo.Constraint(
            model.T, model.S, rule=discharge_rate_rule, doc="Discharge rate limit"
        )

        # Scenario profit calculation
        def scenario_profit_rule(m: pyo.ConcreteModel, s: str) -> pyo.Expression:
            revenue = sum(m.price[t] * m.energy_sold[t, s] for t in m.T)
            degradation = sum(
                m.degradation_cost * (m.energy_stored[t, s] + m.energy_discharged[t, s])
                for t in m.T
            )
            curtailment_cost = sum(
                self.config.curtailment_penalty * m.energy_curtailed[t, s] for t in m.T
            )
            return m.scenario_profit[s] == revenue - degradation - curtailment_cost

        model.profit_def = pyo.Constraint(
            model.S, rule=scenario_profit_rule, doc="Scenario profit definition"
        )

        # CVaR shortfall constraint: u_s >= eta - profit_s
        def cvar_shortfall_rule(m: pyo.ConcreteModel, s: str) -> pyo.Expression:
            return m.cvar_shortfall[s] >= m.var_threshold - m.scenario_profit[s]

        model.cvar_shortfall_con = pyo.Constraint(
            model.S, rule=cvar_shortfall_rule, doc="CVaR shortfall constraint"
        )

        # =============================================================
        # Objective: Risk-adjusted expected profit
        # =============================================================

        def objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
            # Expected profit
            expected_profit = sum(m.probability[s] * m.scenario_profit[s] for s in m.S)

            # CVaR: η - (1/(1-α)) * Σ_s π_s * u_s
            cvar_penalty = (1 / (1 - m.alpha)) * sum(
                m.probability[s] * m.cvar_shortfall[s] for s in m.S
            )
            cvar = m.var_threshold - cvar_penalty

            # Risk-adjusted objective: (1-λ)*E[profit] + λ*CVaR
            return (1 - m.risk_weight) * expected_profit + m.risk_weight * cvar

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        self._model = model
        return model

    def _interpolate_generation(
        self,
        forecast: GenerationForecast,
        scenario: str,
    ) -> float:
        """Interpolate generation for extended scenarios.

        Maps extended scenario labels (P05, P25, P75, P95) to generation
        values by interpolating between P10, P50, P90.
        """
        # Define percentile mapping
        percentile_map = {
            "P05": 0.05,
            "P10": 0.10,
            "P25": 0.25,
            "P50": 0.50,
            "P75": 0.75,
            "P90": 0.90,
            "P95": 0.95,
        }

        # Known values from forecast
        solar_values = {
            0.10: forecast.solar_mw_p10,
            0.50: forecast.solar_mw_p50,
            0.90: forecast.solar_mw_p90,
        }
        wind_values = {
            0.10: forecast.wind_mw_p10,
            0.50: forecast.wind_mw_p50,
            0.90: forecast.wind_mw_p90,
        }

        p = percentile_map.get(scenario, 0.50)

        # Linear interpolation
        solar = self._interpolate_value(p, solar_values)
        wind = self._interpolate_value(p, wind_values)

        return solar + wind

    def _interpolate_value(
        self,
        percentile: float,
        known_values: dict[float, float],
    ) -> float:
        """Linear interpolation between known percentile values."""
        points = sorted(known_values.keys())

        if percentile <= points[0]:
            # Extrapolate below
            slope = (known_values[points[1]] - known_values[points[0]]) / (
                points[1] - points[0]
            )
            return known_values[points[0]] + slope * (percentile - points[0])

        if percentile >= points[-1]:
            # Extrapolate above
            slope = (known_values[points[-1]] - known_values[points[-2]]) / (
                points[-1] - points[-2]
            )
            return known_values[points[-1]] + slope * (percentile - points[-1])

        # Interpolate between known points
        for i in range(len(points) - 1):
            if points[i] <= percentile <= points[i + 1]:
                t = (percentile - points[i]) / (points[i + 1] - points[i])
                return (1 - t) * known_values[points[i]] + t * known_values[
                    points[i + 1]
                ]

        return known_values[points[1]]  # Fallback to P50

    def solve(self, time_limit: float | None = None) -> CVaRResult:
        """Solve the optimization model.

        Args:
            time_limit: Override time limit in seconds.

        Returns:
            CVaRResult with solution values and risk metrics.
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        solver = SolverFactory(self.config.solver_name)
        if solver is None or not solver.available():
            raise RuntimeError(f"Solver '{self.config.solver_name}' not available")

        # Set solver options
        limit = time_limit or self.config.time_limit_seconds
        if hasattr(solver, "options"):
            solver.options["tmlim"] = int(limit)  # GLPK requires integer

        import time

        start_time = time.time()
        results = solver.solve(self._model, tee=False)
        solve_time = time.time() - start_time

        # Check solution status
        status = results.solver.status
        termination = results.solver.termination_condition

        success = (
            status == SolverStatus.ok and termination == TerminationCondition.optimal
        )

        result = CVaRResult(
            success=success,
            solver_status=str(status),
            termination_condition=str(termination),
            objective_value=pyo.value(self._model.objective) if success else None,
            solve_time_seconds=solve_time,
        )

        if success:
            self._extract_solution(result)
            self._compute_risk_metrics(result)

        self._result = result
        return result

    def _extract_solution(self, result: CVaRResult) -> None:
        """Extract solution values from the solved model."""
        model = self._model
        if model is None:
            return

        for t in model.T:
            for s in model.S:
                key = (t, s)
                result.energy_sold[key] = pyo.value(model.energy_sold[t, s])
                result.energy_stored[key] = pyo.value(model.energy_stored[t, s])
                result.energy_discharged[key] = pyo.value(model.energy_discharged[t, s])
                result.energy_curtailed[key] = pyo.value(model.energy_curtailed[t, s])
                result.soc[key] = pyo.value(model.soc[t, s])

    def _compute_risk_metrics(self, result: CVaRResult) -> None:
        """Compute risk metrics from the solution."""
        model = self._model
        if model is None:
            return

        # Get scenario profits
        scenario_profits = {}
        for s in model.S:
            scenario_profits[s] = pyo.value(model.scenario_profit[s])

        profits = np.array(list(scenario_profits.values()))
        probs = np.array(
            [self.config.scenario_probabilities[s] for s in scenario_profits]
        )

        # Expected profit
        expected_profit = float(np.sum(profits * probs))

        # VaR and CVaR from model
        var_threshold = pyo.value(model.var_threshold)

        # Compute empirical CVaR
        sorted_idx = np.argsort(profits)
        sorted_profits = profits[sorted_idx]
        sorted_probs = probs[sorted_idx]
        cumsum_probs = np.cumsum(sorted_probs)

        # Find VaR (quantile at 1 - confidence_level)
        var_level = 1 - self.config.confidence_level
        var_idx = np.searchsorted(cumsum_probs, var_level)
        var_profit = float(sorted_profits[min(var_idx, len(sorted_profits) - 1)])

        # CVaR: expected value below VaR
        below_var_mask = profits <= var_profit
        if below_var_mask.any():
            cvar_profit = float(
                np.average(profits[below_var_mask], weights=probs[below_var_mask])
            )
        else:
            cvar_profit = var_profit

        result.risk_metrics = RiskMetrics(
            expected_profit=expected_profit,
            var_profit=var_threshold,
            cvar_profit=cvar_profit,
            worst_case_profit=float(np.min(profits)),
            best_case_profit=float(np.max(profits)),
            profit_std_dev=float(
                np.sqrt(np.average((profits - expected_profit) ** 2, weights=probs))
            ),
            scenario_profits=scenario_profits,
        )
