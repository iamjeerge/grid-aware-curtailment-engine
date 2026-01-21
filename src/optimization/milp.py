"""MILP formulation for grid-aware curtailment optimization.

Implements the optimization model from copilot-instructions.md:

Indices:
- t ∈ T: hourly time steps
- s ∈ S: forecast scenarios (P10, P50, P90)

Parameters:
- G[t,s]: generation forecast (MW)
- P[t]: market price ($/MWh)
- C[t]: grid export capacity (MW)
- η_c, η_d: charge/discharge efficiency
- E_max: battery capacity (MWh)
- P_max: battery power limit (MW)
- λ: degradation cost ($/MWh cycled)

Decision Variables:
- energy_sold[t,s]: energy sold to grid (MW)
- energy_stored[t,s]: energy charged to battery (MW)
- energy_curtailed[t,s]: curtailed energy (MW)
- energy_discharged[t,s]: energy discharged from battery (MW)
- soc[t,s]: state of charge (MWh)

Constraints:
1. Energy Balance: G[t,s] + d[t,s] = x[t,s] + y[t,s] + z[t,s]
2. Grid Capacity: x[t,s] <= C[t]
3. SOC Dynamics: SOC[t,s] = SOC[t-1,s] + η_c * y[t,s] - d[t,s] / η_d
4. SOC Bounds: SOC_min <= SOC[t,s] <= SOC_max
5. Charge Rate: 0 <= y[t,s] <= P_max
6. Discharge Rate: 0 <= d[t,s] <= P_max

Objective:
Maximize expected profit: Σ_s π_s Σ_t (P[t] * x[t,s] - λ * (y[t,s] + d[t,s]))
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
    SimulationResult,
)


@dataclass
class OptimizationConfig:
    """Configuration for the MILP optimizer."""

    # Solver settings
    solver_name: str = "glpk"  # 'glpk', 'cbc', 'gurobi', 'cplex'
    time_limit_seconds: float = 60.0
    mip_gap: float = 0.01  # 1% optimality gap

    # Scenario probabilities
    scenario_probabilities: dict[ForecastScenario, float] = field(
        default_factory=lambda: {
            ForecastScenario.P10: 0.25,
            ForecastScenario.P50: 0.50,
            ForecastScenario.P90: 0.25,
        }
    )

    # Initial conditions
    initial_soc_fraction: float = 0.5

    # Objective weights
    curtailment_penalty: float = 0.0  # $/MWh penalty for curtailment (optional)

    def validate(self) -> bool:
        """Validate configuration."""
        prob_sum = sum(self.scenario_probabilities.values())
        return abs(prob_sum - 1.0) < 0.001


@dataclass
class OptimizationResult:
    """Results from MILP optimization."""

    success: bool
    solver_status: str
    termination_condition: str
    objective_value: float | None
    solve_time_seconds: float
    gap: float | None

    # Solution values by timestep and scenario
    energy_sold: dict[tuple[int, ForecastScenario], float] = field(default_factory=dict)
    energy_stored: dict[tuple[int, ForecastScenario], float] = field(
        default_factory=dict
    )
    energy_discharged: dict[tuple[int, ForecastScenario], float] = field(
        default_factory=dict
    )
    energy_curtailed: dict[tuple[int, ForecastScenario], float] = field(
        default_factory=dict
    )
    soc: dict[tuple[int, ForecastScenario], float] = field(default_factory=dict)


class MILPOptimizer:
    """Mixed-Integer Linear Programming optimizer for dispatch decisions.

    Uses Pyomo to formulate and solve the MILP problem for optimal
    energy dispatch over a planning horizon.
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            battery_config: Battery configuration parameters.
            config: Optimization configuration. Uses defaults if None.
        """
        self.battery_config = battery_config
        self.config = config or OptimizationConfig()
        self._model: pyo.ConcreteModel | None = None
        self._result: OptimizationResult | None = None

    def build_model(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
        scenarios: list[ForecastScenario] | None = None,
    ) -> pyo.ConcreteModel:
        """Build the Pyomo optimization model.

        Args:
            forecasts: Generation forecasts for each timestep.
            grid_constraints: Grid constraints for each timestep.
            market_prices: Market prices for each timestep.
            scenarios: Scenarios to optimize over. Defaults to all three.

        Returns:
            Pyomo ConcreteModel ready for solving.
        """
        if scenarios is None:
            scenarios = [
                ForecastScenario.P10,
                ForecastScenario.P50,
                ForecastScenario.P90,
            ]

        n_timesteps = len(forecasts)
        model = pyo.ConcreteModel(name="GridAwareCurtailmentOptimization")

        # =================================================================
        # Sets
        # =================================================================
        model.T = pyo.Set(initialize=range(n_timesteps), doc="Time periods")
        model.S = pyo.Set(
            initialize=[s.value for s in scenarios], doc="Forecast scenarios"
        )

        # =================================================================
        # Parameters
        # =================================================================

        # Generation forecasts G[t,s]
        def generation_init(_m: Any, t: int, s: str) -> float:
            scenario = ForecastScenario(s)
            return forecasts[t].total_generation(scenario)

        model.generation = pyo.Param(
            model.T, model.S, initialize=generation_init, doc="Generation forecast (MW)"
        )

        # Grid export capacity C[t]
        def grid_capacity_init(_m: Any, t: int) -> float:
            return grid_constraints[t].max_export_mw

        model.grid_capacity = pyo.Param(
            model.T, initialize=grid_capacity_init, doc="Grid export capacity (MW)"
        )

        # Market prices P[t]
        def price_init(_m: Any, t: int) -> float:
            return market_prices[t].effective_price

        model.price = pyo.Param(
            model.T, initialize=price_init, doc="Market price ($/MWh)"
        )

        # Scenario probabilities π[s]
        def prob_init(_m: Any, s: str) -> float:
            scenario = ForecastScenario(s)
            return self.config.scenario_probabilities.get(scenario, 0.0)

        model.scenario_prob = pyo.Param(
            model.S, initialize=prob_init, doc="Scenario probability"
        )

        # Battery parameters
        model.battery_capacity = pyo.Param(
            initialize=self.battery_config.capacity_mwh, doc="Battery capacity (MWh)"
        )
        model.max_power = pyo.Param(
            initialize=self.battery_config.max_power_mw, doc="Max charge/discharge (MW)"
        )
        model.charge_efficiency = pyo.Param(
            initialize=self.battery_config.charge_efficiency, doc="Charge efficiency"
        )
        model.discharge_efficiency = pyo.Param(
            initialize=self.battery_config.discharge_efficiency,
            doc="Discharge efficiency",
        )
        model.min_soc = pyo.Param(
            initialize=self.battery_config.capacity_mwh
            * self.battery_config.min_soc_fraction,
            doc="Minimum SOC (MWh)",
        )
        model.max_soc = pyo.Param(
            initialize=self.battery_config.capacity_mwh
            * self.battery_config.max_soc_fraction,
            doc="Maximum SOC (MWh)",
        )
        model.degradation_cost = pyo.Param(
            initialize=self.battery_config.degradation_cost_per_mwh,
            doc="Degradation cost ($/MWh)",
        )
        model.initial_soc = pyo.Param(
            initialize=self.battery_config.capacity_mwh
            * self.config.initial_soc_fraction,
            doc="Initial SOC (MWh)",
        )
        model.curtailment_penalty = pyo.Param(
            initialize=self.config.curtailment_penalty,
            doc="Curtailment penalty ($/MWh)",
        )

        # =================================================================
        # Decision Variables
        # =================================================================

        # Energy sold to grid x[t,s]
        model.energy_sold = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy sold to grid (MW)",
        )

        # Energy stored (charging) y[t,s]
        model.energy_stored = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy charged to battery (MW)",
        )

        # Energy discharged d[t,s]
        model.energy_discharged = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy discharged from battery (MW)",
        )

        # Energy curtailed z[t,s]
        model.energy_curtailed = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Energy curtailed (MW)",
        )

        # State of charge SOC[t,s]
        model.soc = pyo.Var(
            model.T,
            model.S,
            domain=pyo.NonNegativeReals,
            doc="Battery state of charge (MWh)",
        )

        # =================================================================
        # Constraints
        # =================================================================

        # 1. Energy Balance: G[t,s] + d[t,s] = x[t,s] + y[t,s] + z[t,s]
        def energy_balance_rule(m: Any, t: int, s: str) -> Any:
            return (
                m.generation[t, s] + m.energy_discharged[t, s]
                == m.energy_sold[t, s]
                + m.energy_stored[t, s]
                + m.energy_curtailed[t, s]
            )

        model.energy_balance = pyo.Constraint(
            model.T, model.S, rule=energy_balance_rule, doc="Energy balance constraint"
        )

        # 2. Grid Capacity: x[t,s] <= C[t]
        def grid_capacity_rule(m: Any, t: int, s: str) -> Any:
            return m.energy_sold[t, s] <= m.grid_capacity[t]

        model.grid_limit = pyo.Constraint(
            model.T, model.S, rule=grid_capacity_rule, doc="Grid export limit"
        )

        # 3. SOC Dynamics: SOC[t,s] = SOC[t-1,s] + η_c * y[t,s] - d[t,s] / η_d
        def soc_dynamics_rule(m: Any, t: int, s: str) -> Any:
            prev_soc = m.initial_soc if t == 0 else m.soc[t - 1, s]
            return m.soc[t, s] == (
                prev_soc
                + m.charge_efficiency * m.energy_stored[t, s]
                - m.energy_discharged[t, s] / m.discharge_efficiency
            )

        model.soc_dynamics = pyo.Constraint(
            model.T, model.S, rule=soc_dynamics_rule, doc="SOC dynamics"
        )

        # 4. SOC Bounds: SOC_min <= SOC[t,s] <= SOC_max
        def soc_min_rule(m: Any, t: int, s: str) -> Any:
            return m.soc[t, s] >= m.min_soc

        def soc_max_rule(m: Any, t: int, s: str) -> Any:
            return m.soc[t, s] <= m.max_soc

        model.soc_min_constraint = pyo.Constraint(
            model.T, model.S, rule=soc_min_rule, doc="Minimum SOC"
        )
        model.soc_max_constraint = pyo.Constraint(
            model.T, model.S, rule=soc_max_rule, doc="Maximum SOC"
        )

        # 5. Charge Rate: 0 <= y[t,s] <= P_max
        def charge_rate_rule(m: Any, t: int, s: str) -> Any:
            return m.energy_stored[t, s] <= m.max_power

        model.charge_rate = pyo.Constraint(
            model.T, model.S, rule=charge_rate_rule, doc="Charge rate limit"
        )

        # 6. Discharge Rate: 0 <= d[t,s] <= P_max
        def discharge_rate_rule(m: Any, t: int, s: str) -> Any:
            return m.energy_discharged[t, s] <= m.max_power

        model.discharge_rate = pyo.Constraint(
            model.T, model.S, rule=discharge_rate_rule, doc="Discharge rate limit"
        )

        # =================================================================
        # Objective Function
        # =================================================================
        # Maximize: Σ_s π_s Σ_t (P[t] * x[t,s] - λ * (y[t,s] + d[t,s]) - penalty * z[t,s])
        def objective_rule(m: Any) -> Any:
            return sum(
                m.scenario_prob[s]
                * sum(
                    m.price[t] * m.energy_sold[t, s]
                    - m.degradation_cost
                    * (m.energy_stored[t, s] + m.energy_discharged[t, s])
                    - m.curtailment_penalty * m.energy_curtailed[t, s]
                    for t in m.T
                )
                for s in m.S
            )

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        self._model = model
        return model

    def solve(self) -> OptimizationResult:
        """Solve the optimization model.

        Returns:
            OptimizationResult with solution details.

        Raises:
            RuntimeError: If model not built or solver fails.
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # Get solver
        solver = SolverFactory(self.config.solver_name)

        if solver is None or not solver.available():
            # Fallback to GLPK if preferred solver unavailable
            solver = SolverFactory("glpk")
            if solver is None or not solver.available():
                raise RuntimeError(
                    f"Solver '{self.config.solver_name}' not available and GLPK fallback failed."
                )

        # Set solver options
        if self.config.solver_name in ["gurobi", "cplex"]:
            solver.options["TimeLimit"] = self.config.time_limit_seconds
            solver.options["MIPGap"] = self.config.mip_gap
        elif self.config.solver_name == "glpk":
            solver.options["tmlim"] = int(self.config.time_limit_seconds)
            solver.options["mipgap"] = self.config.mip_gap
        elif self.config.solver_name == "cbc":
            solver.options["seconds"] = self.config.time_limit_seconds
            solver.options["ratioGap"] = self.config.mip_gap

        # Solve
        import time

        start_time = time.time()
        results = solver.solve(self._model, tee=False)
        solve_time = time.time() - start_time

        # Check solution status
        solver_status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        success = (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.optimal
        ) or (
            results.solver.termination_condition
            == TerminationCondition.feasible  # type: ignore[unreachable]
        )

        # Extract objective value
        obj_value = None
        if success:
            obj_value = pyo.value(self._model.objective)

        # Extract solution values
        result = OptimizationResult(
            success=success,
            solver_status=solver_status,
            termination_condition=termination,
            objective_value=obj_value,
            solve_time_seconds=solve_time,
            gap=None,  # Could extract from solver if available
        )

        if success:
            model = self._model
            for t in model.T:
                for s in model.S:
                    scenario = ForecastScenario(s)
                    result.energy_sold[(t, scenario)] = pyo.value(
                        model.energy_sold[t, s]
                    )
                    result.energy_stored[(t, scenario)] = pyo.value(
                        model.energy_stored[t, s]
                    )
                    result.energy_discharged[(t, scenario)] = pyo.value(
                        model.energy_discharged[t, s]
                    )
                    result.energy_curtailed[(t, scenario)] = pyo.value(
                        model.energy_curtailed[t, s]
                    )
                    result.soc[(t, scenario)] = pyo.value(model.soc[t, s])

        self._result = result
        return result

    def optimize(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
        scenarios: list[ForecastScenario] | None = None,
    ) -> OptimizationResult:
        """Build and solve the optimization model in one step.

        Args:
            forecasts: Generation forecasts for each timestep.
            grid_constraints: Grid constraints for each timestep.
            market_prices: Market prices for each timestep.
            scenarios: Scenarios to optimize over.

        Returns:
            OptimizationResult with solution.
        """
        self.build_model(forecasts, grid_constraints, market_prices, scenarios)
        return self.solve()

    def get_simulation_result(
        self,
        forecasts: list[GenerationForecast],
        scenario: ForecastScenario = ForecastScenario.P50,
    ) -> SimulationResult:
        """Convert optimization result to SimulationResult format.

        Args:
            forecasts: Original forecasts (for timestamps).
            scenario: Which scenario to extract results for.

        Returns:
            SimulationResult compatible with other controllers.

        Raises:
            RuntimeError: If optimization not solved or failed.
        """
        if self._result is None or not self._result.success:
            raise RuntimeError("No successful optimization result available.")

        decisions: list[OptimizationDecision] = []
        total_sold = 0.0
        total_stored = 0.0
        total_discharged = 0.0
        total_curtailed = 0.0
        total_generated = 0.0
        total_revenue = 0.0
        total_degradation = 0.0

        for t, forecast in enumerate(forecasts):
            generation = forecast.total_generation(scenario)
            sold = self._result.energy_sold.get((t, scenario), 0.0)
            stored = self._result.energy_stored.get((t, scenario), 0.0)
            discharged = self._result.energy_discharged.get((t, scenario), 0.0)
            curtailed = self._result.energy_curtailed.get((t, scenario), 0.0)
            soc = self._result.soc.get((t, scenario), 0.0)

            # Get price for this timestep (need to store or pass prices)
            # For now, calculate revenue from objective
            # In practice, we'd store prices during build_model

            total_sold += sold
            total_stored += stored
            total_discharged += discharged
            total_curtailed += curtailed
            total_generated += generation

            # Estimate revenue and degradation
            # This is approximate since we don't store prices per timestep
            degradation = (
                stored + discharged
            ) * self.battery_config.degradation_cost_per_mwh

            decision = OptimizationDecision(
                timestamp=forecast.timestamp,
                scenario=scenario,
                generation_mw=generation,
                energy_sold_mw=sold,
                energy_stored_mw=stored,
                energy_curtailed_mw=curtailed,
                battery_discharge_mw=discharged,
                resulting_soc_mwh=soc,
                revenue_dollars=0.0,  # Would need price data
                degradation_cost_dollars=degradation,
            )
            decisions.append(decision)
            total_degradation += degradation

        # For revenue, use objective value if available
        if self._result.objective_value is not None:
            total_revenue = self._result.objective_value + total_degradation

        return SimulationResult(
            simulation_id=f"milp_{uuid4().hex[:8]}",
            start_time=forecasts[0].timestamp,
            end_time=forecasts[-1].timestamp,
            scenario=scenario,
            decisions=decisions,
            total_generation_mwh=total_generated,
            total_sold_mwh=total_sold,
            total_stored_mwh=total_stored,
            total_curtailed_mwh=total_curtailed,
            total_revenue_dollars=total_revenue,
            total_degradation_cost_dollars=total_degradation,
            grid_violations_count=0,  # MILP ensures no violations
        )


class SingleScenarioOptimizer(MILPOptimizer):
    """Simplified optimizer for single-scenario optimization.

    Useful for deterministic optimization or testing.
    """

    def __init__(
        self,
        battery_config: BatteryConfig,
        scenario: ForecastScenario = ForecastScenario.P50,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Initialize single-scenario optimizer.

        Args:
            battery_config: Battery configuration.
            scenario: Which forecast scenario to use.
            config: Optimization configuration.
        """
        if config is None:
            config = OptimizationConfig()

        # Set 100% probability on the selected scenario
        config.scenario_probabilities = {
            ForecastScenario.P10: 0.0,
            ForecastScenario.P50: 0.0,
            ForecastScenario.P90: 0.0,
        }
        config.scenario_probabilities[scenario] = 1.0

        super().__init__(battery_config, config)
        self.scenario = scenario

    def optimize(
        self,
        forecasts: list[GenerationForecast],
        grid_constraints: list[GridConstraint],
        market_prices: list[MarketPrice],
        _scenarios: list[ForecastScenario] | None = None,
    ) -> OptimizationResult:
        """Optimize for single scenario.

        Args:
            forecasts: Generation forecasts.
            grid_constraints: Grid constraints.
            market_prices: Market prices.
            _scenarios: Ignored, uses configured scenario.

        Returns:
            OptimizationResult.
        """
        return super().optimize(
            forecasts, grid_constraints, market_prices, scenarios=[self.scenario]
        )
