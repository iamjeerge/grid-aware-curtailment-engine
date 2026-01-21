"""Service layer for optimization operations.

This module handles the business logic for running optimizations,
converting between domain models and API schemas.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.api.schemas import (
    BatteryConfigRequest,
    BatteryMetricsResponse,
    ComparisonResponse,
    CurtailmentMetricsResponse,
    DecisionResponse,
    GridComplianceResponse,
    OptimizationResultResponse,
    OptimizationStatus,
    PerformanceSummaryResponse,
    RevenueMetricsResponse,
    ScenarioConfigRequest,
    ScenarioType,
    StrategyResultResponse,
    StrategyType,
)
from src.controllers.naive import NaiveController
from src.domain.models import (
    BatteryConfig,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
)
from src.generators import (
    CongestionPattern,
    GenerationGenerator,
    GridConstraintGenerator,
    MarketPriceGenerator,
    PricePattern,
)
from src.metrics import (
    AnalyzerConfig,
    PerformanceAnalyzer,
    PerformanceSummary,
)
from src.optimization.milp import MILPOptimizer, OptimizationConfig

if TYPE_CHECKING:
    pass


class OptimizationService:
    """Service for running optimizations."""

    def __init__(self) -> None:
        """Initialize the optimization service."""
        self._results: dict[UUID, OptimizationResultResponse] = {}

    def _convert_battery_config(self, config: BatteryConfigRequest) -> BatteryConfig:
        """Convert API battery config to domain model."""
        return BatteryConfig(
            capacity_mwh=config.capacity_mwh,
            max_power_mw=config.max_power_mw,
            charge_efficiency=config.charge_efficiency,
            discharge_efficiency=config.discharge_efficiency,
            min_soc_fraction=config.min_soc_fraction,
            max_soc_fraction=config.max_soc_fraction,
            degradation_cost_per_mwh=config.degradation_cost_per_mwh,
        )

    def _generate_scenario_data(
        self, config: ScenarioConfigRequest
    ) -> tuple[list[GenerationForecast], list[MarketPrice], list[GridConstraint]]:
        """Generate scenario data based on configuration."""
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Determine patterns based on scenario type
        if config.scenario_type == ScenarioType.DUCK_CURVE:
            price_pattern = PricePattern.DUCK_CURVE
            congestion_pattern = CongestionPattern.MIDDAY
        elif config.scenario_type == ScenarioType.HIGH_VOLATILITY:
            price_pattern = PricePattern.HIGH_VOLATILITY
            congestion_pattern = CongestionPattern.RANDOM
        elif config.scenario_type == ScenarioType.CONGESTED_GRID:
            price_pattern = PricePattern.NORMAL
            congestion_pattern = CongestionPattern.EVENING
        else:
            price_pattern = PricePattern.NORMAL
            congestion_pattern = CongestionPattern.NONE

        # Generate forecasts
        gen_generator = GenerationGenerator(
            solar_capacity_mw=config.peak_generation_mw,
            wind_capacity_mw=0.0,
            seed=config.seed,
        )

        forecasts: list[GenerationForecast] = []
        for hour in range(config.horizon_hours):
            timestamp = base_time + timedelta(hours=hour)
            forecast = gen_generator.generate_forecast(timestamp)
            forecasts.append(forecast)

        # Generate prices
        price_gen = MarketPriceGenerator(
            base_price_multiplier=1.0,
            volatility=0.1,
            seed=config.seed,
        )

        prices: list[MarketPrice] = []
        for hour in range(config.horizon_hours):
            timestamp = base_time + timedelta(hours=hour)
            price = price_gen.generate_price(
                timestamp=timestamp,
                pattern=price_pattern,
                include_real_time=True,
            )
            prices.append(price)

        # Generate grid constraints
        grid_gen = GridConstraintGenerator(
            base_export_capacity_mw=config.grid_limit_mw,
            min_export_capacity_mw=config.grid_limit_mw * 0.5,
            max_ramp_rate_mw_per_hour=150.0,
            seed=config.seed,
        )

        constraints: list[GridConstraint] = []
        for hour in range(config.horizon_hours):
            timestamp = base_time + timedelta(hours=hour)
            constraint = grid_gen.generate_constraint(
                timestamp=timestamp,
                pattern=congestion_pattern,
            )
            constraints.append(constraint)

        return forecasts, prices, constraints

    def _run_naive_strategy(
        self,
        forecasts: list[GenerationForecast],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        battery_config: BatteryConfig,
    ) -> tuple[list[OptimizationDecision], PerformanceSummary, float]:
        """Run naive strategy and return results."""
        import time

        start = time.time()

        controller = NaiveController(battery_config)

        analyzer_config = AnalyzerConfig(battery_config=battery_config)
        analyzer = PerformanceAnalyzer(analyzer_config)

        decisions: list[OptimizationDecision] = []
        
        # Initialize the controller
        controller.initialize(
            start_time=forecasts[0].timestamp,
            initial_soc_fraction=0.5,
        )

        for forecast, price, constraint in zip(
            forecasts, prices, constraints, strict=True
        ):
            decision = controller.dispatch(
                timestamp=forecast.timestamp,
                generation_mw=forecast.total_generation(ForecastScenario.P50),
                grid_constraint=constraint,
                market_price=price,
                scenario=ForecastScenario.P50,
            )
            decisions.append(decision)

        summary = analyzer.analyze_decisions(
            decisions=decisions,
            forecasts=forecasts,
            prices=prices,
            constraints=constraints,
            strategy_name="Naive",
        )

        elapsed = time.time() - start
        return decisions, summary, elapsed

    def _run_milp_strategy(
        self,
        forecasts: list[GenerationForecast],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
        battery_config: BatteryConfig,
    ) -> tuple[list[OptimizationDecision], PerformanceSummary, float]:
        """Run MILP strategy and return results."""
        import time

        start = time.time()

        opt_config = OptimizationConfig(
            solver_name="glpk",
            time_limit_seconds=60.0,
            mip_gap=0.01,
        )
        optimizer = MILPOptimizer(battery_config, opt_config)

        analyzer_config = AnalyzerConfig(battery_config=battery_config)
        analyzer = PerformanceAnalyzer(analyzer_config)

        result = optimizer.optimize(
            forecasts=forecasts,
            grid_constraints=constraints,
            market_prices=prices,
        )

        sim_result = optimizer.get_simulation_result(
            forecasts=forecasts,
            scenario=ForecastScenario.P50,
        )
        decisions = sim_result.decisions

        summary = analyzer.analyze_decisions(
            decisions=decisions,
            forecasts=forecasts,
            prices=prices,
            constraints=constraints,
            strategy_name="MILP",
        )

        elapsed = time.time() - start
        return decisions, summary, elapsed

    def _convert_summary_to_response(
        self, summary: PerformanceSummary
    ) -> PerformanceSummaryResponse:
        """Convert domain summary to API response."""
        return PerformanceSummaryResponse(
            strategy_name=summary.strategy_name,
            curtailment=CurtailmentMetricsResponse(
                total_generation_mwh=summary.curtailment.total_generation_mwh,
                total_curtailed_mwh=summary.curtailment.total_curtailed_mwh,
                total_sold_mwh=summary.curtailment.total_sold_mwh,
                total_stored_mwh=summary.curtailment.total_stored_mwh,
                curtailment_rate=summary.curtailment.curtailment_rate,
            ),
            revenue=RevenueMetricsResponse(
                gross_revenue=summary.revenue.total_revenue,
                degradation_cost=summary.revenue.degradation_cost,
                net_profit=summary.revenue.net_profit,
                average_price=summary.revenue.average_price_captured,
            ),
            battery=BatteryMetricsResponse(
                total_charged_mwh=summary.battery.total_charged_mwh,
                total_discharged_mwh=summary.battery.total_discharged_mwh,
                cycles=summary.battery.total_cycles,
                utilization_rate=summary.battery.utilization_rate,
                avg_soc=summary.battery.average_soc,
            ),
            grid_compliance=GridComplianceResponse(
                violation_count=summary.grid_compliance.violation_count,
                total_violation_mwh=summary.grid_compliance.total_violation_mwh,
                max_violation_mw=summary.grid_compliance.max_violation_mw,
                compliance_rate=summary.grid_compliance.compliance_rate,
            ),
        )

    def _convert_decisions_to_response(
        self,
        decisions: list[OptimizationDecision],
        prices: list[MarketPrice],
        constraints: list[GridConstraint],
    ) -> list[DecisionResponse]:
        """Convert domain decisions to API response."""
        responses = []
        for i, decision in enumerate(decisions):
            responses.append(
                DecisionResponse(
                    timestep=i,
                    timestamp=decision.timestamp,
                    generation_mw=decision.generation_mw,
                    energy_sold_mw=decision.energy_sold_mw,
                    energy_stored_mw=decision.energy_stored_mw,
                    energy_curtailed_mw=decision.energy_curtailed_mw,
                    battery_discharge_mw=decision.battery_discharge_mw,
                    soc_mwh=decision.resulting_soc_mwh,
                    price=prices[i].effective_price if i < len(prices) else 0.0,
                    grid_limit_mw=(
                        constraints[i].max_export_mw if i < len(constraints) else 0.0
                    ),
                )
            )
        return responses

    def run_optimization(
        self,
        name: str,
        scenario_config: ScenarioConfigRequest,
        battery_config: BatteryConfigRequest,
        strategies: list[StrategyType],
    ) -> OptimizationResultResponse:
        """Run optimization with specified configuration."""
        optimization_id = uuid4()
        created_at = datetime.now()

        # Convert battery config
        battery = self._convert_battery_config(battery_config)

        # Generate scenario data
        forecasts, prices, constraints = self._generate_scenario_data(scenario_config)

        # Run each strategy
        results: dict[str, StrategyResultResponse] = {}
        summaries: dict[str, PerformanceSummary] = {}

        for strategy in strategies:
            if strategy == StrategyType.NAIVE:
                decisions, summary, elapsed = self._run_naive_strategy(
                    forecasts, prices, constraints, battery
                )
            elif strategy == StrategyType.MILP:
                decisions, summary, elapsed = self._run_milp_strategy(
                    forecasts, prices, constraints, battery
                )
            else:
                # RL and Hybrid not implemented in API yet
                continue

            summaries[strategy.value] = summary
            results[strategy.value] = StrategyResultResponse(
                strategy=strategy,
                summary=self._convert_summary_to_response(summary),
                decisions=self._convert_decisions_to_response(
                    decisions, prices, constraints
                ),
                solve_time_seconds=elapsed,
            )

        # Compute comparison if we have both naive and MILP
        comparison = None
        if "naive" in summaries and "milp" in summaries:
            naive = summaries["naive"]
            milp = summaries["milp"]

            naive_curtail = naive.curtailment.curtailment_rate
            milp_curtail = milp.curtailment.curtailment_rate
            curtail_reduction = (
                (naive_curtail - milp_curtail) / naive_curtail
                if naive_curtail > 0
                else 0.0
            )

            naive_profit = naive.revenue.net_profit
            milp_profit = milp.revenue.net_profit
            revenue_uplift = milp_profit - naive_profit
            revenue_uplift_pct = (
                revenue_uplift / naive_profit if naive_profit > 0 else 0.0
            )

            comparison = ComparisonResponse(
                best_strategy="milp" if milp_profit > naive_profit else "naive",
                curtailment_reduction_pct=curtail_reduction,
                revenue_uplift_pct=revenue_uplift_pct,
                revenue_uplift_dollars=revenue_uplift,
            )

        result = OptimizationResultResponse(
            id=optimization_id,
            name=name,
            status=OptimizationStatus.COMPLETED,
            created_at=created_at,
            completed_at=datetime.now(),
            scenario=scenario_config,
            battery=battery_config,
            results=results,
            comparison=comparison,
        )

        # Store result
        self._results[optimization_id] = result

        return result

    def get_optimization(
        self, optimization_id: UUID
    ) -> OptimizationResultResponse | None:
        """Get optimization result by ID."""
        return self._results.get(optimization_id)

    def list_optimizations(
        self, page: int = 1, page_size: int = 10
    ) -> tuple[list[OptimizationResultResponse], int]:
        """List all optimizations with pagination."""
        all_results = list(self._results.values())
        all_results.sort(key=lambda x: x.created_at, reverse=True)

        total = len(all_results)
        start = (page - 1) * page_size
        end = start + page_size

        return all_results[start:end], total

    def delete_optimization(self, optimization_id: UUID) -> bool:
        """Delete optimization by ID."""
        if optimization_id in self._results:
            del self._results[optimization_id]
            return True
        return False


# Global service instance
optimization_service = OptimizationService()
