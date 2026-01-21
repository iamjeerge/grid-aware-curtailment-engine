"""Tests for metrics module - KPI computation and performance analysis."""

from datetime import datetime, timedelta

import pytest

from src.domain.models import (
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
)
from src.metrics.analyzer import (
    PerformanceAnalyzer,
    compute_kpis_from_decisions,
    quick_kpi_summary,
)
from src.metrics.kpi import (
    BatteryMetrics,
    CurtailmentMetrics,
    GridComplianceMetrics,
    PerformanceSummary,
    RevenueMetrics,
    StrategyComparison,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dispatch_data():
    """Sample dispatch data for testing."""
    return {
        "generations": [100.0, 200.0, 300.0, 250.0, 150.0],
        "sold": [80.0, 150.0, 200.0, 200.0, 120.0],
        "stored": [20.0, 30.0, 50.0, 30.0, 20.0],
        "curtailed": [0.0, 20.0, 50.0, 20.0, 10.0],
        "prices": [50.0, 40.0, -20.0, 80.0, 120.0],
        "capacity_limits": [300.0, 300.0, 200.0, 300.0, 300.0],
        "ramp_limits": [100.0, 100.0, 100.0, 100.0, 100.0],
    }


@pytest.fixture
def sample_decisions(sample_dispatch_data):
    """Create sample OptimizationDecision objects."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    decisions = []
    data = sample_dispatch_data

    for i in range(len(data["sold"])):
        decision = OptimizationDecision(
            timestamp=base_time + timedelta(hours=i),
            scenario=ForecastScenario.P50,
            energy_sold_mw=data["sold"][i],
            energy_stored_mw=data["stored"][i],
            energy_curtailed_mw=data["curtailed"][i],
            generation_mw=data["generations"][i],
            battery_discharge_mw=0.0,
            resulting_soc_mwh=sum(data["stored"][: i + 1]),
            revenue_dollars=data["sold"][i] * data["prices"][i],
        )
        decisions.append(decision)

    return decisions


@pytest.fixture
def sample_forecasts():
    """Create sample GenerationForecast objects."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    generations = [100.0, 200.0, 300.0, 250.0, 150.0]
    forecasts = []

    for i, gen in enumerate(generations):
        forecast = GenerationForecast(
            timestamp=base_time + timedelta(hours=i),
            solar_mw_p50=gen * 0.8,
            wind_mw_p50=gen * 0.2,
        )
        forecasts.append(forecast)

    return forecasts


@pytest.fixture
def sample_prices():
    """Create sample MarketPrice objects."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    prices = [50.0, 40.0, -20.0, 80.0, 120.0]

    return [
        MarketPrice(
            timestamp=base_time + timedelta(hours=i),
            day_ahead_price=p,
            real_time_price=p,
        )
        for i, p in enumerate(prices)
    ]


@pytest.fixture
def sample_constraints():
    """Create sample GridConstraint objects."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    capacities = [300.0, 300.0, 200.0, 300.0, 300.0]

    return [
        GridConstraint(
            timestamp=base_time + timedelta(hours=i),
            max_export_mw=cap,
            max_ramp_up_mw_per_hour=100.0,
            max_ramp_down_mw_per_hour=100.0,
        )
        for i, cap in enumerate(capacities)
    ]


# =============================================================================
# CurtailmentMetrics Tests
# =============================================================================


class TestCurtailmentMetrics:
    """Tests for CurtailmentMetrics."""

    def test_curtailment_metrics_from_dispatch(self, sample_dispatch_data):
        """Test creating metrics from dispatch data."""
        data = sample_dispatch_data
        metrics = CurtailmentMetrics.from_dispatch(
            generations=data["generations"],
            curtailed=data["curtailed"],
            sold=data["sold"],
            stored=data["stored"],
        )

        assert metrics.total_generation_mwh == 1000.0
        assert metrics.total_curtailed_mwh == 100.0
        assert metrics.total_sold_mwh == 750.0
        assert metrics.total_stored_mwh == 150.0
        assert metrics.curtailment_rate == pytest.approx(0.1, rel=1e-5)

    def test_curtailment_metrics_with_baseline(self, sample_dispatch_data):
        """Test curtailment avoided calculation."""
        data = sample_dispatch_data
        baseline_curtailed = [10.0, 50.0, 100.0, 40.0, 20.0]  # 220 MWh total

        metrics = CurtailmentMetrics.from_dispatch(
            generations=data["generations"],
            curtailed=data["curtailed"],
            sold=data["sold"],
            stored=data["stored"],
            baseline_curtailed=baseline_curtailed,
        )

        # 220 - 100 = 120 MWh avoided
        assert metrics.curtailment_avoided_mwh == pytest.approx(120.0, rel=1e-5)
        # 120/220 = 54.5% avoided (as percentage 0-100)
        assert metrics.curtailment_avoided_pct == pytest.approx(54.5, rel=0.1)

    def test_zero_generation(self):
        """Test with zero generation."""
        metrics = CurtailmentMetrics.from_dispatch(
            generations=[0.0, 0.0, 0.0],
            curtailed=[0.0, 0.0, 0.0],
            sold=[0.0, 0.0, 0.0],
            stored=[0.0, 0.0, 0.0],
        )

        assert metrics.curtailment_rate == 0.0
        assert metrics.total_generation_mwh == 0.0


# =============================================================================
# RevenueMetrics Tests
# =============================================================================


class TestRevenueMetrics:
    """Tests for RevenueMetrics."""

    def test_revenue_metrics_from_dispatch(self, sample_dispatch_data):
        """Test revenue calculation from dispatch."""
        data = sample_dispatch_data
        metrics = RevenueMetrics.from_dispatch(
            sold=data["sold"],
            discharged=[0.0] * 5,
            prices=data["prices"],
            degradation_cost=500.0,
            penalty_cost=0.0,
        )

        # Revenue = 80*50 + 150*40 + 200*(-20) + 200*80 + 120*120
        # = 4000 + 6000 - 4000 + 16000 + 14400 = 36400
        assert metrics.total_revenue == pytest.approx(36400.0, rel=1e-5)
        assert metrics.degradation_cost == 500.0
        assert metrics.net_profit == pytest.approx(35900.0, rel=1e-5)

    def test_revenue_with_discharge(self, sample_dispatch_data):
        """Test revenue including battery discharge."""
        data = sample_dispatch_data
        discharged = [0.0, 0.0, 0.0, 20.0, 30.0]  # Discharge during high price

        metrics = RevenueMetrics.from_dispatch(
            sold=data["sold"],
            discharged=discharged,
            prices=data["prices"],
            degradation_cost=100.0,
        )

        # Discharge revenue = 20*80 + 30*120 = 1600 + 3600 = 5200
        assert metrics.revenue_from_discharge == pytest.approx(5200.0, rel=1e-5)
        assert metrics.total_revenue > 36400.0  # More than without discharge

    def test_revenue_uplift(self, sample_dispatch_data):
        """Test revenue uplift calculation."""
        data = sample_dispatch_data
        baseline_revenue = 30000.0

        metrics = RevenueMetrics.from_dispatch(
            sold=data["sold"],
            discharged=[0.0] * 5,
            prices=data["prices"],
            degradation_cost=0.0,
            baseline_revenue=baseline_revenue,
        )

        # Net profit = 36400, baseline = 30000
        # Uplift = (36400 - 30000) / 30000 * 100 = 21.3%
        assert metrics.revenue_uplift_pct == pytest.approx(21.3, rel=0.05)


# =============================================================================
# BatteryMetrics Tests
# =============================================================================


class TestBatteryMetrics:
    """Tests for BatteryMetrics."""

    def test_battery_metrics_from_dispatch(self):
        """Test battery metrics calculation."""
        charged = [20.0, 30.0, 50.0, 30.0, 20.0]
        discharged = [0.0, 0.0, 0.0, 20.0, 30.0]
        soc_values = [20.0, 50.0, 100.0, 110.0, 100.0]
        prices = [50.0, 40.0, -20.0, 80.0, 120.0]

        metrics = BatteryMetrics.from_dispatch(
            charged=charged,
            discharged=discharged,
            soc_values=soc_values,
            prices=prices,
            battery_capacity_mwh=500.0,
            degradation_cost_per_mwh=8.0,
        )

        assert metrics.total_charged_mwh == 150.0
        assert metrics.total_discharged_mwh == 50.0
        # Cycles = 50 / 500 = 0.1
        assert metrics.total_cycles == pytest.approx(0.1, rel=1e-5)
        # Avg SOC = mean([20, 50, 100, 110, 100]) = 76
        assert metrics.average_soc == pytest.approx(76.0, rel=1e-5)
        # Utilization = 76 / 500 = 0.152
        assert metrics.utilization_rate == pytest.approx(0.152, rel=0.01)

    def test_battery_arbitrage_value(self):
        """Test battery arbitrage value calculation."""
        charged = [50.0, 0.0]
        discharged = [0.0, 45.0]  # ~90% round-trip efficiency
        soc_values = [47.5, 0.0]  # After 95% charge eff
        prices = [20.0, 100.0]  # Charge at low, discharge at high

        metrics = BatteryMetrics.from_dispatch(
            charged=charged,
            discharged=discharged,
            soc_values=soc_values,
            prices=prices,
            battery_capacity_mwh=100.0,
        )

        # Arbitrage = (100 - 20) * 45 = 3600
        assert metrics.arbitrage_value > 0


# =============================================================================
# GridComplianceMetrics Tests
# =============================================================================


class TestGridComplianceMetrics:
    """Tests for GridComplianceMetrics."""

    def test_compliance_no_violations(self):
        """Test compliance with no violations."""
        # All values well within limits
        sold = [80.0, 100.0, 120.0, 110.0, 90.0]
        capacity_limits = [300.0, 300.0, 300.0, 300.0, 300.0]
        ramp_limits = [100.0, 100.0, 100.0, 100.0, 100.0]

        metrics = GridComplianceMetrics.from_dispatch(
            sold=sold,
            capacity_limits=capacity_limits,
            ramp_limits=ramp_limits,
        )

        assert metrics.compliance_rate == 1.0
        assert metrics.violation_count == 0
        assert metrics.capacity_violations == 0

    def test_compliance_with_capacity_violations(self):
        """Test compliance with capacity violations."""
        sold = [100.0, 150.0, 260.0, 180.0]  # 260 exceeds 200 limit
        capacity_limits = [150.0, 200.0, 200.0, 200.0]
        ramp_limits = [100.0, 100.0, 100.0, 100.0]

        metrics = GridComplianceMetrics.from_dispatch(
            sold=sold,
            capacity_limits=capacity_limits,
            ramp_limits=ramp_limits,
        )

        assert metrics.capacity_violations == 1
        assert metrics.max_violation_mw == pytest.approx(60.0, rel=1e-5)
        assert metrics.compliance_rate < 1.0

    def test_compliance_with_ramp_violations(self):
        """Test compliance with ramp rate violations."""
        sold = [100.0, 220.0, 150.0, 200.0]  # 100->220 exceeds 50 MW/h ramp
        capacity_limits = [300.0, 300.0, 300.0, 300.0]
        ramp_limits = [50.0, 50.0, 50.0, 50.0]

        metrics = GridComplianceMetrics.from_dispatch(
            sold=sold,
            capacity_limits=capacity_limits,
            ramp_limits=ramp_limits,
        )

        assert metrics.ramp_violations > 0


# =============================================================================
# PerformanceSummary Tests
# =============================================================================


class TestPerformanceSummary:
    """Tests for PerformanceSummary."""

    def test_overall_score_perfect(self):
        """Test perfect score calculation."""
        summary = PerformanceSummary(
            curtailment=CurtailmentMetrics(
                total_generation_mwh=1000.0,
                total_curtailed_mwh=0.0,  # 0% curtailment
                total_sold_mwh=800.0,
                total_stored_mwh=200.0,
            ),
            revenue=RevenueMetrics(
                total_revenue=100000.0,
                net_profit=95000.0,
                revenue_uplift_pct=50.0,  # Good uplift
            ),
            battery=BatteryMetrics(
                total_charged_mwh=200.0,
                total_discharged_mwh=190.0,
                total_cycles=0.4,
                utilization_rate=0.5,
                charge_efficiency_realized=0.95,
            ),
            grid_compliance=GridComplianceMetrics(
                compliance_rate=1.0,  # 100% compliance
                violation_count=0,
                total_timesteps=24,
                compliant_timesteps=24,
            ),
        )

        # Should be high score (0 curtailment, 100% compliance)
        # Score = 30*1.0 + 25*1.0 + 25*(100/100) + 10*0.5 + 10*0.95
        # = 30 + 25 + 25 + 5 + 9.5 = 94.5
        assert summary.overall_score > 80

    def test_overall_score_poor(self):
        """Test poor score calculation."""
        summary = PerformanceSummary(
            curtailment=CurtailmentMetrics(
                total_generation_mwh=1000.0,
                total_curtailed_mwh=400.0,  # 40% curtailment
                total_sold_mwh=400.0,
                total_stored_mwh=200.0,
            ),
            revenue=RevenueMetrics(
                total_revenue=40000.0,
                net_profit=35000.0,
            ),
            battery=BatteryMetrics(
                total_charged_mwh=200.0,
                total_discharged_mwh=50.0,
                total_cycles=0.1,
            ),
            grid_compliance=GridComplianceMetrics(
                compliance_rate=0.5,  # 50% compliance
                violation_count=10,
            ),
        )

        # Should be lower score
        assert summary.overall_score < 70


# =============================================================================
# StrategyComparison Tests
# =============================================================================


class TestStrategyComparison:
    """Tests for StrategyComparison."""

    def test_best_strategy_selection(self):
        """Test best strategy is correctly identified."""
        comparison = StrategyComparison()

        # Add strategies with different scores
        for name, curt_rate, compliance in [
            ("naive", 0.3, 0.8),
            ("milp", 0.1, 1.0),
            ("rl", 0.15, 0.95),
        ]:
            comparison.strategies[name] = PerformanceSummary(
                strategy_name=name,
                curtailment=CurtailmentMetrics(
                    total_generation_mwh=1000.0,
                    total_curtailed_mwh=1000.0 * curt_rate,
                    total_sold_mwh=1000.0 * (1 - curt_rate),
                    total_stored_mwh=0.0,
                ),
                grid_compliance=GridComplianceMetrics(
                    compliance_rate=compliance,
                    total_timesteps=24,
                    compliant_timesteps=int(24 * compliance),
                ),
            )

        # MILP should be best (lowest curtailment, full compliance)
        assert comparison.best_strategy == "milp"

    def test_strategy_ranking(self):
        """Test strategies are correctly ranked."""
        comparison = StrategyComparison()

        # Create strategies with different curtailment rates and efficiency
        # Score inversely proportional to curtailment (and compliance affects it)
        for name, curt_rate, util_rate, eff in [
            ("naive", 0.4, 0.1, 0.5),
            ("milp", 0.05, 0.5, 0.95),
            ("rl", 0.15, 0.3, 0.8),
        ]:
            comparison.strategies[name] = PerformanceSummary(
                strategy_name=name,
                curtailment=CurtailmentMetrics(
                    total_generation_mwh=1000.0,
                    total_curtailed_mwh=1000.0 * curt_rate,
                    total_sold_mwh=1000.0 * (1 - curt_rate),
                ),
                revenue=RevenueMetrics(
                    total_revenue=100000.0 * (1 - curt_rate),
                    net_profit=100000.0 * (1 - curt_rate),
                    revenue_uplift_pct=20.0,  # Give some uplift
                ),
                battery=BatteryMetrics(
                    utilization_rate=util_rate,
                    charge_efficiency_realized=eff,
                ),
                grid_compliance=GridComplianceMetrics(
                    compliance_rate=1.0,
                    total_timesteps=24,
                    compliant_timesteps=24,
                ),
            )

        ranking = comparison.ranking
        assert ranking[0] == "milp"
        assert ranking[1] == "rl"
        assert ranking[2] == "naive"

    def test_compute_deltas(self):
        """Test delta computation between strategies."""
        comparison = StrategyComparison()

        comparison.strategies["naive"] = PerformanceSummary(
            strategy_name="naive",
            curtailment=CurtailmentMetrics(
                total_generation_mwh=1000.0,
                total_curtailed_mwh=300.0,
                total_sold_mwh=700.0,
                curtailment_rate=0.3,  # 30%
            ),
            revenue=RevenueMetrics(total_revenue=50000.0, net_profit=50000.0),
        )
        comparison.strategies["milp"] = PerformanceSummary(
            strategy_name="milp",
            curtailment=CurtailmentMetrics(
                total_generation_mwh=1000.0,
                total_curtailed_mwh=100.0,
                total_sold_mwh=900.0,
                curtailment_rate=0.1,  # 10%
            ),
            revenue=RevenueMetrics(total_revenue=75000.0, net_profit=75000.0),
        )

        # Use get_delta instead of compute_deltas
        deltas = comparison.get_delta("milp", "naive")

        # milp - naive: curtailment 0.1 - 0.3 = -0.2 = -20% points
        assert deltas["curtailment_delta_pct"] == pytest.approx(-20.0, rel=0.1)
        assert deltas["revenue_delta"] == pytest.approx(25000.0, rel=1e-5)


# =============================================================================
# PerformanceAnalyzer Tests
# =============================================================================


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    def test_analyze_decisions(
        self, sample_decisions, sample_forecasts, sample_prices, sample_constraints
    ):
        """Test analyzing optimization decisions."""
        analyzer = PerformanceAnalyzer()

        summary = analyzer.analyze_decisions(
            decisions=sample_decisions,
            forecasts=sample_forecasts,
            prices=sample_prices,
            constraints=sample_constraints,
            strategy_name="test_strategy",
        )

        assert summary.strategy_name == "test_strategy"
        assert summary.horizon_hours == 5
        assert summary.curtailment.total_generation_mwh > 0
        assert summary.revenue.total_revenue > 0
        assert summary.grid_compliance.compliance_rate > 0

    def test_analyze_empty_decisions(
        self, sample_forecasts, sample_prices, sample_constraints
    ):
        """Test analyzing empty decisions."""
        analyzer = PerformanceAnalyzer()

        summary = analyzer.analyze_decisions(
            decisions=[],
            forecasts=sample_forecasts,
            prices=sample_prices,
            constraints=sample_constraints,
        )

        # Empty decisions should have zero generation
        assert summary.curtailment.total_generation_mwh == 0

    def test_compare_strategies(
        self, sample_decisions, sample_forecasts, sample_prices, sample_constraints
    ):
        """Test comparing multiple strategies."""
        analyzer = PerformanceAnalyzer()

        # Create "naive" decisions with higher curtailment
        naive_decisions = []
        for d in sample_decisions:
            naive = OptimizationDecision(
                timestamp=d.timestamp,
                scenario=ForecastScenario.P50,
                energy_sold_mw=d.energy_sold_mw * 0.8,
                energy_stored_mw=d.energy_stored_mw * 0.5,
                energy_curtailed_mw=d.generation_mw
                - d.energy_sold_mw * 0.8
                - d.energy_stored_mw * 0.5,
                generation_mw=d.generation_mw,
                battery_discharge_mw=0.0,
                resulting_soc_mwh=d.resulting_soc_mwh * 0.5,
                revenue_dollars=d.revenue_dollars * 0.8,
            )
            naive_decisions.append(naive)

        comparison = analyzer.compare_strategies(
            strategy_decisions={
                "naive": naive_decisions,
                "optimized": sample_decisions,
            },
            forecasts=sample_forecasts,
            prices=sample_prices,
            constraints=sample_constraints,
            baseline_strategy="naive",
        )

        assert "naive" in comparison.strategies
        assert "optimized" in comparison.strategies
        assert comparison.best_strategy is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_kpis_from_decisions(
        self, sample_decisions, sample_prices, sample_constraints
    ):
        """Test KPI computation convenience function."""
        kpis = compute_kpis_from_decisions(
            decisions=sample_decisions,
            prices=sample_prices,
            constraints=sample_constraints,
        )

        assert "curtailment_rate" in kpis
        assert "revenue" in kpis
        assert "battery_cycles" in kpis
        assert "compliance_rate" in kpis
        assert "overall_score" in kpis

    def test_quick_kpi_summary(self, sample_dispatch_data):
        """Test quick KPI summary function."""
        data = sample_dispatch_data

        kpis = quick_kpi_summary(
            sold=data["sold"],
            stored=data["stored"],
            curtailed=data["curtailed"],
            prices=data["prices"],
            constraints=data["capacity_limits"],
        )

        assert kpis["total_generation_mwh"] == 1000.0
        assert kpis["total_sold_mwh"] == 750.0
        assert kpis["curtailment_rate"] == pytest.approx(0.1, rel=1e-5)
        assert kpis["revenue"] == pytest.approx(36400.0, rel=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_duck_curve_scenario(self):
        """Test metrics with duck curve scenario."""
        # Simulate duck curve: high solar midday, grid constrained
        base_time = datetime(2024, 1, 1, 6, 0, 0)

        # Generation peaks at noon
        generations = [50, 150, 400, 600, 550, 400, 200, 50, 30, 20]
        # Grid constraint limits midday
        capacity_limits = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
        # Negative prices midday, spike evening
        prices = [30, 35, 25, 10, -15, -25, 40, 80, 140, 100]

        # Optimal: store during negative prices, sell during high prices
        sold = [50, 150, 300, 300, 300, 300, 200, 50, 30, 20]
        stored = [0, 0, 100, 200, 150, 100, 0, 0, 0, 0]
        discharged = [0, 0, 0, 0, 0, 0, 0, 50, 70, 30]
        curtailed = [0, 0, 0, 100, 100, 0, 0, 0, 0, 0]

        # Naive: just sell what you can
        naive_sold = [50, 150, 300, 300, 300, 300, 200, 50, 30, 20]
        naive_curtailed = [0, 0, 100, 300, 250, 100, 0, 0, 0, 0]

        # Create decisions
        decisions = []
        naive_decisions = []
        forecasts = []
        prices_obj = []
        constraints = []

        soc = 0.0
        for i in range(len(generations)):
            t = base_time + timedelta(hours=i)

            soc += stored[i] * 0.95
            soc -= discharged[i] / 0.95

            decisions.append(
                OptimizationDecision(
                    timestamp=t,
                    scenario=ForecastScenario.P50,
                    energy_sold_mw=sold[i],
                    energy_stored_mw=stored[i],
                    energy_curtailed_mw=curtailed[i],
                    generation_mw=generations[i],
                    battery_discharge_mw=discharged[i],
                    resulting_soc_mwh=max(0, soc),
                    revenue_dollars=sold[i] * prices[i],
                )
            )

            naive_decisions.append(
                OptimizationDecision(
                    timestamp=t,
                    scenario=ForecastScenario.P50,
                    energy_sold_mw=naive_sold[i],
                    energy_stored_mw=0.0,
                    energy_curtailed_mw=naive_curtailed[i],
                    generation_mw=generations[i],
                    battery_discharge_mw=0.0,
                    resulting_soc_mwh=0.0,
                    revenue_dollars=naive_sold[i] * prices[i],
                )
            )

            forecasts.append(
                GenerationForecast(
                    timestamp=t,
                    solar_mw_p50=generations[i] * 0.8,
                    wind_mw_p50=generations[i] * 0.2,
                )
            )

            prices_obj.append(
                MarketPrice(
                    timestamp=t,
                    day_ahead_price=prices[i],
                    real_time_price=prices[i],
                )
            )

            constraints.append(
                GridConstraint(
                    timestamp=t,
                    max_export_mw=capacity_limits[i],
                    max_ramp_up_mw_per_hour=100.0,
                    max_ramp_down_mw_per_hour=100.0,
                )
            )

        # Analyze both strategies
        analyzer = PerformanceAnalyzer()

        comparison = analyzer.compare_strategies(
            strategy_decisions={
                "naive": naive_decisions,
                "optimized": decisions,
            },
            forecasts=forecasts,
            prices=prices_obj,
            constraints=constraints,
            baseline_strategy="naive",
        )

        # Optimized should beat naive
        naive_summary = comparison.strategies["naive"]
        opt_summary = comparison.strategies["optimized"]

        # Optimized should have less curtailment
        assert (
            opt_summary.curtailment.total_curtailed_mwh
            < naive_summary.curtailment.total_curtailed_mwh
        )

        # Optimized should have better revenue (from battery arbitrage)
        assert opt_summary.revenue.total_revenue > naive_summary.revenue.total_revenue

        # Best strategy should be optimized
        assert comparison.best_strategy == "optimized"
