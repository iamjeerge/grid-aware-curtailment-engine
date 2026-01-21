"""Tests for the visualization module.

Tests cover:
- TimelineVisualizer for optimization decision plots
- ComparisonDashboard for strategy comparison charts
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.domain import (
    BatteryConfig,
    BatteryState,
    ForecastScenario,
    GridConstraint,
    MarketPrice,
    OptimizationDecision,
)
from src.metrics.kpi import (
    BatteryMetrics,
    CurtailmentMetrics,
    GridComplianceMetrics,
    PerformanceSummary,
    RevenueMetrics,
    StrategyComparison,
)
from src.visualization import (
    ComparisonDashboard,
    ComparisonPlotConfig,
    TimelinePlotConfig,
    TimelineVisualizer,
    create_comparison_dashboard,
    create_dispatch_timeline,
)

if TYPE_CHECKING:
    pass

# Use non-interactive backend for testing
matplotlib.use("Agg")


# --- Fixtures ---


@pytest.fixture
def timesteps() -> list[int]:
    """Create timesteps for a 24-hour day."""
    return list(range(24))


@pytest.fixture
def base_time() -> datetime:
    """Create base timestamp."""
    return datetime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def sample_decisions(
    timesteps: list[int], base_time: datetime
) -> list[OptimizationDecision]:
    """Create sample optimization decisions for testing."""
    np.random.seed(42)  # For reproducibility
    decisions = []
    soc = 250.0  # Start at 50% of 500 MWh capacity

    for t in timesteps:
        # Simulate duck curve pattern - solar hours have higher generation
        generation = 400 + 200 * np.sin((t - 6) * np.pi / 12) if 6 <= t <= 18 else 50

        # Grid constraint
        grid_limit = 300

        # Price pattern (high morning/evening, low midday)
        if 7 <= t <= 9 or 17 <= t <= 20:
            price = 80 + 20 * np.random.random()
        elif 11 <= t <= 14:
            price = -10 + 20 * np.random.random()
        else:
            price = 40 + 10 * np.random.random()

        # Optimize dispatch
        sell = min(generation, grid_limit)
        if price < 20 and generation > sell:
            store = min(generation - sell, 100)  # Battery limit
            curtail = generation - sell - store
            soc = min(soc + store * 0.95, 500)
        else:
            store = 0
            curtail = max(0, generation - sell)

        decisions.append(
            OptimizationDecision(
                timestamp=base_time + timedelta(hours=t),
                scenario=ForecastScenario.P50,
                generation_mw=generation,
                energy_sold_mw=sell,
                energy_stored_mw=store,
                energy_curtailed_mw=max(0, curtail),
                resulting_soc_mwh=soc,
                revenue_dollars=sell * price,
            )
        )

    return decisions


@pytest.fixture
def sample_prices(timesteps: list[int], base_time: datetime) -> list[MarketPrice]:
    """Create sample market prices."""
    return [
        MarketPrice(
            timestamp=base_time + timedelta(hours=t),
            day_ahead_price=40 + 20 * np.sin(t * np.pi / 12),
            real_time_price=45 + 25 * np.sin(t * np.pi / 12),
        )
        for t in timesteps
    ]


@pytest.fixture
def sample_constraints(
    timesteps: list[int], base_time: datetime
) -> list[GridConstraint]:
    """Create sample grid constraints."""
    return [
        GridConstraint(
            timestamp=base_time + timedelta(hours=t),
            max_export_mw=300,
            max_ramp_up_mw_per_hour=50,
            max_ramp_down_mw_per_hour=50,
        )
        for t in timesteps
    ]


@pytest.fixture
def sample_battery_states(
    timesteps: list[int], base_time: datetime
) -> list[BatteryState]:
    """Create sample battery states."""
    config = BatteryConfig(
        capacity_mwh=500,
        max_power_mw=150,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    states = []
    soc = 250  # Start at 50%
    for t in timesteps:
        if 10 <= t <= 14:
            # Charging during midday
            charge = 100
            soc = min(soc + charge * 0.95, 500)
        elif 17 <= t <= 20:
            # Discharging during evening
            discharge = 100
            soc = max(soc - discharge / 0.95, 0)
        else:
            charge = 0
            discharge = 0

        states.append(
            BatteryState(
                timestamp=base_time + timedelta(hours=t),
                soc_mwh=soc,
                charge_power_mw=charge if 10 <= t <= 14 else 0,
                discharge_power_mw=discharge if 17 <= t <= 20 else 0,
                config=config,
            )
        )

    return states


@pytest.fixture
def sample_performance_summary() -> PerformanceSummary:
    """Create a sample performance summary."""
    return PerformanceSummary(
        strategy_name="MILP",
        horizon_hours=24,
        curtailment=CurtailmentMetrics(
            total_generation_mwh=2000.0,
            total_curtailed_mwh=150.0,
            total_sold_mwh=1600.0,
            total_stored_mwh=250.0,
            curtailment_rate=0.08,
            curtailment_avoided_mwh=350.0,
            curtailment_avoided_pct=70.0,
        ),
        revenue=RevenueMetrics(
            total_revenue=75000.0,
            revenue_from_sales=72000.0,
            revenue_from_discharge=5000.0,
            total_costs=2000.0,
            degradation_cost=2000.0,
            net_profit=73000.0,
            revenue_uplift_pct=25.0,
            average_price_captured=45.0,
        ),
        battery=BatteryMetrics(
            total_charged_mwh=800.0,
            total_discharged_mwh=750.0,
            total_cycles=1.5,
            utilization_rate=0.65,
            average_soc=325.0,
            min_soc=100.0,
            max_soc=500.0,
            soc_range=400.0,
            throughput_mwh=1550.0,
            degradation_cost=2000.0,
            charge_efficiency_realized=0.94,
            arbitrage_value=5000.0,
        ),
        grid_compliance=GridComplianceMetrics(
            total_timesteps=24,
            compliant_timesteps=24,
            violation_count=0,
            compliance_rate=1.0,
            max_violation_mw=0.0,
            total_violation_mwh=0.0,
            ramp_violations=0,
            capacity_violations=0,
        ),
    )


@pytest.fixture
def sample_strategy_comparison(
    sample_performance_summary: PerformanceSummary,
) -> StrategyComparison:
    """Create a sample strategy comparison."""
    # Create naive summary (worse performance)
    naive = PerformanceSummary(
        strategy_name="Naive",
        horizon_hours=24,
        curtailment=CurtailmentMetrics(
            total_generation_mwh=2000.0,
            total_curtailed_mwh=500.0,
            total_sold_mwh=1500.0,
            total_stored_mwh=0.0,
            curtailment_rate=0.32,
            curtailment_avoided_mwh=0.0,
            curtailment_avoided_pct=0.0,
        ),
        revenue=RevenueMetrics(
            total_revenue=45000.0,
            revenue_from_sales=45000.0,
            revenue_from_discharge=0.0,
            total_costs=3000.0,
            degradation_cost=0.0,
            penalty_cost=3000.0,
            net_profit=42000.0,
            revenue_uplift_pct=0.0,
            average_price_captured=35.0,
        ),
        battery=BatteryMetrics(
            total_charged_mwh=0.0,
            total_discharged_mwh=0.0,
            total_cycles=0.0,
            utilization_rate=0.0,
            average_soc=0.0,
            min_soc=0.0,
            max_soc=0.0,
            soc_range=0.0,
            throughput_mwh=0.0,
            degradation_cost=0.0,
            charge_efficiency_realized=0.0,
            arbitrage_value=0.0,
        ),
        grid_compliance=GridComplianceMetrics(
            total_timesteps=24,
            compliant_timesteps=20,
            violation_count=5,
            compliance_rate=0.85,
            max_violation_mw=75.0,
            total_violation_mwh=150.0,
            ramp_violations=2,
            capacity_violations=3,
        ),
    )

    # Create RL summary (good but not optimal)
    rl = PerformanceSummary(
        strategy_name="RL",
        horizon_hours=24,
        curtailment=CurtailmentMetrics(
            total_generation_mwh=2000.0,
            total_curtailed_mwh=200.0,
            total_sold_mwh=1550.0,
            total_stored_mwh=250.0,
            curtailment_rate=0.12,
            curtailment_avoided_mwh=300.0,
            curtailment_avoided_pct=60.0,
        ),
        revenue=RevenueMetrics(
            total_revenue=68000.0,
            revenue_from_sales=65000.0,
            revenue_from_discharge=4500.0,
            total_costs=1500.0,
            degradation_cost=1500.0,
            penalty_cost=0.0,
            net_profit=67000.0,
            revenue_uplift_pct=15.0,
            average_price_captured=42.0,
        ),
        battery=BatteryMetrics(
            total_charged_mwh=700.0,
            total_discharged_mwh=650.0,
            total_cycles=1.3,
            utilization_rate=0.55,
            average_soc=275.0,
            min_soc=100.0,
            max_soc=450.0,
            soc_range=350.0,
            throughput_mwh=1350.0,
            degradation_cost=1500.0,
            charge_efficiency_realized=0.93,
            arbitrage_value=4500.0,
        ),
        grid_compliance=GridComplianceMetrics(
            total_timesteps=24,
            compliant_timesteps=23,
            violation_count=1,
            compliance_rate=0.96,
            max_violation_mw=10.0,
            total_violation_mwh=10.0,
            ramp_violations=0,
            capacity_violations=1,
        ),
    )

    return StrategyComparison(
        strategies={
            "Naive": naive,
            "MILP": sample_performance_summary,
            "RL": rl,
        },
    )


# --- TimelineVisualizer Tests ---


class TestTimelineVisualizer:
    """Tests for the TimelineVisualizer class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        viz = TimelineVisualizer()
        assert viz.config is not None
        assert isinstance(viz.config, TimelinePlotConfig)

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = TimelinePlotConfig(figsize=(12, 8), dpi=150)
        viz = TimelineVisualizer(config=config)
        assert viz.config.figsize == (12, 8)
        assert viz.config.dpi == 150

    def test_plot_energy_dispatch(
        self, sample_decisions: list[OptimizationDecision]
    ) -> None:
        """Test energy dispatch plot creation."""
        viz = TimelineVisualizer()
        ax = viz.plot_energy_dispatch(sample_decisions)

        assert ax is not None
        assert len(ax.collections) > 0 or len(ax.patches) > 0  # Has plot elements
        plt.close()

    def test_plot_battery_soc(
        self, sample_decisions: list[OptimizationDecision]
    ) -> None:
        """Test battery SOC plot creation."""
        viz = TimelineVisualizer()
        ax = viz.plot_battery_soc(sample_decisions)

        assert ax is not None
        assert len(ax.lines) > 0
        plt.close()

    def test_plot_prices(
        self,
        sample_prices: list[MarketPrice],
    ) -> None:
        """Test price plot creation."""
        viz = TimelineVisualizer()
        ax = viz.plot_prices(sample_prices)

        assert ax is not None
        # Should have bars
        plt.close()

    def test_plot_grid_constraints(
        self,
        sample_decisions: list[OptimizationDecision],
        sample_constraints: list[GridConstraint],
    ) -> None:
        """Test grid constraints plot creation."""
        viz = TimelineVisualizer()
        ax = viz.plot_grid_constraints(sample_decisions, sample_constraints)

        assert ax is not None
        plt.close()

    def test_plot_revenue_breakdown(
        self,
        sample_decisions: list[OptimizationDecision],
        sample_prices: list[MarketPrice],
    ) -> None:
        """Test revenue breakdown plot creation."""
        viz = TimelineVisualizer()
        ax = viz.plot_revenue_breakdown(sample_decisions, sample_prices)

        assert ax is not None
        plt.close()

    def test_plot_full_dashboard(
        self,
        sample_decisions: list[OptimizationDecision],
        sample_prices: list[MarketPrice],
        sample_constraints: list[GridConstraint],
    ) -> None:
        """Test full dashboard creation."""
        viz = TimelineVisualizer()
        fig = viz.plot_full_dashboard(
            decisions=sample_decisions,
            prices=sample_prices,
            constraints=sample_constraints,
            title="Test Dashboard",
        )

        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
        plt.close(fig)

    def test_plot_with_custom_ax(
        self, sample_decisions: list[OptimizationDecision]
    ) -> None:
        """Test plotting to provided axes."""
        viz = TimelineVisualizer()
        fig, ax = plt.subplots()

        returned_ax = viz.plot_energy_dispatch(sample_decisions, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_create_dispatch_timeline_convenience(
        self,
        sample_decisions: list[OptimizationDecision],
        sample_prices: list[MarketPrice],
        sample_constraints: list[GridConstraint],
    ) -> None:
        """Test convenience function."""
        fig = create_dispatch_timeline(
            decisions=sample_decisions,
            prices=sample_prices,
            constraints=sample_constraints,
            title="Quick Timeline",
        )

        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)


# --- ComparisonDashboard Tests ---


class TestComparisonDashboard:
    """Tests for the ComparisonDashboard class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        dashboard = ComparisonDashboard()
        assert dashboard.config is not None
        assert isinstance(dashboard.config, ComparisonPlotConfig)

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ComparisonPlotConfig(figsize=(16, 12), dpi=200)
        dashboard = ComparisonDashboard(config=config)
        assert dashboard.config.figsize == (16, 12)
        assert dashboard.config.dpi == 200

    def test_plot_revenue_comparison(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test revenue comparison plot."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_revenue_comparison(sample_strategy_comparison)

        assert ax is not None
        assert len(ax.patches) > 0  # Has bars
        plt.close()

    def test_plot_curtailment_comparison(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test curtailment comparison plot."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_curtailment_comparison(sample_strategy_comparison)

        assert ax is not None
        assert len(ax.patches) > 0
        plt.close()

    def test_plot_compliance_comparison(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test compliance comparison plot."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_compliance_comparison(sample_strategy_comparison)

        assert ax is not None
        plt.close()

    def test_plot_overall_scores(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test overall scores plot."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_overall_scores(sample_strategy_comparison)

        assert ax is not None
        plt.close()

    def test_plot_radar_comparison(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test radar chart creation."""
        dashboard = ComparisonDashboard()
        fig = dashboard.plot_radar_comparison(sample_strategy_comparison)

        assert fig is not None
        plt.close(fig)

    def test_plot_metrics_heatmap(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test metrics heatmap."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_metrics_heatmap(sample_strategy_comparison)

        assert ax is not None
        plt.close()

    def test_plot_full_comparison(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test full comparison dashboard."""
        dashboard = ComparisonDashboard()
        fig = dashboard.plot_full_comparison(
            sample_strategy_comparison, title="Full Comparison"
        )

        assert fig is not None
        assert len(fig.axes) >= 5  # Multiple subplots
        plt.close(fig)

    def test_plot_improvement_waterfall(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test waterfall chart for improvement."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_improvement_waterfall(
            sample_strategy_comparison, baseline="Naive", target="MILP"
        )

        assert ax is not None
        plt.close()

    def test_create_comparison_dashboard_convenience(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test convenience function."""
        fig = create_comparison_dashboard(
            sample_strategy_comparison, title="Quick Comparison"
        )

        assert fig is not None
        plt.close(fig)


# --- Color Configuration Tests ---


class TestPlotConfigs:
    """Tests for plot configuration classes."""

    def test_timeline_config_defaults(self) -> None:
        """Test TimelinePlotConfig default values."""
        config = TimelinePlotConfig()

        assert config.figsize == (14, 10)
        assert config.dpi == 100
        assert "sold" in config.colors
        assert "curtailed" in config.colors
        assert "stored" in config.colors

    def test_comparison_config_defaults(self) -> None:
        """Test ComparisonPlotConfig default values."""
        config = ComparisonPlotConfig()

        assert config.figsize == (14, 10)
        assert config.dpi == 100
        assert "naive" in config.colors
        assert "milp" in config.colors
        assert "rl" in config.colors

    def test_config_custom_colors(self) -> None:
        """Test custom color configuration."""
        custom_colors = {"sold": "#ff0000", "curtailed": "#00ff00", "stored": "#0000ff"}
        config = TimelinePlotConfig(colors=custom_colors)

        assert config.colors["sold"] == "#ff0000"
        assert config.colors["curtailed"] == "#00ff00"


# --- Edge Case Tests ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_decisions_energy_dispatch(self) -> None:
        """Test handling of empty decision list."""
        viz = TimelineVisualizer()
        ax = viz.plot_energy_dispatch([])

        assert ax is not None
        plt.close()

    def test_single_decision(self) -> None:
        """Test handling of single decision."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        decision = OptimizationDecision(
            timestamp=base_time,
            scenario=ForecastScenario.P50,
            generation_mw=100,
            energy_sold_mw=80,
            energy_stored_mw=10,
            energy_curtailed_mw=10,
            resulting_soc_mwh=100,
            revenue_dollars=4000,
        )
        viz = TimelineVisualizer()
        ax = viz.plot_energy_dispatch([decision])

        assert ax is not None
        plt.close()

    def test_negative_prices(self, base_time: datetime) -> None:
        """Test handling of negative prices."""
        # Create prices with negatives
        prices = [
            MarketPrice(
                timestamp=base_time + timedelta(hours=t),
                day_ahead_price=-20 if 10 <= t <= 14 else 50,
                real_time_price=-20 if 10 <= t <= 14 else 50,
            )
            for t in range(24)
        ]
        viz = TimelineVisualizer()
        ax = viz.plot_prices(prices)

        assert ax is not None
        plt.close()

    def test_missing_strategy_in_waterfall(
        self, sample_strategy_comparison: StrategyComparison
    ) -> None:
        """Test waterfall with missing strategy."""
        dashboard = ComparisonDashboard()
        ax = dashboard.plot_improvement_waterfall(
            sample_strategy_comparison,
            baseline="NonExistent",
            target="MILP",
        )

        assert ax is not None
        plt.close()


# --- Integration Tests ---


class TestIntegration:
    """Integration tests for visualization workflows."""

    def test_timeline_to_file(
        self,
        sample_decisions: list[OptimizationDecision],
        sample_prices: list[MarketPrice],
        sample_constraints: list[GridConstraint],
        tmp_path,
    ) -> None:
        """Test saving timeline to file."""
        viz = TimelineVisualizer()
        save_path = tmp_path / "timeline.png"

        fig = viz.plot_full_dashboard(
            decisions=sample_decisions,
            prices=sample_prices,
            constraints=sample_constraints,
            title="Timeline Test",
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)

    def test_comparison_to_file(
        self,
        sample_strategy_comparison: StrategyComparison,
        tmp_path,
    ) -> None:
        """Test saving comparison dashboard to file."""
        dashboard = ComparisonDashboard()
        save_path = tmp_path / "comparison.png"

        fig = dashboard.plot_full_comparison(
            sample_strategy_comparison,
            title="Comparison Test",
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)

    def test_radar_to_file(
        self,
        sample_strategy_comparison: StrategyComparison,
        tmp_path,
    ) -> None:
        """Test saving radar chart to file."""
        dashboard = ComparisonDashboard()
        save_path = tmp_path / "radar.png"

        fig = dashboard.plot_radar_comparison(
            sample_strategy_comparison,
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)
