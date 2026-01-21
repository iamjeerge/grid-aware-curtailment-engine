"""Tests for the reporting module.

Tests cover:
- ReportGenerator for PDF report creation
- Report sections and content
- HTML report generation
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import pytest

from src.metrics.kpi import (
    BatteryMetrics,
    CurtailmentMetrics,
    GridComplianceMetrics,
    PerformanceSummary,
    RevenueMetrics,
    StrategyComparison,
)
from src.reporting import (
    ReportConfig,
    ReportGenerator,
    ReportSection,
    generate_executive_report,
)
from src.uncertainty.stress_testing import StressTestConfig, StressTestResult

if TYPE_CHECKING:
    pass

# Use non-interactive backend for testing
matplotlib.use("Agg")


# --- Fixtures ---


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

    return StrategyComparison(
        strategies={
            "Naive": naive,
            "MILP": sample_performance_summary,
        },
    )


@pytest.fixture
def sample_stress_result() -> StressTestResult:
    """Create a sample stress test result."""
    config = StressTestConfig(n_simulations=100, seed=42)
    result = StressTestResult(
        config=config,
        n_runs_completed=100,
        total_runtime_seconds=10.0,
        mean_profit=70000.0,
        std_profit=8000.0,
        min_profit=45000.0,
        max_profit=95000.0,
        percentile_5=52000.0,
        percentile_25=62000.0,
        percentile_50=70000.0,
        percentile_75=78000.0,
        percentile_95=86000.0,
        mean_curtailment_rate=0.10,
        max_curtailment_rate=0.15,
        violation_rate=0.02,
        mean_violations_per_run=0.05,
    )
    return result


# --- ReportConfig Tests ---


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReportConfig()

        assert config.title == "Curtailment Optimization Report"
        assert config.author == "Grid Optimization Engine"
        assert config.page_size == "letter"
        assert config.dpi == 150
        assert config.include_assumptions is True
        assert config.include_risk_analysis is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ReportConfig(
            title="Custom Report",
            author="Test Author",
            company="Test Company",
            dpi=300,
        )

        assert config.title == "Custom Report"
        assert config.author == "Test Author"
        assert config.company == "Test Company"
        assert config.dpi == 300


# --- ReportSection Tests ---


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_basic_section(self) -> None:
        """Test creating a basic section."""
        section = ReportSection(
            title="Test Section",
            content="Test content",
        )

        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.figures == []
        assert section.table_data is None

    def test_section_with_figure(self) -> None:
        """Test creating a section with a figure."""
        fig, _ = plt.subplots()
        section = ReportSection(
            title="Test Section",
            figures=[fig],
        )

        assert len(section.figures) == 1
        plt.close(fig)


# --- ReportGenerator Tests ---


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        generator = ReportGenerator()

        assert generator.config is not None
        assert isinstance(generator.config, ReportConfig)
        assert generator.sections == []

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ReportConfig(title="Custom Report")
        generator = ReportGenerator(config=config)

        assert generator.config.title == "Custom Report"

    def test_add_section(self) -> None:
        """Test adding a section."""
        generator = ReportGenerator()
        section = ReportSection(title="Test", content="Content")

        generator.add_section(section)

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Test"

    def test_add_executive_summary(
        self,
        sample_strategy_comparison: StrategyComparison,
    ) -> None:
        """Test adding executive summary section."""
        generator = ReportGenerator()

        generator.add_executive_summary(sample_strategy_comparison)

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Executive Summary"
        assert "KEY FINDINGS" in generator.sections[0].content
        plt.close("all")

    def test_add_executive_summary_missing_strategy(self) -> None:
        """Test executive summary with missing strategies."""
        generator = ReportGenerator()
        comparison = StrategyComparison(strategies={})

        generator.add_executive_summary(comparison)

        assert len(generator.sections) == 1
        assert "not available" in generator.sections[0].content

    def test_add_problem_description(self) -> None:
        """Test adding problem description section."""
        generator = ReportGenerator()

        generator.add_problem_description(
            horizon_hours=48,
            battery_capacity_mwh=1000,
        )

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Problem Description"
        assert "48 hours" in generator.sections[0].content
        assert "1000 MWh" in generator.sections[0].content

    def test_add_methodology(self) -> None:
        """Test adding methodology section."""
        generator = ReportGenerator()

        generator.add_methodology()

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Methodology"
        assert "MILP" in generator.sections[0].content
        assert "RL" in generator.sections[0].content

    def test_add_financial_analysis(
        self,
        sample_performance_summary: PerformanceSummary,
    ) -> None:
        """Test adding financial analysis section."""
        generator = ReportGenerator()

        generator.add_financial_analysis(sample_performance_summary)

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Financial Analysis"
        assert "$73,000" in generator.sections[0].content  # Net profit
        assert len(generator.sections[0].figures) == 1
        plt.close("all")

    def test_add_financial_analysis_with_baseline(
        self,
        sample_performance_summary: PerformanceSummary,
    ) -> None:
        """Test financial analysis with baseline comparison."""
        generator = ReportGenerator()

        generator.add_financial_analysis(
            sample_performance_summary,
            baseline_revenue=50000.0,
        )

        assert "COMPARISON TO BASELINE" in generator.sections[0].content
        plt.close("all")

    def test_add_risk_analysis(
        self,
        sample_stress_result: StressTestResult,
    ) -> None:
        """Test adding risk analysis section."""
        generator = ReportGenerator()

        generator.add_risk_analysis(sample_stress_result)

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Risk Analysis"
        assert "100" in generator.sections[0].content  # n_simulations
        assert "Sharpe Ratio" in generator.sections[0].content
        assert len(generator.sections[0].figures) == 1
        plt.close("all")

    def test_add_assumptions(self) -> None:
        """Test adding assumptions section."""
        generator = ReportGenerator()

        generator.add_assumptions()

        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Assumptions & Limitations"
        assert "GRID ASSUMPTIONS" in generator.sections[0].content
        assert "BATTERY ASSUMPTIONS" in generator.sections[0].content

    def test_generate_pdf(
        self,
        sample_strategy_comparison: StrategyComparison,
        sample_performance_summary: PerformanceSummary,
        tmp_path: Path,
    ) -> None:
        """Test PDF generation."""
        generator = ReportGenerator()
        generator.add_executive_summary(sample_strategy_comparison)
        generator.add_problem_description()
        generator.add_financial_analysis(sample_performance_summary)

        output_path = tmp_path / "test_report.pdf"
        result_path = generator.generate_pdf(output_path)

        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0
        plt.close("all")

    def test_generate_html(
        self,
        sample_strategy_comparison: StrategyComparison,
        tmp_path: Path,
    ) -> None:
        """Test HTML generation."""
        generator = ReportGenerator()
        generator.add_executive_summary(sample_strategy_comparison)
        generator.add_methodology()

        output_path = tmp_path / "test_report.html"
        result_path = generator.generate_html(output_path)

        assert result_path.exists()
        assert result_path.suffix == ".html"

        # Check HTML content
        content = result_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Executive Summary" in content
        assert "Methodology" in content
        plt.close("all")


# --- Convenience Function Tests ---


class TestGenerateExecutiveReport:
    """Tests for the generate_executive_report convenience function."""

    def test_basic_report_generation(
        self,
        sample_strategy_comparison: StrategyComparison,
        sample_performance_summary: PerformanceSummary,
        tmp_path: Path,
    ) -> None:
        """Test basic report generation."""
        output_path = tmp_path / "executive_report.pdf"

        result = generate_executive_report(
            comparison=sample_strategy_comparison,
            summary=sample_performance_summary,
            output_path=output_path,
        )

        assert result.exists()
        assert result.stat().st_size > 0
        plt.close("all")

    def test_report_with_stress_results(
        self,
        sample_strategy_comparison: StrategyComparison,
        sample_performance_summary: PerformanceSummary,
        sample_stress_result: StressTestResult,
        tmp_path: Path,
    ) -> None:
        """Test report generation with stress test results."""
        output_path = tmp_path / "full_report.pdf"

        result = generate_executive_report(
            comparison=sample_strategy_comparison,
            summary=sample_performance_summary,
            stress_result=sample_stress_result,
            output_path=output_path,
        )

        assert result.exists()
        plt.close("all")

    def test_report_with_custom_config(
        self,
        sample_strategy_comparison: StrategyComparison,
        sample_performance_summary: PerformanceSummary,
        tmp_path: Path,
    ) -> None:
        """Test report generation with custom config."""
        config = ReportConfig(
            title="Q4 2024 Analysis",
            author="Energy Analytics Team",
            company="Acme Energy",
        )
        output_path = tmp_path / "custom_report.pdf"

        result = generate_executive_report(
            comparison=sample_strategy_comparison,
            summary=sample_performance_summary,
            output_path=output_path,
            config=config,
        )

        assert result.exists()
        plt.close("all")


# --- Integration Tests ---


class TestReportingIntegration:
    """Integration tests for reporting workflows."""

    def test_full_report_workflow(
        self,
        sample_strategy_comparison: StrategyComparison,
        sample_performance_summary: PerformanceSummary,
        sample_stress_result: StressTestResult,
        tmp_path: Path,
    ) -> None:
        """Test complete report workflow."""
        config = ReportConfig(
            title="Comprehensive Analysis Report",
            author="Test Suite",
            company="Test Corp",
            include_assumptions=True,
            include_risk_analysis=True,
        )

        generator = ReportGenerator(config=config)

        # Add all sections
        generator.add_executive_summary(sample_strategy_comparison)
        generator.add_problem_description()
        generator.add_methodology()
        generator.add_financial_analysis(
            sample_performance_summary,
            baseline_revenue=42000.0,
        )
        generator.add_risk_analysis(sample_stress_result)
        generator.add_assumptions()

        # Verify sections added
        assert len(generator.sections) == 6

        # Generate PDF
        pdf_path = tmp_path / "full_report.pdf"
        generator.generate_pdf(pdf_path)
        assert pdf_path.exists()

        # Generate HTML
        html_path = tmp_path / "full_report.html"
        generator.generate_html(html_path)
        assert html_path.exists()

        plt.close("all")

    def test_minimal_report(
        self,
        sample_performance_summary: PerformanceSummary,
        tmp_path: Path,
    ) -> None:
        """Test minimal report with just financial analysis."""
        generator = ReportGenerator()
        generator.add_financial_analysis(sample_performance_summary)

        output_path = tmp_path / "minimal_report.pdf"
        result = generator.generate_pdf(output_path)

        assert result.exists()
        plt.close("all")
