"""Tests for domain models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.domain.models import (
    BatteryConfig,
    BatteryState,
    ForecastScenario,
    GenerationForecast,
    MarketPrice,
    OptimizationDecision,
    ScenarioConfig,
)


class TestGenerationForecast:
    """Tests for GenerationForecast model."""

    def test_total_generation_p50(self, base_timestamp: datetime) -> None:
        """Test total generation calculation for P50 scenario."""
        forecast = GenerationForecast(
            timestamp=base_timestamp,
            solar_mw_p50=400.0,
            wind_mw_p50=100.0,
        )
        assert forecast.total_generation(ForecastScenario.P50) == 500.0

    def test_total_generation_all_scenarios(self, base_timestamp: datetime) -> None:
        """Test total generation for all scenarios."""
        forecast = GenerationForecast(
            timestamp=base_timestamp,
            solar_mw_p10=80.0,
            solar_mw_p50=100.0,
            solar_mw_p90=120.0,
            wind_mw_p10=16.0,
            wind_mw_p50=20.0,
            wind_mw_p90=24.0,
        )

        assert forecast.total_generation(ForecastScenario.P10) == 96.0
        assert forecast.total_generation(ForecastScenario.P50) == 120.0
        assert forecast.total_generation(ForecastScenario.P90) == 144.0

    def test_immutability(self, base_timestamp: datetime) -> None:
        """Test that GenerationForecast is immutable."""
        forecast = GenerationForecast(timestamp=base_timestamp, solar_mw_p50=100.0)

        with pytest.raises(ValidationError):
            forecast.solar_mw_p50 = 200.0  # type: ignore[misc]


class TestBatteryConfig:
    """Tests for BatteryConfig model."""

    def test_default_values(self) -> None:
        """Test default battery configuration matches spec."""
        config = BatteryConfig()

        assert config.capacity_mwh == 500.0
        assert config.max_power_mw == 150.0
        assert config.charge_efficiency == 0.95
        assert config.discharge_efficiency == 0.95
        assert config.degradation_cost_per_mwh == 8.0

    def test_usable_capacity(self) -> None:
        """Test usable capacity calculation with SOC limits."""
        config = BatteryConfig(
            capacity_mwh=500.0,
            min_soc_fraction=0.1,
            max_soc_fraction=0.9,
        )

        # Usable = 500 * (0.9 - 0.1) = 400 MWh
        assert config.usable_capacity_mwh == 400.0

    def test_efficiency_bounds(self) -> None:
        """Test that efficiency must be between 0 and 1."""
        with pytest.raises(ValidationError):
            BatteryConfig(charge_efficiency=1.5)

        with pytest.raises(ValidationError):
            BatteryConfig(discharge_efficiency=0.0)


class TestBatteryState:
    """Tests for BatteryState model."""

    def test_is_within_bounds(self, base_timestamp: datetime) -> None:
        """Test SOC bounds checking."""
        config = BatteryConfig(min_soc_fraction=0.1, max_soc_fraction=0.9)

        # Within bounds
        state_ok = BatteryState(
            timestamp=base_timestamp,
            soc_mwh=250.0,
            soc_fraction=0.5,
        )
        assert state_ok.is_within_bounds(config) is True

        # Below minimum
        state_low = BatteryState(
            timestamp=base_timestamp,
            soc_mwh=25.0,
            soc_fraction=0.05,
        )
        assert state_low.is_within_bounds(config) is False

        # Above maximum
        state_high = BatteryState(
            timestamp=base_timestamp,
            soc_mwh=475.0,
            soc_fraction=0.95,
        )
        assert state_high.is_within_bounds(config) is False


class TestMarketPrice:
    """Tests for MarketPrice model."""

    def test_negative_pricing(self, base_timestamp: datetime) -> None:
        """Test that negative prices are allowed."""
        price = MarketPrice(
            timestamp=base_timestamp,
            day_ahead_price=-25.0,
            is_negative_pricing=True,
        )
        assert price.day_ahead_price == -25.0
        assert price.effective_price == -25.0

    def test_effective_price_prefers_real_time(self, base_timestamp: datetime) -> None:
        """Test that effective_price uses real-time when available."""
        price = MarketPrice(
            timestamp=base_timestamp,
            day_ahead_price=50.0,
            real_time_price=55.0,
        )
        assert price.effective_price == 55.0


class TestOptimizationDecision:
    """Tests for OptimizationDecision model."""

    def test_energy_balance_valid(self, base_timestamp: datetime) -> None:
        """Test energy balance validation passes for valid decision."""
        decision = OptimizationDecision(
            timestamp=base_timestamp,
            scenario=ForecastScenario.P50,
            generation_mw=500.0,
            energy_sold_mw=300.0,
            energy_stored_mw=150.0,
            energy_curtailed_mw=50.0,
            resulting_soc_mwh=250.0,
            revenue_dollars=15000.0,
        )

        assert decision.validate_energy_balance() is True

    def test_energy_balance_invalid(self, base_timestamp: datetime) -> None:
        """Test energy balance validation fails when not balanced."""
        decision = OptimizationDecision(
            timestamp=base_timestamp,
            scenario=ForecastScenario.P50,
            generation_mw=500.0,
            energy_sold_mw=300.0,
            energy_stored_mw=150.0,
            energy_curtailed_mw=100.0,  # Total = 550 != 500
            resulting_soc_mwh=250.0,
            revenue_dollars=15000.0,
        )

        assert decision.validate_energy_balance() is False

    def test_net_revenue(self, base_timestamp: datetime) -> None:
        """Test net revenue calculation."""
        decision = OptimizationDecision(
            timestamp=base_timestamp,
            scenario=ForecastScenario.P50,
            generation_mw=500.0,
            energy_sold_mw=500.0,
            energy_stored_mw=0.0,
            energy_curtailed_mw=0.0,
            resulting_soc_mwh=250.0,
            revenue_dollars=25000.0,
            degradation_cost_dollars=1200.0,
        )

        assert decision.net_revenue == 23800.0


class TestScenarioConfig:
    """Tests for ScenarioConfig model."""

    def test_default_probabilities_sum_to_one(self) -> None:
        """Test that default scenario probabilities sum to 1."""
        config = ScenarioConfig(name="test")
        assert config.validate_probabilities() is True

    def test_invalid_probabilities(self) -> None:
        """Test that invalid probabilities are detected."""
        config = ScenarioConfig(
            name="test",
            scenario_probabilities={
                ForecastScenario.P10: 0.5,
                ForecastScenario.P50: 0.5,
                ForecastScenario.P90: 0.5,  # Total = 1.5
            },
        )
        assert config.validate_probabilities() is False
