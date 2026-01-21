"""Tests for data generators."""

from datetime import datetime

import pytest

from src.domain.models import ForecastScenario
from src.generators.generation import GenerationGenerator
from src.generators.grid import CongestionPattern, GridConstraintGenerator
from src.generators.prices import MarketPriceGenerator, PricePattern


class TestGenerationGenerator:
    """Tests for GenerationGenerator."""

    @pytest.fixture
    def generator(self) -> GenerationGenerator:
        """Create a seeded generator for reproducibility."""
        return GenerationGenerator(
            solar_capacity_mw=800.0,
            wind_capacity_mw=400.0,
            seed=42,
        )

    def test_solar_zero_at_night(self, generator: GenerationGenerator) -> None:
        """Test that solar generation is negligible at night."""
        midnight = datetime(2025, 7, 15, 0, 0, 0)
        forecast = generator.generate_forecast(midnight)

        # Solar should be negligible at night (may have small noise)
        assert forecast.solar_mw_p10 < 1.0
        assert forecast.solar_mw_p50 < 5.0
        assert forecast.solar_mw_p90 < 20.0

    def test_solar_peaks_midday(self, generator: GenerationGenerator) -> None:
        """Test that solar generation peaks around midday."""
        morning = datetime(2025, 7, 15, 8, 0, 0)
        noon = datetime(2025, 7, 15, 12, 0, 0)
        afternoon = datetime(2025, 7, 15, 16, 0, 0)

        morning_fc = generator.generate_forecast(morning)
        noon_fc = generator.generate_forecast(noon)
        afternoon_fc = generator.generate_forecast(afternoon)

        # Noon should have highest solar
        assert noon_fc.solar_mw_p50 > morning_fc.solar_mw_p50
        assert noon_fc.solar_mw_p50 > afternoon_fc.solar_mw_p50

    def test_p10_less_than_p50_less_than_p90(
        self, generator: GenerationGenerator
    ) -> None:
        """Test that P10 < P50 < P90 for probabilistic bands."""
        noon = datetime(2025, 7, 15, 12, 0, 0)
        forecast = generator.generate_forecast(noon)

        assert forecast.solar_mw_p10 <= forecast.solar_mw_p50 <= forecast.solar_mw_p90
        # Wind may have noise, but generally should follow
        assert forecast.wind_mw_p10 <= forecast.wind_mw_p90

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same results."""
        gen1 = GenerationGenerator(seed=42)
        gen2 = GenerationGenerator(seed=42)

        ts = datetime(2025, 7, 15, 12, 0, 0)
        fc1 = gen1.generate_forecast(ts)
        fc2 = gen2.generate_forecast(ts)

        assert fc1.solar_mw_p50 == fc2.solar_mw_p50
        assert fc1.wind_mw_p50 == fc2.wind_mw_p50

    def test_day_ahead_returns_24_hours(self, generator: GenerationGenerator) -> None:
        """Test that day_ahead returns correct number of forecasts."""
        start = datetime(2025, 7, 15, 0, 0, 0)
        forecasts = generator.generate_day_ahead(start, hours=24)

        assert len(forecasts) == 24

    def test_duck_curve_scenario(self, generator: GenerationGenerator) -> None:
        """Test duck curve scenario matches expected pattern."""
        start = datetime(2025, 7, 15, 0, 0, 0)
        forecasts = generator.generate_duck_curve_scenario(start)

        assert len(forecasts) == 24

        # Check noon peak is around 600 MW
        noon_forecast = forecasts[12]
        assert 550 <= noon_forecast.solar_mw_p50 <= 650

        # Check night has no solar
        midnight_forecast = forecasts[0]
        assert midnight_forecast.solar_mw_p50 == 0.0


class TestGridConstraintGenerator:
    """Tests for GridConstraintGenerator."""

    @pytest.fixture
    def generator(self) -> GridConstraintGenerator:
        """Create a seeded generator."""
        return GridConstraintGenerator(
            base_export_capacity_mw=600.0,
            min_export_capacity_mw=200.0,
            seed=42,
        )

    def test_midday_congestion_pattern(
        self, generator: GridConstraintGenerator
    ) -> None:
        """Test midday congestion reduces capacity."""
        morning = datetime(2025, 7, 15, 8, 0, 0)
        noon = datetime(2025, 7, 15, 12, 0, 0)

        morning_constraint = generator.generate_constraint(
            morning, pattern=CongestionPattern.MIDDAY
        )
        noon_constraint = generator.generate_constraint(
            noon, pattern=CongestionPattern.MIDDAY
        )

        # Noon should have lower capacity due to congestion
        assert noon_constraint.max_export_mw < morning_constraint.max_export_mw
        assert noon_constraint.is_congested is True
        assert morning_constraint.is_congested is False

    def test_no_congestion_pattern(self, generator: GridConstraintGenerator) -> None:
        """Test no congestion maintains full capacity."""
        noon = datetime(2025, 7, 15, 12, 0, 0)
        constraint = generator.generate_constraint(noon, pattern=CongestionPattern.NONE)

        assert constraint.max_export_mw > 550  # Close to 600
        assert constraint.is_congested is False

    def test_emergency_curtailment(self, generator: GridConstraintGenerator) -> None:
        """Test emergency curtailment severely restricts capacity."""
        noon = datetime(2025, 7, 15, 12, 0, 0)
        constraint = generator.generate_constraint(noon, emergency_curtailment=True)

        # Allow small variation due to noise (5% tolerance)
        assert constraint.max_export_mw <= 210.0
        assert constraint.is_congested is True
        assert constraint.congestion_price_adder >= 50.0

    def test_duck_curve_scenario(self, generator: GridConstraintGenerator) -> None:
        """Test duck curve grid scenario."""
        start = datetime(2025, 7, 15, 0, 0, 0)
        constraints = generator.generate_duck_curve_scenario(start)

        assert len(constraints) == 24

        # Check noon constraint is 300 MW
        noon_constraint = constraints[12]
        assert noon_constraint.max_export_mw == 300.0
        assert noon_constraint.is_congested is True

        # Check evening is not congested
        evening_constraint = constraints[20]
        assert evening_constraint.max_export_mw == 600.0
        assert evening_constraint.is_congested is False


class TestMarketPriceGenerator:
    """Tests for MarketPriceGenerator."""

    @pytest.fixture
    def generator(self) -> MarketPriceGenerator:
        """Create a seeded generator."""
        return MarketPriceGenerator(seed=42, volatility=0.05)

    def test_evening_peak_prices(self, generator: MarketPriceGenerator) -> None:
        """Test that evening has highest prices in normal pattern."""
        start = datetime(2025, 7, 15, 0, 0, 0)
        prices = generator.generate_day_ahead(start, pattern=PricePattern.NORMAL)

        # Find max price hour
        max_price = max(prices, key=lambda p: p.day_ahead_price)

        # Peak should be in evening hours (17-20)
        assert 17 <= max_price.timestamp.hour <= 20

    def test_duck_curve_negative_prices(self, generator: MarketPriceGenerator) -> None:
        """Test duck curve has negative midday prices."""
        start = datetime(2025, 7, 15, 0, 0, 0)
        prices = generator.generate_duck_curve_scenario(start)

        # Check noon is negative
        noon_price = prices[12]
        assert noon_price.day_ahead_price < 0
        assert noon_price.is_negative_pricing is True

        # Check evening is high positive
        evening_price = prices[19]
        assert evening_price.day_ahead_price >= 100

    def test_congestion_adder(self) -> None:
        """Test that congestion adds to price."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        # Use zero volatility generators for deterministic comparison
        gen1 = MarketPriceGenerator(seed=42, volatility=0.0)
        gen2 = MarketPriceGenerator(seed=42, volatility=0.0)

        p1 = gen1.generate_price(ts, congestion_adder=0.0)
        p2 = gen2.generate_price(ts, congestion_adder=20.0)

        assert p2.day_ahead_price > p1.day_ahead_price

    def test_reproducibility(self) -> None:
        """Test that same seed produces same prices."""
        gen1 = MarketPriceGenerator(seed=42)
        gen2 = MarketPriceGenerator(seed=42)

        ts = datetime(2025, 7, 15, 12, 0, 0)
        p1 = gen1.generate_price(ts)
        p2 = gen2.generate_price(ts)

        assert p1.day_ahead_price == p2.day_ahead_price

    def test_real_time_price_generation(self, generator: MarketPriceGenerator) -> None:
        """Test real-time price generation when requested."""
        ts = datetime(2025, 7, 15, 12, 0, 0)

        price_da_only = generator.generate_price(ts, include_real_time=False)
        price_with_rt = generator.generate_price(ts, include_real_time=True)

        assert price_da_only.real_time_price is None
        assert price_with_rt.real_time_price is not None


class TestIntegratedScenario:
    """Tests for integrated scenario generation."""

    def test_duck_curve_full_scenario(self) -> None:
        """Test generating a complete duck curve scenario."""
        start = datetime(2025, 7, 15, 0, 0, 0)

        gen_generator = GenerationGenerator(seed=42)
        grid_generator = GridConstraintGenerator(seed=42)
        price_generator = MarketPriceGenerator(seed=42)

        forecasts = gen_generator.generate_duck_curve_scenario(start)
        constraints = grid_generator.generate_duck_curve_scenario(start)
        prices = price_generator.generate_duck_curve_scenario(start)

        # All should have 24 hours
        assert len(forecasts) == len(constraints) == len(prices) == 24

        # Check the problematic hours (10am-3pm)
        for h in [10, 11, 12, 13, 14, 15]:
            gen = forecasts[h].total_generation(ForecastScenario.P50)
            capacity = constraints[h].max_export_mw
            price = prices[h].day_ahead_price

            # Generation exceeds grid capacity (the problem!)
            if h == 12:  # Peak hour
                assert gen > capacity, f"Hour {h}: gen={gen}, capacity={capacity}"

            # Prices are low or negative
            if h in [11, 12, 13]:
                assert price < 0, f"Hour {h}: price should be negative"
