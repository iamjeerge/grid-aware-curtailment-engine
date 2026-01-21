"""Test fixtures for reproducible scenario testing.

Provides standard scenarios as documented in copilot-instructions.md:
- Sunny day scenario
- Congested grid scenario
- Price spike scenario
- Duck curve trap (killer demo)
"""

from datetime import datetime, timedelta

import pytest

from src.domain.models import (
    BatteryConfig,
    BatteryState,
    ForecastScenario,
    GenerationForecast,
    GridConstraint,
    MarketPrice,
    ScenarioConfig,
)

# =============================================================================
# Time Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Standard base timestamp for testing (midnight, summer day)."""
    return datetime(2025, 7, 15, 0, 0, 0)


@pytest.fixture
def hourly_timestamps(base_timestamp: datetime) -> list[datetime]:
    """24-hour horizon of timestamps."""
    return [base_timestamp + timedelta(hours=h) for h in range(24)]


# =============================================================================
# Battery Fixtures
# =============================================================================


@pytest.fixture
def default_battery_config() -> BatteryConfig:
    """Standard 500 MWh / 150 MW battery configuration."""
    return BatteryConfig(
        capacity_mwh=500.0,
        max_power_mw=150.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        min_soc_fraction=0.1,
        max_soc_fraction=0.9,
        degradation_cost_per_mwh=8.0,
    )


@pytest.fixture
def initial_battery_state(
    base_timestamp: datetime, default_battery_config: BatteryConfig
) -> BatteryState:
    """Battery at 50% SOC."""
    return BatteryState(
        timestamp=base_timestamp,
        soc_mwh=default_battery_config.capacity_mwh * 0.5,
        soc_fraction=0.5,
        cumulative_throughput_mwh=0.0,
    )


# =============================================================================
# Generation Forecast Fixtures
# =============================================================================


@pytest.fixture
def sunny_day_forecasts(hourly_timestamps: list[datetime]) -> list[GenerationForecast]:
    """Clear summer day with high solar generation.

    Solar peaks at 600 MW around noon. Wind is relatively low.
    """
    forecasts = []
    for ts in hourly_timestamps:
        hour = ts.hour

        # Solar follows a bell curve peaking at noon
        if 6 <= hour <= 18:
            # Peak at hour 12, symmetric around it
            solar_factor = max(0, 1 - ((hour - 12) / 6) ** 2)
            solar_p50 = 600 * solar_factor
        else:
            solar_p50 = 0.0

        # P10 is 20% lower, P90 is 15% higher
        solar_p10 = solar_p50 * 0.8
        solar_p90 = solar_p50 * 1.15

        # Wind is relatively constant with slight variation
        wind_p50 = 80 + 20 * (hour % 6) / 6
        wind_p10 = wind_p50 * 0.7
        wind_p90 = wind_p50 * 1.3

        forecasts.append(
            GenerationForecast(
                timestamp=ts,
                solar_mw_p10=solar_p10,
                solar_mw_p50=solar_p50,
                solar_mw_p90=solar_p90,
                wind_mw_p10=wind_p10,
                wind_mw_p50=wind_p50,
                wind_mw_p90=wind_p90,
            )
        )

    return forecasts


# =============================================================================
# Grid Constraint Fixtures
# =============================================================================


@pytest.fixture
def uncongested_grid(hourly_timestamps: list[datetime]) -> list[GridConstraint]:
    """Grid with ample export capacity (no congestion)."""
    return [
        GridConstraint(
            timestamp=ts,
            max_export_mw=800.0,  # Well above generation
            max_ramp_up_mw_per_hour=200.0,
            max_ramp_down_mw_per_hour=200.0,
            is_congested=False,
        )
        for ts in hourly_timestamps
    ]


@pytest.fixture
def congested_grid(hourly_timestamps: list[datetime]) -> list[GridConstraint]:
    """Grid with midday congestion (duck curve scenario).

    Export capacity drops to 300 MW during peak solar hours.
    """
    constraints = []
    for ts in hourly_timestamps:
        hour = ts.hour

        # Congestion window: 10am - 3pm
        if 10 <= hour <= 15:
            max_export = 300.0
            is_congested = True
            congestion_adder = 15.0  # $/MWh penalty
        else:
            max_export = 600.0
            is_congested = False
            congestion_adder = 0.0

        constraints.append(
            GridConstraint(
                timestamp=ts,
                max_export_mw=max_export,
                max_ramp_up_mw_per_hour=150.0,
                max_ramp_down_mw_per_hour=150.0,
                is_congested=is_congested,
                congestion_price_adder=congestion_adder,
            )
        )

    return constraints


# =============================================================================
# Market Price Fixtures
# =============================================================================


@pytest.fixture
def standard_prices(hourly_timestamps: list[datetime]) -> list[MarketPrice]:
    """Standard day-ahead prices without negative pricing."""
    # CAISO-style price pattern
    price_profile = {
        0: 35,
        1: 32,
        2: 30,
        3: 28,
        4: 30,
        5: 35,
        6: 45,
        7: 55,
        8: 50,
        9: 45,
        10: 40,
        11: 38,
        12: 35,
        13: 38,
        14: 42,
        15: 50,
        16: 65,
        17: 85,
        18: 95,
        19: 90,
        20: 75,
        21: 60,
        22: 50,
        23: 40,
    }

    return [
        MarketPrice(
            timestamp=ts,
            day_ahead_price=float(price_profile[ts.hour]),
            is_negative_pricing=False,
        )
        for ts in hourly_timestamps
    ]


@pytest.fixture
def duck_curve_prices(hourly_timestamps: list[datetime]) -> list[MarketPrice]:
    """Duck curve scenario with negative midday prices and evening spike.

    Prices go negative during solar peak, spike to $140/MWh in evening.
    """
    # Classic duck curve pattern
    price_profile = {
        0: 35,
        1: 32,
        2: 30,
        3: 28,
        4: 30,
        5: 35,
        6: 40,
        7: 45,
        8: 25,
        9: 10,
        10: -5,
        11: -18,
        12: -25,
        13: -15,
        14: -5,
        15: 15,
        16: 45,
        17: 85,
        18: 120,
        19: 140,
        20: 110,
        21: 80,
        22: 55,
        23: 42,
    }

    return [
        MarketPrice(
            timestamp=ts,
            day_ahead_price=float(price_profile[ts.hour]),
            is_negative_pricing=price_profile[ts.hour] < 0,
        )
        for ts in hourly_timestamps
    ]


# =============================================================================
# Complete Scenario Fixtures
# =============================================================================


@pytest.fixture
def duck_curve_scenario_config() -> ScenarioConfig:
    """The 'Duck Curve Trap' killer demo scenario.

    Reference: copilot-instructions.md Demo Scenario
    - Solar peaks 600 MW at noon, grid limited to 300 MW
    - Negative prices midday (-$25/MWh), evening spike $140/MWh
    - Naive result: 32% curtailment, $420k revenue, grid violations
    - Optimized target: <10% curtailment, >$650k revenue, zero violations
    """
    return ScenarioConfig(
        name="duck_curve_trap",
        description="Classic duck curve with grid congestion and negative pricing",
        horizon_hours=24,
        battery_config=BatteryConfig(),  # Default 500 MWh / 150 MW
        scenario_probabilities={
            ForecastScenario.P10: 0.25,
            ForecastScenario.P50: 0.50,
            ForecastScenario.P90: 0.25,
        },
        random_seed=42,  # Reproducible
    )
