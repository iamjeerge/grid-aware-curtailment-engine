"""Database fixtures for demo data.

This module provides pre-configured demo scenarios that can be
loaded into the database for quick demonstrations.
"""

from __future__ import annotations

from src.api.database import DemoScenarioModel, SessionLocal, init_db


def get_demo_fixtures() -> list[dict]:
    """Get all demo scenario fixtures."""
    return [
        {
            "id": "duck_curve",
            "name": "Duck Curve Challenge",
            "description": (
                "Classic California duck curve with solar peak at noon, "
                "negative midday prices, and evening price spike. "
                "Grid constrained to 300 MW during peak solar. "
                "This scenario demonstrates the value of battery storage "
                "for time-shifting renewable generation."
            ),
            "scenario_type": "duck_curve",
            "config": {
                "scenario_type": "duck_curve",
                "horizon_hours": 24,
                "peak_generation_mw": 600.0,
                "grid_limit_mw": 300.0,
                "seed": 42,
            },
            "battery": {
                "capacity_mwh": 500.0,
                "max_power_mw": 150.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "min_soc_fraction": 0.10,
                "max_soc_fraction": 0.90,
                "degradation_cost_per_mwh": 8.0,
            },
        },
        {
            "id": "high_volatility",
            "name": "High Volatility Market",
            "description": (
                "Extreme price swings with potential for significant "
                "arbitrage opportunities. Tests battery dispatch under "
                "uncertain conditions. Prices can swing from -$30 to +$200/MWh "
                "within hours."
            ),
            "scenario_type": "high_volatility",
            "config": {
                "scenario_type": "high_volatility",
                "horizon_hours": 24,
                "peak_generation_mw": 500.0,
                "grid_limit_mw": 400.0,
                "seed": 123,
            },
            "battery": {
                "capacity_mwh": 400.0,
                "max_power_mw": 200.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "min_soc_fraction": 0.10,
                "max_soc_fraction": 0.90,
                "degradation_cost_per_mwh": 8.0,
            },
        },
        {
            "id": "congested_grid",
            "name": "Congested Grid",
            "description": (
                "Evening congestion scenario with limited export capacity "
                "during peak demand hours (5-9 PM). Requires strategic "
                "battery charging during midday to serve evening demand."
            ),
            "scenario_type": "congested_grid",
            "config": {
                "scenario_type": "congested_grid",
                "horizon_hours": 24,
                "peak_generation_mw": 400.0,
                "grid_limit_mw": 200.0,
                "seed": 456,
            },
            "battery": {
                "capacity_mwh": 300.0,
                "max_power_mw": 100.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "min_soc_fraction": 0.10,
                "max_soc_fraction": 0.90,
                "degradation_cost_per_mwh": 8.0,
            },
        },
        {
            "id": "week_ahead",
            "name": "Week-Ahead Planning",
            "description": (
                "Extended 168-hour (7-day) planning horizon for weekly "
                "optimization. Tests longer-term scheduling decisions "
                "with varying daily patterns."
            ),
            "scenario_type": "duck_curve",
            "config": {
                "scenario_type": "duck_curve",
                "horizon_hours": 168,
                "peak_generation_mw": 800.0,
                "grid_limit_mw": 500.0,
                "seed": 789,
            },
            "battery": {
                "capacity_mwh": 1000.0,
                "max_power_mw": 300.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "min_soc_fraction": 0.10,
                "max_soc_fraction": 0.90,
                "degradation_cost_per_mwh": 8.0,
            },
        },
        {
            "id": "small_battery",
            "name": "Small Battery System",
            "description": (
                "Smaller battery system (50 MWh) demonstrating limitations "
                "of undersized storage. Useful for understanding battery "
                "sizing requirements."
            ),
            "scenario_type": "duck_curve",
            "config": {
                "scenario_type": "duck_curve",
                "horizon_hours": 24,
                "peak_generation_mw": 300.0,
                "grid_limit_mw": 150.0,
                "seed": 101,
            },
            "battery": {
                "capacity_mwh": 50.0,
                "max_power_mw": 25.0,
                "charge_efficiency": 0.92,
                "discharge_efficiency": 0.92,
                "min_soc_fraction": 0.15,
                "max_soc_fraction": 0.85,
                "degradation_cost_per_mwh": 12.0,
            },
        },
    ]


def load_fixtures() -> int:
    """Load demo fixtures into database.

    Returns:
        Number of fixtures loaded.
    """
    # Initialize database tables
    init_db()

    db = SessionLocal()
    try:
        fixtures = get_demo_fixtures()
        count = 0

        for fixture in fixtures:
            # Check if already exists
            existing = (
                db.query(DemoScenarioModel)
                .filter(DemoScenarioModel.id == fixture["id"])
                .first()
            )

            if existing:
                # Update existing
                for key, value in fixture.items():
                    setattr(existing, key, value)
            else:
                # Create new
                scenario = DemoScenarioModel(**fixture)
                db.add(scenario)
                count += 1

        db.commit()
        return count

    finally:
        db.close()


def clear_fixtures() -> int:
    """Clear all demo fixtures from database.

    Returns:
        Number of fixtures deleted.
    """
    db = SessionLocal()
    try:
        count = db.query(DemoScenarioModel).delete()
        db.commit()
        return count
    finally:
        db.close()


if __name__ == "__main__":
    print("Loading demo fixtures...")
    count = load_fixtures()
    print(f"Loaded {count} new fixtures")
