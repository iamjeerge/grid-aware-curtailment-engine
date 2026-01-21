"""Energy market price simulator.

Generates CAISO-style market prices including:
- Day-ahead prices
- Real-time prices
- Negative pricing events during oversupply
- Price volatility tied to congestion and demand
"""

from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from numpy.random import Generator

from src.domain.models import MarketPrice


class PricePattern(str, Enum):
    """Pre-defined price patterns."""

    NORMAL = "normal"  # Standard daily pattern
    DUCK_CURVE = "duck_curve"  # Negative midday, high evening
    HIGH_VOLATILITY = "high_volatility"  # Extreme swings
    FLAT = "flat"  # Relatively constant prices


class MarketPriceGenerator:
    """Generates synthetic energy market prices.

    All prices are in $/MWh.
    """

    # CAISO-style base price profile ($/MWh) for a typical summer day
    BASE_PRICE_PROFILE: dict[int, float] = {
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

    # Duck curve price profile with negative midday prices
    DUCK_CURVE_PROFILE: dict[int, float] = {
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

    def __init__(
        self,
        base_price_multiplier: float = 1.0,
        volatility: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the market price generator.

        Args:
            base_price_multiplier: Multiplier for all base prices.
            volatility: Price volatility (standard deviation as fraction).
            seed: Random seed for reproducibility.
        """
        self.base_price_multiplier = base_price_multiplier
        self.volatility = volatility
        self._rng: Generator = np.random.default_rng(seed)

    def _get_base_price(self, hour: int, pattern: PricePattern) -> float:
        """Get the base price for a given hour and pattern.

        Args:
            hour: Hour of day (0-23).
            pattern: Price pattern to use.

        Returns:
            Base price in $/MWh.
        """
        if pattern == PricePattern.DUCK_CURVE:
            return self.DUCK_CURVE_PROFILE.get(hour, 50.0)
        elif pattern == PricePattern.FLAT:
            return 50.0  # Constant $50/MWh
        elif pattern == PricePattern.HIGH_VOLATILITY:
            # More extreme version of normal
            base = self.BASE_PRICE_PROFILE.get(hour, 50.0)
            return base * 1.5 if hour in [18, 19, 20] else base * 0.8
        else:  # NORMAL
            return self.BASE_PRICE_PROFILE.get(hour, 50.0)

    def _add_noise(self, price: float, volatility_multiplier: float = 1.0) -> float:
        """Add price noise based on volatility."""
        noise_std = abs(price) * self.volatility * volatility_multiplier
        noise_std = max(2.0, noise_std)  # Minimum $2 noise
        return float(price + self._rng.normal(0, noise_std))

    def generate_price(
        self,
        timestamp: datetime,
        pattern: PricePattern = PricePattern.NORMAL,
        congestion_adder: float = 0.0,
        include_real_time: bool = False,
    ) -> MarketPrice:
        """Generate a single timestep market price.

        Args:
            timestamp: The price timestamp.
            pattern: Price pattern to use.
            congestion_adder: Additional price due to congestion ($/MWh).
            include_real_time: If True, generate real-time price too.

        Returns:
            MarketPrice for the timestep.
        """
        hour = timestamp.hour

        # Get base price and apply multiplier
        base_price = self._get_base_price(hour, pattern)
        day_ahead = base_price * self.base_price_multiplier

        # Add congestion premium
        day_ahead += congestion_adder

        # Add noise
        day_ahead = self._add_noise(day_ahead)

        # Real-time price has more volatility
        real_time = None
        if include_real_time:
            real_time = self._add_noise(day_ahead, volatility_multiplier=2.0)

        is_negative = day_ahead < 0

        return MarketPrice(
            timestamp=timestamp,
            day_ahead_price=round(day_ahead, 2),
            real_time_price=round(real_time, 2) if real_time is not None else None,
            is_negative_pricing=is_negative,
        )

    def generate_day_ahead(
        self,
        start_date: datetime,
        hours: int = 24,
        pattern: PricePattern = PricePattern.NORMAL,
        congestion_hours: dict[int, float] | None = None,
    ) -> list[MarketPrice]:
        """Generate day-ahead prices for multiple hours.

        Args:
            start_date: Start timestamp.
            hours: Number of hours.
            pattern: Price pattern to use.
            congestion_hours: Dict of {hour: congestion_adder} for congested hours.

        Returns:
            List of MarketPrice for each hour.
        """
        if congestion_hours is None:
            congestion_hours = {}

        prices = []
        for h in range(hours):
            ts = start_date + timedelta(hours=h)
            congestion_adder = congestion_hours.get(h, 0.0)
            prices.append(
                self.generate_price(
                    ts, pattern=pattern, congestion_adder=congestion_adder
                )
            )

        return prices

    def generate_duck_curve_scenario(
        self,
        start_date: datetime,
    ) -> list[MarketPrice]:
        """Generate prices for the duck curve demo scenario.

        Matches copilot-instructions.md:
        - Negative prices midday (-$25/MWh)
        - Evening spike to $140/MWh

        Args:
            start_date: Start timestamp.

        Returns:
            24-hour prices with duck curve pattern.
        """
        prices = []
        for h in range(24):
            ts = start_date + timedelta(hours=h)
            price = float(self.DUCK_CURVE_PROFILE[h])

            prices.append(
                MarketPrice(
                    timestamp=ts,
                    day_ahead_price=price,
                    is_negative_pricing=price < 0,
                )
            )

        return prices
