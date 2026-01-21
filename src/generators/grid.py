"""Grid constraint simulator.

Generates CAISO-style grid constraints including:
- Maximum export capacity
- Congestion windows
- Ramp rate limits
- Emergency curtailment events
"""

from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from numpy.random import Generator

from src.domain.models import GridConstraint


class CongestionPattern(str, Enum):
    """Pre-defined congestion patterns."""

    NONE = "none"  # No congestion
    MIDDAY = "midday"  # 10am-3pm congestion (duck curve)
    EVENING = "evening"  # 5pm-9pm congestion (peak demand)
    RANDOM = "random"  # Random congestion events


class GridConstraintGenerator:
    """Generates synthetic grid export constraints.

    Models CAISO-style transmission constraints with congestion.
    """

    def __init__(
        self,
        base_export_capacity_mw: float = 600.0,
        min_export_capacity_mw: float = 200.0,
        max_ramp_rate_mw_per_hour: float = 150.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the grid constraint generator.

        Args:
            base_export_capacity_mw: Normal export capacity (MW).
            min_export_capacity_mw: Minimum during congestion (MW).
            max_ramp_rate_mw_per_hour: Maximum ramp rate (MW/hour).
            seed: Random seed for reproducibility.
        """
        self.base_export_capacity_mw = base_export_capacity_mw
        self.min_export_capacity_mw = min_export_capacity_mw
        self.max_ramp_rate_mw_per_hour = max_ramp_rate_mw_per_hour
        self._rng: Generator = np.random.default_rng(seed)

    def _get_congestion_factor(
        self,
        hour: int,
        pattern: CongestionPattern,
    ) -> tuple[float, bool, float]:
        """Calculate congestion factor for a given hour and pattern.

        Args:
            hour: Hour of day (0-23).
            pattern: Congestion pattern to apply.

        Returns:
            Tuple of (capacity_factor, is_congested, congestion_price_adder).
        """
        if pattern == CongestionPattern.NONE:
            return 1.0, False, 0.0

        elif pattern == CongestionPattern.MIDDAY:
            # Duck curve: congestion 10am-3pm
            if 10 <= hour <= 15:
                # Peak congestion at noon
                severity = 1 - 0.3 * abs(hour - 12.5) / 2.5
                capacity_factor = 0.5 + 0.3 * (1 - severity)
                congestion_adder = 15.0 * severity
                return capacity_factor, True, congestion_adder
            return 1.0, False, 0.0

        elif pattern == CongestionPattern.EVENING:
            # Evening peak: congestion 5pm-9pm
            if 17 <= hour <= 21:
                severity = 1 - 0.3 * abs(hour - 19) / 2
                capacity_factor = 0.6 + 0.2 * (1 - severity)
                congestion_adder = 20.0 * severity
                return capacity_factor, True, congestion_adder
            return 1.0, False, 0.0

        else:  # RANDOM
            # Random congestion with 20% probability
            if self._rng.random() < 0.2:
                severity = float(self._rng.uniform(0.3, 0.8))
                capacity_factor = 1 - severity * 0.5
                congestion_adder = float(self._rng.uniform(5, 25))
                return capacity_factor, True, congestion_adder
            return 1.0, False, 0.0

    def generate_constraint(
        self,
        timestamp: datetime,
        pattern: CongestionPattern = CongestionPattern.MIDDAY,
        emergency_curtailment: bool = False,
    ) -> GridConstraint:
        """Generate a single timestep grid constraint.

        Args:
            timestamp: The constraint timestamp.
            pattern: Congestion pattern to apply.
            emergency_curtailment: If True, severely restrict capacity.

        Returns:
            GridConstraint for the timestep.
        """
        hour = timestamp.hour

        if emergency_curtailment:
            # Emergency: drop to 20% capacity
            max_export = self.min_export_capacity_mw
            is_congested = True
            congestion_adder = 50.0  # High penalty
        else:
            capacity_factor, is_congested, congestion_adder = (
                self._get_congestion_factor(hour, pattern)
            )
            max_export = self.base_export_capacity_mw * capacity_factor

        # Add small random variation
        max_export = self._add_variation(max_export, std_fraction=0.05)
        max_export = max(self.min_export_capacity_mw, max_export)

        return GridConstraint(
            timestamp=timestamp,
            max_export_mw=round(max_export, 2),
            max_ramp_up_mw_per_hour=self.max_ramp_rate_mw_per_hour,
            max_ramp_down_mw_per_hour=self.max_ramp_rate_mw_per_hour,
            is_congested=is_congested,
            congestion_price_adder=round(congestion_adder, 2),
        )

    def _add_variation(self, value: float, std_fraction: float) -> float:
        """Add random variation to a value."""
        return float(value * (1 + self._rng.normal(0, std_fraction)))

    def generate_day_ahead(
        self,
        start_date: datetime,
        hours: int = 24,
        pattern: CongestionPattern = CongestionPattern.MIDDAY,
        emergency_hours: list[int] | None = None,
    ) -> list[GridConstraint]:
        """Generate day-ahead grid constraints.

        Args:
            start_date: Start timestamp.
            hours: Number of hours.
            pattern: Congestion pattern to apply.
            emergency_hours: List of hours with emergency curtailment.

        Returns:
            List of GridConstraint for each hour.
        """
        if emergency_hours is None:
            emergency_hours = []

        constraints = []
        for h in range(hours):
            ts = start_date + timedelta(hours=h)
            emergency = h in emergency_hours
            constraints.append(
                self.generate_constraint(
                    ts, pattern=pattern, emergency_curtailment=emergency
                )
            )

        return constraints

    def generate_duck_curve_scenario(
        self,
        start_date: datetime,
    ) -> list[GridConstraint]:
        """Generate grid constraints for the duck curve demo scenario.

        Matches copilot-instructions.md: grid limited to 300 MW during midday.

        Args:
            start_date: Start timestamp.

        Returns:
            24-hour constraints with midday congestion at 300 MW.
        """
        constraints = []
        for h in range(24):
            ts = start_date + timedelta(hours=h)

            # Specific duck curve pattern: 300 MW limit 10am-3pm
            if 10 <= h <= 15:
                max_export = 300.0
                is_congested = True
                congestion_adder = 15.0
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
