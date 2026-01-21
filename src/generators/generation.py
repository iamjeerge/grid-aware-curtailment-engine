"""Solar and wind generation forecast generator.

Produces realistic CAISO-style generation forecasts with:
- Diurnal patterns (solar follows sun, wind peaks overnight/evening)
- Seasonal variations
- Probabilistic bands (P10/P50/P90)
- Reproducible via numpy.random.Generator seeds
"""

from datetime import datetime, timedelta

import numpy as np
from numpy.random import Generator

from src.domain.models import GenerationForecast


class GenerationGenerator:
    """Generates synthetic solar and wind generation forecasts.

    All generation values are in MW (megawatts).
    """

    def __init__(
        self,
        solar_capacity_mw: float = 800.0,
        wind_capacity_mw: float = 400.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the generation generator.

        Args:
            solar_capacity_mw: Installed solar capacity (MW).
            wind_capacity_mw: Installed wind capacity (MW).
            seed: Random seed for reproducibility.
        """
        self.solar_capacity_mw = solar_capacity_mw
        self.wind_capacity_mw = wind_capacity_mw
        self._rng: Generator = np.random.default_rng(seed)

    def _solar_capacity_factor(self, hour: int, day_of_year: int) -> float:
        """Calculate solar capacity factor based on time of day and season.

        Args:
            hour: Hour of day (0-23).
            day_of_year: Day of year (1-365).

        Returns:
            Capacity factor between 0 and 1.
        """
        # No solar at night
        if hour < 5 or hour > 20:
            return 0.0

        # Diurnal pattern: bell curve peaking at solar noon (~13:00)
        solar_noon = 13.0
        hour_factor = max(0, 1 - ((hour - solar_noon) / 7) ** 2)

        # Seasonal pattern: peak in summer (day 172 = June 21)
        summer_solstice = 172
        seasonal_factor = 0.6 + 0.4 * np.cos(
            2 * np.pi * (day_of_year - summer_solstice) / 365
        )

        return float(hour_factor * seasonal_factor)

    def _wind_capacity_factor(self, hour: int, day_of_year: int) -> float:
        """Calculate wind capacity factor based on time and season.

        Wind tends to be stronger at night and in spring/fall.

        Args:
            hour: Hour of day (0-23).
            day_of_year: Day of year (1-365).

        Returns:
            Capacity factor between 0 and 1.
        """
        # Diurnal pattern: higher at night, lower midday
        # Wind typically picks up in evening, peaks overnight
        hour_factor = 0.3 + 0.2 * np.cos(2 * np.pi * (hour - 3) / 24)

        # Seasonal pattern: peak in spring (day 80) and fall (day 265)
        # Lower in summer and winter
        seasonal_factor = 0.5 + 0.3 * np.cos(4 * np.pi * day_of_year / 365)

        return float(hour_factor * seasonal_factor)

    def _add_noise(self, base_value: float, noise_std: float) -> float:
        """Add Gaussian noise to a base value, clamped to non-negative."""
        noisy = base_value + self._rng.normal(0, noise_std)
        return float(max(0, noisy))

    def generate_forecast(
        self,
        timestamp: datetime,
        cloud_cover: float = 0.0,
        wind_speed_factor: float = 1.0,
    ) -> GenerationForecast:
        """Generate a single timestep forecast with P10/P50/P90 bands.

        Args:
            timestamp: The forecast timestamp.
            cloud_cover: Cloud cover fraction (0=clear, 1=overcast).
            wind_speed_factor: Multiplier for wind speed (1.0=normal).

        Returns:
            GenerationForecast with probabilistic bands.
        """
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour

        # Calculate base capacity factors
        solar_cf = self._solar_capacity_factor(hour, day_of_year)
        wind_cf = self._wind_capacity_factor(hour, day_of_year)

        # Apply modifiers
        solar_cf *= 1 - cloud_cover * 0.8  # Clouds reduce solar by up to 80%
        wind_cf *= wind_speed_factor

        # Calculate P50 (median) generation
        solar_p50 = solar_cf * self.solar_capacity_mw
        wind_p50 = wind_cf * self.wind_capacity_mw

        # Add forecast uncertainty for P10/P90
        # Uncertainty is proportional to generation level
        solar_uncertainty = max(10, solar_p50 * 0.15)
        wind_uncertainty = max(15, wind_p50 * 0.25)

        # P10 = low generation scenario (10th percentile)
        # P90 = high generation scenario (90th percentile)
        solar_p10 = max(0, solar_p50 - solar_uncertainty * 1.3)
        solar_p90 = min(self.solar_capacity_mw, solar_p50 + solar_uncertainty * 1.3)

        wind_p10 = max(0, wind_p50 - wind_uncertainty * 1.3)
        wind_p90 = min(self.wind_capacity_mw, wind_p50 + wind_uncertainty * 1.3)

        # Add small random perturbations
        solar_p50 = self._add_noise(solar_p50, solar_uncertainty * 0.1)
        wind_p50 = self._add_noise(wind_p50, wind_uncertainty * 0.1)

        return GenerationForecast(
            timestamp=timestamp,
            solar_mw_p10=round(solar_p10, 2),
            solar_mw_p50=round(solar_p50, 2),
            solar_mw_p90=round(solar_p90, 2),
            wind_mw_p10=round(wind_p10, 2),
            wind_mw_p50=round(wind_p50, 2),
            wind_mw_p90=round(wind_p90, 2),
        )

    def generate_day_ahead(
        self,
        start_date: datetime,
        hours: int = 24,
        cloud_cover_pattern: list[float] | None = None,
    ) -> list[GenerationForecast]:
        """Generate day-ahead forecasts for multiple hours.

        Args:
            start_date: Start timestamp (should be midnight).
            hours: Number of hours to forecast.
            cloud_cover_pattern: Optional hourly cloud cover values.

        Returns:
            List of GenerationForecast for each hour.
        """
        if cloud_cover_pattern is None:
            # Generate random cloud pattern
            cloud_cover_pattern = [
                float(self._rng.uniform(0, 0.3)) for _ in range(hours)
            ]

        forecasts = []
        for h in range(hours):
            ts = start_date + timedelta(hours=h)
            cloud_cover = (
                cloud_cover_pattern[h]
                if h < len(cloud_cover_pattern)
                else cloud_cover_pattern[-1]
            )
            forecasts.append(self.generate_forecast(ts, cloud_cover=cloud_cover))

        return forecasts

    def generate_duck_curve_scenario(
        self,
        start_date: datetime,
    ) -> list[GenerationForecast]:
        """Generate the classic duck curve scenario for testing.

        Creates a high solar day with ~600 MW peak at noon.
        Matches the demo scenario in copilot-instructions.md.

        Args:
            start_date: Start timestamp (should be midnight).

        Returns:
            24-hour forecast with duck curve pattern.
        """
        forecasts = []
        for h in range(24):
            ts = start_date + timedelta(hours=h)

            # Duck curve: high solar, moderate wind
            if 6 <= h <= 18:
                # Bell curve peaking at noon with 600 MW
                solar_factor = max(0, 1 - ((h - 12) / 6) ** 2)
                solar_p50 = 600 * solar_factor
            else:
                solar_p50 = 0.0

            # Uncertainty bands
            solar_p10 = solar_p50 * 0.80
            solar_p90 = solar_p50 * 1.15

            # Moderate wind throughout
            wind_p50 = 80 + 20 * np.sin(2 * np.pi * h / 24)
            wind_p10 = wind_p50 * 0.70
            wind_p90 = wind_p50 * 1.30

            forecasts.append(
                GenerationForecast(
                    timestamp=ts,
                    solar_mw_p10=round(solar_p10, 2),
                    solar_mw_p50=round(solar_p50, 2),
                    solar_mw_p90=round(solar_p90, 2),
                    wind_mw_p10=round(wind_p10, 2),
                    wind_mw_p50=round(wind_p50, 2),
                    wind_mw_p90=round(wind_p90, 2),
                )
            )

        return forecasts
