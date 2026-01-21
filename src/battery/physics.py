"""Battery Energy Storage System (BESS) physics model.

Implements battery operations including:
- State of Charge (SOC) tracking
- Charge/discharge with efficiency losses
- Power and energy limits enforcement
- SOC bounds validation
"""

from datetime import datetime

from src.domain.models import BatteryConfig, BatteryState


class BatteryModel:
    """Physics-based battery model for BESS operations.

    Tracks SOC and enforces physical constraints during charge/discharge cycles.
    All power values in MW, energy values in MWh.
    """

    def __init__(self, config: BatteryConfig) -> None:
        """Initialize the battery model.

        Args:
            config: Battery configuration parameters.
        """
        self.config = config
        self._current_state: BatteryState | None = None

    @property
    def current_state(self) -> BatteryState | None:
        """Get the current battery state."""
        return self._current_state

    def initialize(
        self, timestamp: datetime, initial_soc_fraction: float = 0.5
    ) -> BatteryState:
        """Initialize the battery to a given SOC.

        Args:
            timestamp: Initial timestamp.
            initial_soc_fraction: Initial SOC as fraction of capacity (0-1).

        Returns:
            Initial BatteryState.

        Raises:
            ValueError: If initial_soc_fraction is outside valid bounds.
        """
        if not 0 <= initial_soc_fraction <= 1:
            raise ValueError(
                f"initial_soc_fraction must be 0-1, got {initial_soc_fraction}"
            )

        soc_mwh = self.config.capacity_mwh * initial_soc_fraction

        self._current_state = BatteryState(
            timestamp=timestamp,
            soc_mwh=soc_mwh,
            soc_fraction=initial_soc_fraction,
            cumulative_throughput_mwh=0.0,
        )

        return self._current_state

    def get_max_charge_power(self) -> float:
        """Get maximum allowable charge power given current SOC.

        Returns:
            Maximum charge power in MW.
        """
        if self._current_state is None:
            return 0.0

        # Power limit from battery specs
        power_limit = self.config.max_power_mw

        # Energy limit: how much can we charge before hitting max SOC?
        max_soc_mwh = self.config.capacity_mwh * self.config.max_soc_fraction
        available_capacity = max_soc_mwh - self._current_state.soc_mwh

        # Account for charge efficiency (we need to input more than we store)
        energy_limited_power = available_capacity / self.config.charge_efficiency

        return min(power_limit, max(0, energy_limited_power))

    def get_max_discharge_power(self) -> float:
        """Get maximum allowable discharge power given current SOC.

        Returns:
            Maximum discharge power in MW.
        """
        if self._current_state is None:
            return 0.0

        # Power limit from battery specs
        power_limit = self.config.max_power_mw

        # Energy limit: how much can we discharge before hitting min SOC?
        min_soc_mwh = self.config.capacity_mwh * self.config.min_soc_fraction
        available_energy = self._current_state.soc_mwh - min_soc_mwh

        # Account for discharge efficiency (we get less out than we have stored)
        energy_limited_power = available_energy * self.config.discharge_efficiency

        return min(power_limit, max(0, energy_limited_power))

    def charge(
        self,
        power_mw: float,
        duration_hours: float,
        timestamp: datetime,
    ) -> tuple[BatteryState, float]:
        """Charge the battery.

        Args:
            power_mw: Charge power in MW (must be >= 0).
            duration_hours: Duration of charging in hours.
            timestamp: Timestamp for the new state.

        Returns:
            Tuple of (new BatteryState, actual energy stored in MWh).

        Raises:
            ValueError: If battery not initialized or invalid power.
        """
        if self._current_state is None:
            raise ValueError("Battery not initialized. Call initialize() first.")

        if power_mw < 0:
            raise ValueError(f"Charge power must be >= 0, got {power_mw}")

        # Clamp to maximum allowed
        max_power = self.get_max_charge_power()
        actual_power = min(power_mw, max_power)

        # Calculate energy after efficiency losses
        energy_input_mwh = actual_power * duration_hours
        energy_stored_mwh = energy_input_mwh * self.config.charge_efficiency

        # Update SOC
        new_soc_mwh = self._current_state.soc_mwh + energy_stored_mwh
        max_soc_mwh = self.config.capacity_mwh * self.config.max_soc_fraction
        new_soc_mwh = min(new_soc_mwh, max_soc_mwh)

        new_soc_fraction = new_soc_mwh / self.config.capacity_mwh

        # Update throughput (for degradation tracking)
        new_throughput = (
            self._current_state.cumulative_throughput_mwh + energy_stored_mwh
        )

        self._current_state = BatteryState(
            timestamp=timestamp,
            soc_mwh=round(new_soc_mwh, 4),
            soc_fraction=round(new_soc_fraction, 6),
            cumulative_throughput_mwh=round(new_throughput, 4),
        )

        return self._current_state, round(energy_stored_mwh, 4)

    def discharge(
        self,
        power_mw: float,
        duration_hours: float,
        timestamp: datetime,
    ) -> tuple[BatteryState, float]:
        """Discharge the battery.

        Args:
            power_mw: Discharge power in MW (must be >= 0).
            duration_hours: Duration of discharging in hours.
            timestamp: Timestamp for the new state.

        Returns:
            Tuple of (new BatteryState, actual energy delivered in MWh).

        Raises:
            ValueError: If battery not initialized or invalid power.
        """
        if self._current_state is None:
            raise ValueError("Battery not initialized. Call initialize() first.")

        if power_mw < 0:
            raise ValueError(f"Discharge power must be >= 0, got {power_mw}")

        # Clamp to maximum allowed
        max_power = self.get_max_discharge_power()
        actual_power = min(power_mw, max_power)

        # Calculate energy accounting for efficiency
        energy_delivered_mwh = actual_power * duration_hours
        energy_withdrawn_mwh = energy_delivered_mwh / self.config.discharge_efficiency

        # Update SOC
        new_soc_mwh = self._current_state.soc_mwh - energy_withdrawn_mwh
        min_soc_mwh = self.config.capacity_mwh * self.config.min_soc_fraction
        new_soc_mwh = max(new_soc_mwh, min_soc_mwh)

        new_soc_fraction = new_soc_mwh / self.config.capacity_mwh

        # Update throughput
        new_throughput = (
            self._current_state.cumulative_throughput_mwh + energy_withdrawn_mwh
        )

        self._current_state = BatteryState(
            timestamp=timestamp,
            soc_mwh=round(new_soc_mwh, 4),
            soc_fraction=round(new_soc_fraction, 6),
            cumulative_throughput_mwh=round(new_throughput, 4),
        )

        return self._current_state, round(energy_delivered_mwh, 4)

    def can_charge(self, power_mw: float) -> bool:
        """Check if the battery can accept the given charge power.

        Args:
            power_mw: Desired charge power in MW.

        Returns:
            True if charging is possible (at least partially).
        """
        return self.get_max_charge_power() > 0 and power_mw > 0

    def can_discharge(self, power_mw: float) -> bool:
        """Check if the battery can deliver the given discharge power.

        Args:
            power_mw: Desired discharge power in MW.

        Returns:
            True if discharging is possible (at least partially).
        """
        return self.get_max_discharge_power() > 0 and power_mw > 0

    def get_soc_headroom(self) -> tuple[float, float]:
        """Get available headroom for charging and discharging.

        Returns:
            Tuple of (charge_headroom_mwh, discharge_headroom_mwh).
        """
        if self._current_state is None:
            return 0.0, 0.0

        max_soc_mwh = self.config.capacity_mwh * self.config.max_soc_fraction
        min_soc_mwh = self.config.capacity_mwh * self.config.min_soc_fraction

        charge_headroom = max_soc_mwh - self._current_state.soc_mwh
        discharge_headroom = self._current_state.soc_mwh - min_soc_mwh

        return max(0, charge_headroom), max(0, discharge_headroom)

    def reset(self) -> None:
        """Reset the battery state (requires re-initialization)."""
        self._current_state = None
