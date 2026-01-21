"""Battery degradation cost model.

Implements cycle-based degradation cost calculation:
- Cost per equivalent full cycle
- Throughput-based degradation tracking
- Penalty for aggressive cycling in optimization
"""

from src.domain.models import BatteryConfig, BatteryState


class DegradationModel:
    """Cycle-based battery degradation cost model.

    Calculates degradation costs based on energy throughput and cycling depth.
    Uses equivalent full cycle (EFC) methodology.
    """

    def __init__(self, config: BatteryConfig) -> None:
        """Initialize the degradation model.

        Args:
            config: Battery configuration with degradation cost parameter.
        """
        self.config = config

    @property
    def cost_per_mwh(self) -> float:
        """Get degradation cost per MWh of throughput."""
        return self.config.degradation_cost_per_mwh

    def calculate_cycle_cost(self, energy_mwh: float) -> float:
        """Calculate degradation cost for a given energy throughput.

        Args:
            energy_mwh: Energy throughput in MWh.

        Returns:
            Degradation cost in dollars.
        """
        return energy_mwh * self.config.degradation_cost_per_mwh

    def calculate_efc(self, throughput_mwh: float) -> float:
        """Calculate equivalent full cycles from cumulative throughput.

        One EFC = one full charge + one full discharge = 2 * capacity.

        Args:
            throughput_mwh: Cumulative energy throughput in MWh.

        Returns:
            Number of equivalent full cycles.
        """
        # EFC = throughput / (2 * usable_capacity)
        # We use usable capacity since that's what's actually cycled
        usable_capacity = self.config.usable_capacity_mwh
        if usable_capacity <= 0:
            return 0.0

        return throughput_mwh / (2 * usable_capacity)

    def calculate_total_degradation_cost(self, state: BatteryState) -> float:
        """Calculate total degradation cost from battery state.

        Args:
            state: Current battery state with cumulative throughput.

        Returns:
            Total degradation cost in dollars.
        """
        return self.calculate_cycle_cost(state.cumulative_throughput_mwh)

    def calculate_incremental_cost(
        self,
        initial_state: BatteryState,
        final_state: BatteryState,
    ) -> float:
        """Calculate incremental degradation cost between two states.

        Args:
            initial_state: Battery state before operation.
            final_state: Battery state after operation.

        Returns:
            Incremental degradation cost in dollars.
        """
        throughput_delta = (
            final_state.cumulative_throughput_mwh
            - initial_state.cumulative_throughput_mwh
        )
        return self.calculate_cycle_cost(max(0, throughput_delta))

    def estimate_remaining_life_cycles(
        self,
        state: BatteryState,
        total_cycle_life: float = 4000.0,
    ) -> float:
        """Estimate remaining battery life in cycles.

        Args:
            state: Current battery state.
            total_cycle_life: Total expected cycle life (default: 4000 EFC).

        Returns:
            Estimated remaining cycles.
        """
        used_cycles = self.calculate_efc(state.cumulative_throughput_mwh)
        return max(0, total_cycle_life - used_cycles)

    def calculate_depth_of_discharge_factor(
        self,
        discharge_mwh: float,
    ) -> float:
        """Calculate depth of discharge factor for cycle aging.

        Deeper discharges cause more degradation per MWh. This factor
        can be used to penalize deep cycling in optimization.

        Args:
            discharge_mwh: Energy to be discharged in MWh.

        Returns:
            DoD factor (1.0 = normal, >1.0 = accelerated degradation).
        """
        usable_capacity = self.config.usable_capacity_mwh
        if usable_capacity <= 0:
            return 1.0

        dod = discharge_mwh / usable_capacity

        # Non-linear degradation: deeper cycles are worse
        # Based on typical Li-ion cycle life curves
        if dod <= 0.5:
            return 1.0  # Shallow cycles: normal degradation
        elif dod <= 0.8:
            return 1.2  # Medium cycles: 20% more degradation
        else:
            return 1.5  # Deep cycles: 50% more degradation

    def calculate_adjusted_cycle_cost(
        self,
        energy_mwh: float,
        depth_of_discharge: float,
    ) -> float:
        """Calculate degradation cost adjusted for depth of discharge.

        Args:
            energy_mwh: Energy throughput in MWh.
            depth_of_discharge: DoD as fraction of usable capacity (0-1).

        Returns:
            Adjusted degradation cost in dollars.
        """
        dod_energy = depth_of_discharge * self.config.usable_capacity_mwh
        dod_factor = self.calculate_depth_of_discharge_factor(dod_energy)
        return energy_mwh * self.config.degradation_cost_per_mwh * dod_factor


class OptimizationDegradationPenalty:
    """Degradation penalty calculator for use in MILP/RL optimization.

    Provides penalty terms that can be incorporated into objective functions.
    """

    def __init__(
        self,
        config: BatteryConfig,
        penalty_multiplier: float = 1.0,
    ) -> None:
        """Initialize the optimization penalty calculator.

        Args:
            config: Battery configuration.
            penalty_multiplier: Multiplier to adjust penalty severity.
        """
        self.config = config
        self.degradation_model = DegradationModel(config)
        self.penalty_multiplier = penalty_multiplier

    def calculate_charge_penalty(self, charge_mwh: float) -> float:
        """Calculate penalty for charging operation.

        Args:
            charge_mwh: Energy to be charged in MWh.

        Returns:
            Penalty cost in dollars.
        """
        base_cost = self.degradation_model.calculate_cycle_cost(charge_mwh)
        return base_cost * self.penalty_multiplier

    def calculate_discharge_penalty(
        self,
        discharge_mwh: float,
        current_soc_mwh: float,
    ) -> float:
        """Calculate penalty for discharging operation.

        Includes DoD-adjusted penalty for deep discharges.

        Args:
            discharge_mwh: Energy to be discharged in MWh.
            current_soc_mwh: Current SOC in MWh.

        Returns:
            Penalty cost in dollars.
        """
        # Calculate effective DoD
        min_soc_mwh = self.config.capacity_mwh * self.config.min_soc_fraction
        available = current_soc_mwh - min_soc_mwh
        usable = self.config.usable_capacity_mwh
        dod = 1.0 if usable <= 0 or available <= 0 else min(1.0, discharge_mwh / usable)

        adjusted_cost = self.degradation_model.calculate_adjusted_cycle_cost(
            discharge_mwh, dod
        )
        return adjusted_cost * self.penalty_multiplier

    def calculate_cycling_penalty(
        self,
        charge_mwh: float,
        discharge_mwh: float,
    ) -> float:
        """Calculate total penalty for a charge-discharge cycle.

        Penalizes rapid cycling to discourage aggressive arbitrage
        that damages the battery.

        Args:
            charge_mwh: Energy charged in MWh.
            discharge_mwh: Energy discharged in MWh.

        Returns:
            Total cycling penalty in dollars.
        """
        # Base throughput cost
        total_throughput = charge_mwh + discharge_mwh
        base_penalty = self.degradation_model.calculate_cycle_cost(total_throughput)

        # Additional penalty for rapid cycling (charge + discharge in same period)
        # This discourages simultaneous charge/discharge arbitrage
        cycling_intensity = min(charge_mwh, discharge_mwh)
        rapid_cycling_penalty = (
            cycling_intensity * self.config.degradation_cost_per_mwh * 0.5
        )

        return (base_penalty + rapid_cycling_penalty) * self.penalty_multiplier
