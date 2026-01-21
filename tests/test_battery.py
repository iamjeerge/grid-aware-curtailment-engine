"""Tests for battery physics and degradation models."""

from datetime import datetime

import pytest

from src.battery.degradation import DegradationModel, OptimizationDegradationPenalty
from src.battery.physics import BatteryModel
from src.domain.models import BatteryConfig, BatteryState


class TestBatteryModel:
    """Tests for BatteryModel physics."""

    @pytest.fixture
    def config(self) -> BatteryConfig:
        """Standard battery configuration."""
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
    def battery(self, config: BatteryConfig) -> BatteryModel:
        """Initialized battery at 50% SOC."""
        model = BatteryModel(config)
        model.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5)
        return model

    def test_initialize(self, config: BatteryConfig) -> None:
        """Test battery initialization."""
        model = BatteryModel(config)
        state = model.initialize(
            datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.5
        )

        assert state.soc_mwh == 250.0  # 500 * 0.5
        assert state.soc_fraction == 0.5
        assert state.cumulative_throughput_mwh == 0.0

    def test_initialize_invalid_soc(self, config: BatteryConfig) -> None:
        """Test initialization with invalid SOC raises error."""
        model = BatteryModel(config)

        with pytest.raises(ValueError):
            model.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=1.5)

        with pytest.raises(ValueError):
            model.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=-0.1)

    def test_charge_updates_soc(self, battery: BatteryModel) -> None:
        """Test that charging increases SOC correctly."""
        initial_soc = battery.current_state.soc_mwh

        ts = datetime(2025, 7, 15, 1, 0, 0)
        new_state, energy_stored = battery.charge(100.0, 1.0, ts)

        # Energy stored = 100 MW * 1 hour * 0.95 efficiency = 95 MWh
        assert energy_stored == pytest.approx(95.0, rel=0.01)
        assert new_state.soc_mwh == pytest.approx(initial_soc + 95.0, rel=0.01)

    def test_discharge_updates_soc(self, battery: BatteryModel) -> None:
        """Test that discharging decreases SOC correctly."""
        initial_soc = battery.current_state.soc_mwh

        ts = datetime(2025, 7, 15, 1, 0, 0)
        new_state, energy_delivered = battery.discharge(100.0, 1.0, ts)

        # Energy delivered = 100 MW * 1 hour = 100 MWh
        # Energy withdrawn from battery = 100 / 0.95 = ~105.26 MWh
        assert energy_delivered == pytest.approx(100.0, rel=0.01)
        expected_soc = initial_soc - (100.0 / 0.95)
        assert new_state.soc_mwh == pytest.approx(expected_soc, rel=0.01)

    def test_charge_respects_max_soc(
        self, battery: BatteryModel, config: BatteryConfig
    ) -> None:
        """Test that charging stops at max SOC."""
        # Initialize near max SOC
        battery.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.88)

        ts = datetime(2025, 7, 15, 1, 0, 0)
        new_state, _ = battery.charge(150.0, 1.0, ts)  # Try to charge a lot

        # Should not exceed 90% SOC
        max_soc_mwh = config.capacity_mwh * config.max_soc_fraction
        assert new_state.soc_mwh <= max_soc_mwh

    def test_discharge_respects_min_soc(
        self, battery: BatteryModel, config: BatteryConfig
    ) -> None:
        """Test that discharging stops at min SOC."""
        # Initialize near min SOC
        battery.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.12)

        ts = datetime(2025, 7, 15, 1, 0, 0)
        new_state, _ = battery.discharge(150.0, 1.0, ts)  # Try to discharge a lot

        # Should not go below 10% SOC
        min_soc_mwh = config.capacity_mwh * config.min_soc_fraction
        assert new_state.soc_mwh >= min_soc_mwh

    def test_get_max_charge_power(self, battery: BatteryModel) -> None:
        """Test max charge power calculation."""
        # At 50% SOC, should be able to charge at full power
        max_power = battery.get_max_charge_power()
        assert max_power == 150.0  # Limited by power rating

    def test_get_max_discharge_power(self, battery: BatteryModel) -> None:
        """Test max discharge power calculation."""
        # At 50% SOC, should be able to discharge at full power
        max_power = battery.get_max_discharge_power()
        assert max_power == 150.0  # Limited by power rating

    def test_charge_power_limited_near_max_soc(self, battery: BatteryModel) -> None:
        """Test that charge power is limited when near max SOC."""
        # Initialize at 85% SOC (only 5% headroom to max)
        battery.initialize(datetime(2025, 7, 15, 0, 0, 0), initial_soc_fraction=0.85)

        max_power = battery.get_max_charge_power()

        # Available capacity: 500 * (0.9 - 0.85) = 25 MWh
        # Accounting for efficiency: 25 / 0.95 = ~26.3 MW max
        assert max_power < 150.0
        assert max_power == pytest.approx(25 / 0.95, rel=0.01)

    def test_throughput_tracking(self, battery: BatteryModel) -> None:
        """Test that cumulative throughput is tracked correctly."""
        ts1 = datetime(2025, 7, 15, 1, 0, 0)
        ts2 = datetime(2025, 7, 15, 2, 0, 0)

        # Charge 50 MWh
        battery.charge(50.0, 1.0, ts1)
        throughput_after_charge = battery.current_state.cumulative_throughput_mwh

        # Discharge 30 MWh
        battery.discharge(30.0, 1.0, ts2)
        throughput_after_discharge = battery.current_state.cumulative_throughput_mwh

        # Throughput should increase with both operations
        assert throughput_after_charge > 0
        assert throughput_after_discharge > throughput_after_charge

    def test_soc_headroom(self, battery: BatteryModel) -> None:
        """Test SOC headroom calculation."""
        charge_headroom, discharge_headroom = battery.get_soc_headroom()

        # At 50% SOC with 10%-90% bounds:
        # Charge headroom: 500 * (0.9 - 0.5) = 200 MWh
        # Discharge headroom: 500 * (0.5 - 0.1) = 200 MWh
        assert charge_headroom == pytest.approx(200.0, rel=0.01)
        assert discharge_headroom == pytest.approx(200.0, rel=0.01)

    def test_uninitialized_battery_raises(self, config: BatteryConfig) -> None:
        """Test that operations on uninitialized battery raise errors."""
        model = BatteryModel(config)

        with pytest.raises(ValueError):
            model.charge(50.0, 1.0, datetime.now())

        with pytest.raises(ValueError):
            model.discharge(50.0, 1.0, datetime.now())


class TestDegradationModel:
    """Tests for DegradationModel."""

    @pytest.fixture
    def config(self) -> BatteryConfig:
        """Standard battery configuration."""
        return BatteryConfig(
            capacity_mwh=500.0,
            degradation_cost_per_mwh=8.0,
            min_soc_fraction=0.1,
            max_soc_fraction=0.9,
        )

    @pytest.fixture
    def model(self, config: BatteryConfig) -> DegradationModel:
        """Degradation model instance."""
        return DegradationModel(config)

    def test_calculate_cycle_cost(self, model: DegradationModel) -> None:
        """Test basic cycle cost calculation."""
        cost = model.calculate_cycle_cost(100.0)
        assert cost == 800.0  # 100 MWh * $8/MWh

    def test_calculate_efc(self, model: DegradationModel) -> None:
        """Test equivalent full cycle calculation."""
        # Usable capacity = 500 * (0.9 - 0.1) = 400 MWh
        # One EFC = 2 * 400 = 800 MWh throughput
        efc = model.calculate_efc(800.0)
        assert efc == pytest.approx(1.0, rel=0.01)

        efc_half = model.calculate_efc(400.0)
        assert efc_half == pytest.approx(0.5, rel=0.01)

    def test_calculate_incremental_cost(self, model: DegradationModel) -> None:
        """Test incremental cost between states."""
        ts = datetime(2025, 7, 15, 0, 0, 0)

        initial_state = BatteryState(
            timestamp=ts,
            soc_mwh=250.0,
            soc_fraction=0.5,
            cumulative_throughput_mwh=100.0,
        )
        final_state = BatteryState(
            timestamp=ts,
            soc_mwh=300.0,
            soc_fraction=0.6,
            cumulative_throughput_mwh=150.0,
        )

        cost = model.calculate_incremental_cost(initial_state, final_state)
        assert cost == 400.0  # 50 MWh delta * $8/MWh

    def test_dod_factor_shallow_cycle(self, model: DegradationModel) -> None:
        """Test DoD factor for shallow cycles."""
        # Shallow discharge (< 50% of usable capacity)
        factor = model.calculate_depth_of_discharge_factor(100.0)  # 100 / 400 = 25% DoD
        assert factor == 1.0

    def test_dod_factor_deep_cycle(self, model: DegradationModel) -> None:
        """Test DoD factor for deep cycles."""
        # Deep discharge (> 80% of usable capacity)
        factor = model.calculate_depth_of_discharge_factor(
            350.0
        )  # 350 / 400 = 87.5% DoD
        assert factor == 1.5

    def test_estimate_remaining_life(self, model: DegradationModel) -> None:
        """Test remaining life estimation."""
        ts = datetime(2025, 7, 15, 0, 0, 0)

        # 800 MWh throughput = 1 EFC
        state = BatteryState(
            timestamp=ts,
            soc_mwh=250.0,
            soc_fraction=0.5,
            cumulative_throughput_mwh=800.0,
        )

        remaining = model.estimate_remaining_life_cycles(state, total_cycle_life=4000.0)
        assert remaining == pytest.approx(3999.0, rel=0.01)


class TestOptimizationDegradationPenalty:
    """Tests for OptimizationDegradationPenalty."""

    @pytest.fixture
    def config(self) -> BatteryConfig:
        """Standard battery configuration."""
        return BatteryConfig(
            capacity_mwh=500.0,
            degradation_cost_per_mwh=8.0,
            min_soc_fraction=0.1,
            max_soc_fraction=0.9,
        )

    @pytest.fixture
    def penalty_calc(self, config: BatteryConfig) -> OptimizationDegradationPenalty:
        """Penalty calculator instance."""
        return OptimizationDegradationPenalty(config, penalty_multiplier=1.0)

    def test_charge_penalty(self, penalty_calc: OptimizationDegradationPenalty) -> None:
        """Test charge penalty calculation."""
        penalty = penalty_calc.calculate_charge_penalty(100.0)
        assert penalty == 800.0  # 100 MWh * $8/MWh

    def test_discharge_penalty_includes_dod(
        self, penalty_calc: OptimizationDegradationPenalty
    ) -> None:
        """Test discharge penalty includes DoD adjustment."""
        # Shallow discharge from 50% SOC (50/400 = 12.5% DoD)
        penalty_shallow = penalty_calc.calculate_discharge_penalty(50.0, 250.0)

        # Very deep discharge - 350 MWh (350/400 = 87.5% DoD)
        penalty_deep = penalty_calc.calculate_discharge_penalty(350.0, 400.0)

        # Deep discharge should have higher penalty per MWh (1.5x factor)
        assert penalty_deep / 350.0 > penalty_shallow / 50.0

    def test_cycling_penalty(
        self, penalty_calc: OptimizationDegradationPenalty
    ) -> None:
        """Test cycling penalty calculation."""
        # Charge and discharge in same period
        penalty = penalty_calc.calculate_cycling_penalty(100.0, 80.0)

        # Should include base throughput cost + rapid cycling penalty
        base_cost = (100.0 + 80.0) * 8.0  # $1440
        assert penalty > base_cost

    def test_penalty_multiplier(self, config: BatteryConfig) -> None:
        """Test that penalty multiplier affects calculations."""
        calc_1x = OptimizationDegradationPenalty(config, penalty_multiplier=1.0)
        calc_2x = OptimizationDegradationPenalty(config, penalty_multiplier=2.0)

        penalty_1x = calc_1x.calculate_charge_penalty(100.0)
        penalty_2x = calc_2x.calculate_charge_penalty(100.0)

        assert penalty_2x == 2 * penalty_1x
