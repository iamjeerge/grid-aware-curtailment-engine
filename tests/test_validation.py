"""Tests for the validation module.

Tests cover:
- Assumption documentation
- AssumptionRegistry functionality
- Validation report generation
"""

from __future__ import annotations

from src.validation import (
    Assumption,
    AssumptionCategory,
    AssumptionRegistry,
    ValidationReport,
    ValidationSeverity,
    get_all_assumptions,
    get_assumption_registry,
    validate_assumptions,
)
from src.validation.assumptions import ValidationResult

# --- Assumption Tests ---


class TestAssumption:
    """Tests for the Assumption dataclass."""

    def test_basic_assumption(self) -> None:
        """Test creating a basic assumption."""
        assumption = Assumption(
            id="TEST-001",
            category=AssumptionCategory.GRID,
            title="Test Assumption",
            description="This is a test assumption.",
        )

        assert assumption.id == "TEST-001"
        assert assumption.category == AssumptionCategory.GRID
        assert assumption.title == "Test Assumption"
        assert assumption.description == "This is a test assumption."
        assert assumption.rationale == ""
        assert assumption.evidence == []
        assert assumption.limitations == []

    def test_full_assumption(self) -> None:
        """Test creating a fully specified assumption."""
        assumption = Assumption(
            id="TEST-002",
            category=AssumptionCategory.BATTERY,
            title="Full Test Assumption",
            description="Complete assumption with all fields.",
            rationale="Testing all fields",
            evidence=["Evidence 1", "Evidence 2"],
            limitations=["Limitation 1"],
            impact_if_violated="Significant impact",
            validation_method="Compare to actuals",
            default_value=100,
            valid_range=(50, 200),
            unit="MW",
        )

        assert assumption.id == "TEST-002"
        assert len(assumption.evidence) == 2
        assert len(assumption.limitations) == 1
        assert assumption.default_value == 100
        assert assumption.valid_range == (50, 200)
        assert assumption.unit == "MW"


# --- AssumptionCategory Tests ---


class TestAssumptionCategory:
    """Tests for AssumptionCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test that all expected categories exist."""
        categories = list(AssumptionCategory)

        assert AssumptionCategory.GRID in categories
        assert AssumptionCategory.BATTERY in categories
        assert AssumptionCategory.MARKET in categories
        assert AssumptionCategory.FORECAST in categories
        assert AssumptionCategory.REGULATORY in categories
        assert AssumptionCategory.OPERATIONAL in categories

    def test_category_values(self) -> None:
        """Test category string values."""
        assert AssumptionCategory.GRID.value == "grid"
        assert AssumptionCategory.BATTERY.value == "battery"
        assert AssumptionCategory.MARKET.value == "market"


# --- ValidationResult Tests ---


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(
            assumption_id="TEST-001",
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Assumption is valid",
        )

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.INFO

    def test_invalid_result_with_details(self) -> None:
        """Test creating an invalid result with details."""
        result = ValidationResult(
            assumption_id="TEST-002",
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            message="Value outside range",
            actual_value=1000,
            expected_range=(50, 500),
        )

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.actual_value == 1000
        assert result.expected_range == (50, 500)


# --- ValidationReport Tests ---


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty validation report."""
        report = ValidationReport()

        assert report.total_assumptions == 0
        assert report.valid_count == 0
        assert report.is_valid is True
        assert "0/0 valid" in report.summary

    def test_add_valid_result(self) -> None:
        """Test adding a valid result."""
        report = ValidationReport()
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Valid",
            )
        )

        assert report.total_assumptions == 1
        assert report.valid_count == 1
        assert report.is_valid is True

    def test_add_warning_result(self) -> None:
        """Test adding a warning result."""
        report = ValidationReport()
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Warning",
            )
        )

        assert report.total_assumptions == 1
        assert report.valid_count == 0
        assert report.warning_count == 1
        assert report.is_valid is True  # Warnings don't fail validation

    def test_add_error_result(self) -> None:
        """Test adding an error result."""
        report = ValidationReport()
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Error",
            )
        )

        assert report.error_count == 1
        assert report.is_valid is False

    def test_add_critical_result(self) -> None:
        """Test adding a critical result."""
        report = ValidationReport()
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Critical",
            )
        )

        assert report.critical_count == 1
        assert report.is_valid is False

    def test_mixed_results(self) -> None:
        """Test report with mixed results."""
        report = ValidationReport()

        # Add valid
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Valid",
            )
        )

        # Add warning
        report.add_result(
            ValidationResult(
                assumption_id="TEST-002",
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Warning",
            )
        )

        # Add error
        report.add_result(
            ValidationResult(
                assumption_id="TEST-003",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Error",
            )
        )

        assert report.total_assumptions == 3
        assert report.valid_count == 1
        assert report.warning_count == 1
        assert report.error_count == 1
        assert report.is_valid is False

    def test_summary_format(self) -> None:
        """Test summary string format."""
        report = ValidationReport()
        report.add_result(
            ValidationResult(
                assumption_id="TEST-001",
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Valid",
            )
        )

        summary = report.summary
        assert "1/1 valid" in summary
        assert "0 warnings" in summary
        assert "0 errors" in summary


# --- AssumptionRegistry Tests ---


class TestAssumptionRegistry:
    """Tests for AssumptionRegistry class."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes with assumptions."""
        registry = AssumptionRegistry()
        assumptions = registry.get_all()

        assert len(assumptions) > 0

    def test_get_assumption_by_id(self) -> None:
        """Test getting assumption by ID."""
        registry = AssumptionRegistry()

        # Get a known assumption
        assumption = registry.get("GRID-001")

        assert assumption is not None
        assert assumption.id == "GRID-001"
        assert assumption.category == AssumptionCategory.GRID

    def test_get_nonexistent_assumption(self) -> None:
        """Test getting a nonexistent assumption."""
        registry = AssumptionRegistry()

        assumption = registry.get("NONEXISTENT-001")

        assert assumption is None

    def test_get_by_category_grid(self) -> None:
        """Test getting grid assumptions."""
        registry = AssumptionRegistry()

        grid_assumptions = registry.get_by_category(AssumptionCategory.GRID)

        assert len(grid_assumptions) > 0
        for a in grid_assumptions:
            assert a.category == AssumptionCategory.GRID

    def test_get_by_category_battery(self) -> None:
        """Test getting battery assumptions."""
        registry = AssumptionRegistry()

        battery_assumptions = registry.get_by_category(AssumptionCategory.BATTERY)

        assert len(battery_assumptions) > 0
        for a in battery_assumptions:
            assert a.category == AssumptionCategory.BATTERY

    def test_get_by_category_market(self) -> None:
        """Test getting market assumptions."""
        registry = AssumptionRegistry()

        market_assumptions = registry.get_by_category(AssumptionCategory.MARKET)

        assert len(market_assumptions) > 0
        for a in market_assumptions:
            assert a.category == AssumptionCategory.MARKET

    def test_get_by_category_forecast(self) -> None:
        """Test getting forecast assumptions."""
        registry = AssumptionRegistry()

        forecast_assumptions = registry.get_by_category(AssumptionCategory.FORECAST)

        assert len(forecast_assumptions) > 0
        for a in forecast_assumptions:
            assert a.category == AssumptionCategory.FORECAST

    def test_get_by_category_regulatory(self) -> None:
        """Test getting regulatory assumptions."""
        registry = AssumptionRegistry()

        regulatory_assumptions = registry.get_by_category(AssumptionCategory.REGULATORY)

        assert len(regulatory_assumptions) > 0
        for a in regulatory_assumptions:
            assert a.category == AssumptionCategory.REGULATORY

    def test_get_by_category_operational(self) -> None:
        """Test getting operational assumptions."""
        registry = AssumptionRegistry()

        operational_assumptions = registry.get_by_category(
            AssumptionCategory.OPERATIONAL
        )

        assert len(operational_assumptions) > 0
        for a in operational_assumptions:
            assert a.category == AssumptionCategory.OPERATIONAL

    def test_register_custom_assumption(self) -> None:
        """Test registering a custom assumption."""
        registry = AssumptionRegistry()

        custom = Assumption(
            id="CUSTOM-001",
            category=AssumptionCategory.GRID,
            title="Custom Assumption",
            description="Test custom assumption",
        )
        registry.register(custom)

        retrieved = registry.get("CUSTOM-001")
        assert retrieved is not None
        assert retrieved.title == "Custom Assumption"

    def test_to_markdown(self) -> None:
        """Test exporting assumptions to markdown."""
        registry = AssumptionRegistry()

        markdown = registry.to_markdown()

        assert "# Model Assumptions Documentation" in markdown
        assert "## Grid Assumptions" in markdown
        assert "## Battery Assumptions" in markdown
        assert "GRID-001" in markdown
        assert "BATT-001" in markdown


# --- Module Functions Tests ---


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_assumption_registry(self) -> None:
        """Test getting the global registry."""
        registry = get_assumption_registry()

        assert isinstance(registry, AssumptionRegistry)
        assert len(registry.get_all()) > 0

    def test_get_assumption_registry_singleton(self) -> None:
        """Test registry is a singleton."""
        registry1 = get_assumption_registry()
        registry2 = get_assumption_registry()

        assert registry1 is registry2

    def test_get_all_assumptions(self) -> None:
        """Test getting all assumptions."""
        assumptions = get_all_assumptions()

        assert isinstance(assumptions, list)
        assert len(assumptions) > 0
        assert all(isinstance(a, Assumption) for a in assumptions)


# --- validate_assumptions Tests ---


class TestValidateAssumptions:
    """Tests for the validate_assumptions function."""

    def test_validate_grid_capacity_valid(self) -> None:
        """Test validating grid capacity within range."""
        report = validate_assumptions(grid_capacity_mw=300)

        assert report.total_assumptions == 1
        result = report.results[0]
        assert result.assumption_id == "GRID-001"
        assert result.is_valid is True

    def test_validate_grid_capacity_invalid(self) -> None:
        """Test validating grid capacity outside range."""
        report = validate_assumptions(grid_capacity_mw=5000)

        result = report.results[0]
        assert result.is_valid is False
        assert result.severity == ValidationSeverity.WARNING

    def test_validate_battery_efficiency_valid(self) -> None:
        """Test validating battery efficiency within range."""
        report = validate_assumptions(battery_efficiency=0.90)

        result = report.results[0]
        assert result.assumption_id == "BATT-001"
        assert result.is_valid is True

    def test_validate_battery_efficiency_invalid(self) -> None:
        """Test validating battery efficiency outside range."""
        report = validate_assumptions(battery_efficiency=0.50)

        result = report.results[0]
        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR

    def test_validate_battery_power_valid(self) -> None:
        """Test validating battery power within range."""
        report = validate_assumptions(battery_power_mw=150)

        result = report.results[0]
        assert result.assumption_id == "BATT-004"
        assert result.is_valid is True

    def test_validate_degradation_cost_valid(self) -> None:
        """Test validating degradation cost within range."""
        report = validate_assumptions(degradation_cost=8.0)

        result = report.results[0]
        assert result.assumption_id == "BATT-002"
        assert result.is_valid is True

    def test_validate_forecast_horizon_valid(self) -> None:
        """Test validating forecast horizon within range."""
        report = validate_assumptions(forecast_horizon_hours=24)

        result = report.results[0]
        assert result.assumption_id == "FCST-002"
        assert result.is_valid is True

    def test_validate_multiple_params(self) -> None:
        """Test validating multiple parameters at once."""
        report = validate_assumptions(
            grid_capacity_mw=300,
            battery_efficiency=0.90,
            battery_power_mw=150,
            degradation_cost=8.0,
            forecast_horizon_hours=24,
        )

        assert report.total_assumptions == 5
        assert report.valid_count == 5
        assert report.is_valid is True

    def test_validate_with_invalid_params(self) -> None:
        """Test validation with some invalid parameters."""
        report = validate_assumptions(
            grid_capacity_mw=300,  # Valid
            battery_efficiency=0.50,  # Invalid - too low
            battery_power_mw=150,  # Valid
        )

        assert report.total_assumptions == 3
        assert report.valid_count == 2
        assert report.error_count == 1
        assert report.is_valid is False


# --- Assumption Content Tests ---


class TestAssumptionContent:
    """Tests to verify assumption content quality."""

    def test_all_assumptions_have_descriptions(self) -> None:
        """Test that all assumptions have descriptions."""
        assumptions = get_all_assumptions()

        for a in assumptions:
            assert a.description, f"{a.id} missing description"

    def test_all_assumptions_have_unique_ids(self) -> None:
        """Test that all assumption IDs are unique."""
        assumptions = get_all_assumptions()
        ids = [a.id for a in assumptions]

        assert len(ids) == len(set(ids)), "Duplicate assumption IDs found"

    def test_assumption_id_format(self) -> None:
        """Test assumption ID format."""
        assumptions = get_all_assumptions()

        for a in assumptions:
            parts = a.id.split("-")
            assert len(parts) == 2, f"{a.id} has invalid format"
            assert parts[0].isupper(), f"{a.id} prefix should be uppercase"
            assert parts[1].isdigit(), f"{a.id} suffix should be numeric"

    def test_minimum_assumptions_per_category(self) -> None:
        """Test that each category has at least one assumption."""
        registry = get_assumption_registry()

        for category in AssumptionCategory:
            assumptions = registry.get_by_category(category)
            assert len(assumptions) >= 1, f"{category.value} has no assumptions"
