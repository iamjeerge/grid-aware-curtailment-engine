"""Validation module for documenting real-world assumptions and constraints.

This module provides structured documentation of the assumptions, limitations,
and constraints that underpin the optimization model. It serves as both
documentation and validation infrastructure.

Exports:
    AssumptionCategory: Enum for categorizing assumptions.
    Assumption: Individual assumption with evidence and limitations.
    AssumptionRegistry: Central registry of all documented assumptions.
    ValidationReport: Structured report of assumption validity.
    validate_assumptions: Check assumptions against real-world data.
"""

from src.validation.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionRegistry,
    ValidationReport,
    ValidationSeverity,
    get_all_assumptions,
    get_assumption_registry,
    validate_assumptions,
)

__all__ = [
    "Assumption",
    "AssumptionCategory",
    "AssumptionRegistry",
    "ValidationReport",
    "ValidationSeverity",
    "get_all_assumptions",
    "get_assumption_registry",
    "validate_assumptions",
]
