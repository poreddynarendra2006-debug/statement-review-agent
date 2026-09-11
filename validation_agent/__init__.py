"""
validation_agent package

Exports the primary ValidationAgent, core accounting validation entrypoint run_all_validations,
the ValidationResult dataclass, and independent data-quality checks.
"""

from validation_agent.models import ValidationResult
from validation_agent.accounting import run_all_validations
from validation_agent.validator import ValidationAgent
from validation_agent.rules import (
    REQUIRED_COLUMNS,
    STANDARD_COLUMNS,
    NUMERIC_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    NOT_VALIDATABLE_METRICS,
    STATUS_PASS,
    STATUS_WARNING,
    STATUS_BLOCK,
    STATUS_FAIL,
    STATUS_SKIPPED,
    SEVERITY_NONE,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO
)
from validation_agent.checks import (
    validate_schema,
    validate_data_quality,
    validate_duplicates,
    validate_financial_formulas,
    validate_domain_sanity
)

__all__ = [
    "ValidationAgent",
    "ValidationResult",
    "run_all_validations",
    "REQUIRED_COLUMNS",
    "STANDARD_COLUMNS",
    "NUMERIC_COLUMNS",
    "NON_NEGATIVE_COLUMNS",
    "NOT_VALIDATABLE_METRICS",
    "STATUS_PASS",
    "STATUS_WARNING",
    "STATUS_BLOCK",
    "STATUS_FAIL",
    "STATUS_SKIPPED",
    "SEVERITY_NONE",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_MEDIUM",
    "SEVERITY_LOW",
    "SEVERITY_INFO",
    "validate_schema",
    "validate_data_quality",
    "validate_duplicates",
    "validate_financial_formulas",
    "validate_domain_sanity"
]
