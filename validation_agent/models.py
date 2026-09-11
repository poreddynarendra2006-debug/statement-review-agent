"""
validation_agent/models.py

Data models for the Validation Agent.
Defines the standardized ValidationResult dataclass expected by the orchestrator
and downstream Risk engines.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ValidationResult:
    """
    Validation finding returned for each accounting and ratio check.

    Attributes:
        rule_id: Unique rule identifier (e.g. VAL_BS_01, VAL_RATIO_ROE).
        rule_name: Human-readable name of the accounting rule.
        company: Company identifier / ticker.
        year: Fiscal year under audit.
        status: Validation status - PASS, FAIL, or SKIPPED.
        expected: The mathematically computed value, or None if skipped.
        actual: The reported financial value, or None if skipped.
        difference: actual - expected, or None if skipped.
        severity: Severity level - NONE, LOW, MEDIUM, HIGH, or CRITICAL.
        evidence: Human-readable string detailing the actual figures.
        formula: Mathematical formula defining the check.
        message: Neutral, explainable description of the outcome.
    """
    rule_id: str
    rule_name: str
    company: str
    year: int
    status: str            # PASS | FAIL | SKIPPED
    expected: Optional[float]
    actual: Optional[float]
    difference: Optional[float]
    severity: str          # NONE | LOW | MEDIUM | HIGH | CRITICAL
    evidence: str          # the actual numbers, readable
    formula: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize finding to a plain dictionary."""
        return asdict(self)
