"""Data models for the FinSight Review Agent.

Defines the structured ReviewResult output contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReviewResult:
    """Structured review narrative and findings synthesis produced by the Review Agent.

    ASSUMPTION — TEAM REVIEW REQUIRED:
    As no pre-existing ReviewResult contract was provided in the project repository,
    this minimal dataclass defines the structured review output reflecting the exact
    analytical sections mandated by the project requirements.
    """
    summary: str
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    trend_deviations: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_observations: List[Dict[str, Any]] = field(default_factory=list)
    recurring_issues: List[Dict[str, Any]] = field(default_factory=list)
    document_security_observations: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_priorities: List[str] = field(default_factory=list)
    limitations_and_uncertainties: List[str] = field(default_factory=list)
    generation_mode: str = "deterministic_fallback"  # "llm" or "deterministic_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewResult to dictionary."""
        return asdict(self)
