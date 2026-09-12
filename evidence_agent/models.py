"""Data models for the FinSight Evidence Agent.

Defines standardized, typed evidence items for each upstream source
(Validation, Trend, Anomaly) and the unified EvidencePacket for the Review Agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceSource(str, Enum):
    """Originating upstream agent source."""
    VALIDATION = "validation"
    TREND = "trend"
    ANOMALY = "anomaly"
    RECURRING = "recurring"

    def __str__(self) -> str:
        return self.value


@dataclass
class ValidationEvidenceItem:
    """Normalized evidence item representing an upstream accounting or ratio validation failure.

    Preserves the exact figures and messages computed by the Validation Agent.
    Does not recalculate expected, actual, or difference values.
    """
    rule_id: str
    rule_name: str
    company: str
    year: int
    status: str
    severity: str
    expected: Optional[float]
    actual: Optional[float]
    difference: Optional[float]
    evidence_text: str
    formula: str
    message: str
    source: str = EvidenceSource.VALIDATION.value
    raw_finding: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item to dictionary."""
        return asdict(self)


@dataclass
class TrendDeviationEvidenceItem:
    """Normalized evidence item representing a material actual vs. forecast deviation.

    Preserves the exact forecasting, actuals, and deviation metrics computed by the Trend Agent.
    Does not recalculate deviation, deviation percentages, or ratios.
    """
    company: str
    year: int
    metric: str
    actual: float
    forecast: float
    deviation: float
    deviation_percent: Optional[float]
    absolute_deviation: float
    absolute_deviation_percent: Optional[float]
    actual_to_forecast_ratio: Optional[float]
    direction: str
    material_deviation: bool
    materiality_threshold: float
    notes: Optional[str] = None
    source: str = EvidenceSource.TREND.value
    raw_finding: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item to dictionary."""
        return asdict(self)


@dataclass
class AnomalyEvidenceItem:
    """Normalized evidence item representing a statistically unusual financial pattern.

    Preserves all detection metadata, relevant features, deviations, and explanations
    provided by the Anomaly Agent. Never makes accusations of fraud, misconduct,
    or manipulation.
    """
    company: Optional[str]
    year: Optional[int]
    record_id: Optional[str]
    anomaly_type: str
    score: float
    severity: str
    confidence: float
    relevant_features: Dict[str, float] = field(default_factory=dict)
    contributing_features: Dict[str, float] = field(default_factory=dict)
    deviations: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommendation: str = ""
    model_name: str = "IsolationForest"
    model_mode: str = "GENERIC_LOCAL"
    percentile: Optional[float] = None
    actual_values: Dict[str, Any] = field(default_factory=dict)
    normal_ranges: Dict[str, Any] = field(default_factory=dict)
    source: str = EvidenceSource.ANOMALY.value
    raw_finding: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item to dictionary."""
        return asdict(self)


@dataclass
class RecurringIssueEvidenceItem:
    """Normalized evidence item representing a recurring financial issue across fiscal years.

    Supplied by the upstream recurring issues agent. Preserves exact upstream fields
    without recalculating or fabricating values.
    """
    company: str
    source: str
    key: str
    issue: str
    years: List[int]
    consecutive: bool
    severity: str
    evidence: str
    occurrences: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence item to dictionary."""
        return asdict(self)


@dataclass
class ScreenedDocumentEvidence:
    """Screened user document text and security flags from the orchestrator guardrail.

    Strict security rule: Only screened_text (from screened_text_for_prompt)
    is exposed for downstream LLM/model prompts. Original unscreened text
    is strictly omitted to prevent prompt injection and instruction override.
    """
    screened_text: str
    is_suspicious: bool = False
    categories: List[str] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document evidence to dictionary."""
        return asdict(self)


@dataclass
class EvidencePacket:
    """Consolidated, structured Evidence Packet for consumption by the Review Agent.

    Organizes findings strictly by upstream source (validation, trend, anomaly, recurring),
    retaining original upstream values, and preserves screened document text
    with prompt-injection markers separately.
    """
    validation_findings: List[ValidationEvidenceItem] = field(default_factory=list)
    trend_findings: List[TrendDeviationEvidenceItem] = field(default_factory=list)
    anomaly_findings: List[AnomalyEvidenceItem] = field(default_factory=list)
    recurring_issues: List[RecurringIssueEvidenceItem] = field(default_factory=list)
    screened_documents: List[ScreenedDocumentEvidence] = field(default_factory=list)
    security_flags: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_findings_count(self) -> int:
        """Total count of findings across all upstream analytical agents."""
        return (
            len(self.validation_findings)
            + len(self.trend_findings)
            + len(self.anomaly_findings)
            + len(self.recurring_issues)
        )

    def companies_covered(self) -> List[str]:
        """Return a sorted list of unique company tickers/names with findings."""
        comps = set()
        for v in self.validation_findings:
            if v.company:
                comps.add(v.company)
        for t in self.trend_findings:
            if t.company:
                comps.add(t.company)
        for a in self.anomaly_findings:
            if a.company:
                comps.add(a.company)
        for r in self.recurring_issues:
            if r.company:
                comps.add(r.company)
        return sorted(comps)

    def get_company_findings(self, company: str) -> Dict[str, List[Any]]:
        """Filter findings for a specific company."""
        comp_clean = str(company).strip().upper()
        return {
            "validation": [
                v for v in self.validation_findings if str(v.company).strip().upper() == comp_clean
            ],
            "trend": [
                t for t in self.trend_findings if str(t.company).strip().upper() == comp_clean
            ],
            "anomaly": [
                a for a in self.anomaly_findings if str(a.company).strip().upper() == comp_clean
            ],
            "recurring": [
                r for r in self.recurring_issues if str(r.company).strip().upper() == comp_clean
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete evidence packet to dictionary."""
        return {
            "total_findings": self.total_findings_count,
            "companies_covered": self.companies_covered(),
            "metadata": self.metadata,
            "validation_findings": [f.to_dict() for f in self.validation_findings],
            "trend_findings": [f.to_dict() for f in self.trend_findings],
            "anomaly_findings": [f.to_dict() for f in self.anomaly_findings],
            "recurring_issues": [r.to_dict() for r in self.recurring_issues],
            "screened_documents": [d.to_dict() for d in self.screened_documents],
            "security_flags": self.security_flags,
        }
