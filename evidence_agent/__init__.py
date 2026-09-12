"""FinSight Evidence Agent package.

Normalizes and consolidates audit findings across Validation, Trend, and Anomaly agents,
incorporating screened document text safely under orchestrator security guardrails.
"""

from evidence_agent.models import (
    AnomalyEvidenceItem,
    EvidencePacket,
    EvidenceSource,
    RecurringIssueEvidenceItem,
    ScreenedDocumentEvidence,
    TrendDeviationEvidenceItem,
    ValidationEvidenceItem,
)
from evidence_agent.agent import (
    EvidenceAgent,
    compile_all_findings,
)

__all__ = [
    "EvidenceAgent",
    "compile_all_findings",
    "EvidencePacket",
    "EvidenceSource",
    "ValidationEvidenceItem",
    "TrendDeviationEvidenceItem",
    "AnomalyEvidenceItem",
    "RecurringIssueEvidenceItem",
    "ScreenedDocumentEvidence",
]
