"""Shared objects and settings for the API.

The orchestrator is built once and reused. Constructing it per request would
reload any model weights on every call, which is the difference between a
fast endpoint and an unusable one.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from agents.orchestrator import ReviewOrchestrator

VERSION = "0.1.0"


class Record:
    """A submitted company-year, as the analysis agents expect it.

    Attribute access rather than dictionary access, because every component
    reads records with getattr. Unknown fields are kept so a column we do not
    model yet still reaches whatever wants it.
    """

    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> None:
        # Any field not supplied reads as None, so a check whose input is
        # missing skips rather than raising.
        return None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Record {getattr(self, 'company', '?')} {getattr(self, 'year', '?')}>"


def to_records(payload: Sequence[Any]) -> List[Record]:
    """Turn validated request models into records the agents can read."""
    out: List[Record] = []
    for item in payload:
        fields = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        out.append(Record(**fields))
    return out


def build_orchestrator() -> ReviewOrchestrator:
    """Wire in whichever components are importable.

    Components arrive at different times. Anything not yet available is left
    out, and the planner reports it as skipped rather than the API failing -
    so the endpoint is usable from the first day rather than the last.
    """
    components: Dict[str, Any] = {}

    try:
        from analysis.validation import run_all_validations
        components["validation"] = run_all_validations
    except Exception:
        pass

    try:
        from analysis.trend import run_trend_analysis
        components["trend"] = run_trend_analysis
    except Exception:
        pass

    try:
        from analysis.anomaly_detection import run_all_anomaly_detection
        components["anomaly"] = run_all_anomaly_detection
    except Exception:
        pass

    try:
        from analysis.recurring_issues import detect_recurring_issues
        components["recurring"] = detect_recurring_issues
    except Exception:
        pass

    try:
        from agents.evidence_agent import compile_all_findings
        components["evidence"] = compile_all_findings
    except Exception:
        pass

    try:
        from agents.review_agent import write_review
        components["review"] = write_review
    except Exception:
        pass

    try:
        from risk_reporting.risk_engine import calculate_risk
        components["risk"] = calculate_risk
    except Exception:
        pass

    return ReviewOrchestrator(**components)


@lru_cache(maxsize=1)
def get_orchestrator() -> ReviewOrchestrator:
    """The single shared orchestrator instance."""
    return build_orchestrator()


def component_status() -> Dict[str, bool]:
    """Which components are wired in. Reported by the health endpoint."""
    orchestrator = get_orchestrator()
    status = {name: fn is not None for name, fn in orchestrator._components.items()}
    status["evidence"] = orchestrator._evidence is not None
    status["review"] = orchestrator._review is not None
    status["risk"] = orchestrator._risk is not None
    return status


def settings() -> Dict[str, Any]:
    """Runtime configuration, all from the environment."""
    return {
        "version": VERSION,
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "heuristic"),
    }
