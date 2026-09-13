"""Connects the Anomaly agent in finsight/ to the review pipeline.

The orchestrator hands over records as objects, while finsight reads rows as
dictionaries. finsight also learns what normal looks like from the upload
itself, so a model is trained on each submission before it is scored. The
planner only runs this when there are enough company-years for that to mean
something.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from finsight.analysis.anomaly_detection import detect_anomalies, train_model

from .forensic_flags import detect_forensic_flags


def _as_row(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return {k: v for k, v in vars(record).items() if not k.startswith("_")}


def _identity(finding: Any) -> tuple:
    kind = getattr(finding, "anomaly_type", None)
    return (getattr(finding, "company", None), getattr(finding, "year", None),
            getattr(kind, "value", kind))


def run_all_anomaly_detection(records: Sequence[Any]) -> List[Any]:
    """Anomaly findings for a submission, most unusual first.

    Two detectors, not one. The model reports what is unusual *for this file*;
    the forensic rules report patterns that are suspicious whatever the rest of
    the file looks like - revenue that does not turn into cash, profit with
    negative cash flow, a margin that collapses. A model trained on a single
    upload cannot learn those, because if several companies share the pattern it
    stops looking unusual.
    """
    rows = [_as_row(r) for r in records]
    findings = list(detect_anomalies(rows, train_model(rows)) or [])

    seen = {_identity(f) for f in findings}
    for flag in detect_forensic_flags(rows):
        if _identity(flag) not in seen:
            findings.append(flag)
            seen.add(_identity(flag))
    return findings
