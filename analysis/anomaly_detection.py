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


def _as_row(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return {k: v for k, v in vars(record).items() if not k.startswith("_")}


def run_all_anomaly_detection(records: Sequence[Any]) -> List[Any]:
    """Anomaly findings for a submission, most unusual first."""
    rows = [_as_row(r) for r in records]
    return detect_anomalies(rows, train_model(rows))
