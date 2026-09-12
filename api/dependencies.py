"""Shared objects and settings for the API.

The orchestrator is built once and reused. Constructing it per request would
reload any model weights on every call, which is the difference between a
fast endpoint and an unusable one.

Every teammate's component is found by trying the places it can live, rather
than by a hard import. A component that isn't installed yet is simply left
out: the planner reports its agent as skipped, and endpoints that need it
answer 503 with a clear message instead of the whole service failing to start.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agents.orchestrator import AnalysisResult, ReviewOrchestrator

logger = logging.getLogger("api.dependencies")

VERSION = "0.2.0"

#: Largest upload accepted, from MAX_UPLOAD_MB. Statement files are small;
#: this stops an accidental multi-gigabyte upload exhausting container memory.
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

ALLOWED_UPLOAD_SUFFIXES: Tuple[str, ...] = (".csv", ".xlsx")

#: Where the reviewer's screens live. Served from / when index.html exists.
UI_DIR = os.environ.get("UI_DIR") or str(Path(__file__).resolve().parent.parent / "ui")

#: Where each component's entry point may live. First match wins, so a
#: teammate's package plugs in wherever it lands without editing this file.
COMPONENT_ENTRY_POINTS: Dict[str, List[Tuple[str, str]]] = {
    "validation": [
        ("validation_agent", "run_all_validations"),
        ("validation_agent.accounting", "run_all_validations"),
        ("analysis.validation", "run_all_validations"),
    ],
    "trend": [("analysis.trend", "run_trend_analysis")],
    "anomaly": [("analysis.anomaly_detection", "run_all_anomaly_detection")],
    "recurring": [("analysis.recurring_issues", "detect_recurring_issues")],
    "peer": [("analysis.peer_comparison", "compare_with_peers")],
    "evidence": [("agents.evidence_agent", "compile_all_findings")],
    "review": [("agents.review_agent", "write_review")],
}

RISK_ENTRY_POINTS: List[Tuple[str, str]] = [
    ("risk_reporting.risk_engine", "calculate_risk"),
    ("risk_reporting", "calculate_risk"),
]

INGESTION_ENTRY_POINTS: List[Tuple[str, str]] = [
    ("extraction", "ingest_financial_statement"),
    ("extraction.pipeline", "ingest_financial_statement"),
    ("ingestion", "ingest_financial_statement"),
    ("ingestion.pipeline", "ingest_financial_statement"),
]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


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
        # missing skips rather than raising. Dunder lookups must still fail,
        # or copy and pickle recurse forever.
        if name.startswith("__"):
            raise AttributeError(name)
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


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def records_from_ingestion(result: Any) -> List[Any]:
    """Records from an ingestion result.

    Uses Ingestion's own to_records() when it has one. Otherwise falls back to
    its DataFrame, turning blank cells into None so they skip checks rather
    than poisoning arithmetic with NaN.
    """
    if hasattr(result, "to_records"):
        return list(result.to_records())
    data = getattr(result, "data", None)
    if data is None:
        return []
    return [
        Record(**{k: (None if _is_missing(v) else v) for k, v in row.items()})
        for row in data.to_dict(orient="records")
    ]


def ingestion_warnings(result: Any) -> List[str]:
    """Warnings Ingestion raised while reading the file, for the reviewer."""
    metadata = getattr(result, "metadata", None)
    warnings = getattr(metadata, "warnings", None) or []
    return [str(w) for w in warnings]


def full_result(review: Dict[str, Any]) -> Dict[str, Any]:
    """The complete stored review result, however the database returns it."""
    for key in ("result", "result_json"):
        value = review.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                pass
    return review


# ---------------------------------------------------------------------------
# Locating components
# ---------------------------------------------------------------------------


def _try_import(module_name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(module_name)
    except Exception:  # not installed, or failing on import - both mean absent
        return None


def _first_available(candidates: Sequence[Tuple[str, str]]) -> Optional[Callable[..., Any]]:
    for module_name, attr in candidates:
        module = _try_import(module_name)
        fn = getattr(module, attr, None) if module is not None else None
        if callable(fn):
            return fn
    return None


def risk_adapter(calculate_risk: Callable[..., Any]) -> Callable[[AnalysisResult], Any]:
    """Adapt the risk engine to the orchestrator.

    The orchestrator hands over the whole AnalysisResult; the risk engine
    expects findings grouped by the agent that produced them. Only findings
    that can carry risk are passed: failed checks, anomalies and material
    deviations.

    Three kinds of finding are deliberately left out, each measured rather than
    assumed, so that nobody later "fixes" the omission and breaks the score:

    - **Peer comparison.** Being unlike one's peers is not a defect; a
      specialist lender genuinely carries more debt than a software firm.
      Including it took clean books from 44 MEDIUM to 82 CRITICAL.
    - **Ratio health.** The trend agent grades roughly 210 ratios CRITICAL on
      both our datasets, because a high debt ratio is a characteristic, not an
      error. Including them took both files to 100 CRITICAL, clean and
      defective alike, which tells a reviewer nothing.
    - **Recurring issues.** They are derived from failures already scored here,
      so counting them again would charge the same problem twice.

    All three still reach the reviewer as findings. The score is for things
    that are wrong; those are things worth a look.
    """

    def score(result: AnalysisResult) -> Any:
        return calculate_risk({
            "validation": result.failed_validations,
            "anomaly": result.anomalies,
            "trend": result.material_deviations,
        })

    return score


def validation_adapter(run_all_validations: Callable[..., Any]) -> Callable[..., List[Any]]:
    """Run the accounting checks, then the data-quality ones.

    The Validation package ships both, but its entry point runs only the
    accounting formulas, so negative revenue, a duplicated company-year and a
    blank company name went unreported. They are failures of fact rather than
    judgement, so they belong in the same list as a failed identity.
    """

    def validate(records: Sequence[Any], **options: Any) -> List[Any]:
        results = list(run_all_validations(records, **options) or [])
        try:
            from agents import data_quality
        except Exception:  # noqa: BLE001 - the checks are part of the same package
            return results
        return results + data_quality.run_data_quality_checks(records)

    # The planner passes only the options a function accepts, by signature, so
    # the wrapper must keep the original's.
    validate.__signature__ = inspect.signature(run_all_validations)
    return validate


def build_orchestrator() -> ReviewOrchestrator:
    """Wire in whichever components are installed."""
    components: Dict[str, Any] = {
        name: _first_available(candidates)
        for name, candidates in COMPONENT_ENTRY_POINTS.items()
    }
    if components.get("validation") is not None:
        components["validation"] = validation_adapter(components["validation"])

    calculate_risk = _first_available(RISK_ENTRY_POINTS)
    components["risk"] = risk_adapter(calculate_risk) if calculate_risk else None

    wired = sorted(name for name, fn in components.items() if fn is not None)
    logger.info("review components wired: %s", ", ".join(wired) or "none yet")
    return ReviewOrchestrator(**{k: v for k, v in components.items() if v is not None})


@lru_cache(maxsize=1)
def get_orchestrator() -> ReviewOrchestrator:
    """The single shared orchestrator instance."""
    return build_orchestrator()


@lru_cache(maxsize=1)
def get_ingestion() -> Optional[Callable[..., Any]]:
    """Data Ingestion's entry point, or None until it is installed."""
    return _first_available(INGESTION_ENTRY_POINTS)


class Reporting:
    """Risk & Reporting's storage, PDF and monitoring modules."""

    def __init__(self, database: ModuleType, monitoring: Optional[ModuleType],
                 report: Optional[ModuleType]) -> None:
        self.database = database
        self.monitoring = monitoring
        self.report = report


@lru_cache(maxsize=1)
def get_reporting() -> Optional[Reporting]:
    """Risk & Reporting, or None until its database module is installed."""
    database = _try_import("database.database")
    if database is None or not hasattr(database, "save_review"):
        return None
    return Reporting(
        database=database,
        monitoring=_try_import("database.monitoring"),
        report=_try_import("reports.report_generator"),
    )


def prepare_storage() -> None:
    """Create tables and seed demo reviews at startup.

    Container filesystems are wiped on every restart, so without seeding a
    freshly started service would show an empty history to whoever opens it
    first. Failure here is logged, never fatal: reviews still run.
    """
    reporting = get_reporting()
    if reporting is None:
        logger.info("Risk & Reporting not installed; review history disabled")
        return
    try:
        reporting.database.init_db()
        reporting.database.seed_demo_data()
    except Exception:  # noqa: BLE001 - storage must not stop the service
        logger.exception("could not prepare review storage")


def component_status() -> Dict[str, bool]:
    """Which components are installed. Reported by the health endpoint."""
    orchestrator = get_orchestrator()
    status = {name: fn is not None for name, fn in orchestrator._components.items()}
    status["evidence"] = orchestrator._evidence is not None
    status["review"] = orchestrator._review is not None
    status["risk"] = orchestrator._risk is not None
    status["ingestion"] = get_ingestion() is not None
    status["reporting"] = get_reporting() is not None
    return status
