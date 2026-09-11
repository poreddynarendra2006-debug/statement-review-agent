"""Explainable risk scoring for financial-statement review outputs.

The scorer accepts either the component dataclass objects or dictionaries with
the same fields.  It intentionally does not require the upstream components to
be imported, so it can also be used as a small standalone service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log1p
from typing import Any, Iterable, Mapping, Sequence

from .sample_data.risk_rules import SEVERITY_SCORES, get_risk_level


# Additional findings of the same severity have a diminishing effect.
# This keeps a large number of medium findings from simply saturating at 100.
DIMINISHING_EXPONENT = 0.35

# A small density adjustment makes a case with many findings slightly riskier,
# without allowing count alone to dominate severity.
DENSITY_WEIGHT = 0.10
DENSITY_BASE = 10


@dataclass(frozen=True)
class RiskContributor:
    reason: str
    points: int
    source_agent: str
    finding_reference: Any


@dataclass(frozen=True)
class RiskScoreResult:
    score: int
    risk_level: str
    badge_color: str
    summary: str
    contributors: list[RiskContributor]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return asdict(self)

    # Backward-compatible aliases for callers of the old dictionary API.
    @property
    def risk_score(self) -> int:
        return self.score

    @property
    def findings(self) -> list[dict[str, Any]]:
        return [
            {
                "agent": c.source_agent,
                "finding": c.finding_reference,
                "score": c.points,
                "reason": c.reason,
            }
            for c in self.contributors
        ]


@dataclass(frozen=True)
class _Finding:
    source_agent: str
    severity: str
    reason: str
    reference: Any


def _get(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _status(value: Any) -> str:
    return str(value).upper() if value is not None else ""


def _normalise_severity(value: Any) -> str:
    severity = _status(value)
    return severity if severity in SEVERITY_SCORES else "LOW"


def _reference(obj: Any) -> Any:
    """Create a stable, serialisable reference for a finding."""
    if isinstance(obj, Mapping):
        # Keep the useful identifying fields, including the original finding
        # text when supplied by legacy callers.
        keys = (
            "finding", "rule_id", "rule_name", "year", "company", "record_id",
            "anomaly_type", "metric", "ratio_name",
        )
        return {k: obj[k] for k in keys if k in obj}
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else {
        k: getattr(obj, k) for k in dir(obj)
        if not k.startswith("_") and not callable(getattr(obj, k))
    }
    keys = (
        "finding", "rule_id", "rule_name", "year", "company", "record_id",
        "anomaly_type", "metric", "ratio_name",
    )
    return {k: data[k] for k in keys if k in data}


def _reason(source: str, obj: Any, severity: str) -> str:
    if source == "validation":
        rule = _get(obj, "rule_name") or _get(obj, "rule_id") or "validation rule"
        message = _get(obj, "message")
        return f"{rule} failed" + (f": {message}" if message else "")
    if source == "anomaly":
        anomaly = _get(obj, "anomaly_type") or "anomaly"
        explanation = _get(obj, "explanation")
        return f"{anomaly} ({severity})" + (f": {explanation}" if explanation else "")
    if source == "deviation":
        metric = _get(obj, "metric") or "metric"
        pct = _get(obj, "deviation_percent")
        detail = f" ({pct}% deviation)" if pct is not None else ""
        return f"Material deviation in {metric}{detail}"
    if source == "ratio":
        name = _get(obj, "ratio_name") or "ratio"
        return f"Critical {name} ratio"
    return str(_get(obj, "finding") or "Risk finding")


def _collect(agent_outputs: Mapping[str, Iterable[Any]]) -> list[_Finding]:
    findings: list[_Finding] = []

    for agent_name, agent_findings in agent_outputs.items():
        source = str(agent_name).lower()
        for obj in agent_findings or []:
            if source in {"validation", "validations", "validation_agent"}:
                # Real ValidationResult objects have a status. Legacy dicts
                # with only finding/severity should remain supported.
                if _get(obj, "status") is not None:
                    if _status(_get(obj, "status")) != "FAIL":
                        continue
                    severity = _normalise_severity(_get(obj, "severity", "LOW"))
                    if severity == "NONE":
                        continue
                    findings.append(_Finding(
                        source_agent="validation",
                        severity=severity,
                        reason=_reason("validation", obj, severity),
                        reference=_reference(obj),
                    ))
                else:
                    severity = _normalise_severity(_get(obj, "severity", "LOW"))
                    if _get(obj, "finding") is None or severity == "NONE":
                        continue
                    findings.append(_Finding(
                        source_agent=str(agent_name),
                        severity=severity,
                        reason=_reason(str(agent_name), obj, severity),
                        reference=_reference(obj),
                    ))

            elif source in {"anomaly", "anomalies", "anomaly_agent"}:
                severity = _normalise_severity(_get(obj, "severity", "LOW"))
                if severity == "NONE":
                    continue
                findings.append(_Finding(
                    source_agent="anomaly",
                    severity=severity,
                    reason=_reason("anomaly", obj, severity),
                    reference=_reference(obj),
                ))

            elif source in {"deviation", "deviations", "trend", "trend_agent"}:
                # DeviationRecord has no severity field. Material deviations
                # are scored according to their materiality/impact.
                if _get(obj, "material_deviation") is not True:
                    continue
                pct = _get(obj, "deviation_percent")
                try:
                    magnitude = abs(float(pct)) if pct is not None else 0.0
                except (TypeError, ValueError):
                    magnitude = 0.0
                severity = "CRITICAL" if magnitude >= 50 else (
                    "HIGH" if magnitude >= 25 else (
                        "MEDIUM" if magnitude >= 10 else "LOW"
                    )
                )
                findings.append(_Finding(
                    source_agent="deviation",
                    severity=severity,
                    reason=_reason("deviation", obj, severity),
                    reference=_reference(obj),
                ))

            elif source in {"ratio", "ratios", "ratio_agent"}:
                status = _status(_get(obj, "status"))
                if status != "CRITICAL":
                    continue
                findings.append(_Finding(
                    source_agent="ratio",
                    severity="HIGH",
                    reason=_reason("ratio", obj, "HIGH"),
                    reference=_reference(obj),
                ))

            else:
                # Legacy/plain-dict input remains supported. It only scores
                # entries that explicitly have a finding and severity.
                severity = _normalise_severity(_get(obj, "severity", "LOW"))
                if _get(obj, "finding") is None or severity == "NONE":
                    continue
                findings.append(_Finding(
                    source_agent=str(agent_name),
                    severity=severity,
                    reason=_reason(str(agent_name), obj, severity),
                    reference=_reference(obj),
                ))

    return findings


def _raw_contribution(base: int, occurrence: int) -> float:
    """Cumulative-score contribution for occurrence N of one severity.

    The aggregate for N equal-severity findings is base * N**exponent.
    Each individual point attribution is the incremental amount from N-1 to N.
    """
    previous = (occurrence - 1) ** DIMINISHING_EXPONENT if occurrence > 1 else 0.0
    current = occurrence ** DIMINISHING_EXPONENT
    return base * (current - previous)


def _allocate_integer_points(raw_points: Sequence[float], target: int) -> list[int]:
    """Allocate integer attribution points that sum exactly to ``target``.

    The allocation is proportional to the raw contributions and is performed
    *after* the final 0-100 target has been determined.  This is important
    when the unclamped score is above 100: the waterfall must never sum to
    more than the displayed score.
    """
    if not raw_points:
        return []

    target = max(0, int(target))
    raw_total = sum(max(0.0, value) for value in raw_points)
    if target == 0 or raw_total <= 0:
        return [0] * len(raw_points)

    scaled = [max(0.0, value) * target / raw_total for value in raw_points]
    points = [int(value) for value in scaled]
    remainder = target - sum(points)

    # Largest-remainder allocation gives deterministic integer points while
    # preserving the relative contribution of each finding as closely as
    # possible.
    fractions = sorted(
        range(len(scaled)),
        key=lambda i: (scaled[i] - points[i], -i),
        reverse=True,
    )
    for i in fractions[:remainder]:
        points[i] += 1

    return points


def calculate_risk(agent_outputs: Mapping[str, Iterable[Any]]) -> RiskScoreResult:
    """Calculate an explainable 0-100 risk score.

    Expected real inputs are a mapping such as::

        {
            "validation": [ValidationResult(...)],
            "anomaly": [AnomalyFinding(...)],
            "deviation": [DeviationRecord(...)],
            "ratio": [RatioResult(...)],
        }

    Legacy dictionaries with ``finding``/``severity`` are also accepted.
    """
    if not agent_outputs:
        return RiskScoreResult(0, "LOW", "green", "No risk findings detected.", [])

    findings = _collect(agent_outputs)
    if not findings:
        return RiskScoreResult(0, "LOW", "green", "No scoring findings detected.", [])

    # Count each severity independently so repeated findings of one severity
    # have diminishing impact.
    occurrences: dict[str, int] = {}
    raw_points: list[float] = []
    for finding in findings:
        occurrences[finding.severity] = occurrences.get(finding.severity, 0) + 1
        raw_points.append(
            _raw_contribution(
                SEVERITY_SCORES[finding.severity],
                occurrences[finding.severity],
            )
        )

    # Finding density is a modest multiplier. Severity mix remains the main
    # driver, while a high-density review gets a small additional penalty.
    density_factor = 1.0 + DENSITY_WEIGHT * (
        log1p(len(findings)) / log1p(DENSITY_BASE)
    )
    adjusted_raw = [value * density_factor for value in raw_points]
    # Clamp only after density scaling and every other adjustment.
    # Allocation is then performed against this final displayed total.
    total = max(0, min(100, round(sum(adjusted_raw))))
    points = _allocate_integer_points(adjusted_raw, total)

    contributors = [
        RiskContributor(
            reason=f.reason,
            points=p,
            source_agent=f.source_agent,
            finding_reference=f.reference,
        )
        for f, p in zip(findings, points)
        if p > 0
    ]

    # Very small contributors can round to zero; their points cannot appear in
    # the waterfall, so recompute total from visible contributors.
    total = sum(c.points for c in contributors)
    level = get_risk_level(total)
    badge = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "orange",
        "CRITICAL": "red",
    }[level]

    if total == 0:
        summary = "No material risk contribution detected."
    else:
        summary = (
            f"{level} risk: {len(findings)} scoring finding(s) produced "
            f"a {total}/100 explainable risk score."
        )

    return RiskScoreResult(
        score=total,
        risk_level=level,
        badge_color=badge,
        summary=summary,
        contributors=contributors,
    )
