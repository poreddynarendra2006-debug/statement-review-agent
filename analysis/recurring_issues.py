"""Finds problems that keep coming back for the same company.

A single failed check can be a one-off slip. The same check failing for the
same company year after year is a pattern, and a reviewer should see it as one
finding rather than scattered through a long list. This runs after the other
analysis agents and reads what they found:

    validation   the same accounting check failing
    anomaly      the same kind of anomaly being flagged
    trend        the same figure differing materially from its forecast

in at least MIN_YEARS different years. It computes no figures of its own:
every issue points back to the findings it was built from.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Different years a problem must appear in before it counts as recurring.
MIN_YEARS = 3

#: From this many years a recurring issue is HIGH, whatever the single findings were.
HIGH_FROM_YEARS = 4

SEVERITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass
class RecurringIssue:
    """One problem found for the same company in several years."""

    company: Optional[str]
    source: str          # validation | anomaly | trend
    key: str             # rule id, anomaly type or metric
    issue: str           # one readable line
    years: List[int]
    consecutive: bool
    severity: str        # MEDIUM | HIGH | CRITICAL
    evidence: str        # what was found in each year
    occurrences: int     # findings behind it; a year can hold more than one

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reading findings, whether they arrive as objects or dicts
# ---------------------------------------------------------------------------


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _text(value: Any) -> str:
    """Plain strings and enum members (such as Severity.HIGH) alike."""
    return str(getattr(value, "value", value) or "")


def _year(item: Any) -> Optional[int]:
    try:
        return int(float(_get(item, "year")))
    except (TypeError, ValueError):
        return None


def _label(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.2f}"


def _first_sentence(text: Any, limit: int = 140) -> str:
    sentence = str(text or "").split(". ")[0].strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Building issues
# ---------------------------------------------------------------------------

Groups = Dict[Tuple[Optional[str], str], Dict[int, List[Any]]]


def _group(items: Iterable[Any], key_of: Callable[[Any], str]) -> Groups:
    groups: Groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        year, key = _year(item), key_of(item)
        if year is not None and key:
            groups[(_get(item, "company"), key)][year].append(item)
    return groups


def _severity(year_count: int, findings: Iterable[Any]) -> str:
    level = "HIGH" if year_count >= HIGH_FROM_YEARS else "MEDIUM"
    worst = max((_text(_get(f, "severity")).upper() for f in findings),
                key=lambda s: SEVERITY_RANK.get(s, 0), default="NONE")
    return worst if SEVERITY_RANK.get(worst, 0) > SEVERITY_RANK[level] else level


def _when(years: List[int], consecutive: bool) -> str:
    if consecutive:
        return f"in {len(years)} consecutive years ({years[0]}-{years[-1]})"
    return f"in {len(years)} years ({', '.join(str(y) for y in years)})"


def _issues(groups: Groups, source: str,
            name_of: Callable[[str, List[Any]], str],
            evidence_of: Callable[[int, Any], str]) -> List[RecurringIssue]:
    issues = []
    for (company, key), by_year in groups.items():
        years = sorted(by_year)
        if len(years) < MIN_YEARS:
            continue
        consecutive = years == list(range(years[0], years[-1] + 1))
        findings = [f for year in years for f in by_year[year]]
        issues.append(RecurringIssue(
            company=company,
            source=source,
            key=key,
            issue=f"{name_of(key, findings)} {_when(years, consecutive)}",
            years=years,
            consecutive=consecutive,
            severity=_severity(len(years), findings),
            evidence=" | ".join(evidence_of(year, by_year[year][0]) for year in years),
            occurrences=len(findings),
        ))
    return issues


def _failed_checks(results: Any) -> List[RecurringIssue]:
    failed = [r for r in results or [] if _text(_get(r, "status")).upper() == "FAIL"]
    return _issues(
        _group(failed, lambda r: _text(_get(r, "rule_id"))), "validation",
        lambda key, found: f"{next((_get(f, 'rule_name') for f in found if _get(f, 'rule_name')), key)} check failed",
        lambda year, r: f"FY{year}: {_get(r, 'evidence') or _get(r, 'message') or 'check failed'}",
    )


def _repeated_anomalies(findings: Any) -> List[RecurringIssue]:
    return _issues(
        _group(findings or [], lambda a: _text(_get(a, "anomaly_type"))), "anomaly",
        lambda key, found: f"{_label(key)} flagged",
        lambda year, a: f"FY{year}: {_text(_get(a, 'severity')) or 'flagged'} - {_first_sentence(_get(a, 'explanation'))}",
    )


def _repeated_deviations(trend_output: Any) -> List[RecurringIssue]:
    deviations = trend_output.get("deviations") if isinstance(trend_output, Mapping) else None
    material = [d for d in deviations or [] if _get(d, "material_deviation")]

    def evidence(year: int, d: Any) -> str:
        percent = _get(d, "deviation_percent")
        shown = f" ({float(percent):+.1f}%)" if isinstance(percent, (int, float)) else ""
        return f"FY{year}: actual {_number(_get(d, 'actual'))} vs forecast {_number(_get(d, 'forecast'))}{shown}"

    return _issues(
        _group(material, lambda d: _text(_get(d, "metric"))), "trend",
        lambda key, found: f"{_label(key)} differed materially from its forecast",
        evidence,
    )


def detect_recurring_issues(records: Sequence[Any],
                            outputs: Optional[Mapping[str, Any]] = None) -> List[RecurringIssue]:
    """Problems found for the same company in at least MIN_YEARS different years.

    Args:
        records: The submission. The planner hands it to every agent; the
            findings below already carry the company and year.
        outputs: What the analysis agents that ran before this one returned, by
            name ("validation", "anomaly", "trend"). The planner supplies it.

    Returns:
        Recurring issues, most serious first.
    """
    outputs = outputs or {}
    issues = (_failed_checks(outputs.get("validation"))
              + _repeated_anomalies(outputs.get("anomaly"))
              + _repeated_deviations(outputs.get("trend")))
    issues.sort(key=lambda i: (-SEVERITY_RANK.get(i.severity, 0), -len(i.years),
                               str(i.company), i.source, i.key))
    return issues
