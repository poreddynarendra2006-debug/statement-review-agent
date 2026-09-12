"""Data-quality checks: the failures that are not a matter of judgement.

The five accounting rules ask whether the figures agree with each other. They
say nothing about whether the file is fit to review in the first place. Negative
revenue, the same company-year twice, a blank company name - none of those are
unusual-looking, they are wrong, and no amount of statistical judgement should
be involved in saying so.

The Validation role wrote these four checks. They were never called: the entry
point the pipeline uses runs only the accounting formulas, so the checks sat in
the package doing nothing. This connects them.

One of them needs narrowing. `validate_data_quality` reports every empty cell in
every column, which on our own clean dataset is 5,280 findings - our files
legitimately carry no market capitalisation or employee count. Reported in full
it would bury the 147 real failures. So only the fields a review genuinely
cannot proceed without are kept: the company, the year and the revenue.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from validation_agent.checks import (validate_data_quality, validate_domain_sanity,
                                     validate_duplicates, validate_schema)
from validation_agent.models import ValidationResult

#: Without these a row cannot be reviewed at all, so a blank one is a failure
#: rather than a skipped check.
REQUIRED_FIELDS = frozenset({"company", "year", "revenue"})

#: Rule numbers continue from the five accounting identities (VAL_BS_01 to
#: VAL_CF_05), so a reviewer sees one numbered list.
RULES: Dict[str, Dict[str, str]] = {
    "schema": {
        "id": "VAL_SC_06",
        "name": "Required columns present",
        "formula": "the file carries company, year and revenue columns",
    },
    "quality": {
        "id": "VAL_DQ_07",
        "name": "Required fields populated",
        "formula": "company, year and revenue are present on every row",
    },
    "duplicate": {
        "id": "VAL_DU_08",
        "name": "One row per company-year",
        "formula": "company and year together appear once",
    },
    "domain": {
        "id": "VAL_DS_09",
        "name": "Values possible in the real world",
        "formula": "revenue and other counted quantities are not negative; the year is real",
    },
}


def _as_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {k: v for k, v in vars(record).items() if not k.startswith("_")}


def _frame(records: Sequence[Any]) -> pd.DataFrame:
    return pd.DataFrame([_as_dict(r) for r in records])


def _result(kind: str, finding: Dict[str, Any]) -> ValidationResult:
    rule = RULES[kind]
    message = str(finding.get("message") or rule["name"])
    year = finding.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    return ValidationResult(
        rule_id=rule["id"],
        rule_name=rule["name"],
        company=finding.get("company") or "UNKNOWN",
        year=year,
        status="FAIL",
        expected=None,
        actual=None,
        difference=None,
        severity=str(finding.get("severity") or "HIGH").upper(),
        evidence=message,
        formula=rule["formula"],
        message=message,
    )


def _required_only(findings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blank optional columns are normal; blank required ones are not."""
    kept = []
    for finding in findings:
        field = str(finding.get("field") or "").strip().lower()
        if field in REQUIRED_FIELDS:
            finding = dict(finding, severity="HIGH")
            kept.append(finding)
    return kept


def run_data_quality_checks(records: Sequence[Any]) -> List[ValidationResult]:
    """Every data-quality failure in the submission, as validation results.

    Returns an empty list for an empty submission, and never raises: a check
    that cannot run must not take the accounting checks down with it.
    """
    if not records:
        return []

    frame = _frame(records)
    if frame.empty:
        return []

    results: List[ValidationResult] = []

    for kind, check, narrow in (
        ("schema", validate_schema, False),
        ("quality", validate_data_quality, True),
        ("duplicate", validate_duplicates, False),
        ("domain", validate_domain_sanity, False),
    ):
        try:
            _, findings = check(frame)
        except Exception:  # noqa: BLE001 - one failing check must not stop the rest
            continue
        if narrow:
            findings = _required_only(findings)
        results.extend(_result(kind, finding) for finding in findings or [])

    return results
