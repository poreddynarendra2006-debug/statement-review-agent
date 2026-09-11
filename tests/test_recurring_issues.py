"""Recurring-issue detection must find the same problem across years, and nothing else."""

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.orchestrator import ReviewOrchestrator
from agents.planner import Planner
from analysis.recurring_issues import HIGH_FROM_YEARS, MIN_YEARS, detect_recurring_issues
from api.dependencies import (
    COMPONENT_ENTRY_POINTS,
    INGESTION_ENTRY_POINTS,
    _first_available,
    records_from_ingestion,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def failed(company, year, rule_id="VAL_NI_04", rule_name="Net Income", severity="LOW", status="FAIL"):
    return {"company": company, "year": year, "rule_id": rule_id, "rule_name": rule_name,
            "status": status, "severity": severity,
            "evidence": f"FY{year}: Net Income 90 vs Pre-tax Income 120 - Taxes 20 = 100"}


def issues_from(**outputs):
    return detect_recurring_issues([], outputs=outputs)


# --- failed checks ------------------------------------------------------------


def test_a_check_failing_in_three_years_is_one_recurring_issue():
    issues = issues_from(validation=[failed("Acme", year) for year in (2019, 2020, 2021)])

    assert len(issues) == 1
    issue = issues[0]
    assert (issue.company, issue.source, issue.key) == ("Acme", "validation", "VAL_NI_04")
    assert issue.issue == "Net Income check failed in 3 consecutive years (2019-2021)"
    assert issue.years == [2019, 2020, 2021] and issue.consecutive
    assert issue.severity == "MEDIUM"
    assert "FY2020: Net Income 90" in issue.evidence


def test_two_years_is_not_recurring():
    assert MIN_YEARS == 3
    assert issues_from(validation=[failed("Acme", 2019), failed("Acme", 2020)]) == []


def test_years_with_gaps_still_count():
    issue = issues_from(validation=[failed("Acme", year) for year in (2016, 2018, 2020)])[0]

    assert not issue.consecutive
    assert issue.issue.endswith("in 3 years (2016, 2018, 2020)")


def test_different_companies_and_different_checks_are_kept_apart():
    assert issues_from(validation=[failed("Acme", 2019), failed("Acme", 2020), failed("Beta", 2021)]) == []
    assert issues_from(validation=[failed("Acme", 2019), failed("Acme", 2020),
                                   failed("Acme", 2021, rule_id="VAL_BS_01")]) == []


def test_passing_and_skipped_checks_are_ignored():
    results = [failed("Acme", 2019, status="PASS"), failed("Acme", 2020, status="SKIPPED"),
               failed("Acme", 2021)]
    assert issues_from(validation=results) == []


def test_several_failures_in_one_year_count_as_one_year():
    assert issues_from(validation=[failed("Acme", 2019), failed("Acme", 2019), failed("Acme", 2020)]) == []

    issue = issues_from(validation=[failed("Acme", 2019), failed("Acme", 2019),
                                    failed("Acme", 2020), failed("Acme", 2021)])[0]
    assert issue.years == [2019, 2020, 2021]
    assert issue.occurrences == 4


def test_more_years_or_worse_findings_raise_severity():
    four_years = issues_from(validation=[failed("Acme", y) for y in range(2018, 2018 + HIGH_FROM_YEARS)])
    assert four_years[0].severity == "HIGH"

    with_critical = issues_from(validation=[failed("Acme", 2019), failed("Acme", 2020),
                                            failed("Acme", 2021, severity="CRITICAL")])
    assert with_critical[0].severity == "CRITICAL"


# --- anomalies and trend deviations -----------------------------------------------


def test_the_same_anomaly_type_across_years_is_found():
    findings = [SimpleNamespace(company="Acme", year=year, anomaly_type="temporal_anomaly",
                                severity=SimpleNamespace(value="LOW"),
                                explanation=f"Acme ({year}): ROE fell sharply. It is a potential anomaly.")
                for year in (2019, 2020, 2021)]
    issue = issues_from(anomaly=findings)[0]

    assert issue.source == "anomaly"
    assert issue.issue == "Temporal anomaly flagged in 3 consecutive years (2019-2021)"
    assert "FY2019: LOW - Acme (2019): ROE fell sharply" in issue.evidence


def test_material_deviations_on_the_same_figure_are_found():
    deviations = [dict(company="Acme", year=year, metric="net_profit_margin", actual=120.0,
                       forecast=100.0, deviation_percent=20.0, material_deviation=True)
                  for year in (2019, 2020, 2021)]
    deviations.append(dict(company="Acme", year=2022, metric="net_profit_margin", actual=101.0,
                           forecast=100.0, deviation_percent=1.0, material_deviation=False))
    issue = issues_from(trend={"deviations": deviations})[0]

    assert issue.source == "trend"
    assert issue.years == [2019, 2020, 2021], "the non-material year must not count"
    assert issue.issue.startswith("Net profit margin differed materially from its forecast")
    assert "FY2020: actual 120 vs forecast 100 (+20.0%)" in issue.evidence


def test_nothing_to_read_means_no_issues():
    assert detect_recurring_issues([]) == []
    assert issues_from(validation=None, anomaly=None, trend=None) == []


def test_most_serious_issues_come_first():
    issues = issues_from(validation=[failed("Beta", y) for y in (2019, 2020, 2021)]
                         + [failed("Acme", y, rule_id="VAL_BS_01") for y in (2018, 2019, 2020, 2021)])

    assert [i.severity for i in issues] == ["HIGH", "MEDIUM"]


def test_issues_serialise_for_the_api():
    issue = issues_from(validation=[failed("Acme", year) for year in (2019, 2020, 2021)])[0]

    data = json.loads(json.dumps(issue.to_dict()))
    assert set(data) == {"company", "source", "key", "issue", "years", "consecutive",
                         "severity", "evidence", "occurrences"}


# --- wiring into the planner and the orchestrator ------------------------------------


def test_planner_gives_earlier_results_only_to_tools_that_ask(financial_records):
    seen = {}
    planner = Planner()
    planner.register("validation", lambda records: ["checked"])
    planner.register("plain", lambda records, materiality=0.1: seen.setdefault("plain", materiality))

    def reader(records, outputs):
        seen["outputs"] = outputs
        return []

    planner.register("reader", reader)
    planner.run(financial_records, options={"materiality": 0.2})

    assert seen["plain"] == 0.2
    assert seen["outputs"] == {"validation": ["checked"], "plain": 0.2}


def test_orchestrator_reports_recurring_failures(financial_records):
    orchestrator = ReviewOrchestrator(
        validation=lambda records: [failed("Acme Corporation", y) for y in (2021, 2022, 2023)],
        recurring=detect_recurring_issues,
    )
    result = orchestrator.run(financial_records)

    assert "recurring" in result.coverage["selected"]
    assert [i.key for i in result.recurring_issues] == ["VAL_NI_04"]
    assert result.to_dict()["recurring_issues"][0]["years"] == [2021, 2022, 2023]
    assert "recurring" in result.timings


def test_recurring_detection_is_skipped_with_fewer_than_three_years(financial_records):
    orchestrator = ReviewOrchestrator(validation=lambda records: [], recurring=detect_recurring_issues)
    result = orchestrator.run(financial_records[:2])

    assert "3 periods" in result.coverage["skipped"]["recurring"]
    assert result.recurring_issues == []


def test_the_app_finds_the_detector():
    assert _first_available(COMPONENT_ENTRY_POINTS["recurring"]) is detect_recurring_issues


def test_real_validation_results_match_the_answer_key():
    validate = _first_available(COMPONENT_ENTRY_POINTS["validation"])
    ingest = _first_available(INGESTION_ENTRY_POINTS)
    if validate is None or ingest is None:
        pytest.skip("needs the Validation and Ingestion components")

    planted = defaultdict(set)
    for defect in json.loads((DATA / "dummy_statements_labels.json").read_text(encoding="utf-8"))["defects"]:
        planted[(defect["company"], defect["rule_id"])].add(int(defect["year"]))
    expected = {(company, rule, tuple(sorted(years)))
                for (company, rule), years in planted.items() if len(years) >= MIN_YEARS}

    records = records_from_ingestion(ingest(str(DATA / "dummy_statements_defective.csv")))
    issues = detect_recurring_issues(records, outputs={"validation": validate(records)})

    assert expected, "the answer key should hold some recurring errors"
    assert {(i.company, i.key, tuple(i.years)) for i in issues} == expected
