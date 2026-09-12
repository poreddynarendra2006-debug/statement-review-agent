"""Data-quality failures: negative revenue, duplicate rows, blank required fields.

Raised by the TL on 12 September: these were going unreported, because the
Validation package's checks for them were never called. They are failures of
fact, so they must appear as failed checks, not as anomaly-detection judgement
calls - and equally they must not fire on an ordinary file.
"""

import pytest

from agents.data_quality import run_data_quality_checks
from api.dependencies import Record, build_orchestrator


def row(**fields):
    base = dict(company="Acme Corporation", year=2023, revenue=1000.0,
                net_income=100.0, total_assets=2000.0)
    base.update(fields)
    return base


def rule_ids(results):
    return {r.rule_id for r in results}


def failures_for(rows):
    return run_data_quality_checks([Record(**r) for r in rows])


# --- the three the TL named -------------------------------------------------


def test_negative_revenue_is_a_failure():
    results = failures_for([row(), row(company="Beta Ltd", revenue=-500.0)])

    assert "VAL_DS_09" in rule_ids(results)
    bad = next(r for r in results if r.rule_id == "VAL_DS_09")
    assert bad.company == "Beta Ltd"
    assert bad.status == "FAIL"
    assert "negative" in bad.message.lower()


def test_the_same_company_and_year_twice_is_a_failure():
    results = failures_for([row(), row(revenue=1200.0)])

    assert "VAL_DU_08" in rule_ids(results)
    assert all(r.company == "Acme Corporation"
               for r in results if r.rule_id == "VAL_DU_08")


def test_a_blank_required_field_is_a_failure():
    results = failures_for([row(), row(company="Beta Ltd", revenue=None)])

    assert "VAL_DQ_07" in rule_ids(results)
    assert any("revenue" in r.message.lower()
               for r in results if r.rule_id == "VAL_DQ_07")


@pytest.mark.parametrize("missing", ["company", "revenue"])
def test_every_required_field_is_checked(missing):
    results = failures_for([row(), row(**{missing: None})])

    assert "VAL_DQ_07" in rule_ids(results)


# --- and what it must not do ------------------------------------------------


def test_an_ordinary_file_produces_nothing():
    results = failures_for([row(year=y) for y in range(2019, 2024)])

    assert results == []


def test_a_blank_optional_field_is_not_a_failure():
    # Our own datasets carry no market capitalisation or employee count. The
    # underlying check reports every empty cell - 5,280 of them on the clean
    # file - which would bury the real failures.
    results = failures_for([row(year=2022, market_cap_b_usd=None, number_of_employees=None),
                            row(year=2023, market_cap_b_usd=None, number_of_employees=None)])

    assert results == []


def test_a_legitimate_negative_figure_is_not_a_failure():
    # A loss, or cash used by investing, is ordinary. Only quantities that
    # cannot be negative are checked.
    results = failures_for([row(net_income=-250.0, cash_flow_investing=-80.0),
                            row(year=2022, net_income=-90.0)])

    assert results == []


def test_the_same_company_in_different_years_is_not_a_duplicate():
    results = failures_for([row(year=2022), row(year=2023)])

    assert "VAL_DU_08" not in rule_ids(results)


def test_different_companies_in_the_same_year_are_not_duplicates():
    results = failures_for([row(), row(company="Beta Ltd")])

    assert "VAL_DU_08" not in rule_ids(results)


def test_an_empty_submission_is_answered_with_nothing():
    assert run_data_quality_checks([]) == []


# --- through the pipeline ----------------------------------------------------


def test_these_failures_reach_the_reviewer_with_the_accounting_ones():
    records = [Record(**row(company="Acme Corporation", year=2023, revenue=-400.0,
                            cost_of_revenue=300.0, gross_profit=100.0)),
               Record(**row(company="Beta Ltd", year=2023))]

    result = build_orchestrator().run(records)

    failed = {getattr(v, "rule_id", None) for v in result.failed_validations}
    assert "VAL_DS_09" in failed, "a data-quality failure belongs in the same list"


def test_an_ordinary_submission_gains_no_extra_failures():
    records = [Record(**row(company="Acme Corporation", year=y)) for y in (2022, 2023)]

    result = build_orchestrator().run(records)

    assert not [v for v in result.failed_validations
                if getattr(v, "rule_id", "").startswith(("VAL_SC", "VAL_DQ", "VAL_DU", "VAL_DS"))]
