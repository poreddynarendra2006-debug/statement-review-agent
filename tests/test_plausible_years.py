"""A year that has not begun is an error, and it must not stretch the review period.

The TL's retest put a 2099 row in a file. The Validation rule accepts 1900 to
2100, so nothing was reported, and the report period read "FY2016 - FY2099".
"""

from datetime import date
from types import SimpleNamespace

import pytest

from agents.years import EARLIEST_VALID_YEAR, is_plausible_year, latest_valid_year


# --- what counts as a real year ----------------------------------------------


def test_the_latest_year_is_next_year_so_fiscal_naming_works():
    assert latest_valid_year(date(2026, 9, 13)) == 2027


@pytest.mark.parametrize("value", [2016, 2023, "2024", 2024.0, EARLIEST_VALID_YEAR])
def test_real_years_are_plausible(value):
    assert is_plausible_year(value, today=date(2026, 9, 13))


@pytest.mark.parametrize("value", [2099, 2028, 1492, 3025, None, "n/a", float("nan")])
def test_years_no_statement_can_cover_are_not(value):
    assert not is_plausible_year(value, today=date(2026, 9, 13))


# --- validation reports it ---------------------------------------------------


def _rules_for(rows):
    from api.dependencies import Record
    from agents.data_quality import run_data_quality_checks

    return run_data_quality_checks([Record(**row) for row in rows])


def row(**fields):
    return {"company": "Acme Corporation", "year": 2023, "revenue": 1000.0,
            "net_income": 100.0, "total_assets": 2000.0, **fields}


def test_a_future_year_is_reported():
    results = _rules_for([row(), row(company="FutureCo", year=2099)])

    future = [r for r in results if r.company == "FutureCo"]
    assert future, "2099 passed validation"
    assert future[0].rule_id == "VAL_DS_09"
    assert "2099" in future[0].message


def test_next_year_is_not_reported():
    results = _rules_for([row(company="EarlyFiler", year=latest_valid_year())])

    assert not [r for r in results if r.company == "EarlyFiler"]


def test_a_year_past_2100_is_reported_once_not_twice():
    # The Validation rule already reports these; the new check must not repeat it.
    results = _rules_for([row(company="FarFuture", year=3025)])

    assert len([r for r in results if r.company == "FarFuture"]) == 1


def test_ordinary_years_report_nothing():
    assert _rules_for([row(year=y) for y in range(2019, 2024)]) == []


# --- and the period ignores it -----------------------------------------------


def _period(years):
    from agents.orchestrator import ReviewOrchestrator

    records = [SimpleNamespace(company="Acme", year=y, currency="$") for y in years]
    return ReviewOrchestrator._identify(records)[1]


def test_a_future_year_does_not_stretch_the_period():
    assert _period([2016, 2020, 2023, 2099]) == "FY2016 - FY2023"


def test_an_ancient_year_does_not_stretch_it_either():
    assert _period([1492, 2021, 2023]) == "FY2021 - FY2023"


def test_a_file_of_only_impossible_years_still_shows_what_it_says():
    # Better to show the file's own years than an empty period.
    assert _period([2099]) == "FY2099"


def test_an_ordinary_period_is_unchanged():
    assert _period([2019, 2020, 2021]) == "FY2019 - FY2021"
