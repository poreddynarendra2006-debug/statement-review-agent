"""Forensic red flags: patterns that are suspicious whatever the rest of the file looks like.

The point of these rules is that they stay quiet. A red flag that fires often is
not a red flag, so most of these tests are about what they do *not* report.
"""

import pytest

from analysis.forensic_flags import detect_forensic_flags
from api.dependencies import Record


def year(y, **fields):
    base = dict(company="Acme Corporation", year=y, revenue=1000.0, gross_profit=400.0,
                net_income=100.0, cash_flow_operating=120.0, shareholder_equity=1000.0,
                debt=400.0, total_assets=2000.0)
    base.update(fields)
    return base


def flags_for(rows):
    return detect_forensic_flags([Record(**r) for r in rows])


def types(findings):
    return {str(getattr(f.anomaly_type, "value", f.anomaly_type)) for f in findings}


# --- each rule fires on the pattern it is for --------------------------------


def test_revenue_that_does_not_turn_into_cash():
    found = flags_for([year(2022), year(2023, revenue=2000.0, cash_flow_operating=121.0)])

    assert "revenue_profit_mismatch" in types(found)
    assert "100%" in found[0].explanation or "revenue" in found[0].explanation.lower()


def test_profit_reported_while_cash_flow_is_negative():
    found = flags_for([year(2022), year(2023, net_income=300.0, cash_flow_operating=-50.0)])

    assert "cash_flow_profit_divergence" in types(found)


def test_gross_margin_collapsing_in_one_year():
    found = flags_for([year(2022), year(2023, gross_profit=40.0)])

    assert "margin_anomaly" in types(found)


def test_debt_to_equity_more_than_trebling():
    found = flags_for([year(2022), year(2023, debt=1600.0)])

    assert "leverage_anomaly" in types(found)


def test_equity_wiped_out():
    found = flags_for([year(2022), year(2023, shareholder_equity=200.0)])

    assert "sudden_financial_change" in types(found)


# --- and stay quiet otherwise -------------------------------------------------


def test_an_ordinary_pair_of_years_reports_nothing():
    assert flags_for([year(2022), year(2023, revenue=1100.0, net_income=110.0,
                                       cash_flow_operating=130.0)]) == []


def test_healthy_growth_with_cash_following_is_not_flagged():
    # Revenue up 60%, but cash flow follows. That is a good year, not a red flag.
    found = flags_for([year(2022), year(2023, revenue=1600.0, gross_profit=640.0,
                                        net_income=160.0, cash_flow_operating=200.0)])

    assert found == []


def test_a_loss_with_negative_cash_flow_is_not_the_accrual_flag():
    # The flag is profit *without* cash. A loss with negative cash is consistent.
    found = flags_for([year(2022), year(2023, net_income=-80.0, cash_flow_operating=-60.0)])

    assert "cash_flow_profit_divergence" not in types(found)


def test_a_margin_that_was_already_thin_is_not_a_collapse():
    # From a 5% base there is nothing to collapse; that is a low-margin business.
    found = flags_for([year(2022, gross_profit=50.0), year(2023, gross_profit=15.0)])

    assert "margin_anomaly" not in types(found)


def test_one_year_alone_produces_nothing():
    # Every rule compares against the company's own previous year, so the same
    # figures that flag with a prior year must stay silent without one.
    alone = [year(2023, net_income=300.0, cash_flow_operating=-50.0)]

    assert flags_for(alone) == []
    assert flags_for([year(2022)] + alone) != [], "and they do flag once there is a prior year"


def test_no_records_is_answered_with_nothing():
    assert detect_forensic_flags([]) == []


def test_years_are_compared_in_order_not_file_order():
    rows = [year(2023, net_income=300.0, cash_flow_operating=-50.0), year(2022)]

    assert "cash_flow_profit_divergence" in types(flags_for(rows))


def test_companies_are_not_compared_with_each_other():
    rows = [year(2022, company="Acme Corporation"),
            year(2023, company="Beta Ltd", shareholder_equity=100.0)]

    assert flags_for(rows) == []


@pytest.mark.parametrize("missing", ["revenue", "cash_flow_operating", "shareholder_equity"])
def test_a_missing_figure_is_skipped_not_guessed(missing):
    rows = [year(2022), year(2023, **{missing: None})]

    # No exception, and no finding invented from an absent number.
    for finding in flags_for(rows):
        assert finding.explanation


# --- what a reviewer receives -------------------------------------------------


def test_a_finding_explains_itself_and_says_what_to_do():
    found = flags_for([year(2022), year(2023, net_income=300.0, cash_flow_operating=-50.0)])[0]

    assert found.company == "Acme Corporation" and found.year == 2023
    assert found.model_name == "ForensicRedFlags"
    assert "300" in found.explanation
    assert found.recommendation
    assert found.actual_values["operating_cash_flow"] == -50.0


def test_nothing_it_says_accuses_anyone():
    rows = [year(2022), year(2023, revenue=2000.0, cash_flow_operating=121.0,
                             net_income=300.0, gross_profit=40.0)]

    words = " ".join(f"{f.explanation} {f.recommendation}" for f in flags_for(rows)).lower()

    for accusation in ("fraud", "fraudulent", "manipulat", "falsif", "misconduct", "illegal"):
        assert accusation not in words


# --- through the anomaly agent -------------------------------------------------


def test_the_anomaly_agent_reports_both_kinds():
    from analysis.anomaly_detection import run_all_anomaly_detection

    # Enough company-years for the model, plus one planted accrual divergence.
    rows = [year(y, company=f"Company {c}") for c in range(6) for y in range(2019, 2024)]
    rows.append(year(2024, company="Company 0", net_income=500.0, cash_flow_operating=-200.0))

    found = run_all_anomaly_detection([Record(**r) for r in rows])

    assert any(getattr(f, "model_name", "") == "ForensicRedFlags" for f in found)
