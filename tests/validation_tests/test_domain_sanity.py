"""
tests/test_domain_sanity.py

Tests for Domain Sanity Validation (Check 5):
- Valid values pass domain sanity
- Invalid Year (< 1900 or > 2100) flagged with HIGH severity
- Negative Number of Employees flagged with HIGH severity
- Negative Revenue flagged with HIGH severity
- Legitimate negative values (Cash Flow from Investing, Financing, Net Income) NOT flagged
"""

import pytest
import pandas as pd

from validation_agent.rules import (
    SEVERITY_HIGH,
    STATUS_PASS
)
from validation_agent.checks import validate_domain_sanity


@pytest.fixture
def sane_df():
    row = {
        "year": 2022,
        "company": "SANE_CO",
        "category": "IT",
        "number_of_employees": 15000,
        "revenue": 50000.0,
        "net_income": 5000.0,
        "net_profit_margin": 10.0,
        "cash_flow_investing": -8500.0,
        "cash_flow_financing": -4200.0
    }
    return pd.DataFrame([row])


def test_valid_domain_sanity(sane_df):
    """Valid business record passes domain sanity checks."""
    summary, findings = validate_domain_sanity(sane_df)
    assert summary["status"] == "PASS"
    assert len(findings) == 0


def test_invalid_year(sane_df):
    """Year outside [1900, 2100] is rejected."""
    df_bad_year = sane_df.copy()
    df_bad_year.at[0, "year"] = 1840

    summary, findings = validate_domain_sanity(df_bad_year)
    assert summary["status"] == "FAIL"
    assert len(findings) == 1
    assert findings[0]["field"] == "year"
    assert findings[0]["severity"] == SEVERITY_HIGH


def test_negative_employees(sane_df):
    """Number of employees cannot be negative in the physical world."""
    df_bad_emp = sane_df.copy()
    df_bad_emp.at[0, "number_of_employees"] = -250

    summary, findings = validate_domain_sanity(df_bad_emp)
    assert summary["status"] == "FAIL"
    assert len(findings) == 1
    assert findings[0]["field"] == "number_of_employees"
    assert findings[0]["severity"] == SEVERITY_HIGH


def test_negative_revenue(sane_df):
    """Corporate revenue cannot be negative."""
    df_bad_rev = sane_df.copy()
    df_bad_rev.at[0, "revenue"] = -10000.0

    summary, findings = validate_domain_sanity(df_bad_rev)
    assert summary["status"] == "FAIL"
    assert len(findings) == 1
    assert findings[0]["field"] == "revenue"
    assert findings[0]["severity"] == SEVERITY_HIGH


def test_legitimate_negative_cash_flows_and_income(sane_df):
    """Legitimate negative values like CapEx (Investing CF) and Net Losses must NOT be flagged."""
    df_legit_neg = sane_df.copy()
    df_legit_neg.at[0, "net_income"] = -1200.0
    df_legit_neg.at[0, "cash_flow_investing"] = -25000.0
    df_legit_neg.at[0, "cash_flow_financing"] = -15000.0

    summary, findings = validate_domain_sanity(df_legit_neg)
    assert summary["status"] == "PASS"
    assert len(findings) == 0
