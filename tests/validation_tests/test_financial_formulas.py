"""
tests/test_financial_formulas.py

Tests for Financial Formula Validation (Check 4):
- Mathematically correct formula (Net Profit Margin = Net Income / Revenue * 100) passes
- Incorrect formula (e.g. expected 10.0%, actual 18.0%) fails with HIGH severity
- Tolerance behavior (small deviations within tolerance pass, outside fail)
- Metrics lacking inputs (ROA, Current Ratio, etc.) cataloged as NOT_VALIDATABLE without false failure
"""

import pytest
import pandas as pd

from validation_agent.rules import (
    SEVERITY_HIGH,
    SEVERITY_INFO,
    STATUS_BLOCK,
    STATUS_PASS,
    NOT_VALIDATABLE_METRICS
)
from validation_agent.checks import validate_financial_formulas
from validation_agent.validator import ValidationAgent


@pytest.fixture
def base_formula_df():
    row = {
        "year": 2023,
        "company": "FORMULA_TEST",
        "category": "IT",
        "number_of_employees": 1000,
        "revenue": 1000.0,
        "net_income": 100.0,
        "net_profit_margin": 10.0  # Exactly (100 / 1000) * 100
    }
    return pd.DataFrame([row])


def test_correct_formula(base_formula_df):
    """Accurate Net Profit Margin calculation passes validation."""
    summary, findings = validate_financial_formulas(base_formula_df, tolerance=0.5)
    failing_findings = [f for f in findings if f.get("status") == "FAIL"]

    assert summary["status"] == "PASS"
    assert len(failing_findings) == 0

    agent = ValidationAgent(tolerance=0.5)
    res = agent.validate(base_formula_df)
    assert res["checks"]["financial_formula"]["status"] == "PASS"


def test_incorrect_formula(base_formula_df):
    """Deliberately corrupted Net Profit Margin triggers HIGH severity finding."""
    df_bad = base_formula_df.copy()
    # Expected is 10.0%, report 18.0%
    df_bad.at[0, "net_profit_margin"] = 18.0

    summary, findings = validate_financial_formulas(df_bad, tolerance=0.5)
    failing_findings = [f for f in findings if f.get("status") == "FAIL"]

    assert summary["status"] == "FAIL"
    assert len(failing_findings) == 1
    f = failing_findings[0]
    assert f["check"] == "FINANCIAL_FORMULA"
    assert f["severity"] == SEVERITY_HIGH
    assert f["metric"] == "Net Profit Margin"
    assert f["expected"] == 10.0
    assert f["actual"] == 18.0
    assert f["difference"] == 8.0

    agent = ValidationAgent(tolerance=0.5)
    res = agent.validate(df_bad)
    # Formula failure is HIGH severity, triggering BLOCK
    assert res["status"] == STATUS_BLOCK
    assert res["confidence"] <= 90.0


def test_tolerance_behavior(base_formula_df):
    """Formula deviation within tolerance passes; outside tolerance fails."""
    df_dev = base_formula_df.copy()
    # Expected is 10.0%, actual is 10.3% (diff = 0.3%)
    df_dev.at[0, "net_profit_margin"] = 10.3

    # With tolerance = 0.5%, difference 0.3% should PASS
    sum_pass, findings_pass = validate_financial_formulas(df_dev, tolerance=0.5)
    fail_pass = [f for f in findings_pass if f.get("status") == "FAIL"]
    assert sum_pass["status"] == "PASS"
    assert len(fail_pass) == 0

    # With tolerance = 0.1%, difference 0.3% should FAIL
    sum_fail, findings_fail = validate_financial_formulas(df_dev, tolerance=0.1)
    fail_fail = [f for f in findings_fail if f.get("status") == "FAIL"]
    assert sum_fail["status"] == "FAIL"
    assert len(fail_fail) == 1
    assert fail_fail[0]["severity"] == SEVERITY_HIGH


def test_not_validatable_metrics(base_formula_df):
    """Metrics whose inputs are absent must be marked NOT_VALIDATABLE rather than FAIL."""
    summary, findings = validate_financial_formulas(base_formula_df)
    nv_findings = [f for f in findings if f.get("status") == "NOT_VALIDATABLE"]

    assert len(nv_findings) == len(NOT_VALIDATABLE_METRICS)
    nv_metrics = {f["metric"] for f in nv_findings}
    assert "ROA" in nv_metrics
    assert "Current Ratio" in nv_metrics
    assert "Debt/Equity Ratio" in nv_metrics
    assert "ROI" in nv_metrics

    # NOT_VALIDATABLE metrics have SEVERITY_INFO and do not cause BLOCK or confidence drops
    for f in nv_findings:
        assert f["severity"] == SEVERITY_INFO
