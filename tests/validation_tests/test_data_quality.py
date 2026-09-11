"""
tests/test_data_quality.py

Tests for Data Quality Validation (Check 2):
- Missing values (NaN/None) detected as MEDIUM severity
- Infinite values (+inf, -inf) detected as HIGH severity
- Malformed/unparseable numeric values detected as HIGH severity
- Clean data produces zero data quality findings
"""

import pytest
import numpy as np
import pandas as pd

from validation_agent.rules import (
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    STATUS_PASS,
    STATUS_WARNING
)
from validation_agent.checks import validate_data_quality
from validation_agent.validator import ValidationAgent


@pytest.fixture
def clean_df():
    """Returns a valid 2-row DataFrame."""
    rows = []
    for i in range(2):
        r = {
            "year": 2020 + i,
            "company": "ACME",
            "revenue": 1000.0,
            "net_income": 100.0,
            "net_profit_margin": 10.0,
            "category": "IT",
            "number_of_employees": 1000,
            "Market Cap(in B USD)": 50.0,
            "Current Ratio": 1.5
        }
        rows.append(r)
    return pd.DataFrame(rows)


def test_clean_data_quality(clean_df):
    """Clean data should yield 0 findings and PASS status."""
    summary, findings = validate_data_quality(clean_df)
    assert summary["status"] == "PASS"
    assert len(findings) == 0


def test_missing_value_detection(clean_df):
    """Missing value should be flagged with MEDIUM severity."""
    df_missing = clean_df.copy()
    df_missing.at[0, "Market Cap(in B USD)"] = np.nan

    summary, findings = validate_data_quality(df_missing)
    assert summary["status"] == "FAIL"
    assert len(findings) == 1
    assert findings[0]["check"] == "DATA_QUALITY"
    assert findings[0]["severity"] == SEVERITY_MEDIUM
    assert findings[0]["field"] == "Market Cap(in B USD)"

    agent = ValidationAgent()
    res = agent.validate(df_missing)
    # A single MEDIUM finding should trigger WARNING status, not BLOCK
    assert res["status"] == STATUS_WARNING
    assert res["confidence"] < 100.0


def test_infinite_value_detection(clean_df):
    """Infinite value should be flagged with HIGH severity."""
    df_inf = clean_df.copy()
    df_inf.at[1, "Current Ratio"] = np.inf

    summary, findings = validate_data_quality(df_inf)
    assert summary["status"] == "FAIL"
    inf_findings = [f for f in findings if "Infinite" in f["message"]]
    assert len(inf_findings) == 1
    assert inf_findings[0]["severity"] == SEVERITY_HIGH
    assert inf_findings[0]["field"] == "Current Ratio"


def test_invalid_string_numeric_detection(clean_df):
    """Malformed string inside a numeric column should be flagged with HIGH severity."""
    df_corrupt = clean_df.copy()
    df_corrupt["revenue"] = df_corrupt["revenue"].astype(object)
    df_corrupt.at[0, "revenue"] = "NOT_A_VALID_NUMBER"

    summary, findings = validate_data_quality(df_corrupt)
    assert summary["status"] == "FAIL"
    malformed = [f for f in findings if "Invalid numeric value" in f["message"]]
    assert len(malformed) == 1
    assert malformed[0]["severity"] == SEVERITY_HIGH
    assert malformed[0]["field"] == "revenue"
