"""
tests/test_schema.py

Tests for Schema Validation (Check 1):
- Valid schema with minimal required columns (year, company, revenue)
- Missing required column(s) triggering CRITICAL severity and BLOCK status
- Empty DataFrame and None handling
"""

import pytest
import pandas as pd

from validation_agent.rules import (
    REQUIRED_COLUMNS,
    SEVERITY_CRITICAL,
    STATUS_BLOCK,
    STATUS_PASS
)
from validation_agent.checks import validate_schema
from validation_agent.validator import ValidationAgent


@pytest.fixture
def clean_sample_df():
    """Create a minimal DataFrame with required columns."""
    row = {
        "year": 2022,
        "company": "TEST",
        "revenue": 100.0,
        "category": "IT",
        "number_of_employees": 5000,
        "net_profit_margin": 10.0,
        "net_income": 10.0
    }
    return pd.DataFrame([row])


def test_valid_schema(clean_sample_df):
    """Test that a DataFrame with all required columns passes schema validation."""
    summary, findings = validate_schema(clean_sample_df)
    assert summary["status"] == "PASS"
    assert len(findings) == 0

    agent = ValidationAgent()
    res = agent.validate(clean_sample_df)
    assert res["checks"]["schema"]["status"] == "PASS"


def test_missing_single_column(clean_sample_df):
    """Test that removing a required column triggers CRITICAL severity and BLOCK status."""
    df_missing = clean_sample_df.drop(columns=["revenue"])
    summary, findings = validate_schema(df_missing)

    assert summary["status"] == "FAIL"
    assert len(findings) == 1
    f = findings[0]
    assert f["check"] == "SCHEMA"
    assert f["severity"] == SEVERITY_CRITICAL
    assert "revenue" in f["message"]

    agent = ValidationAgent()
    res = agent.validate(df_missing)
    assert res["status"] == STATUS_BLOCK
    assert res["checks"]["schema"]["status"] == "FAIL"


def test_missing_multiple_columns(clean_sample_df):
    """Test that multiple missing columns produce structured findings for each."""
    cols_to_drop = ["company", "revenue"]
    df_missing = clean_sample_df.drop(columns=cols_to_drop)
    summary, findings = validate_schema(df_missing)

    assert summary["status"] == "FAIL"
    assert len(findings) == 2
    dropped_found = [f["field"] for f in findings]
    for col in cols_to_drop:
        assert col in dropped_found


def test_empty_dataframe():
    """Test that an empty DataFrame without columns triggers CRITICAL schema failure."""
    empty_df = pd.DataFrame()
    summary, findings = validate_schema(empty_df)

    assert summary["status"] == "FAIL"
    assert any(f["severity"] == SEVERITY_CRITICAL for f in findings)

    agent = ValidationAgent()
    res = agent.validate(empty_df)
    assert res["status"] == STATUS_BLOCK


def test_none_dataframe():
    """Test that passing None triggers CRITICAL schema failure without raising exception."""
    summary, findings = validate_schema(None)
    assert summary["status"] == "FAIL"
    assert findings[0]["severity"] == SEVERITY_CRITICAL

    agent = ValidationAgent()
    res = agent.validate(None)
    assert res["status"] == STATUS_BLOCK
    assert res["confidence"] < 100.0


def test_validate_file_missing():
    """Test that validating a non-existent file path returns BLOCK gracefully."""
    agent = ValidationAgent()
    res = agent.validate_file("data/non_existent_file_path.csv")
    assert res["status"] == STATUS_BLOCK
    assert res["confidence"] == 0.0
    assert len(res["findings"]) == 1
    assert "File not found" in res["findings"][0]["message"]
