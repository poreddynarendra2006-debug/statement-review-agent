"""
tests/test_duplicates.py

Tests for Duplicate Validation (Check 3):
- Valid unique Company-Year records pass
- Duplicate Company-Year records flagged with MEDIUM severity
- Same company different years pass
- Different companies same year pass
"""

import pytest
import pandas as pd

from validation_agent.rules import (
    REQUIRED_COLUMNS,
    SEVERITY_MEDIUM,
    STATUS_PASS,
    STATUS_WARNING
)
from validation_agent.checks import validate_duplicates
from validation_agent.validator import ValidationAgent


@pytest.fixture
def base_row():
    row = {c: 10.0 for c in REQUIRED_COLUMNS}
    row["Category"] = "IT"
    row["Number of Employees"] = 1000
    row["Revenue"] = 1000.0
    row["Net Income"] = 100.0
    row["Net Profit Margin"] = 10.0
    return row


def test_unique_company_year(base_row):
    """Different companies and different years should pass duplicate validation."""
    r1 = dict(base_row, **{"Company ": "AAPL", "Year": 2021})
    r2 = dict(base_row, **{"Company ": "AAPL", "Year": 2022})
    r3 = dict(base_row, **{"Company ": "MSFT", "Year": 2021})
    r4 = dict(base_row, **{"Company ": "MSFT", "Year": 2022})

    df = pd.DataFrame([r1, r2, r3, r4])
    summary, findings = validate_duplicates(df)

    assert summary["status"] == "PASS"
    assert len(findings) == 0


def test_duplicate_company_year(base_row):
    """Identical Company and Year across multiple records should be flagged."""
    r1 = dict(base_row, **{"Company ": "GOOG", "Year": 2022, "Revenue": 150000.0, "Net Income": 15000.0, "Net Profit Margin": 10.0})
    r2 = dict(base_row, **{"Company ": "GOOG", "Year": 2022, "Revenue": 150000.0, "Net Income": 15000.0, "Net Profit Margin": 10.0})
    r3 = dict(base_row, **{"Company ": "AMZN", "Year": 2022, "Revenue": 200000.0, "Net Income": 20000.0, "Net Profit Margin": 10.0})

    df = pd.DataFrame([r1, r2, r3])
    summary, findings = validate_duplicates(df)

    assert summary["status"] == "FAIL"
    # Both duplicate records are identified
    assert len(findings) == 2
    for f in findings:
        assert f["check"] == "DUPLICATE"
        assert f["company"] == "GOOG"
        assert f["year"] == 2022
        assert f["severity"] == SEVERITY_MEDIUM

    agent = ValidationAgent()
    res = agent.validate(df)
    assert res["checks"]["duplicates"]["status"] == "FAIL"
    # Duplicates alone with no high/critical issues trigger WARNING
    assert res["status"] == STATUS_WARNING
