"""Unit tests for AnomalyFinding construction, explainability, and categorization."""

import numpy as np
import pytest

from finsight.analysis.anomaly_scoring import grade_severities
from finsight.analysis.explanation import (
    build_anomaly_finding,
    categorize_anomaly_type,
    format_feature_value,
)
from finsight.core.models import AnomalyFinding, AnomalyType, Severity


def test_format_feature_value():
    """Test formatting of growth, margin, and ratio values."""
    assert format_feature_value("revenue_growth", 0.152) == "15.2%"
    assert format_feature_value("gross_margin", 0.4285) == "42.9%"
    assert format_feature_value("gross_margin_change", 0.052) == "+5.2 percentage points"
    assert format_feature_value("gross_margin_change", -0.031) == "-3.1 percentage points"
    assert format_feature_value("debt_to_assets", 1.45) == "1.45x"
    assert format_feature_value("revenue_growth", float("nan")) == "N/A"


def test_categorize_anomaly_type():
    """Test classification of historical, peer, cross-figure, and statistical anomalies."""
    # Cross figure pattern: revenue down, net income surging
    type1 = categorize_anomaly_type(
        {"revenue_growth": -0.25, "net_income_growth": 1.50},
        raw_record={},
    )
    assert type1 in (AnomalyType.CROSS_FIGURE_PATTERN, AnomalyType.REVENUE_PROFIT_MISMATCH)

    # Historical / growth outlier: big YoY growth shock
    type2 = categorize_anomaly_type(
        {"revenue_growth": 3.50, "net_income_growth": 3.20},
        raw_record={},
    )
    assert type2 in (AnomalyType.HISTORICAL_OUTLIER, AnomalyType.GROWTH_ANOMALY, AnomalyType.TEMPORAL_ANOMALY)

    # Peer / profitability outlier: static ratio outlier
    type3 = categorize_anomaly_type(
        {"gross_margin": 0.95, "roe": 1.20},
        raw_record={},
    )
    assert type3 in (AnomalyType.PEER_OUTLIER, AnomalyType.MARGIN_ANOMALY, AnomalyType.PROFITABILITY_ANOMALY)


def test_determine_severity():
    """Test severity classification with top 10% HIGH, next 20% MEDIUM, remainder LOW rule."""
    # Test percentiles directly
    percentiles = np.array([95.0, 75.0, 50.0])
    scores = np.array([0.9, 0.5, 0.2])
    severities = grade_severities(scores, percentiles=percentiles)
    assert severities[0] == Severity.HIGH
    assert severities[1] == Severity.MEDIUM
    assert severities[2] == Severity.LOW

    # Test automatic ranking across an array of scores
    test_scores = np.linspace(0.0, 1.0, 10)
    auto_severities = grade_severities(test_scores)
    assert auto_severities[-1] == Severity.HIGH
    assert Severity.MEDIUM in auto_severities
    assert auto_severities[0] == Severity.LOW


def test_build_anomaly_finding():
    """Test full finding generation with human-readable explanation."""
    finding = build_anomaly_finding(
        company="ACME_CORP",
        year=2023,
        score=0.22,
        feature_values={
            "revenue_growth": 4.50,
            "net_income_growth": 0.10,
            "gross_margin": 0.35,
        },
        feature_zscores={
            "revenue_growth": 5.2,
            "net_income_growth": 0.3,
            "gross_margin": 0.1,
        },
        raw_record={"revenue": 50000.0, "net_income": 3000.0},
        score_percentile=0.992,
        model_name="IsolationForest",
    )

    assert isinstance(finding, AnomalyFinding)
    assert finding.company == "ACME_CORP"
    assert finding.year == 2023
    assert finding.severity == Severity.HIGH
    assert finding.anomaly_type in (
        AnomalyType.CROSS_FIGURE_PATTERN,
        AnomalyType.REVENUE_PROFIT_MISMATCH,
        AnomalyType.HISTORICAL_OUTLIER,
        AnomalyType.GROWTH_ANOMALY,
    )
    assert "ACME_CORP" in finding.explanation
    assert "Revenue Growth" in finding.explanation
    assert len(finding.recommendation) > 10

    # Serialization test
    d = finding.to_dict()
    assert d["company"] == "ACME_CORP"
    assert d["year"] == 2023
    assert d["severity"] == "HIGH"
    assert "score" in d


def test_explanation_safety_and_conservative_language():
    """Test that generated findings never make unwarranted claims of fraud, manipulation, or misconduct."""
    finding = build_anomaly_finding(
        company="TEST_CORP",
        year=2022,
        score=0.35,
        feature_values={"revenue_growth": -0.40, "net_income_growth": 3.50},
        feature_zscores={"revenue_growth": -3.5, "net_income_growth": 4.2},
        raw_record={"revenue": 1000.0, "net_income": 300.0},
    )

    combined_text = (finding.explanation + " " + finding.recommendation).lower()

    # Forbidden claim keywords
    forbidden_words = ["fraud", "manipulation", "accounting error", "misconduct", "accounting fraud"]
    for word in forbidden_words:
        assert word not in combined_text, f"Forbidden word '{word}' found in finding text!"

    # Verify conservative explanation
    assert "statistically unusual" in combined_text
