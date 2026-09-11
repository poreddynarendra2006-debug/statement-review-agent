"""Regression tests: cross-company scale neutrality and plain-language explanations."""

import re
from pathlib import Path

import numpy as np
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.dataset_profiler import profile_dataset
from finsight.analysis.explanation import build_anomaly_finding
from finsight.analysis.feature_engineering import RAW_LEVEL_REASON, extract_dynamic_features
from finsight.core.models import FinancialRecord
from finsight.data.loader import load_financial_file

RAW_LEVELS = {
    "revenue", "gross_profit", "net_income", "ebitda", "eps", "shareholder_equity", "total_assets",
    "total_liabilities", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
}
STATEMENTS = Path(__file__).resolve().parents[1] / "data" / "Financial Statements.csv"


def _records(companies=6, years=range(2015, 2023), giant_scale=1000.0):
    """Companies with similar ratios; the first one is far larger than the rest."""
    rng = np.random.default_rng(7)
    records = []
    for c in range(companies):
        revenue = 1000.0 * (giant_scale if c == 0 else 1.0)
        for year in years:
            revenue *= 1.0 + rng.uniform(0.03, 0.08)
            records.append(FinancialRecord(
                company=f"Company {c}",
                year=year,
                revenue=revenue,
                gross_profit=revenue * rng.uniform(0.38, 0.42),
                net_income=revenue * rng.uniform(0.09, 0.11),
                total_assets=revenue * rng.uniform(1.9, 2.1),
                total_liabilities=revenue * rng.uniform(0.9, 1.1),
                shareholder_equity=revenue * rng.uniform(0.9, 1.1),
                operating_cash_flow=revenue * rng.uniform(0.11, 0.13),
            ))
    return records


def test_cross_company_review_leaves_out_company_size_amounts():
    result = AnomalyDetector(model_mode="GENERIC_LOCAL").analyze_dataset(_records())

    used = set(result.dataset_summary.used_feature_names)
    assert not used & RAW_LEVELS
    assert {"net_profit_margin", "revenue_growth"} <= used
    assert result.dataset_summary.ignored_features["revenue"] == RAW_LEVEL_REASON
    for finding in result.anomalies:
        assert not set(finding.relevant_features) & RAW_LEVELS


def test_single_company_keeps_amounts_for_review_over_time():
    records = _records(companies=1, years=range(2008, 2023))
    result = AnomalyDetector(model_mode="GENERIC_LOCAL").analyze_dataset(records)

    assert "revenue" in result.dataset_summary.used_feature_names


def test_amounts_kept_when_too_few_ratios_to_compare_on():
    records = [
        {"company": f"Company {i}", "revenue": 1000.0 * (i + 1), "net_income": 100.0 * (i + 1)}
        for i in range(8)
    ]
    result = AnomalyDetector(model_mode="GENERIC_LOCAL").analyze_dataset(records)

    assert "revenue" in result.dataset_summary.used_feature_names


@pytest.mark.skipif(not STATEMENTS.exists(), reason="data/Financial Statements.csv not present")
def test_statements_upload_uses_only_engineered_features():
    loaded = load_financial_file(STATEMENTS)
    result = AnomalyDetector(model_mode="GENERIC_LOCAL").analyze_dataset(loaded.records, allow_local_fit=True)

    used = set(result.dataset_summary.used_feature_names)
    assert not used & RAW_LEVELS
    assert len(used) == 27
    for finding in result.anomalies:
        assert not set(finding.relevant_features) & RAW_LEVELS


def test_explanation_describes_growth_gap_in_financial_terms():
    finding = build_anomaly_finding(
        company="Example Motors",
        year=2017,
        feature_values={
            "revenue_growth": 0.421,
            "net_income_growth": 7.566,
            "operating_cash_flow_growth": 7.633,
            "revenue_profit_growth_spread": 7.145,
        },
        feature_zscores={
            "revenue_profit_growth_spread": 32.9,
            "net_income_growth": 30.1,
            "operating_cash_flow_growth": 26.9,
            "revenue_growth": 3.9,
        },
    )
    text = finding.explanation

    assert text.startswith(
        "Example Motors (2017): Revenue increased 42.1% year over year, while net income increased 756.6% "
        "and operating cash flow increased 763.3%"
    )
    assert "potential revenue profit mismatch requiring review" in text
    lowered = text.lower()
    for jargon in ("std", "normal benchmark", "normalbenchmark", "driven by", "drivenby", "deviation"):
        assert jargon not in lowered
    assert not re.search(r"\bmad\b", lowered)
    # The statistics stay available as supporting evidence
    assert finding.deviations["revenue_profit_growth_spread"] == 32.9


def test_margin_change_uses_percentage_points():
    finding = build_anomaly_finding(
        company="Example Retail",
        year=2020,
        feature_values={"gross_margin_change": -0.12},
        feature_zscores={"gross_margin_change": -6.0},
        feature_medians={"gross_margin_change": 0.005},
        feature_iqrs={"gross_margin_change": 0.02},
    )

    assert "Gross margin fell by 12.0 percentage points from the prior year" in finding.explanation
    assert "typical change of +0.5 percentage points" in finding.explanation
    assert "%" not in finding.explanation


def test_record_label_is_not_repeated():
    finding = build_anomaly_finding(
        record_id="Record #4",
        feature_values={"net_profit_margin": -0.8},
        feature_zscores={"net_profit_margin": -9.0},
    )

    assert finding.explanation.startswith("Record #4: Net profit margin was -80.0%, unusually low")


def test_normal_ranges_are_in_the_features_own_units():
    records = _records()
    result = AnomalyDetector(model_mode="GENERIC_LOCAL").analyze_dataset(records, include_normal=True)
    _, features, _, _ = extract_dynamic_features(records, profile=profile_dataset(records))

    checked = 0
    for finding in result.anomalies:
        for name, (low, high) in finding.normal_ranges.items():
            median = float(features[name].dropna().median())
            assert np.isclose((low + high) / 2, median, atol=1e-3)
            checked += 1
    assert checked > 0
