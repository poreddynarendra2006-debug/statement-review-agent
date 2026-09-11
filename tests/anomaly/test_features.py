"""Unit tests for financial feature engineering on the 23-column dataset."""

import numpy as np
import pandas as pd
import pytest

from finsight.analysis.anomaly_features import (
    FEATURE_REGISTRY,
    FeaturePipeline,
    _safe_divide,
    compute_financial_features,
)
from finsight.core.models import FinancialRecord


@pytest.fixture
def sample_company_records() -> list[FinancialRecord]:
    """Sample records for two companies with cash flows and balance sheet ratios."""
    return [
        FinancialRecord(
            company="CORP_A",
            year=2020,
            revenue=1000.0,
            gross_profit=400.0,
            net_income=100.0,
            ebitda=200.0,
            shareholder_equity=500.0,
            operating_cash_flow=120.0,
            investing_cash_flow=-40.0,
            financing_cash_flow=-20.0,
            current_ratio=1.50,
            debt_to_equity=0.60,
            roe=0.20,
            roa=0.10,
            roi=0.12,
            net_profit_margin=0.10,
            free_cash_flow_per_share=2.50,
            return_on_tangible_equity=0.22,
        ),
        FinancialRecord(
            company="CORP_A",
            year=2021,
            revenue=1200.0,
            gross_profit=500.0,
            net_income=150.0,
            ebitda=250.0,
            shareholder_equity=600.0,
            operating_cash_flow=180.0,
            investing_cash_flow=-50.0,
            financing_cash_flow=-30.0,
            current_ratio=1.60,
            debt_to_equity=0.55,
            roe=0.25,
            roa=0.12,
            roi=0.14,
            net_profit_margin=0.125,
            free_cash_flow_per_share=3.20,
            return_on_tangible_equity=0.28,
        ),
        FinancialRecord(
            company="CORP_B",
            year=2021,
            revenue=5000.0,
            gross_profit=1500.0,
            net_income=500.0,
            ebitda=800.0,
            shareholder_equity=2000.0,
            operating_cash_flow=600.0,
            investing_cash_flow=-200.0,
            financing_cash_flow=-100.0,
            current_ratio=2.00,
            debt_to_equity=0.80,
            roe=0.25,
            roa=0.15,
            roi=0.16,
            net_profit_margin=0.10,
            free_cash_flow_per_share=8.00,
            return_on_tangible_equity=0.29,
        ),
        FinancialRecord(
            company="CORP_B",
            year=2022,
            revenue=5500.0,
            gross_profit=1650.0,
            net_income=550.0,
            ebitda=880.0,
            shareholder_equity=2200.0,
            operating_cash_flow=660.0,
            investing_cash_flow=-220.0,
            financing_cash_flow=-110.0,
            current_ratio=2.10,
            debt_to_equity=0.75,
            roe=0.25,
            roa=0.14,
            roi=0.15,
            net_profit_margin=0.10,
            free_cash_flow_per_share=8.80,
            return_on_tangible_equity=0.28,
        ),
    ]


def test_safe_divide():
    """Test safe division behavior."""
    assert _safe_divide(10.0, 2.0) == 5.0
    assert np.isnan(_safe_divide(10.0, 0.0))
    assert np.isnan(_safe_divide(10.0, 1e-12))


def test_profitability_and_cash_flow_ratios(sample_company_records):
    """Test calculation of profitability and cash flow ratios."""
    meta, feat_df, active = compute_financial_features(sample_company_records)

    row0 = feat_df.iloc[0]
    assert pytest.approx(row0["gross_margin"], 1e-4) == 0.40
    assert pytest.approx(row0["net_profit_margin"], 1e-4) == 0.10
    assert pytest.approx(row0["ebitda_margin"], 1e-4) == 0.20
    assert pytest.approx(row0["roe"], 1e-4) == 0.20
    assert pytest.approx(row0["current_ratio"], 1e-4) == 1.50
    assert pytest.approx(row0["debt_equity_ratio"], 1e-4) == 0.60
    assert pytest.approx(row0["operating_cash_flow_to_net_income"], 1e-4) == 1.20


def test_growth_and_cash_flow_features(sample_company_records):
    """Test YoY growth across revenue, net income, and operating cash flows."""
    meta, feat_df, active = compute_financial_features(sample_company_records)

    # First year for CORP_A -> NaNs
    assert pd.isna(feat_df.iloc[0]["revenue_growth"])
    assert pd.isna(feat_df.iloc[0]["operating_cash_flow_growth"])

    # Second year for CORP_A -> (1200 - 1000)/1000 = 0.20
    assert pytest.approx(feat_df.iloc[1]["revenue_growth"], 1e-4) == 0.20
    # OCF: (180 - 120)/120 = 0.50 (50%)
    assert pytest.approx(feat_df.iloc[1]["operating_cash_flow_growth"], 1e-4) == 0.50

    # Cross-Figure spread: Net Income growth (50%) - Revenue growth (20%) = +30 percentage points
    assert pytest.approx(feat_df.iloc[1]["revenue_profit_growth_spread"], 1e-4) == 0.30

    # First year for CORP_B -> Must be NaN and not compared to CORP_A
    assert pd.isna(feat_df.iloc[2]["revenue_growth"])
    assert pd.isna(feat_df.iloc[2]["operating_cash_flow_growth"])


def test_ratio_change_features(sample_company_records):
    """Test calculation of historical ratio shifts."""
    meta, feat_df, active = compute_financial_features(sample_company_records)

    # CORP_A 2021: current_ratio shift = 1.60 - 1.50 = +0.10
    assert pytest.approx(feat_df.iloc[1]["current_ratio_change"], 1e-4) == 0.10
    # debt_equity_change = 0.55 - 0.60 = -0.05
    assert pytest.approx(feat_df.iloc[1]["debt_equity_change"], 1e-4) == -0.05


def test_pipeline_imputation_and_transform(sample_company_records):
    """Test that FeaturePipeline cleans all NaNs/Infs."""
    meta, feat_df, _ = compute_financial_features(sample_company_records)

    pipeline = FeaturePipeline()
    matrix = pipeline.fit_transform(feat_df)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == feat_df.shape
    assert not np.isnan(matrix).any()
    assert not np.isinf(matrix).any()
    assert pipeline.fitted_
