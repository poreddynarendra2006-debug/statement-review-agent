"""Comprehensive test suite for Adaptive Financial Anomaly Detection Agent.

Tests all 13 required scenarios:
1. Small dataset (20 records, 4 financial columns)
2. Medium dataset (500 records, 8 financial columns)
3. Large dataset (10,000+ records, scalable execution)
4. Dataset with 20+ financial features
5. Dataset with only Revenue, Net Income, and Assets
6. Dataset with Revenue, Profit, Debt, and Equity under custom column names
7. Dataset with Company + Year + financial data (temporal panel)
8. Dataset without Company and Year (cross-sectional)
9. Dataset containing extreme financial outliers (High severity & confidence)
10. Dataset with missing optional financial columns
11. Dataset with irrelevant categorical / text columns
12. Dataset with constant / near-constant features
13. Dataset with insufficient usable numerical features (graceful return)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.schema_mapper import SchemaMapper
from finsight.core.models import (
    AnalysisResult,
    AnomalyFinding,
    AnomalyType,
    FinancialRecord,
    Severity,
)
from finsight.data.loader import load_dataset, load_financial_file


# ---------------------------------------------------------------------------
# TEST 1: Small dataset with 20 records and 4 financial columns
# ---------------------------------------------------------------------------
def test_scenario_01_small_dataset_robust_statistics():
    """Test 1: Small dataset (20 records) uses robust statistical methods and conservative decisions."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(20):
        rev = float(rng.uniform(1000, 5000))
        net = float(rev * rng.uniform(0.08, 0.15))
        assets = float(rev * rng.uniform(1.2, 2.5))
        liab = float(assets * rng.uniform(0.3, 0.6))
        rows.append({
            "Revenue": rev,
            "Net_Income": net,
            "Total_Assets": assets,
            "Total_Liabilities": liab,
        })

    # Add 1 slight outlier in row 19
    rows[19]["Net_Income"] = float(rows[19]["Revenue"] * -0.50)  # Heavy loss

    df = pd.DataFrame(rows)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert isinstance(result, AnalysisResult)
    assert result.dataset_summary.records == 20
    assert "RobustStatistical" in result.dataset_summary.strategy_selected or "Small Dataset" in result.dataset_summary.strategy_selected
    assert result.dataset_summary.features_used > 0
    # Conservative decisions: confidence should not be exaggerated
    for finding in result.anomalies:
        assert isinstance(finding, AnomalyFinding)
        assert finding.confidence <= 0.85
        assert len(finding.relevant_features) > 0
        assert finding.explanation != ""


# ---------------------------------------------------------------------------
# TEST 2: Medium dataset with 500 records and 8 financial columns
# ---------------------------------------------------------------------------
def test_scenario_02_medium_dataset_ensemble():
    """Test 2: Medium dataset (500 records) with 8 financial columns."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(500):
        rev = float(rng.uniform(10000, 50000))
        gp = float(rev * rng.uniform(0.30, 0.50))
        net = float(gp * rng.uniform(0.20, 0.40))
        opex = float(gp - net)
        assets = float(rev * rng.uniform(1.5, 3.0))
        liab = float(assets * rng.uniform(0.3, 0.7))
        cash = float(assets * rng.uniform(0.05, 0.20))
        debt = float(liab * rng.uniform(0.4, 0.8))
        rows.append({
            "Revenue": rev,
            "Gross_Profit": gp,
            "Net_Income": net,
            "Operating_Expenses": opex,
            "Total_Assets": assets,
            "Total_Liabilities": liab,
            "Cash": cash,
            "Total_Debt": debt,
        })

    # Plant 3 distinct anomalies
    rows[50]["Gross_Profit"] = float(rows[50]["Revenue"] * 0.02)  # Margin collapse
    rows[150]["Net_Income"] = float(rows[150]["Revenue"] * -1.2)  # Extreme negative net margin
    rows[350]["Total_Liabilities"] = float(rows[350]["Total_Assets"] * 5.0)  # Extreme debt

    df = pd.DataFrame(rows)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.records == 500
    assert result.anomaly_summary.total_anomalies > 0
    # Anti-overreporting check: anomaly rate should stay reasonable (< 6%)
    assert result.anomaly_summary.anomaly_rate < 0.06
    # Verify planted anomalies were identified
    flagged_ids = {a.record_id for a in result.anomalies}
    assert "Record #51" in flagged_ids or "Record #151" in flagged_ids or "Record #351" in flagged_ids


# ---------------------------------------------------------------------------
# TEST 3: Large dataset with 10,000+ records (scalable processing)
# ---------------------------------------------------------------------------
def test_scenario_03_large_dataset_scalable():
    """Test 3: Large dataset (10,000+ records) processes scalably without memory errors."""
    rng = np.random.default_rng(42)
    n_records = 10_000
    rev = rng.uniform(5000, 50000, size=n_records)
    net = rev * rng.uniform(0.05, 0.20, size=n_records)
    assets = rev * rng.uniform(1.0, 3.0, size=n_records)
    liab = assets * rng.uniform(0.2, 0.8, size=n_records)

    df = pd.DataFrame({
        "Revenue": rev,
        "Net_Income": net,
        "Total_Assets": assets,
        "Total_Liabilities": liab,
    })

    # Add 5 extreme outliers
    for idx in [100, 2500, 5000, 7500, 9999]:
        df.loc[idx, "Net_Income"] = df.loc[idx, "Revenue"] * -3.0

    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.records == 10_000
    assert result.anomaly_summary.total_anomalies > 0
    assert result.anomaly_summary.anomaly_rate < 0.05
    assert len(result.anomalies) > 0


# ---------------------------------------------------------------------------
# TEST 4: Dataset with 20+ financial features
# ---------------------------------------------------------------------------
def test_scenario_04_high_dimensional_features():
    """Test 4: Dataset with 20+ financial features."""
    rng = np.random.default_rng(42)
    n = 100
    data = {}
    feature_names = [
        "Revenue", "Gross_Profit", "Net_Income", "Operating_Income", "Operating_Expenses",
        "EBITDA", "Total_Assets", "Total_Liabilities", "Shareholder_Equity", "Cash",
        "Operating_Cash_Flow", "Investing_Cash_Flow", "Financing_Cash_Flow", "Current_Ratio",
        "Debt_to_Equity", "ROE", "ROA", "ROI", "Gross_Margin", "Net_Profit_Margin",
        "EBITDA_Margin", "Free_Cash_Flow_per_Share", "Return_on_Tangible_Equity", "Asset_Turnover"
    ]
    for feat in feature_names:
        data[feat] = rng.normal(100.0, 15.0, size=n)

    # Invert one record completely
    data["ROE"][10] = -500.0
    data["Net_Profit_Margin"][10] = -40.0

    df = pd.DataFrame(data)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.features_used >= 15
    assert result.anomaly_summary.total_anomalies > 0
    top_finding = result.anomalies[0]
    assert len(top_finding.relevant_features) > 0
    assert len(top_finding.deviations) > 0


# ---------------------------------------------------------------------------
# TEST 5: Dataset with only Revenue, Net Income, and Assets
# ---------------------------------------------------------------------------
def test_scenario_05_minimal_three_columns():
    """Test 5: Dataset with only Revenue, Net Income, and Assets computes valid ratios."""
    df = pd.DataFrame({
        "Revenue": [1000.0, 1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 1000.0, 5000.0, 6000.0],
        "Net_Income": [100.0, 120.0, 150.0, -800.0, 250.0, 300.0, 350.0, 100.0, 500.0, 600.0],
        "Total_Assets": [2000.0, 2400.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 2000.0, 10000.0, 12000.0],
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.records == 10
    assert "net_profit_margin" in result.dataset_summary.derived_feature_names or "roa" in result.dataset_summary.derived_feature_names
    assert result.anomaly_summary.total_anomalies > 0
    # Record #4 (index 3 with net income -800) should be identified
    flagged_ids = {a.record_id for a in result.anomalies}
    assert "Record #4" in flagged_ids


# ---------------------------------------------------------------------------
# TEST 6: Custom Column Names (Revenue, Profit, Debt, Equity)
# ---------------------------------------------------------------------------
def test_scenario_06_custom_column_names():
    """Test 6: Dataset with non-standard column headers."""
    df = pd.DataFrame({
        "Topline Sales": [5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 12000.0, 15000.0],
        "Bottomline Profit": [500.0, 600.0, 700.0, 800.0, -4000.0, 1000.0, 1200.0, 1500.0],
        "Total Indebtedness": [2000.0, 2400.0, 2800.0, 3200.0, 12000.0, 4000.0, 4800.0, 6000.0],
        "Book Value of Equity": [4000.0, 4800.0, 5600.0, 6400.0, 1000.0, 8000.0, 9600.0, 12000.0],
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.records == 8
    assert result.dataset_summary.features_used > 0
    assert result.anomaly_summary.total_anomalies > 0
    # Record #5 (index 4) has profit -4000 and debt 12000
    flagged_ids = {a.record_id for a in result.anomalies}
    assert "Record #5" in flagged_ids


# ---------------------------------------------------------------------------
# TEST 7: Dataset with Company + Year + Financial data (Temporal Active)
# ---------------------------------------------------------------------------
def test_scenario_07_temporal_panel_dataset():
    """Test 7: Multi-company panel dataset with years computes YoY metrics."""
    records = []
    for yr in range(2018, 2024):
        records.append(FinancialRecord(
            company="Alpha Corp",
            year=yr,
            revenue=1000.0 * (1.1 ** (yr - 2018)),
            gross_profit=400.0 * (1.1 ** (yr - 2018)),
            net_income=100.0 * (1.1 ** (yr - 2018)) if yr != 2022 else -500.0,  # Sudden collapse
            total_assets=2000.0,
            total_liabilities=800.0,
        ))
        records.append(FinancialRecord(
            company="Beta Inc",
            year=yr,
            revenue=5000.0 * (1.05 ** (yr - 2018)),
            gross_profit=2000.0 * (1.05 ** (yr - 2018)),
            net_income=500.0 * (1.05 ** (yr - 2018)),
            total_assets=10000.0,
            total_liabilities=4000.0,
        ))

    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(records)

    assert result.dataset_summary.temporal_analysis is True
    assert "revenue_growth" in result.dataset_summary.derived_feature_names or "net_income_growth" in result.dataset_summary.derived_feature_names
    # Alpha Corp 2022 shock should be flagged
    alpha_2022_findings = [a for a in result.anomalies if a.company == "Alpha Corp" and a.year == 2022]
    assert len(alpha_2022_findings) > 0


# ---------------------------------------------------------------------------
# TEST 8: Dataset without Company and Year (Cross-sectional)
# ---------------------------------------------------------------------------
def test_scenario_08_no_company_no_year():
    """Test 8: Cross-sectional dataset with no company and no year does NOT fake temporal ordering."""
    df = pd.DataFrame({
        "Revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0],
        "Net_Income": [100.0, 200.0, 300.0, 400.0, -1500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "Total_Assets": [2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0],
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.temporal_analysis is False
    assert result.dataset_summary.peer_analysis is False
    for a in result.anomalies:
        assert a.company is None
        assert a.year is None
        assert a.record_id.startswith("Record #")


# ---------------------------------------------------------------------------
# TEST 9: Extreme Financial Outliers (High Severity & Confidence)
# ---------------------------------------------------------------------------
def test_scenario_09_extreme_financial_outliers():
    """Test 9: Extreme financial outliers receive HIGH severity and HIGH confidence."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(100):
        rev = float(rng.uniform(10000, 50000))
        net = float(rev * rng.uniform(0.08, 0.15))
        assets = float(rev * rng.uniform(1.5, 3.0))
        rows.append({"Revenue": rev, "Net_Income": net, "Total_Assets": assets})

    # Extreme outlier
    rows[50]["Net_Income"] = float(rows[50]["Revenue"] * -15.0)  # -1500% profit margin

    df = pd.DataFrame(rows)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    outlier_finding = [a for a in result.anomalies if a.record_id == "Record #51"]
    assert len(outlier_finding) > 0
    assert outlier_finding[0].severity == Severity.HIGH
    assert outlier_finding[0].confidence >= 0.85
    assert len(outlier_finding[0].explanation) > 0


# ---------------------------------------------------------------------------
# TEST 10: Missing Optional Financial Columns
# ---------------------------------------------------------------------------
def test_scenario_10_missing_optional_columns():
    """Test 10: Missing optional columns does not cause crash or fabricate values."""
    df = pd.DataFrame({
        "Revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "Net_Income": [100.0, 200.0, 300.0, 400.0, -1000.0],
        # EBITDA, Cash Flow, Debt, Equity, Market Cap are all omitted
    })
    loaded = load_financial_file(df)
    assert "ebitda" in loaded.missing_fields
    assert "operating_cash_flow" in loaded.missing_fields

    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)
    assert result.dataset_summary.records == 5
    assert result.dataset_summary.features_used > 0


# ---------------------------------------------------------------------------
# TEST 11: Irrelevant Categorical and Text Columns
# ---------------------------------------------------------------------------
def test_scenario_11_irrelevant_categorical_columns():
    """Test 11: Categorical and text columns are ignored from ML features."""
    df = pd.DataFrame({
        "Company": ["Corp A", "Corp B", "Corp C", "Corp D", "Corp E"],
        "Revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "Net_Income": [100.0, 200.0, 300.0, 400.0, -1000.0],
        "Auditor_Name": ["PwC", "Deloitte", "EY", "KPMG", "BDO"],
        "HQ_City": ["New York", "London", "Tokyo", "Paris", "Berlin"],
        "Internal_Audit_Notes": ["Clean", "Clean", "Pending", "Clean", "Requires Investigation"],
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    used_feats = result.dataset_summary.used_feature_names
    assert "Auditor_Name" not in used_feats
    assert "HQ_City" not in used_feats
    assert "Internal_Audit_Notes" not in used_feats
    assert "Auditor_Name" in result.dataset_summary.ignored_features or "auditor_name" in result.dataset_summary.ignored_features


# ---------------------------------------------------------------------------
# TEST 12: Constant / Near-Constant Features
# ---------------------------------------------------------------------------
def test_scenario_12_constant_features():
    """Test 12: Dataset with constant / near-constant columns does not divide by zero or crash."""
    df = pd.DataFrame({
        "Revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        "Net_Income": [100.0, 200.0, 300.0, 400.0, 500.0, -2000.0],
        "Inflation_Rate": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],  # Constant
        "Statutory_Tax_Rate": [0.21, 0.21, 0.21, 0.21, 0.21, 0.21],  # Constant
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert result.dataset_summary.records == 6
    assert isinstance(result.anomalies, list)


# ---------------------------------------------------------------------------
# TEST 13: Insufficient Usable Numerical Features (Graceful Return)
# ---------------------------------------------------------------------------
def test_scenario_13_insufficient_features_graceful_exit():
    """Test 13: Dataset with zero numerical financial columns gracefully returns meaningful message without crashing."""
    df = pd.DataFrame({
        "Company": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
        "Country": ["USA", "UK", "Germany"],
        "Auditor": ["PwC", "EY", "KPMG"],
        "Status": ["Active", "Active", "Inactive"],
    })

    # Schema mapping detection
    mapper = SchemaMapper()
    mapping = mapper.map_columns(list(df.columns))
    assert len(mapping.available_financial_fields) == 0

    # Passing records with 0 numerical features into AnomalyDetector
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    records = [FinancialRecord(company=c) for c in df["Company"]]
    result = detector.analyze_dataset(records)

    assert result.dataset_summary.records == 3
    assert result.dataset_summary.features_used == 0
    assert result.anomaly_summary.total_anomalies == 0
    assert "Insufficient usable numerical financial features" in result.dataset_summary.message
