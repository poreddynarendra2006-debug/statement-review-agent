"""Unit and regression tests specifically covering fixes for BUG 1 through BUG 7."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.explanation import format_feature_value
from finsight.analysis.schema_mapper import SchemaMapper
from finsight.core.config import AnomalyConfig, IsolationForestConfig
from finsight.core.models import Severity
from finsight.data.loader import generate_reference_dataset, load_financial_file


# ---------------------------------------------------------------------------
# BUG 1 FIX TESTS: KeyError on columns with leading/trailing whitespace
# ---------------------------------------------------------------------------
def test_bug1_csv_header_with_trailing_whitespace(tmp_path: Path):
    """BUG 1: Verify CSV with trailing and leading whitespace in headers loads without KeyError."""
    csv_file = tmp_path / "whitespace_headers.csv"
    csv_content = (
        "Company , Year , Revenue , Gross Profit , Net Income , Total Assets , Total Liabilities \n"
        "Alpha Corp , 2022 , 10000.0 , 4000.0 , 1000.0 , 20000.0 , 8000.0 \n"
        "Alpha Corp , 2023 , 12000.0 , 4800.0 , 1200.0 , 24000.0 , 9000.0 \n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    loaded = load_financial_file(csv_file)
    assert len(loaded.records) == 2
    assert loaded.records[0].company == "Alpha Corp"
    assert loaded.records[0].year == 2022
    assert loaded.records[0].revenue == 10000.0
    assert loaded.records[0].gross_profit == 4000.0
    assert loaded.records[0].net_income == 1000.0
    assert loaded.records[0].total_assets == 20000.0
    assert loaded.records[0].total_liabilities == 8000.0


# ---------------------------------------------------------------------------
# BUG 2 FIX TESTS: Over-flagging reduction and configurable contamination
# ---------------------------------------------------------------------------
def test_bug2_clean_161_row_dataset_low_false_positive_rate():
    """BUG 2: Verify clean 161-row dataset flags roughly 5% anomalies instead of 71%."""
    df_clean = generate_reference_dataset(seed=42)
    assert len(df_clean) >= 160

    loaded = load_financial_file(df_clean)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    # Contamination should flag roughly 5% (~7-9 anomalies out of 161)
    anomaly_rate = result.anomaly_summary.anomaly_rate
    assert 0.03 <= anomaly_rate <= 0.07, f"Expected ~5% anomaly rate, got {anomaly_rate*100:.1f}%"
    assert result.anomaly_summary.total_anomalies <= 12


def test_bug2_clean_480_row_dataset_low_false_positive_rate():
    """BUG 2: Verify clean 480-row dataset flags roughly 5% anomalies instead of 27%."""
    rng = np.random.default_rng(123)
    n = 480
    rev = rng.uniform(10000, 50000, size=n)
    gp = rev * rng.uniform(0.35, 0.55, size=n)
    net = gp * rng.uniform(0.15, 0.35, size=n)
    assets = rev * rng.uniform(1.2, 2.5, size=n)
    liab = assets * rng.uniform(0.3, 0.6, size=n)

    df_clean = pd.DataFrame({
        "Revenue": rev,
        "Gross Profit": gp,
        "Net Income": net,
        "Total Assets": assets,
        "Total Liabilities": liab,
    })

    loaded = load_financial_file(df_clean)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    anomaly_rate = result.anomaly_summary.anomaly_rate
    assert 0.03 <= anomaly_rate <= 0.07, f"Expected ~5% anomaly rate on 480-row dataset, got {anomaly_rate*100:.1f}%"


def test_bug2_configurable_contamination():
    """BUG 2: Verify contamination rate is configurable (e.g. 2% flags ~2%)."""
    df_clean = generate_reference_dataset(seed=42)
    loaded = load_financial_file(df_clean)

    cfg = AnomalyConfig(contamination=0.02)
    detector = AnomalyDetector(config=cfg, model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    anomaly_rate = result.anomaly_summary.anomaly_rate
    assert 0.01 <= anomaly_rate <= 0.035, f"Expected ~2% anomaly rate, got {anomaly_rate*100:.1f}%"


# ---------------------------------------------------------------------------
# BUG 3 FIX TESTS: Severity varies (HIGH, MEDIUM, LOW) based on percentiles
# ---------------------------------------------------------------------------
def test_bug3_severity_grades_all_three_levels_on_realistic_dataset():
    """BUG 3: Verify all three severity levels (HIGH, MEDIUM, LOW) appear on a realistic dataset."""
    df_sample = generate_reference_dataset(seed=42)
    loaded = load_financial_file(df_sample)

    # Use default detector
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    findings = result.anomalies
    assert len(findings) >= 3

    severities = {f.severity for f in findings}
    assert Severity.HIGH in severities, "HIGH severity findings missing"
    assert Severity.MEDIUM in severities, "MEDIUM severity findings missing"
    assert Severity.LOW in severities, "LOW severity findings missing"

    # Check breakdown matches summary
    assert result.anomaly_summary.high > 0
    assert result.anomaly_summary.medium > 0
    assert result.anomaly_summary.low > 0
    assert result.anomaly_summary.high + result.anomaly_summary.medium + result.anomaly_summary.low == len(findings)


# ---------------------------------------------------------------------------
# BUG 4 FIX TESTS: Percentage double-scaling detection and formatting
# ---------------------------------------------------------------------------
def test_bug4_format_fractional_and_percentage_values():
    """BUG 4: Verify format_feature_value handles fractional and percentage formats correctly by feature name."""
    # Fractional values (margins, returns, growth)
    assert format_feature_value("roa", 0.2829) == "28.3%"
    assert format_feature_value("roe", 0.155) == "15.5%"
    assert format_feature_value("gross_margin", 0.4285) == "42.9%"
    assert format_feature_value("revenue_growth", 1.50) == "150.0%"
    assert format_feature_value("gross_margin_change", 0.052) == "+5.2 percentage points"
    assert format_feature_value("gross_margin_change", -0.031) == "-3.1 percentage points"

    # Percentage-named fields (already in percent scale)
    assert format_feature_value("inflation_rate", 3.2) == "3.2%"
    assert format_feature_value("tax_rate_percent", 21.0) == "21.0%"
    assert format_feature_value("gross_margin_pct", 42.9) == "42.9%"


# ---------------------------------------------------------------------------
# BUG 5 FIX TESTS: Silent unmapped columns and missing aliases
# ---------------------------------------------------------------------------
def test_bug5_missing_column_aliases_mapped_properly():
    """BUG 5: Verify 'Cash Flow from Operating', 'Cash Flow from Financial Activities', and 'Inflation Rate(in US)' map properly."""
    raw_columns = [
        "Company",
        "Year",
        "Revenue",
        "Cash Flow from Operating",
        "Cash Flow from Financial Activities",
        "Inflation Rate(in US)",
    ]
    mapper = SchemaMapper()
    result = mapper.map_columns(raw_columns)

    assert result.canonical_to_raw.get("operating_cash_flow") == "Cash Flow from Operating"
    assert result.canonical_to_raw.get("financing_cash_flow") == "Cash Flow from Financial Activities"
    assert result.canonical_to_raw.get("inflation_rate") == "Inflation Rate(in US)"


def test_bug5_unmapped_columns_emit_warning():
    """BUG 5: Verify unmapped columns emit a warning rather than being dropped silently."""
    raw_columns = [
        "Company",
        "Year",
        "Revenue",
        "Net Income",
        "Unrecognized Custom Column XYZ",
        "Another Mystery Header 123",
    ]
    mapper = SchemaMapper()
    result = mapper.map_columns(raw_columns)

    assert "Unrecognized Custom Column XYZ" in result.unmapped_columns
    assert "Another Mystery Header 123" in result.unmapped_columns
    assert len(result.warnings) > 0
    assert any("Unmapped columns" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# BUG 6 FIX TESTS: dataset_summary count fields match lists
# ---------------------------------------------------------------------------
def test_bug6_dataset_summary_counts_match_lists():
    """BUG 6: Verify dataset_summary raw_columns_count and derived_features_count match their respective lists."""
    df_sample = generate_reference_dataset(seed=42)
    loaded = load_financial_file(df_sample)

    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    d_sum = result.dataset_summary
    assert d_sum.raw_columns_count == len(d_sum.raw_column_names)
    assert d_sum.raw_columns_count > 0

    assert d_sum.derived_features_count == len(d_sum.derived_feature_names)
    assert d_sum.derived_features_count > 0

    assert d_sum.features_available == len(d_sum.available_feature_names)
    assert d_sum.features_used == len(d_sum.used_feature_names)

    # Verify dictionary serialization
    d_dict = d_sum.to_dict()
    assert d_dict["raw_columns_count"] == len(d_dict["feature_details"]["raw_columns"])
    assert d_dict["derived_features_count"] == len(d_dict["feature_details"]["derived"])
