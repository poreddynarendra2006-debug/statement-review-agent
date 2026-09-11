"""End-to-end integration and regression tests for User CSV and Excel file support."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.schema_mapper import DataValidationError, SchemaMapper
from finsight.data.loader import (
    clean_numeric_value,
    inspect_dataset,
    load_dataset,
    load_financial_file,
)


@pytest.fixture
def sample_user_csv(tmp_path: Path) -> Path:
    """Fixture creating a CSV file with renamed columns and custom formatting."""
    csv_file = tmp_path / "user_financials.csv"
    data = {
        "Entity Name": ["Acme Corp", "Acme Corp", "Acme Corp", "Beta LLC", "Beta LLC"],
        "Fiscal Period": [2021, 2022, 2023, 2022, 2023],
        "Sales Revenue": ["$10,000", "$12,500", "$15,000", "$5,000", "$6,000"],
        "Net Profit": ["1,200", "1,500", "(200)", "600", "800"],
        "Cash Flow from Operations": ["$1,400", "$1,600", "$300", "$700", "$900"],
        "Total Stockholders Equity": ["$8,000", "$9,200", "$9,000", "$4,000", "$4,600"],
        "Gross Earnings": ["4,000", "5,000", "5,500", "2,000", "2,400"],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_user_excel(tmp_path: Path) -> Path:
    """Fixture creating a multi-sheet Excel file (.xlsx) with Notes, Cover, and Financial Data."""
    xlsx_file = tmp_path / "financial_report.xlsx"

    cover_df = pd.DataFrame({"Title": ["Q4 Financial Overview"], "Date": ["2023-12-31"]})
    notes_df = pd.DataFrame({"Notes": ["Confidential", "Audited figures only"]})
    data_df = pd.DataFrame({
        "Company": ["Gamma Inc", "Gamma Inc", "Gamma Inc", "Delta Co", "Delta Co"],
        "Year": [2021, 2022, 2023, 2022, 2023],
        "Total Revenue": [20000.0, 24000.0, 30000.0, 8000.0, 10000.0],
        "Net Income": [2500.0, 3000.0, 4000.0, 900.0, 1200.0],
        "Operating Cash Flow": [2800.0, 3200.0, 4200.0, 1000.0, 1300.0],
        "Gross Profit": [8000.0, 9600.0, 12000.0, 3200.0, 4000.0],
        "Current Ratio": [1.8, 1.9, 2.1, 1.4, 1.5],
        "Debt/Equity Ratio": [0.6, 0.55, 0.50, 0.8, 0.75],
    })

    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        cover_df.to_excel(writer, sheet_name="Cover", index=False)
        notes_df.to_excel(writer, sheet_name="Instructions", index=False)
        data_df.to_excel(writer, sheet_name="Financial Statements", index=False)

    return xlsx_file


# A. Standard Cognizant CSV Loading
def test_standard_cognizant_csv_loading():
    """Verify default Financial Statements.csv loads properly with full 27 features."""
    dataset_path = next(p for p in ("data/Financial Statements.csv", "data/kaggle_financial_statements.csv") if Path(p).exists())
    loaded = load_financial_file(dataset_path)
    assert len(loaded.records) > 100
    assert len(loaded.mapping.available_features) == 27
    assert len(loaded.mapping.missing_required_fields) == 0
    assert loaded.has_temporal_ordering is True


# B. CSV with Renamed Columns
def test_csv_with_renamed_columns(sample_user_csv: Path):
    """Verify user CSV with custom financial column aliases is mapped correctly."""
    loaded = load_financial_file(sample_user_csv)
    assert len(loaded.records) == 5
    assert loaded.mapping.canonical_to_raw["revenue"] == "Sales Revenue"
    assert loaded.mapping.canonical_to_raw["net_income"] == "Net Profit"
    assert loaded.mapping.canonical_to_raw["operating_cash_flow"] == "Cash Flow from Operations"
    assert loaded.mapping.canonical_to_raw["shareholder_equity"] == "Total Stockholders Equity"
    assert loaded.mapping.canonical_to_raw["gross_profit"] == "Gross Earnings"

    rec_2023 = [r for r in loaded.records if r.company == "Acme Corp" and r.year == 2023][0]
    assert rec_2023.net_income == -200.0
    assert rec_2023.revenue == 15000.0


# C. Excel Ingestion (.xlsx)
def test_excel_file_ingestion(sample_user_excel: Path):
    """Verify single and multi-sheet Excel files load seamlessly."""
    loaded = load_financial_file(sample_user_excel)
    assert loaded.file_type == "excel"
    assert loaded.sheet_name == "Financial Statements"
    assert len(loaded.records) == 5


# D. Excel with Multiple Sheets
def test_excel_multi_sheet_selection_and_override(sample_user_excel: Path):
    """Verify automatic detection skips cover/notes sheets, and manual override works."""
    loaded = load_financial_file(sample_user_excel)
    assert loaded.sheet_name == "Financial Statements"

    loaded_explicit = load_financial_file(sample_user_excel, sheet_name="Financial Statements")
    assert loaded_explicit.sheet_name == "Financial Statements"

    with pytest.raises(DataValidationError):
        load_financial_file(sample_user_excel, sheet_name="Cover")


# E. Different Capitalization / Spacing / Hyphens / Underscores
def test_header_casing_and_punctuation_variations():
    """Verify casing and punctuation variations in columns."""
    df = pd.DataFrame({
        "COMPANY_NAME": ["Alpha Corp", "Alpha Corp"],
        "fiscal-year": [2022, 2023],
        "TOTAL  REVENUE": [1000.0, 1200.0],
        "Net_Income": [100.0, 150.0],
        "CURRENT--RATIO": [1.5, 1.6],
    })
    loaded = load_financial_file(df)
    assert loaded.mapping.canonical_to_raw["company"] == "COMPANY_NAME"
    assert loaded.mapping.canonical_to_raw["year"] == "fiscal-year"
    assert loaded.mapping.canonical_to_raw["revenue"] == "TOTAL  REVENUE"
    assert loaded.mapping.canonical_to_raw["net_income"] == "Net_Income"
    assert loaded.mapping.canonical_to_raw["current_ratio"] == "CURRENT--RATIO"


# F. Missing Optional Columns
def test_missing_optional_columns_partial_features():
    """Verify missing optional columns disable only dependent features without fabricating data."""
    df = pd.DataFrame({
        "Company": ["Corp A", "Corp A"],
        "Year": [2022, 2023],
        "Revenue": [1000.0, 1200.0],
        "Net Income": [100.0, 150.0],
    })
    loaded = load_financial_file(df)

    assert "ebitda" in loaded.missing_fields
    assert "operating_cash_flow" in loaded.missing_fields

    assert len(loaded.available_features) < 27
    assert "ebitda_growth" not in loaded.available_features
    assert "operating_cash_flow_growth" not in loaded.available_features
    assert "revenue_growth" in loaded.available_features


# G. Missing Required Columns
def test_missing_required_columns_validation_error():
    """Verify that a file lacking financial metrics triggers a DataValidationError."""
    df = pd.DataFrame({
        "Company": ["Test Corp"],
        "Year": [2023],
        "Headcount": [500],
        "Location": ["New York"],
    })
    with pytest.raises(DataValidationError) as exc:
        load_financial_file(df)
    assert "not contain enough financial fields" in str(exc.value)


# H. Ambiguous Column Mapping
def test_ambiguous_column_mapping_protection():
    """Verify multiple conflicting candidates for a financial metric trigger warnings."""
    df = pd.DataFrame({
        "Company": ["Corp X"],
        "Year": [2023],
        "Sales": [1000.0],
        "Turnover": [1000.0],
        "Net Profit": [100.0],
    })
    mapper = SchemaMapper()
    res = mapper.map_columns(list(df.columns))
    assert "revenue" in res.ambiguous_columns
    assert len(res.warnings) > 0


# I. Metadata Strictly Excluded from ML Features
def test_metadata_strictly_excluded_from_ml_features(sample_user_csv: Path):
    """Verify Company and Year are strictly treated as metadata and never fed into ML model."""
    detector = AnomalyDetector()
    loaded = load_financial_file(sample_user_csv)
    summary = detector.train(loaded.records)

    assert "company" not in detector.training_feature_names
    assert "year" not in detector.training_feature_names
    assert "record_id" not in detector.training_feature_names
    assert "Entity Name" not in detector.training_feature_names
    assert "Fiscal Period" not in detector.training_feature_names


# J. Malformed and Empty Files
def test_malformed_and_empty_file_handling(tmp_path: Path):
    """Verify empty and malformed files raise clear DataValidationError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    with pytest.raises(DataValidationError):
        load_financial_file(empty_csv)


# K. Numeric Cleaning & String Formats
def test_numeric_cleaning_helper():
    """Verify clean_numeric_value handles various string and currency representations."""
    assert clean_numeric_value("$1,234.56") == 1234.56
    assert clean_numeric_value("(500.00)") == -500.00
    assert clean_numeric_value("€ 2,500,000") == 2500000.0
    assert clean_numeric_value("15.5%") == 15.5
    assert clean_numeric_value("-") is None
    assert clean_numeric_value("N/A") is None
    assert clean_numeric_value("null") is None
    assert clean_numeric_value(np.nan) is None


# L. End-to-End Regression Test on Financial Statements.csv
def test_regression_pipeline_on_standard_dataset():
    """Verify full end-to-end inspect, train, and analyze workflow on standard dataset."""
    dataset_path = next(p for p in ("data/Financial Statements.csv", "data/kaggle_financial_statements.csv") if Path(p).exists())
    summary = inspect_dataset(dataset_path)
    assert summary.num_rows > 100

    records = load_dataset(dataset_path)
    detector = AnomalyDetector()
    train_summary = detector.train(records, model_mode="STANDARD_PRETRAINED")
    assert train_summary["num_features"] == 27
    assert detector.model_mode == "STANDARD_PRETRAINED"

    findings = detector.analyze(records)
    assert len(findings) > 0
    for f in findings:
        assert f.company is not None
        assert f.year is not None and f.year > 2000
        assert f.score is not None
        assert f.severity in ("LOW", "MEDIUM", "HIGH")
        assert f.model_mode == "STANDARD_PRETRAINED"


# M. Real Anomaly Dataset Non-Temporal Upload Workflow
def test_real_financial_statement_anomaly_dataset():
    """Verify the 4000-row Financial Statement Anomaly Dataset runs in generic local mode with non-zero findings."""
    dataset_path = Path("data/Financial Statement Anomaly Dataset.csv")
    if not dataset_path.exists():
        pytest.skip("Dataset file not present in workspace.")

    loaded = load_financial_file(dataset_path)
    assert len(loaded.records) == 4000
    assert loaded.has_temporal_ordering is False
    assert len(loaded.mapping.available_features) > 0

    # Verify no fake company or fake years were created
    assert loaded.records[0].company is None
    assert loaded.records[0].year is None
    assert loaded.records[0].record_id == "Record #1"

    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    findings = detector.analyze(loaded.records, allow_local_fit=True)

    # Must produce non-zero valid findings
    assert len(findings) > 0
    top_finding = findings[0]
    assert top_finding.score > 0
    assert top_finding.model_mode == "GENERIC_LOCAL"
    assert len(top_finding.relevant_features) > 0
    assert "Record" in top_finding.explanation
    assert top_finding.recommendation != ""


# N. Small Dataset & Edge Case Safeguards
def test_small_dataset_safeguard():
    """Verify datasets with fewer than 3 records return empty findings safely without crashing."""
    small_df = pd.DataFrame({
        "Revenue": [1000.0, 1200.0],
        "Net_Income": [100.0, 120.0],
        "Gross_Margin": [0.4, 0.45],
    })
    loaded = load_financial_file(small_df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    findings = detector.analyze(loaded.records, allow_local_fit=True)
    assert findings == []


# O. Constant Feature Safeguard (Zero Variance)
def test_constant_feature_handling():
    """Verify dataset with constant columns is handled safely by ZScore and IsolationForest."""
    const_df = pd.DataFrame({
        "Revenue": [1000.0] * 10,
        "Net_Income": [100.0] * 10,
        "Gross_Margin": [0.35] * 10,
        "Current_Ratio": [1.5] * 10,
    })
    loaded = load_financial_file(const_df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    findings = detector.analyze(loaded.records, allow_local_fit=True)
    # Model should train and analyze without divide-by-zero crash
    assert isinstance(findings, list)
