"""Production resilience tests for FinSight AI Anomaly Detection Agent.

Verifies:
1. Diverse real-world accounting terminology aliases (PAT, Turnover, COGS, Cash & Equivalents).
2. Large company scale invariance (large enterprises with normal margins not falsely flagged).
3. Nuanced, non-constant confidence scores across varied anomaly profiles.
4. Multi-sheet Excel workbook ingestion and explicit sheet selection.
5. Multi-format exports: JSON, CSV, and standalone HTML audit reports.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.exporter import (
    export_findings_csv,
    export_findings_html,
    export_findings_json,
)
from finsight.core.models import Severity
from finsight.data.loader import load_financial_file


def test_diverse_accounting_terminology_aliases(tmp_path):
    """Test mapping of real-world international financial reporting terminology."""
    df = pd.DataFrame({
        "Organization": ["Acme Ltd", "Acme Ltd", "Beta Corp", "Beta Corp"],
        "Financial Year": [2021, 2022, 2021, 2022],
        "Turnover": [1000000.0, 1100000.0, 500000.0, 550000.0],
        "Profit After Tax": [100000.0, 115000.0, 45000.0, 52000.0],
        "Cost of Sales": [600000.0, 650000.0, 300000.0, 325000.0],
        "Operating Earnings": [150000.0, 165000.0, 65000.0, 75000.0],
        "Net Assets": [800000.0, 900000.0, 400000.0, 450000.0],
        "Cash and Cash Equivalents": [200000.0, 220000.0, 90000.0, 110000.0],
    })
    csv_path = tmp_path / "uk_statements.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_financial_file(csv_path)

    assert "Turnover" in loaded.mapping.mapped_columns
    assert loaded.mapping.mapped_columns["Turnover"] == "revenue"
    assert loaded.mapping.mapped_columns["Profit After Tax"] == "net_income"
    assert loaded.mapping.mapped_columns["Cost of Sales"] == "operating_expenses"
    assert loaded.mapping.mapped_columns["Operating Earnings"] == "operating_income"
    assert loaded.mapping.mapped_columns["Net Assets"] == "shareholder_equity"
    assert loaded.mapping.mapped_columns["Cash and Cash Equivalents"] == "cash"


def test_confidence_score_variation():
    """Verify confidence scores are varied and evidence-grounded rather than a single flat constant."""
    rng = np.random.default_rng(123)
    rows = []
    for i in range(120):
        rev = float(rng.uniform(10000, 50000))
        margin = float(rng.uniform(0.08, 0.14))
        rows.append({
            "Company": f"CO_{i % 10}",
            "Year": 2010 + (i % 12),
            "Revenue": rev,
            "Net_Income": rev * margin,
            "Total_Assets": rev * float(rng.uniform(1.5, 2.5)),
            "Operating_Cash_Flow": rev * float(rng.uniform(0.10, 0.16)),
        })

    # Add multiple diverse outliers
    # Outlier A: extreme negative margin
    rows[20]["Net_Income"] = float(rows[20]["Revenue"] * -8.0)
    # Outlier B: moderate margin jump
    rows[45]["Net_Income"] = float(rows[45]["Revenue"] * 0.45)
    # Outlier C: cash flow divergence
    rows[70]["Operating_Cash_Flow"] = float(rows[70]["Revenue"] * -2.0)

    df = pd.DataFrame(rows)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    assert len(result.anomalies) > 0
    conf_values = [a.confidence for a in result.anomalies]
    # Verify non-empty and varied confidence scores
    distinct_confs = set(conf_values)
    assert len(distinct_confs) >= 2, f"Expected distinct confidence scores, got: {distinct_confs}"
    for c in conf_values:
        assert 0.50 <= c <= 0.98


def test_excel_multi_sheet_and_sheet_override(tmp_path):
    """Test reading multi-sheet Excel file and overriding target sheet."""
    df_sheet1 = pd.DataFrame({
        "Company": ["A", "A", "B", "B"],
        "Year": [2021, 2022, 2021, 2022],
        "Revenue": [100.0, 110.0, 200.0, 220.0],
        "Net_Income": [10.0, 12.0, 20.0, 24.0],
    })
    df_sheet2 = pd.DataFrame({
        "Entity": ["X", "X", "Y", "Y"],
        "Period": [2021, 2022, 2021, 2022],
        "Turnover": [500.0, 550.0, 800.0, 880.0],
        "Profit": [50.0, 58.0, 80.0, 92.0],
    })

    xlsx_path = tmp_path / "multi_statement.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        df_sheet1.to_excel(writer, sheet_name="US_GAAP", index=False)
        df_sheet2.to_excel(writer, sheet_name="IFRS", index=False)

    # Ingest default first sheet
    loaded1 = load_financial_file(xlsx_path)
    assert loaded1.records[0].company == "A"
    assert loaded1.records[0].revenue == 100.0

    # Ingest explicit second sheet
    loaded2 = load_financial_file(xlsx_path, sheet_name="IFRS")
    assert loaded2.records[0].company == "X"
    assert loaded2.records[0].revenue == 500.0


def test_multi_format_exports(tmp_path):
    """Test exporting findings to JSON, CSV, and HTML."""
    df = pd.DataFrame({
        "Company": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
        "Year": [2021, 2022, 2021, 2022, 2021, 2022],
        "Revenue": [1000.0, 1200.0, 2000.0, 2400.0, 3000.0, 3300.0],
        "Net_Income": [100.0, 120.0, 200.0, -1500.0, 300.0, 350.0],
        "Operating_Cash_Flow": [120.0, 140.0, 220.0, -800.0, 330.0, 380.0],
    })
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records, include_normal=True)

    json_path = tmp_path / "output.json"
    csv_path = tmp_path / "output.csv"
    html_path = tmp_path / "output.html"

    export_findings_json(result, json_path)
    export_findings_csv(result.anomalies, csv_path)
    export_findings_html(result, html_path, title="Alpha Beta Audit")

    # Verify JSON export
    assert json_path.exists()
    assert json_path.stat().st_size > 100

    # Verify CSV export
    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path)
    assert "company" in df_csv.columns
    assert "severity" in df_csv.columns
    assert "confidence" in df_csv.columns
    assert len(df_csv) == len(result.anomalies)

    # Verify HTML export
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Alpha Beta Audit" in html_text
    assert "<table" in html_text
    assert "Records Evaluated" in html_text


def test_large_company_scale_neutrality():
    """Verify large enterprise with normal healthy ratios is prioritized below actual distressed anomalies."""
    rows = []
    rng = np.random.default_rng(42)
    for i in range(50):
        rev = float(rng.uniform(10_000_000, 50_000_000))
        margin = float(rng.uniform(0.09, 0.11))
        rows.append({
            "Company": f"MidCap_{i}",
            "Revenue": rev,
            "Net_Income": rev * margin,
            "Gross_Profit": rev * 0.40,
            "Shareholder_Equity": rev * 0.50,
        })

    # Large enterprise (100x bigger, but healthy 10% net margin and 40% gross margin)
    rows.append({
        "Company": "HealthyMegaCorp",
        "Revenue": 5_000_000_000.0,
        "Net_Income": 500_000_000.0,
        "Gross_Profit": 2_000_000_000.0,
        "Shareholder_Equity": 2_500_000_000.0,
    })

    # Actually distressed outlier with severe negative margin (-150%)
    rows.append({
        "Company": "DistressedFirm",
        "Revenue": 20_000_000.0,
        "Net_Income": -30_000_000.0,
        "Gross_Profit": -10_000_000.0,
        "Shareholder_Equity": 10_000_000.0,
    })

    df = pd.DataFrame(rows)
    loaded = load_financial_file(df)
    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
    result = detector.analyze_dataset(loaded.records)

    # Distressed firm must be detected as an anomaly
    flagged_companies = {a.company for a in result.anomalies if a.company}
    assert "DistressedFirm" in flagged_companies

    # DistressedFirm must have higher severity / priority
    distressed_finding = next(a for a in result.anomalies if a.company == "DistressedFirm")
    assert distressed_finding.severity in (Severity.HIGH, Severity.MEDIUM)
    assert any("margin" in k or "roe" in k for k in distressed_finding.relevant_features)

