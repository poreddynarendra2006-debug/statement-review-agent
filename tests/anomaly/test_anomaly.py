"""End-to-end integration tests for FinSight AI Anomaly Detection Agent."""

from pathlib import Path

import pytest

from finsight.analysis.anomaly_benchmark import run_anomaly_benchmark
from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.core.models import AnomalyFinding, Severity
from finsight.data.loader import generate_reference_dataset, inspect_dataset, load_dataset


def test_dataset_inspection():
    """Test inspection of 23-column dataset reporting all required fields."""
    df = generate_reference_dataset(seed=42)
    summary = inspect_dataset(df)

    assert summary.num_rows >= 160
    assert summary.num_columns == 23
    assert summary.company_identifier == "Company"
    assert summary.year_field == "Year"
    assert len(summary.unique_companies) >= 10
    assert len(summary.years_available) >= 14
    assert "Operating Cash Flow" in summary.numerical_fields
    assert "Debt/Equity Ratio" in summary.numerical_fields
    assert "Current Ratio" in summary.numerical_fields
    assert summary.duplicate_rows == 0


def test_end_to_end_anomaly_agent_workflow(tmp_path: Path):
    """Test full lifecycle: data loading -> feature engineering -> training -> inference -> persistence -> findings."""
    df = generate_reference_dataset(seed=42)
    records = load_dataset(df)
    assert len(records) >= 160

    # 1. Train detector
    detector = AnomalyDetector()
    train_summary = detector.train(records)

    assert train_summary["num_records"] == len(records)
    assert train_summary["num_features"] >= 15
    assert detector.is_trained

    # 2. Analyze records with IsolationForest
    findings = detector.analyze(records)
    assert isinstance(findings, list)

    for finding in findings:
        assert isinstance(finding, AnomalyFinding)
        assert finding.model_name == "IsolationForest"
        assert finding.company != ""
        assert finding.year > 0
        assert isinstance(finding.score, float)
        assert isinstance(finding.severity, Severity)
        assert len(finding.relevant_features) > 0
        assert len(finding.explanation) > 0
        # Verify conservative wording
        assert "fraud" not in finding.explanation.lower()
        assert "manipulation" not in finding.explanation.lower()

    # 3. Analyze records with Z-score baseline
    z_findings = detector.analyze(records, model_type="zscore")
    assert isinstance(z_findings, list)
    for zf in z_findings:
        assert zf.model_name == "ZScoreBaseline"

    # 4. Model persistence and reload
    model_path = tmp_path / "models" / "isolation_forest.joblib"
    detector.save(model_path)
    assert model_path.exists()

    reloaded_detector = AnomalyDetector.load(model_path)
    assert reloaded_detector.is_trained
    assert reloaded_detector.training_feature_names == detector.training_feature_names

    reloaded_findings = reloaded_detector.analyze(records)
    assert len(reloaded_findings) == len(findings)
    if len(findings) > 0:
        assert reloaded_findings[0].company == findings[0].company
        assert reloaded_findings[0].year == findings[0].year
        assert pytest.approx(reloaded_findings[0].score, 1e-4) == findings[0].score


def test_planted_anomaly_benchmark_execution():
    """Test that planted anomaly benchmark runs offline and computes valid precision/recall/F1 metrics."""
    results = run_anomaly_benchmark(seed=42)

    assert results["dataset_size"] >= 160
    assert results["planted_anomalies_count"] > 0

    iforest = results["isolation_forest"]
    assert 0.0 <= iforest["precision"] <= 1.0
    assert 0.0 <= iforest["recall"] <= 1.0
    assert 0.0 <= iforest["f1"] <= 1.0
    assert iforest["true_positives"] + iforest["false_negatives"] == results["planted_anomalies_count"]

    zscore = results["zscore"]
    assert 0.0 <= zscore["precision"] <= 1.0
    assert 0.0 <= zscore["recall"] <= 1.0
    assert 0.0 <= zscore["f1"] <= 1.0
