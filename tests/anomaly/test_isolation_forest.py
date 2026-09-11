"""Unit tests for IsolationForest model, anomaly scoring, and model persistence."""

from pathlib import Path

import numpy as np
import pytest

from finsight.analysis.anomaly_detection import (
    AnomalyDetector,
    IsolationForestModel,
    load_model,
    save_model,
)
from finsight.core.config import IsolationForestConfig
from finsight.data.loader import generate_reference_dataset, load_dataset


def test_isolation_forest_fit_and_predict():
    """Test IsolationForest training and deterministic prediction shape/values."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0.0, 1.0, size=(100, 5))

    model = IsolationForestModel(IsolationForestConfig(n_estimators=100, random_state=42))
    model.fit(X_train)

    assert model.fitted_
    preds = model.predict(X_train)
    assert len(preds) == 100
    assert set(preds).issubset({1, -1})

    # Scores
    scores = model.anomaly_score(X_train)
    assert len(scores) == 100
    assert isinstance(scores, np.ndarray)


def test_isolation_forest_score_direction_and_interpretation():
    """Test that anomaly_score is -decision_function, higher for outliers, and not a probability."""
    rng = np.random.default_rng(42)
    # 100 tight normal points around origin
    X_normal = rng.normal(0.0, 0.2, size=(100, 3))

    model = IsolationForestModel(IsolationForestConfig(n_estimators=100, random_state=42))
    model.fit(X_normal)

    # Inlier point at center
    inlier = np.array([[0.0, 0.0, 0.0]])
    # Extreme outlier point far away
    outlier = np.array([[15.0, -20.0, 30.0]])

    df_inlier = model.decision_function(inlier)[0]
    df_outlier = model.decision_function(outlier)[0]

    score_inlier = model.anomaly_score(inlier)[0]
    score_outlier = model.anomaly_score(outlier)[0]

    # Verify negation relationship: Score = - decision_function
    assert pytest.approx(score_inlier, 1e-6) == -df_inlier
    assert pytest.approx(score_outlier, 1e-6) == -df_outlier

    # In scikit-learn: lower decision_function = more anomalous
    assert df_outlier < df_inlier
    # In our transformed score: higher anomaly_score = more anomalous
    assert score_outlier > score_inlier


def test_no_data_leakage_in_ml_features():
    """Test that metadata, identifiers, and non-financial fields NEVER enter ML features."""
    df_sample = generate_reference_dataset(seed=42)
    records = load_dataset(df_sample)

    detector = AnomalyDetector()
    detector.train(records)

    prohibited_substrings = [
        "company", "year", "category", "market_cap", "employees", "inflation",
        "ground_truth", "label", "target", "id", "index", "planted"
    ]

    for feat in detector.training_feature_names:
        for prob in prohibited_substrings:
            assert prob != feat.lower(), f"Data leakage detected! '{feat}' in ML feature list."

    # Verify transformed matrix is numeric 2D float without non-feature columns
    _, feature_df, _ = detector.pipeline.transform, detector.training_feature_names, None
    X = detector.pipeline.transform(df_sample)
    assert X.shape[1] == len(detector.training_feature_names)
    assert X.dtype == np.float64


def test_isolation_forest_determinism():
    """Test that model output is strictly deterministic given fixed random_state."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(50, 4))

    m1 = IsolationForestModel(IsolationForestConfig(n_estimators=100, random_state=42)).fit(X)
    m2 = IsolationForestModel(IsolationForestConfig(n_estimators=100, random_state=42)).fit(X)

    preds1 = m1.predict(X)
    preds2 = m2.predict(X)
    np.testing.assert_array_equal(preds1, preds2)

    scores1 = m1.anomaly_score(X)
    scores2 = m2.anomaly_score(X)
    np.testing.assert_allclose(scores1, scores2)


def test_detector_save_and_load(tmp_path: Path):
    """Test full model and metadata persistence roundtrip with joblib."""
    df_sample = generate_reference_dataset(seed=42)
    records = load_dataset(df_sample)

    detector = AnomalyDetector()
    detector.train(records)

    # Save
    save_file = tmp_path / "models" / "isolation_forest.joblib"
    save_model(detector, save_file)
    assert save_file.exists()

    # Load
    loaded_detector = load_model(save_file)
    assert loaded_detector.is_trained
    assert loaded_detector.training_feature_names == detector.training_feature_names

    # Verify identical findings on test data
    orig_findings = detector.analyze(records[:10], include_normal=True)
    loaded_findings = loaded_detector.analyze(records[:10], include_normal=True)

    assert len(orig_findings) == len(loaded_findings)
    for f_orig, f_load in zip(orig_findings, loaded_findings):
        assert f_orig.company == f_load.company
        assert f_orig.year == f_load.year
        assert pytest.approx(f_orig.score, 1e-4) == f_load.score
        assert f_orig.anomaly_type == f_load.anomaly_type
        assert f_orig.severity == f_load.severity
