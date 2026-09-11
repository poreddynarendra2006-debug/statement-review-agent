"""Unit tests for Z-Score statistical baseline."""

import numpy as np
import pytest

from finsight.analysis.anomaly_detection import ZScoreBaseline
from finsight.core.config import ZScoreConfig


def test_zscore_fit_and_predict_normal():
    """Test standard Z-score calculation on normally distributed data."""
    rng = np.random.default_rng(42)
    # Generate 100 normal samples with mean=10, std=2
    X_train = rng.normal(10.0, 2.0, size=(100, 3))

    zscore = ZScoreBaseline(ZScoreConfig(threshold=3.0))
    zscore.fit(X_train)

    assert zscore.fitted_
    assert zscore.means_ is not None
    assert zscore.stds_ is not None
    assert pytest.approx(zscore.means_[0], abs=0.5) == 10.0

    # Normal test point (mean value)
    X_normal = np.array([[10.0, 10.0, 10.0]])
    preds = zscore.predict(X_normal)
    assert preds[0] == 1  # 1 = normal

    # Extreme test point (6 standard deviations away)
    X_extreme = np.array([[30.0, 10.0, 10.0]])
    preds_extreme = zscore.predict(X_extreme)
    assert preds_extreme[0] == -1  # -1 = anomaly


def test_zscore_zero_std_safeguard():
    """Test that zero standard deviation is handled safely without division by zero or errors."""
    # Constant matrix
    X_constant = np.ones((50, 4)) * 5.0

    zscore = ZScoreBaseline(ZScoreConfig(threshold=3.0))
    zscore.fit(X_constant)

    z_scores = zscore.compute_zscores(X_constant)
    assert not np.isnan(z_scores).any()
    assert not np.isinf(z_scores).any()
    assert np.all(z_scores == 0.0)

    preds = zscore.predict(X_constant)
    assert np.all(preds == 1)  # All normal


def test_zscore_configurable_threshold():
    """Test that adjusting Z_SCORE_THRESHOLD alters anomaly sensitivity."""
    X_train = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ])
    X_test = np.array([[8.0, 8.0]])  # ~3.5 stds away

    # High threshold (5.0) -> Not an anomaly
    z_strict = ZScoreBaseline(ZScoreConfig(threshold=5.0)).fit(X_train)
    assert z_strict.predict(X_test)[0] == 1

    # Normal threshold (2.0) -> Anomaly
    z_sensitive = ZScoreBaseline(ZScoreConfig(threshold=2.0)).fit(X_train)
    assert z_sensitive.predict(X_test)[0] == -1
