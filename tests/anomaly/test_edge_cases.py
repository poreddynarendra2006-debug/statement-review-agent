"""Unit tests for edge cases: empty datasets, single record, identical records, NaNs, and Infs."""

import numpy as np
import pytest

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.anomaly_features import compute_financial_features
from finsight.core.models import FinancialRecord
from finsight.data.loader import load_dataset


def test_empty_dataset_handling():
    """Test behavior when given an empty sequence of records."""
    meta, feat_df, active = compute_financial_features([])
    assert feat_df.empty
    assert len(active) == 0

    records = load_dataset([])
    assert records == []

    detector = AnomalyDetector()
    with pytest.raises(ValueError, match="empty"):
        detector.train([])


def test_single_record():
    """Test behavior with only one record (no historical comparison possible)."""
    record = FinancialRecord(
        company="SOLO_CORP",
        year=2023,
        revenue=1000.0,
        gross_profit=400.0,
        net_income=100.0,
        ebitda=200.0,
        shareholder_equity=500.0,
    )

    meta, feat_df, active = compute_financial_features([record])
    assert len(feat_df) == 1
    # YoY growth should be NaN for a single record without prior history
    assert np.isnan(feat_df["revenue_growth"].iloc[0])
    # Static ratios should still compute correctly
    assert pytest.approx(feat_df["gross_margin"].iloc[0], 1e-4) == 0.40


def test_identical_records():
    """Test behavior when all records have identical financial figures."""
    records = [
        FinancialRecord(
            company="CLONE_CORP",
            year=2015 + i,
            revenue=1000.0,
            gross_profit=500.0,
            net_income=100.0,
            ebitda=200.0,
            shareholder_equity=500.0,
        )
        for i in range(10)
    ]

    detector = AnomalyDetector()
    summary = detector.train(records)
    assert summary["num_records"] == 10

    findings = detector.analyze(records)
    # Identical records should have identical scores and zero variance
    scores = [f.score for f in detector.analyze(records, include_normal=True)]
    assert len(set(scores)) <= 2  # At most 1 or 2 distinct score levels due to first year vs subsequent


def test_extreme_nan_and_inf_handling():
    """Test that records with zero revenue, negative equity, and NaNs do not cause crash."""
    records = [
        FinancialRecord(
            company="EDGE_CORP",
            year=2020,
            revenue=0.0,  # Zero revenue
            gross_profit=0.0,
            net_income=-500.0,
            ebitda=None,
            shareholder_equity=-200.0,  # Negative equity
        ),
        FinancialRecord(
            company="EDGE_CORP",
            year=2021,
            revenue=1000.0,
            gross_profit=500.0,
            net_income=100.0,
            ebitda=200.0,
            shareholder_equity=500.0,
        ),
    ]

    meta, feat_df, active = compute_financial_features(records)
    assert not feat_df.empty

    detector = AnomalyDetector()
    # Adding extra reference records for minimum training set
    from finsight.data.loader import generate_reference_dataset
    ref_records = load_dataset(generate_reference_dataset(seed=42))
    detector.train(ref_records)

    findings = detector.analyze(records)
    assert len(findings) == 2
    for f in findings:
        assert isinstance(f.score, float)
        assert not np.isnan(f.score)
