"""Calibrated Anomaly Scoring, Confidence Estimation, and Severity Classification.

Provides calibrated continuous scores in [0.0, 1.0], multi-signal confidence estimation,
percentile ranking, and empirical severity grading for financial anomalies.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from finsight.core.models import Severity

logger = logging.getLogger(__name__)


def calibrate_continuous_score(
    robust_z_scores: np.ndarray,
    if_scores: Optional[np.ndarray] = None,
    peer_z_scores: Optional[np.ndarray] = None,
    temporal_z_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute calibrated continuous anomaly scores in [0.0, 1.0].

    Parameters
    ----------
    robust_z_scores : np.ndarray
        Max absolute robust Z-scores per observation.
    if_scores : np.ndarray, optional
        Isolation forest continuous raw scores.
    peer_z_scores : np.ndarray, optional
        Peer deviation max absolute Z-scores.
    temporal_z_scores : np.ndarray, optional
        Temporal YoY deviation Z-scores.

    Returns
    -------
    np.ndarray
        Calibrated score array where higher values represent higher anomaly probability.
    """
    n_records = len(robust_z_scores)
    if n_records == 0:
        return np.array([], dtype=float)

    # 1. Calibrate Robust Z-scores via tanh scaling: tanh(z / 4.0) maps [0, inf) -> [0, 1)
    rz_calib = np.tanh(np.maximum(0.0, robust_z_scores) / 3.5)

    # 2. Calibrate Isolation Forest scores if present
    if if_scores is not None and len(if_scores) == n_records:
        if_min = np.min(if_scores)
        if_max = np.max(if_scores)
        if if_max > if_min:
            if_calib = (if_scores - if_min) / (if_max - if_min)
        else:
            if_calib = np.full(n_records, 0.5)
        # Combined score: weighted blend
        combined = 0.55 * if_calib + 0.45 * rz_calib
    else:
        combined = rz_calib

    # 3. Boost with Peer or Temporal shock signals if present
    if peer_z_scores is not None and len(peer_z_scores) == n_records:
        peer_boost = np.tanh(np.maximum(0.0, peer_z_scores) / 5.0) * 0.15
        combined = np.clip(combined + peer_boost, 0.0, 1.0)

    if temporal_z_scores is not None and len(temporal_z_scores) == n_records:
        temp_boost = np.tanh(np.maximum(0.0, temporal_z_scores) / 5.0) * 0.15
        combined = np.clip(combined + temp_boost, 0.0, 1.0)

    return np.clip(combined, 0.0, 1.0)


def compute_confidence_scores(
    robust_z_scores: np.ndarray,
    if_flags: Optional[np.ndarray] = None,
    rz_flags: Optional[np.ndarray] = None,
    num_contributing_features: Optional[np.ndarray] = None,
    temporal_z_scores: Optional[np.ndarray] = None,
    peer_z_scores: Optional[np.ndarray] = None,
    dataset_size: Optional[int] = None,
    anomaly_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute continuous, evidence-grounded confidence scores in [0.50, 0.98]."""
    n_records = len(robust_z_scores)
    if n_records == 0:
        return np.array([], dtype=float)

    # Base confidence
    conf = np.full(n_records, 0.55, dtype=float)

    # 1. Anomaly score alignment
    if anomaly_scores is not None:
        scores_clean = np.clip(np.nan_to_num(anomaly_scores, nan=0.5), 0.0, 1.0)
        conf += scores_clean * 0.15

    # 2. Statistical strength: continuous scaling with deviation magnitude
    abs_rz = np.abs(np.nan_to_num(robust_z_scores, nan=0.0))
    stat_strength = (np.clip(abs_rz, 0.0, 30.0) / 30.0) * 0.12
    conf += stat_strength

    # 3. Algorithm agreement: +0.06 when both Isolation Forest and Robust Z-score flag
    if if_flags is not None and rz_flags is not None:
        both_flagged = (if_flags == -1) & (rz_flags == -1)
        conf = np.where(both_flagged, conf + 0.06, conf)

    # 4. Temporal evidence: +0.04 when year-over-year shift diverges significantly
    if temporal_z_scores is not None:
        temp_abs = np.abs(np.nan_to_num(temporal_z_scores, nan=0.0))
        conf = np.where(temp_abs >= 2.5, conf + 0.04, conf)

    # 5. Peer divergence: +0.04 when cohort peer deviations are abnormal
    if peer_z_scores is not None:
        peer_abs = np.abs(np.nan_to_num(peer_z_scores, nan=0.0))
        conf = np.where(peer_abs >= 2.5, conf + 0.04, conf)

    # 6. Multi-feature breadth: +0.02 per additional abnormal contributing feature (up to +0.06)
    if num_contributing_features is not None:
        feat_counts = np.maximum(0, np.nan_to_num(num_contributing_features, nan=0.0) - 1)
        breadth_boost = np.clip(feat_counts * 0.02, 0.0, 0.06)
        conf += breadth_boost

    # Extreme outlier boost (very clear anomalous signal)
    extreme_outlier = abs_rz >= 8.0
    conf = np.where(extreme_outlier, np.maximum(conf, 0.88), conf)

    # 7. Small dataset safeguard: capped at <= 0.85 when dataset < 50 rows
    eff_size = dataset_size if dataset_size is not None else n_records
    if eff_size < 50:
        conf = np.minimum(conf, 0.85)

    return np.round(np.clip(conf, 0.50, 0.98), 2)


def grade_severities(
    scores: np.ndarray,
    percentiles: Optional[np.ndarray] = None,
    is_anomaly: Optional[np.ndarray] = None,
) -> list[Severity]:
    """Grade anomaly severity based on score percentiles and deviation evidence.

    Distribution:
    - HIGH: Top 10% (percentile >= 90.0)
    - MEDIUM: Next 20% (70.0 <= percentile < 90.0)
    - LOW: Remainder (percentile < 70.0)
    """
    n_records = len(scores)
    if n_records == 0:
        return []

    if percentiles is not None:
        pcts = percentiles
        severities: list[Severity] = []
        for i in range(n_records):
            pct = pcts[i]
            if pct >= 90.0:
                severities.append(Severity.HIGH)
            elif pct >= 70.0:
                severities.append(Severity.MEDIUM)
            else:
                severities.append(Severity.LOW)
        return severities

    if n_records == 1:
        return [Severity.HIGH]
    if n_records == 2:
        return [Severity.HIGH, Severity.MEDIUM] if scores[0] >= scores[1] else [Severity.MEDIUM, Severity.HIGH]

    scores_arr = np.asarray(scores, dtype=float)
    order = np.argsort(-scores_arr)
    high_cutoff = max(1, round(0.10 * n_records))
    med_cutoff = max(high_cutoff + 1, round(0.30 * n_records))

    severities = [Severity.LOW] * n_records
    for rank, orig_idx in enumerate(order):
        if rank < high_cutoff:
            severities[orig_idx] = Severity.HIGH
        elif rank < med_cutoff:
            severities[orig_idx] = Severity.MEDIUM
        else:
            severities[orig_idx] = Severity.LOW

    return severities
