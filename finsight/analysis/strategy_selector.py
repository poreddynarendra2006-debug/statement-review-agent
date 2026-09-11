"""Adaptive Strategy Selection layer for FinSight AI Anomaly Detection Agent.

Determines the optimal anomaly detection algorithms, statistical baselines,
temporal models, and cohort peer comparisons based on dataset scale, structure,
and dimensional characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from finsight.core.models import DatasetProfile


@dataclass
class StrategyDecision:
    """Strategy decision configuration selected for dataset evaluation."""

    strategy_name: str
    description: str
    primary_model: str  # "RobustStatistical", "EnsembleStatisticalML", "ScalableIsolationForest"
    enable_isolation_forest: bool
    enable_robust_zscore: bool
    enable_multivariate_distance: bool
    enable_temporal_analysis: bool
    temporal_status_message: str
    enable_peer_analysis: bool
    peer_status_message: str
    recommended_contamination: float = 0.048
    n_estimators: int = 100
    batch_size: Optional[int] = None


def select_strategy(profile: DatasetProfile, user_contamination: Optional[float] = None) -> StrategyDecision:
    """Select the optimal anomaly detection strategy based on dataset profiling.

    Parameters
    ----------
    profile : DatasetProfile
        Dataset profile describing dimensions, entities, temporal series, and scale.
    user_contamination : float, optional
        User-specified contamination threshold if provided.

    Returns
    -------
    StrategyDecision
        Selected strategy configuration.
    """
    contam = user_contamination if user_contamination is not None else 0.048

    # 1. Dataset Scale Algorithm Selection
    if profile.num_rows < 3:
        return StrategyDecision(
            strategy_name="InsufficientRecords",
            description="Dataset contains fewer than 3 records. Anomaly detection is not statistically meaningful.",
            primary_model="None",
            enable_isolation_forest=False,
            enable_robust_zscore=False,
            enable_multivariate_distance=False,
            enable_temporal_analysis=False,
            temporal_status_message=profile.temporal_reason or "Insufficient records for temporal analysis.",
            enable_peer_analysis=False,
            peer_status_message=profile.peer_reason or "Insufficient records for peer analysis.",
            recommended_contamination=contam,
        )

    if profile.num_rows < 50:
        # Small dataset: Robust statistics (Median, MAD, IQR, Robust Z-score)
        strategy_name = "RobustStatistical"
        description = "Small dataset (<50 rows): Using robust non-parametric statistics (Median, MAD, IQR) to prevent overfitting."
        primary_model = "RobustStatistical"
        enable_if = False
        enable_rz = True
        enable_multi = True
        n_est = 50
        batch_sz = None
    elif profile.num_rows <= 1000:
        # Medium dataset: Ensemble of Robust Stats + Isolation Forest
        strategy_name = "EnsembleStatisticalML"
        description = "Medium dataset (50-1,000 rows): Combining robust non-parametric statistics with Isolation Forest ensemble."
        primary_model = "EnsembleStatisticalML"
        enable_if = True
        enable_rz = True
        enable_multi = True
        n_est = 100
        batch_sz = None
    else:
        # Large dataset: Scalable Isolation Forest + Fast Matrix Vectorization
        strategy_name = "ScalableIsolationForest"
        description = f"Large dataset ({profile.num_rows:,} rows): Using vectorized matrix operations and scalable Isolation Forest."
        primary_model = "ScalableIsolationForest"
        enable_if = True
        enable_rz = True
        enable_multi = False  # Avoid expensive O(N^2) pairwise calculations
        n_est = 100
        batch_sz = 10000

    # 2. Temporal Analysis Determination
    temporal_enabled = profile.has_temporal_ordering
    temporal_msg = profile.temporal_reason if profile.temporal_reason else (
        "Temporal analysis enabled." if temporal_enabled else "Temporal analysis unavailable: only one period is present."
    )

    # 3. Peer Analysis Determination
    peer_enabled = profile.has_peer_grouping
    peer_msg = profile.peer_reason if profile.peer_reason else (
        "Peer analysis enabled." if peer_enabled else "Peer Analysis: NO (Dataset-Wide Only)."
    )

    return StrategyDecision(
        strategy_name=strategy_name,
        description=description,
        primary_model=primary_model,
        enable_isolation_forest=enable_if,
        enable_robust_zscore=enable_rz,
        enable_multivariate_distance=enable_multi,
        enable_temporal_analysis=temporal_enabled,
        temporal_status_message=temporal_msg,
        enable_peer_analysis=peer_enabled,
        peer_status_message=peer_msg,
        recommended_contamination=contam,
        n_estimators=n_est,
        batch_size=batch_sz,
    )
