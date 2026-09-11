"""Configuration and hyperparameters for FinSight AI Anomaly Detection Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ZScoreConfig:
    """Configuration for statistical Z-score baseline and robust deviations."""

    threshold: float = 3.0
    robust_threshold: float = 3.0
    min_std: float = 1e-6
    selected_features: Optional[list[str]] = None


@dataclass
class IsolationForestConfig:
    """Configuration for Scikit-Learn IsolationForest model."""

    n_estimators: int = 100
    random_state: int = 42
    contamination: Union[str, float] = 0.048
    max_samples: Union[str, float, int] = "auto"
    bootstrap: bool = False
    n_jobs: int = 1


@dataclass
class SeverityThresholds:
    """Score thresholds for classifying statistical unusualness severity."""

    # Direct score fallback thresholds if percentiles not applicable
    high_score: float = 0.20
    medium_score: float = 0.10
    low_score: float = 0.05


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive thresholding and multi-signal consensus to prevent overreporting."""

    iqr_multiplier: float = 2.5
    min_anomaly_rate: float = 0.0
    max_anomaly_rate: float = 0.08  # Default upper ceiling for unsupervised anomaly rate (8%)
    confidence_min_threshold: float = 0.50
    require_statistical_confirmation: bool = True
    peer_zscore_threshold: float = 3.0
    temporal_zscore_threshold: float = 3.0


@dataclass
class FeatureConfig:
    """Configuration for feature engineering, discovery, and pruning."""

    clip_growth_bounds: tuple[float, float] = (-10.0, 50.0)  # Safe bounds for YoY growth
    clip_margin_bounds: tuple[float, float] = (-5.0, 5.0)  # Safe bounds for profitability margins
    min_history_required: int = 1
    impute_strategy: str = "median"  # 'median', 'mean', 'zero'
    scaler: str = "robust"
    variance_threshold: float = 1e-8  # Minimum variance required to keep feature
    max_missing_ratio: float = 0.90  # Maximum missing fraction before feature is pruned


@dataclass
class AnomalyConfig:
    """Master configuration for the Anomaly Agent."""

    zscore: ZScoreConfig = field(default_factory=ZScoreConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    severity: SeverityThresholds = field(default_factory=SeverityThresholds)
    adaptive: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model_save_path: str = "models/isolation_forest.joblib"
    random_seed: int = 42
    contamination: Optional[Union[str, float]] = None

    def __post_init__(self) -> None:
        if self.contamination is not None:
            self.isolation_forest.contamination = self.contamination
