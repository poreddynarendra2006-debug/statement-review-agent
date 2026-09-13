"""Anomaly Detection models, adaptive strategy selection, and main agent interface for FinSight AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from finsight.analysis.anomaly_scoring import (
    calibrate_continuous_score,
    compute_confidence_scores,
    grade_severities,
)
from finsight.analysis.dataset_profiler import profile_dataset
from finsight.analysis.explanation import build_anomaly_finding
from finsight.analysis.feature_engineering import (
    FeaturePipeline,
    extract_dynamic_features,
)
from finsight.analysis.strategy_selector import select_strategy
from finsight.core.config import (
    AnomalyConfig,
    IsolationForestConfig,
    ZScoreConfig,
)
from finsight.core.models import (
    AnalysisResult,
    AnomalyFinding,
    AnomalySummaryInfo,
    DatasetSummaryInfo,
    FinancialRecord,
    Severity,
)

logger = logging.getLogger(__name__)


class ZScoreBaseline:
    """Statistical and Robust Z-score anomaly baseline using Mean/Std and Median/MAD."""

    def __init__(self, config: Optional[ZScoreConfig] = None):
        self.config = config or ZScoreConfig()
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.medians_: Optional[np.ndarray] = None
        self.mads_: Optional[np.ndarray] = None
        self.iqrs_: Optional[np.ndarray] = None
        self.feature_names_: list[str] = []
        self.fitted_: bool = False

    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> ZScoreBaseline:
        """Fit feature means, stds, medians, and Median Absolute Deviations (MAD)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")

        if len(X_arr) == 0:
            raise ValueError("Cannot fit ZScoreBaseline on empty dataset.")

        if X_arr.shape[1] == 0:
            self.means_ = np.array([])
            self.stds_ = np.array([])
            self.medians_ = np.array([])
            self.mads_ = np.array([])
            self.iqrs_ = np.array([])
            self.feature_names_ = []
            self.fitted_ = True
            return self

        # 1. Standard Mean & Std
        self.means_ = np.nanmean(X_arr, axis=0)
        self.stds_ = np.nanstd(X_arr, axis=0)
        self.stds_ = np.where(self.stds_ < self.config.min_std, 1.0, self.stds_)

        # 2. Robust Median, MAD, and IQR
        self.medians_ = np.nanmedian(X_arr, axis=0)
        abs_diff = np.abs(X_arr - self.medians_)
        raw_mad = np.nanmedian(abs_diff, axis=0)
        self.mads_ = raw_mad * 1.4826  # Normal consistency scaling
        self.mads_ = np.where(self.mads_ < self.config.min_std, self.stds_, self.mads_)
        self.mads_ = np.where(self.mads_ < self.config.min_std, 1.0, self.mads_)

        q75 = np.nanpercentile(X_arr, 75, axis=0)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        raw_iqr = q75 - q25
        self.iqrs_ = np.where(raw_iqr < self.config.min_std, self.mads_, raw_iqr)

        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        self.fitted_ = True
        return self

    def compute_zscores(self, X: np.ndarray) -> np.ndarray:
        """Compute standard Z-scores for all features in X."""
        if not self.fitted_ or self.means_ is None or self.stds_ is None:
            raise RuntimeError("ZScoreBaseline must be fitted before computing z-scores.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] == 0:
            return np.empty((len(X_arr), 0), dtype=np.float64)

        z_matrix = (X_arr - self.means_) / self.stds_
        z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=50.0, neginf=-50.0)
        return z_matrix

    def compute_robust_zscores(self, X: np.ndarray) -> np.ndarray:
        """Compute robust Z-scores using median and MAD for resilience to heavy tails."""
        medians = getattr(self, "medians_", None)
        mads = getattr(self, "mads_", None)
        if not self.fitted_ or medians is None or mads is None:
            if self.means_ is not None and self.stds_ is not None:
                return self.compute_zscores(X)
            raise RuntimeError("ZScoreBaseline must be fitted before computing robust z-scores.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] == 0:
            return np.empty((len(X_arr), 0), dtype=np.float64)

        rz_matrix = (X_arr - medians) / mads
        rz_matrix = np.nan_to_num(rz_matrix, nan=0.0, posinf=50.0, neginf=-50.0)
        return rz_matrix

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels: 1 for normal, -1 for anomaly."""
        rz_scores = self.compute_robust_zscores(X)
        if rz_scores.shape[1] == 0:
            return np.ones(len(X), dtype=int)
        max_abs_rz = np.max(np.abs(rz_scores), axis=1)
        preds = np.where(max_abs_rz > self.config.threshold, -1, 1)
        return preds

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return max absolute robust Z-score as the continuous anomaly score."""
        rz_scores = self.compute_robust_zscores(X)
        if rz_scores.shape[1] == 0:
            return np.zeros(len(X), dtype=float)
        return np.max(np.abs(rz_scores), axis=1)


class IsolationForestModel:
    """Production-grade Isolation Forest anomaly detector wrapper with scalable processing."""

    def __init__(self, config: Optional[IsolationForestConfig] = None):
        self.config = config or IsolationForestConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
        )
        self.feature_names_: list[str] = []
        self.fitted_: bool = False

    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> IsolationForestModel:
        """Fit Isolation Forest on numeric feature matrix."""
        X_arr = np.asarray(X, dtype=np.float64)
        if len(X_arr) == 0:
            raise ValueError("Cannot fit IsolationForest on empty dataset.")

        if X_arr.shape[1] == 0:
            self.feature_names_ = []
            self.fitted_ = True
            return self

        # Handle very large datasets efficiently
        if len(X_arr) > 100_000:
            self.model.max_samples = min(10000, len(X_arr))

        self.model.fit(X_arr)
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: 1 for normal, -1 for anomaly."""
        if not self.fitted_:
            raise RuntimeError("IsolationForestModel must be fitted before calling predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] == 0:
            return np.ones(len(X_arr), dtype=int)
        return self.model.predict(X_arr)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw decision function from sklearn."""
        if not self.fitted_:
            raise RuntimeError("IsolationForestModel must be fitted before calling decision_function.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] == 0:
            return np.zeros(len(X_arr), dtype=float)
        return self.model.decision_function(X_arr)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return ranking anomaly score where higher score = more statistically unusual."""
        raw_df = self.decision_function(X)
        return -raw_df


#: Robust z beyond which a record is reported whatever the contamination cap
#: allows. The ordinary statistical threshold is 3, which on noisy data flags a
#: third of an honest file, so this is set far higher: the "no longer a
#: judgement call" line - 30 deviations is territory like net income at four
#: times revenue, not an unusual-looking year.
#:
#: Measured on all three datasets: the clean dummy file and the Kaggle file
#: both still flag 5.0%, exactly as before, while a 40-row file with 10 blatant
#: anomalies now reports all 10 instead of 2. Lower bars were tried and
#: rejected: at 12 the Kaggle file jumped to 29.8%, because pooling companies of
#: wildly different sizes makes ordinary rows look extreme.
CERTAIN_ROBUST_Z = 30.0


def _impossible_values(record: dict[str, Any]) -> bool:
    """True when a record holds figures no real set of accounts can have.

    Kept whatever the alert budget allows, like CERTAIN_ROBUST_Z. A record can be
    impossible without being statistically extreme: net income above revenue
    looks ordinary beside its peers if the rest of the file is small, so the
    budget used to drop it. These are facts, not judgement calls.

    None of these fires on the clean, defective or Kaggle datasets, so the
    ordinary alert rate is unchanged.
    """
    def number(*names: str) -> Optional[float]:
        for name in names:
            value = record.get(name)
            try:
                if value is not None and value == value:  # value == value drops NaN
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    revenue = number("revenue")
    assets = number("total_assets")
    net_income = number("net_income")
    gross_profit = number("gross_profit")
    liabilities = number("total_liabilities")
    equity = number("shareholder_equity", "total_equity")

    if assets is not None and assets <= 0:
        return True
    if revenue is not None and revenue > 0:
        if net_income is not None and net_income > revenue:
            return True
        if gross_profit is not None and gross_profit > revenue * 1.0001:
            return True
    if assets is not None and assets > 0:
        if liabilities is not None and liabilities > 5 * assets:
            return True
        if equity is not None and equity < -assets:
            return True
    return False


class AnomalyDetector:
    """Production-quality Universal Financial Anomaly Detection Agent."""

    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        model_mode: str = "GENERIC_LOCAL",
    ):
        self.config = config or AnomalyConfig()
        self.pipeline = FeaturePipeline(self.config.features)
        self.zscore_baseline = ZScoreBaseline(self.config.zscore)
        self.isolation_forest = IsolationForestModel(self.config.isolation_forest)
        self.is_trained: bool = False
        self.training_feature_names: list[str] = []
        self.model_mode: str = model_mode

    def _compute_peer_deviations(
        self,
        metadata_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        group_col: str = "category",
        min_group_size: int = 3,
    ) -> Optional[np.ndarray]:
        """Compute peer-relative robust Z-scores when grouping information exists."""
        if feature_df.empty:
            return None

        target_col = None
        if group_col in metadata_df.columns:
            target_col = group_col
        elif "company" in metadata_df.columns:
            target_col = "company"
        else:
            for c in metadata_df.columns:
                if any(k in str(c).lower() for k in ("category", "industry", "sector", "peer")):
                    target_col = c
                    break

        if not target_col:
            return None

        groups = metadata_df[target_col].astype(str)
        valid_groups = groups[groups.ne("None") & groups.ne("nan") & groups.ne("")]
        if len(valid_groups) == 0:
            return None

        group_counts = groups.value_counts()
        usable_groups = set(group_counts[group_counts >= min_group_size].index.tolist())
        if not usable_groups:
            return None

        peer_z = np.zeros(feature_df.shape, dtype=float)
        for g_name in usable_groups:
            mask = (groups == g_name).to_numpy()
            sub_feats = feature_df[mask].to_numpy()
            g_median = np.nanmedian(sub_feats, axis=0)
            g_mad = np.nanmedian(np.abs(sub_feats - g_median), axis=0) * 1.4826
            g_mad = np.where(g_mad < 1e-6, 1.0, g_mad)
            peer_z[mask] = np.nan_to_num((sub_feats - g_median) / g_mad, nan=0.0)

        return peer_z

    def train(
        self,
        records: Sequence[Union[FinancialRecord, dict]],
        model_mode: str = "STANDARD_PRETRAINED",
    ) -> dict[str, Any]:
        """Train the Anomaly Detection pipeline on financial records."""
        if not records or len(records) == 0:
            raise ValueError("Training dataset cannot be empty.")

        self.model_mode = model_mode
        standard_only = (model_mode == "STANDARD_PRETRAINED")

        # 1. Profile & compute financial features
        profile = profile_dataset(records)
        metadata_df, raw_feature_df, active_features, provenance = extract_dynamic_features(
            records, self.config.features, profile=profile, standard_features_only=standard_only
        )

        if raw_feature_df.empty or len(active_features) == 0:
            # If candidate features exist but all had zero variance, train gracefully without raising
            if provenance.get("candidates") and len(provenance["candidates"]) > 0:
                self.training_feature_names = []
                self.is_trained = True
                summary = {
                    "model_mode": self.model_mode,
                    "num_records": len(records),
                    "num_features": 0,
                    "features": [],
                    "anomalies_detected_in_train": 0,
                    "anomaly_rate": 0.0,
                    "warning": "All features have zero variance across records. Zero anomalies detected.",
                }
                logger.info("AnomalyDetector (%s) trained with zero-variance features: %s", self.model_mode, summary)
                return summary
            raise ValueError("No valid financial features could be computed from input records.")

        if len(records) < 3:
            raise ValueError(f"Dataset has too few records ({len(records)} < 3) to train IsolationForest.")

        # 2. Fit preprocessor pipeline
        X = self.pipeline.fit_transform(raw_feature_df)
        self.training_feature_names = self.pipeline.feature_names_

        # 3. Fit Z-score baseline
        self.zscore_baseline.fit(X, feature_names=self.training_feature_names)

        # 4. Fit Isolation Forest
        self.isolation_forest.fit(X, feature_names=self.training_feature_names)
        self.is_trained = True

        train_preds = self.isolation_forest.predict(X)
        num_anomalies = int((train_preds == -1).sum())

        summary = {
            "model_mode": self.model_mode,
            "num_records": len(records),
            "num_features": len(self.training_feature_names),
            "features": self.training_feature_names,
            "anomalies_detected_in_train": num_anomalies,
            "anomaly_rate": round(num_anomalies / len(records), 4),
        }
        logger.info("AnomalyDetector (%s) training complete: %s", self.model_mode, summary)
        return summary

    def analyze(
        self,
        records: Sequence[Union[FinancialRecord, dict]],
        model_type: str = "isolation_forest",
        include_normal: bool = False,
        allow_local_fit: bool = True,
    ) -> list[AnomalyFinding]:
        """Analyze financial records and generate explainable AnomalyFinding objects."""
        res = self.analyze_dataset(
            records=records,
            model_type=model_type,
            include_normal=include_normal,
            allow_local_fit=allow_local_fit,
        )
        return res.anomalies

    def analyze_dataset(
        self,
        records: Sequence[Union[FinancialRecord, dict]],
        model_type: str = "isolation_forest",
        include_normal: bool = False,
        allow_local_fit: bool = True,
    ) -> AnalysisResult:
        """Comprehensive analysis returning structured AnalysisResult with summary and findings."""
        if not records or len(records) == 0:
            return AnalysisResult(
                dataset_summary=DatasetSummaryInfo(
                    records=0,
                    features_available=0,
                    features_used=0,
                    strategy_selected="EmptyDataset",
                    message="The input dataset contains 0 records.",
                ),
                anomaly_summary=AnomalySummaryInfo(),
                anomalies=[],
            )

        num_records = len(records)

        # 1. Profile Dataset
        profile = profile_dataset(records)

        # 2. Select Strategy
        strategy = select_strategy(profile, user_contamination=self.config.isolation_forest.contamination)

        standard_only = (self.model_mode == "STANDARD_PRETRAINED" and self.is_trained)

        # 3. Dynamic Feature Engineering
        metadata_df, raw_feature_df, active_features, provenance = extract_dynamic_features(
            records, self.config.features, profile=profile, standard_features_only=standard_only
        )

        has_temporal = strategy.enable_temporal_analysis
        has_peer = strategy.enable_peer_analysis
        strategy_selected = strategy.strategy_name

        # Handle 0 numerical features or zero-variance edge case
        if len(active_features) == 0 or raw_feature_df.empty:
            is_zero_variance = bool(provenance.get("candidates") and len(provenance["candidates"]) > 0)
            msg = (
                "All financial features have zero variance across records. Zero anomalies detected."
                if is_zero_variance
                else "Insufficient usable numerical financial features for reliable anomaly detection."
            )

            findings: list[AnomalyFinding] = []
            if include_normal and num_records > 0:
                for idx, r in enumerate(records):
                    r_dict = r.to_dict() if isinstance(r, FinancialRecord) else (r if isinstance(r, dict) else dict(r))
                    rec_id = str(r_dict.get("record_id") or r_dict.get("id") or f"Record #{idx+1}")
                    f = build_anomaly_finding(
                        row_idx=idx,
                        raw_record=r_dict,
                        feature_row={},
                        deviations_row={},
                        score=0.0,
                        severity=Severity.LOW,
                        confidence=0.50,
                        model_name="ZeroVarianceBaseline",
                        model_mode=self.model_mode,
                        record_id=rec_id,
                    )
                    findings.append(f)

            return AnalysisResult(
                dataset_summary=DatasetSummaryInfo(
                    records=num_records,
                    raw_columns_count=len(provenance.get("raw_columns", [])),
                    derived_features_count=len(provenance.get("derived", [])),
                    candidate_features_count=len(provenance.get("candidates", [])),
                    features_available=len(provenance.get("available", [])),
                    features_used=0,
                    temporal_analysis=has_temporal,
                    peer_analysis=has_peer,
                    strategy_selected="ZeroVariance" if is_zero_variance else "NoUsableFeatures",
                    raw_column_names=provenance.get("raw_columns", []),
                    available_feature_names=provenance.get("available", []),
                    derived_feature_names=provenance.get("derived", []),
                    candidate_feature_names=provenance.get("candidates", []),
                    used_feature_names=[],
                    ignored_features=provenance.get("ignored", {}),
                    message=msg,
                ),
                anomaly_summary=AnomalySummaryInfo(
                    total_anomalies=0,
                    high=0,
                    medium=0,
                    low=0,
                    anomaly_rate=0.0,
                ),
                anomalies=findings,
            )

        # Handle very small dataset edge case (< 3 records) ONLY if detector is not already trained
        if num_records < 3 and not (self.is_trained and len(self.training_feature_names) > 0):
            msg = f"Dataset has too few records ({num_records} < 3) for reliable anomaly detection."
            return AnalysisResult(
                dataset_summary=DatasetSummaryInfo(
                    records=num_records,
                    raw_columns_count=len(provenance.get("raw_columns", [])),
                    derived_features_count=len(provenance.get("derived", [])),
                    candidate_features_count=len(provenance.get("candidates", [])),
                    features_available=len(provenance.get("available", [])),
                    features_used=len(active_features),
                    temporal_analysis=has_temporal,
                    peer_analysis=has_peer,
                    strategy_selected="InsufficientRecords",
                    raw_column_names=provenance.get("raw_columns", []),
                    available_feature_names=provenance.get("available", []),
                    derived_feature_names=provenance.get("derived", []),
                    candidate_feature_names=provenance.get("candidates", []),
                    used_feature_names=active_features,
                    ignored_features=provenance.get("ignored", {}),
                    message=msg,
                ),
                anomaly_summary=AnomalySummaryInfo(),
                anomalies=[],
            )

        # 4. Model Selection & Local Fitting
        detector_to_use = self
        if not self.is_trained or self.model_mode == "GENERIC_LOCAL":
            if allow_local_fit:
                detector_to_use = AnomalyDetector(config=self.config, model_mode="GENERIC_LOCAL")
                detector_to_use.train(records, model_mode="GENERIC_LOCAL")
            else:
                raise RuntimeError("Detector is not trained and allow_local_fit is False.")
        else:
            detector_to_use = self

        # 5. Transform Features
        X = detector_to_use.pipeline.transform(raw_feature_df)
        features_used = detector_to_use.training_feature_names

        # 6. Multi-Signal Deviations & Continuous Scores
        rz_matrix = detector_to_use.zscore_baseline.compute_robust_zscores(X)

        is_ratio_feat = np.array([
            any(kw in f.lower() for kw in ("margin", "roe", "roa", "roi", "ratio", "growth", "change", "spread", "divergence", "rate", "percent", "%"))
            for f in features_used
        ])
        has_ratio_features = bool(np.sum(is_ratio_feat) >= 2)

        if has_ratio_features and self.model_mode == "GENERIC_LOCAL":
            # Scale-neutral: prioritize financial health ratios and down-weight raw enterprise volume metrics
            weights = np.where(is_ratio_feat, 1.0, 0.25)
            effective_rz_matrix = rz_matrix * weights
        else:
            effective_rz_matrix = rz_matrix

        max_abs_rz = np.max(np.abs(effective_rz_matrix), axis=1) if effective_rz_matrix.shape[1] > 0 else np.zeros(num_records)

        # Peer analysis
        peer_z_matrix = self._compute_peer_deviations(metadata_df, raw_feature_df) if has_peer else None
        max_peer_z = np.max(np.abs(peer_z_matrix), axis=1) if peer_z_matrix is not None and peer_z_matrix.shape[1] > 0 else None

        # Isolation Forest Scores
        if_scores = None
        if_preds = None
        enable_if = (strategy.enable_isolation_forest or (self.is_trained and self.model_mode != "GENERIC_LOCAL")) and detector_to_use.isolation_forest.fitted_
        if enable_if:
            if_scores = detector_to_use.isolation_forest.anomaly_score(X)
            if_preds = detector_to_use.isolation_forest.predict(X)

        # Calibrate Continuous Anomaly Score in [0.0, 1.0]
        calibrated_scores = calibrate_continuous_score(
            robust_z_scores=max_abs_rz,
            if_scores=if_scores,
            peer_z_scores=max_peer_z,
        )

        # 7. Anomaly Flagging Decision
        contam = float(self.config.isolation_forest.contamination) if isinstance(self.config.isolation_forest.contamination, (int, float)) else 0.048
        target_flag_count = max(1, int(np.ceil(num_records * contam)))
        stat_flags = (max_abs_rz >= self.config.zscore.threshold)

        if if_preds is not None:
            raw_if_flags = (if_preds == -1)
            score_cutoff = np.percentile(calibrated_scores, max(0.0, 100.0 * (1.0 - contam)))
            is_anomaly_mask = raw_if_flags | (calibrated_scores >= score_cutoff) | stat_flags
        else:
            rz_threshold = self.config.zscore.threshold
            score_cutoff = np.percentile(calibrated_scores, max(0.0, 100.0 * (1.0 - contam)))
            is_anomaly_mask = ((calibrated_scores >= score_cutoff) & (max_abs_rz >= rz_threshold)) | stat_flags
            if is_anomaly_mask.sum() == 0 and num_records >= 10:
                is_anomaly_mask = (calibrated_scores >= score_cutoff)

        # Cap how much the model may flag, so an ordinary file is not filled
        # with borderline findings.
        #
        # The exception is a record so far from the rest that it is not a
        # judgement call - CERTAIN_ROBUST_Z deviations, against the ordinary
        # z-test's 3. Those are kept whatever the budget allows. Without this,
        # the cap silently dropped real findings: a file with ten obvious
        # anomalies in forty rows reported two, because two was 4.8% of forty.
        #
        # Records holding impossible figures are kept too. Being far from the
        # rest only catches what is extreme; a file with ten planted problems in
        # fifty-five rows still reported three, because net income above
        # revenue or negative assets need not be statistically extreme at all.
        raw_dicts: list[dict[str, Any]] = []
        for r in records:
            if isinstance(r, FinancialRecord):
                raw_dicts.append(r.to_dict())
            elif isinstance(r, dict):
                raw_dicts.append(r)
            else:
                raw_dicts.append(dict(r))
        impossible = np.array([_impossible_values(d) for d in raw_dicts], dtype=bool)
        if impossible.shape[0] != num_records:
            impossible = np.zeros(num_records, dtype=bool)

        certain = (max_abs_rz >= CERTAIN_ROBUST_Z) | impossible
        if is_anomaly_mask.sum() > target_flag_count and num_records >= 20:
            budget = max(0, target_flag_count - int(certain.sum()))
            discretionary = np.where(is_anomaly_mask & ~certain)[0]
            kept = np.zeros(num_records, dtype=bool)
            if budget > 0 and discretionary.size:
                strongest = discretionary[np.argsort(calibrated_scores[discretionary])[-budget:]]
                kept[strongest] = True
            is_anomaly_mask = certain | kept
        else:
            # Under budget the cap is not applied, but an impossible record may
            # not have scored high enough to be flagged at all - include it.
            is_anomaly_mask = is_anomaly_mask | impossible

        # Compute Confidence Scores
        yoy_col_indices = [i for i, c in enumerate(features_used) if c.startswith("yoy_") or c.endswith("_growth") or c.endswith("_change")]
        temporal_z = np.max(np.abs(rz_matrix[:, yoy_col_indices]), axis=1) if yoy_col_indices and rz_matrix.shape[1] > 0 else None
        num_contrib = np.sum(np.abs(rz_matrix) >= 2.5, axis=1) if rz_matrix.shape[1] > 0 else np.zeros(num_records)

        confidences = compute_confidence_scores(
            robust_z_scores=max_abs_rz,
            if_flags=if_preds,
            rz_flags=np.where(max_abs_rz >= self.config.zscore.threshold, -1, 1),
            num_contributing_features=num_contrib,
            temporal_z_scores=temporal_z,
            peer_z_scores=max_peer_z,
            dataset_size=num_records,
            anomaly_scores=calibrated_scores,
        )

        # 8. Build Anomaly Findings

        # Typical values and normal ranges in each feature's own unit. The z-score
        # baseline is fitted on scaled model inputs, so its medians can't be shown to reviewers.
        feature_medians: dict[str, float] = {}
        feature_iqrs: dict[str, float] = {}
        pipe = detector_to_use.pipeline
        for f_name in features_used:
            if f_name in pipe.scale_medians_:
                feature_medians[f_name] = float(pipe.scale_medians_[f_name])
                feature_iqrs[f_name] = float(pipe.scale_iqrs_.get(f_name, 1.0))

        findings: list[AnomalyFinding] = []
        for idx in range(num_records):
            if not is_anomaly_mask[idx] and not include_normal:
                continue

            r_dict = raw_dicts[idx] if idx < len(raw_dicts) else {}
            feat_vals = {f: float(raw_feature_df.iloc[idx].get(f, 0.0)) for f in features_used}
            feat_devs = {f: float(rz_matrix[idx, f_i]) for f_i, f in enumerate(features_used)}

            is_peer_dev = bool(max_peer_z is not None and max_peer_z[idx] >= 2.5)

            active_model_name = "IsolationForest" if (enable_if and model_type != "zscore") else ("ZScoreBaseline" if model_type == "zscore" else strategy.primary_model)
            finding = build_anomaly_finding(
                row_idx=idx,
                raw_record=r_dict,
                feature_row=feat_vals,
                deviations_row=feat_devs,
                score=float(calibrated_scores[idx]),
                severity=Severity.LOW,  # Assigned after ranking below
                confidence=float(confidences[idx]),
                model_name=active_model_name,
                model_mode=detector_to_use.model_mode,
                is_peer_deviation=is_peer_dev,
                feature_medians=feature_medians,
                feature_iqrs=feature_iqrs,
            )
            findings.append(finding)

        # Sort findings strictly by anomaly score descending
        findings.sort(key=lambda f: f.score, reverse=True)

        # Grade finding severities by anomaly score using unified rule from anomaly_scoring
        n_findings = len(findings)
        if n_findings > 0:
            finding_scores = np.array([f.score for f in findings])
            assigned_severities = grade_severities(finding_scores)
            for rank, (f, sev) in enumerate(zip(findings, assigned_severities)):
                f.severity = sev
                f.percentile = 1.0 if n_findings == 1 else round(float(1.0 - (rank / n_findings)), 4)

        # 9. Summary breakdown
        high_cnt = sum(1 for f in findings if f.severity == Severity.HIGH or f.severity.value == "HIGH")
        med_cnt = sum(1 for f in findings if f.severity == Severity.MEDIUM or f.severity.value == "MEDIUM")
        low_cnt = sum(1 for f in findings if f.severity == Severity.LOW or f.severity.value == "LOW")

        anomaly_summary = AnomalySummaryInfo(
            total_anomalies=len(findings),
            high=high_cnt,
            medium=med_cnt,
            low=low_cnt,
            anomaly_rate=len(findings) / num_records if num_records > 0 else 0.0,
        )

        msg = None
        if not findings:
            if not has_temporal and not has_peer:
                msg = strategy.temporal_status_message
            else:
                msg = "No statistically unusual financial anomalies identified above review threshold."

        dataset_summary = DatasetSummaryInfo(
            records=num_records,
            raw_columns_count=len(provenance.get("raw_columns", [])),
            derived_features_count=len(provenance.get("derived", [])),
            candidate_features_count=len(provenance.get("candidates", [])),
            features_available=len(provenance.get("available", [])),
            features_used=len(features_used),
            temporal_analysis=has_temporal,
            peer_analysis=has_peer,
            strategy_selected=strategy_selected,
            raw_column_names=provenance.get("raw_columns", []),
            available_feature_names=provenance.get("available", []),
            derived_feature_names=provenance.get("derived", []),
            candidate_feature_names=provenance.get("candidates", []),
            used_feature_names=features_used,
            ignored_features=provenance.get("ignored", {}),
            message=msg,
        )

        return AnalysisResult(
            dataset_summary=dataset_summary,
            anomaly_summary=anomaly_summary,
            anomalies=findings,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the trained detector, feature preprocessor, and configuration metadata."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained AnomalyDetector.")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config,
            "pipeline": self.pipeline,
            "zscore_baseline": self.zscore_baseline,
            "isolation_forest": self.isolation_forest,
            "training_feature_names": self.training_feature_names,
            "model_mode": self.model_mode,
            "is_trained": self.is_trained,
        }
        joblib.dump(payload, save_path)
        logger.info("Saved AnomalyDetector (%s) to: %s", self.model_mode, save_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> AnomalyDetector:
        """Load a persisted AnomalyDetector from disk."""
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        payload = joblib.load(load_path)
        detector = cls(config=payload["config"])
        detector.pipeline = payload["pipeline"]
        detector.zscore_baseline = payload["zscore_baseline"]
        detector.isolation_forest = payload["isolation_forest"]
        detector.training_feature_names = payload["training_feature_names"]
        detector.model_mode = payload.get("model_mode", "STANDARD_PRETRAINED")
        detector.is_trained = payload["is_trained"]
        logger.info("Loaded AnomalyDetector (%s) with %d features from: %s", detector.model_mode, len(detector.training_feature_names), load_path)
        return detector


def train_model(
    records: Sequence[Union[FinancialRecord, dict]],
    config: Optional[AnomalyConfig] = None,
    model_mode: str = "STANDARD_PRETRAINED",
) -> AnomalyDetector:
    """Functional interface to train an AnomalyDetector."""
    detector = AnomalyDetector(config=config, model_mode=model_mode)
    detector.train(records, model_mode=model_mode)
    return detector


def detect_anomalies(
    records: Sequence[Union[FinancialRecord, dict]],
    detector: AnomalyDetector,
    model_type: str = "isolation_forest",
) -> list[AnomalyFinding]:
    """Functional interface to detect anomalies using a trained detector."""
    return detector.analyze(records, model_type=model_type)


def save_model(detector: AnomalyDetector, filepath: Union[str, Path]) -> None:
    """Functional interface to save a detector."""
    detector.save(filepath)


def load_model(filepath: Union[str, Path]) -> AnomalyDetector:
    """Functional interface to load a detector."""
    return AnomalyDetector.load(filepath)
