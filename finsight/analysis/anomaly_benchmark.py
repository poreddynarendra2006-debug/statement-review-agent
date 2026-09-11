"""Planted Anomaly Benchmark Suite for FinSight AI Anomaly Detection Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.anomaly_features import compute_financial_features
from finsight.core.config import AnomalyConfig
from finsight.core.models import FinancialRecord
from finsight.data.loader import generate_reference_dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Evaluation metrics for an anomaly detection model."""

    model_name: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    total_anomalies: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_samples": self.total_samples,
            "total_anomalies": self.total_anomalies,
        }


def inject_planted_anomalies(
    clean_records: list[FinancialRecord],
    anomaly_fraction: float = 0.08,
    seed: int = 42,
) -> tuple[list[FinancialRecord], np.ndarray, list[dict[str, Any]]]:
    """Inject controlled, realistic financial anomalies across all statement dimensions.

    Types of planted anomalies:
    1. Revenue Explosion (+400% top-line surge with flat bottom-line and flat cash flows)
    2. Gross Margin Collapse (Gross Profit collapses to 4% of revenue)
    3. Cash Flow vs Accrual Earnings Divergence (Net income surges 3x while Operating Cash Flow turns negative)
    4. Debt/Equity & Leverage Explosion (Debt/Equity surges 5x, interest burden spikes)
    5. Profitability & ROE Shock (Net loss erasing 80% of shareholder equity, ROE collapses)

    Returns:
        1. benchmark_records: list of FinancialRecord instances with injected anomalies
        2. ground_truth_labels: binary array (0 = normal, 1 = anomaly)
        3. anomaly_log: list of dicts detailing what was injected where
    """
    rng = np.random.default_rng(seed)
    records = [
        FinancialRecord.from_dict(r.to_dict()) for r in clean_records
    ]

    min_year_by_comp = {}
    for r in records:
        if r.company not in min_year_by_comp or r.year < min_year_by_comp[r.company]:
            min_year_by_comp[r.company] = r.year

    candidate_indices = [
        i for i, r in enumerate(records) if r.year > min_year_by_comp[r.company]
    ]

    num_anomalies = max(1, int(len(records) * anomaly_fraction))
    chosen_indices = rng.choice(candidate_indices, size=num_anomalies, replace=False)

    ground_truth = np.zeros(len(records), dtype=int)
    anomaly_log = []

    anomaly_types = [
        "revenue_explosion",
        "margin_collapse",
        "ocf_net_income_divergence",
        "debt_equity_explosion",
        "roe_profitability_collapse",
    ]

    for idx_num, idx in enumerate(chosen_indices):
        rec = records[idx]
        atype = anomaly_types[idx_num % len(anomaly_types)]
        ground_truth[idx] = 1

        orig_dict = rec.to_dict()

        if atype == "revenue_explosion":
            rec.revenue = rec.revenue * 5.0
            rec.gross_profit = (rec.gross_profit or rec.revenue * 0.3) * 1.5
            desc = f"Revenue surged 5x to {rec.revenue} while gross profit grew only modestly"

        elif atype == "margin_collapse":
            rec.gross_profit = max(10.0, rec.revenue * 0.04)
            rec.net_income = -abs(rec.revenue * 0.15)
            rec.ebitda = -abs(rec.revenue * 0.10)
            desc = "Gross margin collapsed to 4% of revenue"

        elif atype == "ocf_net_income_divergence":
            rec.net_income = max(100.0, (rec.net_income or rec.revenue * 0.1) * 3.5)
            rec.operating_cash_flow = -abs(rec.revenue * 0.20)
            desc = f"Accrual net income surged 3.5x to {rec.net_income} while operating cash flow turned negative ({rec.operating_cash_flow})"

        elif atype == "debt_equity_explosion":
            cur_de = rec.debt_to_equity or 1.0
            rec.debt_to_equity = cur_de * 6.0
            rec.current_ratio = max(0.4, (rec.current_ratio or 1.5) * 0.35)
            desc = f"Debt/Equity ratio surged 6x from {cur_de} to {rec.debt_to_equity}"

        elif atype == "roe_profitability_collapse":
            cur_eq = rec.shareholder_equity or 5000.0
            rec.net_income = -abs(cur_eq * 1.2)
            rec.shareholder_equity = max(50.0, cur_eq * 0.20)
            rec.roe = rec.net_income / rec.shareholder_equity
            desc = f"Severe operational loss of {rec.net_income} causing ROE to collapse to {rec.roe:.2f}"

        anomaly_log.append({
            "index": idx,
            "company": rec.company,
            "year": rec.year,
            "type": atype,
            "description": desc,
        })

    return records, ground_truth, anomaly_log


def evaluate_model_predictions(
    ground_truth: np.ndarray,
    raw_predictions: np.ndarray,
    model_name: str,
) -> BenchmarkMetrics:
    """Evaluate model predictions against binary ground truth labels (0=normal, 1=anomaly)."""
    pred_binary = np.where(raw_predictions == -1, 1, 0)

    tp = int(np.sum((pred_binary == 1) & (ground_truth == 1)))
    fp = int(np.sum((pred_binary == 1) & (ground_truth == 0)))
    tn = int(np.sum((pred_binary == 0) & (ground_truth == 0)))
    fn = int(np.sum((pred_binary == 0) & (ground_truth == 1)))

    prec = float(precision_score(ground_truth, pred_binary, zero_division=0))
    rec = float(recall_score(ground_truth, pred_binary, zero_division=0))
    f1 = float(f1_score(ground_truth, pred_binary, zero_division=0))
    acc = float(accuracy_score(ground_truth, pred_binary))

    return BenchmarkMetrics(
        model_name=model_name,
        precision=prec,
        recall=rec,
        f1=f1,
        accuracy=acc,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=len(ground_truth),
        total_anomalies=int(np.sum(ground_truth)),
    )


def run_anomaly_benchmark(
    records: Optional[list[FinancialRecord]] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    anomaly_fraction: float = 0.08,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full Planted Anomaly Benchmark comparing Z-Score Baseline and IsolationForest."""
    if records is None:
        if dataset_path and Path(dataset_path).exists():
            records = load_dataset(Path(dataset_path))
        elif Path("data/Financial Statements.csv").exists():
            records = load_dataset(Path("data/Financial Statements.csv"))
        else:
            df_sample = generate_reference_dataset(seed=seed)
            records = load_dataset(df_sample)

    # 1. Inject planted anomalies
    bench_records, ground_truth, anomaly_log = inject_planted_anomalies(
        records, anomaly_fraction=anomaly_fraction, seed=seed
    )

    # 2. Train Detector on clean dataset
    detector = AnomalyDetector(config=AnomalyConfig(random_seed=seed))
    detector.train(records)

    # 3. Predict on benchmark dataset using both models
    _, feature_df, _ = compute_financial_features(bench_records, detector.config.features)
    X_bench = detector.pipeline.transform(feature_df)

    zscore_preds = detector.zscore_baseline.predict(X_bench)
    iforest_preds = detector.isolation_forest.predict(X_bench)

    # 4. Evaluate metrics
    zscore_metrics = evaluate_model_predictions(ground_truth, zscore_preds, "Z-Score Baseline")
    iforest_metrics = evaluate_model_predictions(ground_truth, iforest_preds, "IsolationForest")

    results = {
        "dataset_size": len(bench_records),
        "planted_anomalies_count": int(np.sum(ground_truth)),
        "planted_anomalies_details": anomaly_log,
        "features_used": detector.training_feature_names,
        "zscore": zscore_metrics.to_dict(),
        "isolation_forest": iforest_metrics.to_dict(),
    }

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_anomaly_benchmark()
    print("\n" + "=" * 70)
    print("FINSIGHT AI - SYNTHETIC PLANTED ANOMALY BENCHMARK RESULTS")
    print("=" * 70)
    print("Methodology Note:")
    print("  - Planted anomalies are synthetically injected financial shocks (e.g., revenue surges,")
    print("    margin collapses, cash flow divergences) and do NOT represent real fraud or error labels.")
    print("  - Precision, Recall, and F1 measure detection performance against this controlled artificial benchmark.")
    print("  - Ground-truth labels are completely separated and NEVER enter model training or feature matrices.")
    print("-" * 70)
    print(f"Total Observations: {results['dataset_size']} | Injected Ground-Truth Anomalies: {results['planted_anomalies_count']}\n")

    print(f"{'Metric':<22} | {'Z-Score Baseline':<18} | {'Isolation Forest':<18}")
    print("-" * 70)
    for metric in ["precision", "recall", "f1", "accuracy", "true_positives", "false_positives", "false_negatives"]:
        z_val = results["zscore"][metric]
        if_val = results["isolation_forest"][metric]
        print(f"{metric:<22} | {str(z_val):<18} | {str(if_val):<18}")
    print("=" * 70)
