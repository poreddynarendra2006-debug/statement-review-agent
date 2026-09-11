"""Actual vs Forecast deviation analysis module for FinSight Trend Agent.

Calculates:
- deviation = actual - forecast
- deviation_percent = ((actual - forecast) / abs(forecast)) * 100
- absolute_deviation = abs(deviation)
- absolute_deviation_percent = abs(deviation_percent)
- actual_to_forecast_ratio = actual / forecast
- material_deviation = abs(deviation_percent) >= (threshold * 100)
- direction: "above_forecast", "below_forecast", "within_expected_range"

Rules:
- Configurable materiality threshold (default 0.10 / 10%).
- Safe zero-division handling when forecast == 0.
- Completely deterministic Python arithmetic.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import DEFAULT_MATERIALITY_THRESHOLD
from analysis.forecasting import BacktestPrediction

logger = logging.getLogger(__name__)


@dataclass
class DeviationRecord:
    company: str
    year: int
    metric: str
    actual: float
    forecast: float
    deviation: float
    deviation_percent: Optional[float]
    absolute_deviation: float
    absolute_deviation_percent: Optional[float]
    actual_to_forecast_ratio: Optional[float]
    direction: str  # "above_forecast", "below_forecast", "within_expected_range"
    material_deviation: bool
    materiality_threshold: float
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeviationAnalyzer:
    """Evaluates backtested forecasts against actual observed results with dual-condition materiality."""

    def __init__(self, materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD):
        self.materiality_threshold = float(materiality_threshold)

    def analyze_prediction(
        self,
        pred: BacktestPrediction,
        walk_forward_mae: Optional[float] = None,
    ) -> DeviationRecord:
        """Analyze a single backtest prediction point.
        
        Material deviation requires BOTH:
        1. abs(deviation_percent) > materiality_threshold * 100
        2. abs(actual - forecast) > 2 * walk_forward_MAE
        """
        actual = float(pred.actual)
        forecast = float(pred.forecast)
        dev = actual - forecast
        abs_dev = abs(dev)

        # Safe division handling for zero forecast
        if forecast == 0.0:
            return DeviationRecord(
                company=pred.company,
                year=pred.year,
                metric=pred.metric,
                actual=round(actual, 4),
                forecast=0.0,
                deviation=round(dev, 4),
                deviation_percent=None,
                absolute_deviation=round(abs_dev, 4),
                absolute_deviation_percent=None,
                actual_to_forecast_ratio=None,
                direction="within_expected_range",
                material_deviation=False,
                materiality_threshold=self.materiality_threshold,
                notes="Forecast is exactly 0.0; percentage deviation undefined.",
            )

        dev_pct = (dev / abs(forecast)) * 100.0
        abs_dev_pct = abs(dev_pct)
        ratio = actual / forecast

        # Condition 1: percentage exceeds materiality threshold
        threshold_pct = self.materiality_threshold * 100.0
        pct_exceeded = abs_dev_pct > threshold_pct

        # Condition 2: error exceeds 2 * walk-forward MAE
        notes = None
        if walk_forward_mae is not None and not np.isnan(walk_forward_mae):
            range_exceeded = abs_dev > (2.0 * float(walk_forward_mae))
            is_material = bool(pct_exceeded and range_exceeded)
        else:
            is_material = False
            notes = "Walk-forward MAE unavailable; expected error range condition cannot be evaluated."

        if is_material:
            direction = "above_forecast" if dev > 0 else "below_forecast"
        else:
            direction = "within_expected_range"

        return DeviationRecord(
            company=pred.company,
            year=pred.year,
            metric=pred.metric,
            actual=round(actual, 4),
            forecast=round(forecast, 4),
            deviation=round(dev, 4),
            deviation_percent=round(dev_pct, 4),
            absolute_deviation=round(abs_dev, 4),
            absolute_deviation_percent=round(abs_dev_pct, 4),
            actual_to_forecast_ratio=round(ratio, 4),
            direction=direction,
            material_deviation=is_material,
            materiality_threshold=self.materiality_threshold,
            notes=notes,
        )

    def analyze_all(
        self,
        predictions: List[BacktestPrediction],
        mae_lookup: Optional[Dict[Any, float]] = None,
        evaluations: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Analyze all backtest predictions and return DataFrame.
        
        Supports passing MAE lookup directly or passing an EvaluationResult list / DataFrame.
        """
        if not predictions:
            return pd.DataFrame()

        lookup: Dict[Tuple[str, str], float] = {}
        if mae_lookup:
            for k, v in mae_lookup.items():
                if isinstance(k, tuple):
                    lookup[(str(k[0]), str(k[1]))] = float(v) if v is not None else None
                elif isinstance(k, str) and "_" in k:
                    parts = k.split("_", 1)
                    lookup[(parts[0], parts[1])] = float(v) if v is not None else None
                elif isinstance(k, str):
                    lookup[(k, "")] = float(v) if v is not None else None
        elif evaluations is not None:
            if isinstance(evaluations, pd.DataFrame) and not evaluations.empty:
                for _, row in evaluations.iterrows():
                    c = str(row.get("company", ""))
                    m = str(row.get("metric", ""))
                    mae_v = row.get("mae")
                    if pd.notna(mae_v):
                        lookup[(c, m)] = float(mae_v)
            elif isinstance(evaluations, list):
                for ev in evaluations:
                    if hasattr(ev, "company") and hasattr(ev, "metric") and hasattr(ev, "mae"):
                        if ev.mae is not None:
                            lookup[(str(ev.company), str(ev.metric))] = float(ev.mae)

        records = []
        for p in predictions:
            mae_val = lookup.get((str(p.company), str(p.metric)))
            records.append(self.analyze_prediction(p, walk_forward_mae=mae_val).to_dict())

        return pd.DataFrame(records)
