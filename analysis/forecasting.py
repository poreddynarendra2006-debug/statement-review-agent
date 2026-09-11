"""Time-series forecasting and historical backtesting engine for FinSight Trend Agent.

Key Features:
1. Company-wise separation: each company's historical records are modeled as an independent time series.
2. Chronological ordering: strictly avoids time-travel or future data leakage.
3. Chronological backtesting: holds out the most recent historical period to evaluate MAE, RMSE, and R².
4. Unbounded R²: respects that R² can be negative for a poor model, never forcing it to [0, 1].
5. Full-history retraining: retrains on all available observations to predict next_year = max(year) + 1.
6. Handles insufficient data or missing values gracefully without failing.
7. Completely deterministic: uses scikit-learn LinearRegression with no random seeding or stochastic variation.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import (
    BACKTEST_TEST_RATIO,
    DEFAULT_FORECAST_METHOD,
    MIN_OBSERVATIONS_FOR_EVAL,
    MIN_OBSERVATIONS_FOR_FORECAST,
    WALK_FORWARD_MIN_TRAIN,
    R2_MIN_TEST_POINTS,
)

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Structure for next-year forecast per company and metric."""
    company: str
    forecast_year: int
    metric: str
    last_actual_year: int
    last_actual_value: float
    forecast_value: float
    forecast_method: str
    slope: Optional[float] = None
    intercept: Optional[float] = None
    status: str = "success"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Structure for historical walk-forward backtesting evaluation metrics."""
    company: str
    metric: str
    model: str
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    evaluation_status: str  # "success", "insufficient_data", "constant_target"
    train_years_count: int
    test_years_count: int
    train_years_range: str
    test_years_range: str
    notes: Optional[str] = None
    mape: Optional[float] = None
    baseline_mae: Optional[float] = None
    beats_baseline: Optional[bool] = None
    test_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestPrediction:
    """Individual prediction during chronological backtest."""
    company: str
    year: int
    metric: str
    actual: float
    forecast: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrendForecaster:
    """Independent time-series forecasting engine with chronological walk-forward backtesting."""

    def __init__(
        self,
        method: str = DEFAULT_FORECAST_METHOD,
        test_ratio: float = BACKTEST_TEST_RATIO,
        min_eval_obs: int = MIN_OBSERVATIONS_FOR_EVAL,
        min_forecast_obs: int = MIN_OBSERVATIONS_FOR_FORECAST,
        min_train_obs: int = WALK_FORWARD_MIN_TRAIN,
        r2_min_test_points: int = R2_MIN_TEST_POINTS,
    ):
        self.method = method
        self.test_ratio = test_ratio
        self.min_eval_obs = min_eval_obs
        self.min_forecast_obs = min_forecast_obs
        self.min_train_obs = min_train_obs
        self.r2_min_test_points = r2_min_test_points

    def backtest_series(
        self,
        company: str,
        metric: str,
        years: List[int],
        values: List[float],
    ) -> Tuple[EvaluationResult, List[BacktestPrediction]]:
        """Run chronological walk-forward backtesting starting from the 4th observation."""
        valid_pairs = [
            (y, v) for y, v in zip(years, values)
            if v is not None and not np.isnan(v)
        ]

        if len(valid_pairs) < self.min_eval_obs:
            eval_res = EvaluationResult(
                company=company,
                metric=metric,
                model=self.method,
                mae=None,
                rmse=None,
                r2=None,
                evaluation_status="insufficient_data",
                train_years_count=len(valid_pairs),
                test_years_count=0,
                train_years_range=f"{valid_pairs[0][0]}-{valid_pairs[-1][0]}" if valid_pairs else "none",
                test_years_range="none",
                notes=(
                    f"Insufficient observations ({len(valid_pairs)} < {self.min_eval_obs}). "
                    f"Walk-forward evaluation requires at least {self.min_eval_obs} observations."
                ),
                mape=None,
                baseline_mae=None,
                beats_baseline=None,
                test_points=0,
            )
            return eval_res, []

        y_arr = np.array([p[0] for p in valid_pairs], dtype=int)
        v_arr = np.array([p[1] for p in valid_pairs], dtype=float)
        n = len(valid_pairs)

        model_errors: List[float] = []
        model_squared_errors: List[float] = []
        baseline_errors: List[float] = []
        mapes: List[float] = []
        test_actuals: List[float] = []
        test_predictions: List[float] = []
        backtest_preds: List[BacktestPrediction] = []

        # Chronological walk-forward / rolling-origin evaluation starting from 4th observation
        for i in range(self.min_train_obs, n):
            x_train = y_arr[:i]
            y_train = v_arr[:i]

            reg = LinearRegression()
            reg.fit(x_train.reshape(-1, 1), y_train)

            test_year = int(y_arr[i])
            actual_val = float(v_arr[i])
            pred_val = float(reg.predict(np.array([[test_year]]))[0])
            naive_pred = float(v_arr[i - 1])  # Naive baseline: next year equals current year

            err = abs(actual_val - pred_val)
            model_errors.append(err)
            model_squared_errors.append((actual_val - pred_val) ** 2)
            baseline_errors.append(abs(actual_val - naive_pred))

            if actual_val != 0.0:
                mapes.append((err / abs(actual_val)) * 100.0)

            test_actuals.append(actual_val)
            test_predictions.append(pred_val)

            backtest_preds.append(
                BacktestPrediction(
                    company=company,
                    year=test_year,
                    metric=metric,
                    actual=actual_val,
                    forecast=pred_val,
                )
            )

        test_points = len(test_actuals)
        mae_val = float(np.mean(model_errors))
        rmse_val = float(np.sqrt(np.mean(model_squared_errors)))
        baseline_mae_val = float(np.mean(baseline_errors))
        beats_baseline_val = bool(mae_val < baseline_mae_val)
        mape_val = float(np.mean(mapes)) if mapes else None

        # R² evaluation: only report R² when test_points >= 5
        r2_val: Optional[float] = None
        if test_points >= self.r2_min_test_points:
            var_test = float(np.var(test_actuals))
            if var_test > 1e-9:
                r2_val = float(r2_score(test_actuals, test_predictions))
                r2_note = "Walk-forward backtest successful."
            else:
                r2_note = "R² null: test targets have near-zero variance."
        else:
            r2_note = (
                f"R² null: requires at least {self.r2_min_test_points} test points "
                f"in walk-forward evaluation (found {test_points})."
            )

        eval_res = EvaluationResult(
            company=company,
            metric=metric,
            model=self.method,
            mae=round(mae_val, 4),
            rmse=round(rmse_val, 4),
            r2=round(r2_val, 4) if r2_val is not None else None,
            evaluation_status="success",
            train_years_count=n - 1,
            test_years_count=test_points,
            train_years_range=f"{y_arr[0]}-{y_arr[-2]}",
            test_years_range=f"{y_arr[self.min_train_obs]}-{y_arr[-1]}",
            notes=r2_note,
            mape=round(mape_val, 4) if mape_val is not None else None,
            baseline_mae=round(baseline_mae_val, 4),
            beats_baseline=beats_baseline_val,
            test_points=test_points,
        )

        return eval_res, backtest_preds

    def forecast_next_year(
        self,
        company: str,
        metric: str,
        years: List[int],
        values: List[float],
    ) -> ForecastResult:
        """Retrain model on ALL historical observations and predict next_year = max(year) + 1."""
        valid_pairs = [
            (y, v) for y, v in zip(years, values)
            if v is not None and not np.isnan(v)
        ]

        if len(valid_pairs) < self.min_forecast_obs:
            last_yr = valid_pairs[-1][0] if valid_pairs else 0
            last_val = valid_pairs[-1][1] if valid_pairs else 0.0
            return ForecastResult(
                company=company,
                forecast_year=last_yr + 1 if last_yr else 0,
                metric=metric,
                last_actual_year=last_yr,
                last_actual_value=last_val,
                forecast_value=np.nan,
                forecast_method=self.method,
                status="not_forecastable",
                notes=(
                    f"Requires at least {self.min_forecast_obs} observations to fit a trend. "
                    f"Found {len(valid_pairs)}."
                ),
            )

        y_arr = np.array([p[0] for p in valid_pairs], dtype=int)
        v_arr = np.array([p[1] for p in valid_pairs], dtype=float)

        latest_actual_year = int(y_arr[-1])
        latest_actual_value = float(v_arr[-1])
        next_year = latest_actual_year + 1

        # Check for near-zero variance across historical target values
        var_val = float(np.var(v_arr))
        if var_val <= 1e-9:
            constant_val = float(v_arr[0])
            return ForecastResult(
                company=company,
                forecast_year=next_year,
                metric=metric,
                last_actual_year=latest_actual_year,
                last_actual_value=round(latest_actual_value, 4),
                forecast_value=np.nan,
                forecast_method=self.method,
                slope=0.0,
                intercept=round(constant_val, 4),
                status="not_forecastable",
                notes="Target has near-zero variance; next-year forecast is not meaningful.",
            )

        # Full retraining on ALL historical observations
        reg = LinearRegression()
        reg.fit(y_arr.reshape(-1, 1), v_arr)

        pred_val = float(reg.predict(np.array([[next_year]]))[0])
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)

        return ForecastResult(
            company=company,
            forecast_year=next_year,
            metric=metric,
            last_actual_year=latest_actual_year,
            last_actual_value=round(latest_actual_value, 4),
            forecast_value=round(pred_val, 4),
            forecast_method=self.method,
            slope=round(slope, 6),
            intercept=round(intercept, 4),
            status="success",
            notes=f"Retrained on {len(valid_pairs)} observations from {y_arr[0]} to {latest_actual_year}.",
        )

    def process_all_companies(
        self,
        df: pd.DataFrame,
        target_metrics: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[BacktestPrediction]]:
        """Run backtesting and next-year forecasting across all companies in the dataset."""
        forecast_records: List[Dict[str, Any]] = []
        eval_records: List[Dict[str, Any]] = []
        all_backtest_preds: List[BacktestPrediction] = []

        if df.empty or "company" not in df.columns or "year" not in df.columns:
            return pd.DataFrame(), pd.DataFrame(), []

        if target_metrics is None:
            target_metrics = ["revenue", "cost_of_revenue", "net_profit_margin"]

        sorted_df = df.sort_values(by=["company", "year"], ascending=[True, True])

        for company, comp_group in sorted_df.groupby("company"):
            years = comp_group["year"].tolist()

            for metric in target_metrics:
                if metric not in comp_group.columns:
                    continue

                values = comp_group[metric].tolist()

                # 1. Walk-Forward Backtest
                eval_res, backtest_preds = self.backtest_series(
                    company=str(company),
                    metric=metric,
                    years=years,
                    values=values,
                )
                eval_records.append(eval_res.to_dict())
                all_backtest_preds.extend(backtest_preds)

                # 2. Next-Year Forecast
                forecast_res = self.forecast_next_year(
                    company=str(company),
                    metric=metric,
                    years=years,
                    values=values,
                )
                forecast_records.append(forecast_res.to_dict())

        forecast_df = pd.DataFrame(forecast_records)
        eval_df = pd.DataFrame(eval_records)
        return forecast_df, eval_df, all_backtest_preds
