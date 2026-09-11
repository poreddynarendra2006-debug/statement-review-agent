"""In-memory entry point for FinSight Trend Agent.

Provides pure, in-memory trend analysis for consumption by Orchestrator
and other peer agents without filesystem side effects or file I/O.
"""

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.deviation_analysis import DeviationAnalyzer, DeviationRecord
from analysis.forecasting import (
    BacktestPrediction,
    EvaluationResult,
    ForecastResult,
    TrendForecaster,
)
from analysis.ratio_analysis import RatioAnalyzer, RatioResult
from analysis.yoy_analysis import YoYResult, calculate_yoy_for_series

logger = logging.getLogger(__name__)

STANDARD_FIELDS = [
    "year",
    "company",
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "net_income",
    "operating_expenses",
    "shareholder_equity",
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "roe",
    "roa",
    "roi",
    "current_ratio",
    "debt_equity_ratio",
    "net_profit_margin",
    "return_on_tangible_equity",
]

CANONICAL_ALIASES = {
    "expenses": "cost_of_revenue",
    "cost_of_revenue": "cost_of_revenue",
    "cost of revenue": "cost_of_revenue",
    "margin": "net_profit_margin",
    "net_profit_margin": "net_profit_margin",
    "net profit margin": "net_profit_margin",
    "profit_margin": "net_profit_margin",
    "profit margin": "net_profit_margin",
    "debt/equity_ratio": "debt_equity_ratio",
    "debt_to_equity": "debt_equity_ratio",
    "current ratio": "current_ratio",
    "return on equity": "roe",
    "return on assets": "roa",
    "return on investment": "roi",
    "return on tangible equity": "return_on_tangible_equity",
}


def _normalize_records(records: Any) -> pd.DataFrame:
    """Normalize input records (DataFrame or iterable of record objects) into a DataFrame.

    Ensures all standard fields exist and missing fields are set to None.
    Replaces any legacy metric names ('expenses', 'margin') with canonical
    names ('cost_of_revenue', 'net_profit_margin').
    """
    if records is None:
        return pd.DataFrame()

    if isinstance(records, pd.DataFrame):
        if records.empty:
            return pd.DataFrame()
        df = records.copy()

        # Map existing columns to canonical standard fields where applicable
        rename_map = {}
        for col in df.columns:
            cleaned = str(col).strip().lower()
            if cleaned in CANONICAL_ALIASES:
                target = CANONICAL_ALIASES[cleaned]
                if target not in df.columns or df[target].isna().all():
                    rename_map[col] = target
        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure all standard fields exist in DataFrame
        for f in STANDARD_FIELDS:
            if f not in df.columns:
                matched_col = None
                cleaned_target = re.sub(r"[^\w]", "", f.lower())
                for col in df.columns:
                    if re.sub(r"[^\w]", "", str(col).lower()) == cleaned_target:
                        matched_col = col
                        break
                if matched_col is not None:
                    df[f] = df[matched_col]
                else:
                    df[f] = None

    elif isinstance(records, (list, tuple)):
        if len(records) == 0:
            return pd.DataFrame()
        rows = []
        for rec in records:
            row: Dict[str, Any] = {}
            if isinstance(rec, dict):
                # First copy all raw keys normalized
                for k, v in rec.items():
                    cleaned_k = str(k).strip().lower()
                    canonical_k = CANONICAL_ALIASES.get(cleaned_k, cleaned_k)
                    row[canonical_k] = v

                # Ensure standard fields
                for f in STANDARD_FIELDS:
                    if f not in row or row[f] is None:
                        val = rec.get(f)
                        if val is None:
                            cleaned_target = re.sub(r"[^\w]", "", f.lower())
                            for k, v in rec.items():
                                if re.sub(r"[^\w]", "", str(k).lower()) == cleaned_target:
                                    val = v
                                    break
                        row[f] = val
            else:
                # Object whose fields are accessed by attributes
                for attr in dir(rec):
                    if attr.startswith("_"):
                        continue
                    cleaned_attr = attr.lower()
                    if cleaned_attr in CANONICAL_ALIASES:
                        row[CANONICAL_ALIASES[cleaned_attr]] = getattr(rec, attr, None)

                for f in STANDARD_FIELDS:
                    val = getattr(rec, f, None)
                    if val is None:
                        cleaned_target = re.sub(r"[^\w]", "", f.lower())
                        for attr in dir(rec):
                            if attr.startswith("_"):
                                continue
                            if re.sub(r"[^\w]", "", attr.lower()) == cleaned_target:
                                val = getattr(rec, attr, None)
                                break
                    row[f] = val

                if hasattr(rec, "__dict__"):
                    for k, v in vars(rec).items():
                        if not k.startswith("_") and k not in row:
                            row[k] = v

            rows.append(row)
        df = pd.DataFrame(rows)

    else:
        try:
            df = pd.DataFrame(records)
            for f in STANDARD_FIELDS:
                if f not in df.columns:
                    df[f] = None
        except Exception as e:
            logger.error(f"Cannot convert records to DataFrame: {e}")
            return pd.DataFrame()

    if df.empty or "company" not in df.columns or "year" not in df.columns:
        return pd.DataFrame()

    # Clean company and year
    df["company"] = df["company"].astype(str).str.strip()
    valid_mask = (
        df["company"].ne("")
        & df["company"].ne("None")
        & df["company"].ne("nan")
        & df["year"].notna()
    )
    df = df[valid_mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["year"] = df["year"].astype(int)

    # Convert numeric fields
    numeric_fields = [
        f for f in STANDARD_FIELDS if f not in ["company", "year"] and f in df.columns
    ]
    for f in numeric_fields:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    # Bridge aliases: cost_of_revenue vs expenses
    if ("cost_of_revenue" not in df.columns or df["cost_of_revenue"].isna().all()) and "expenses" in df.columns:
        df["cost_of_revenue"] = pd.to_numeric(df["expenses"], errors="coerce")
    if "cost_of_revenue" in df.columns and df["cost_of_revenue"].isna().all():
        if "revenue" in df.columns and "gross_profit" in df.columns:
            if not (df["revenue"].isna().all() or df["gross_profit"].isna().all()):
                df["cost_of_revenue"] = df["revenue"] - df["gross_profit"]

    # Bridge aliases: net_profit_margin vs margin
    if ("net_profit_margin" not in df.columns or df["net_profit_margin"].isna().all()) and "margin" in df.columns:
        df["net_profit_margin"] = pd.to_numeric(df["margin"], errors="coerce")
    if "net_profit_margin" in df.columns and df["net_profit_margin"].isna().all():
        if "net_income" in df.columns and "revenue" in df.columns:
            if not (df["net_income"].isna().all() or df["revenue"].isna().all()):
                with np.errstate(divide="ignore", invalid="ignore"):
                    derived_npm = np.where(
                        df["revenue"] != 0,
                        (df["net_income"] / df["revenue"]) * 100.0,
                        np.nan,
                    )
                df["net_profit_margin"] = derived_npm

    # Deduplicate and sort chronologically
    df = df.drop_duplicates(subset=["company", "year"], keep="last")
    df = df.sort_values(by=["company", "year"], ascending=[True, True]).reset_index(drop=True)

    # Strictly eliminate non-canonical metric columns 'expenses' and 'margin' from df
    cols_to_drop = [c for c in ["expenses", "margin"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def run_trend_analysis(records: Any, materiality: float = 0.10) -> Dict[str, Any]:
    """Pure in-memory entry point for financial trend analysis.

    Parameters:
        records: A pandas DataFrame or a list of record objects whose fields
                 are accessed by attributes.
        materiality: Configurable deviation materiality threshold (default 0.10 / 10%).

    Returns:
        A dictionary with exactly five keys containing project dataclass instances:
        {
            "yoy": List[YoYResult],
            "ratios": List[RatioResult],
            "forecasts": List[ForecastResult],
            "evaluations": List[EvaluationResult],
            "deviations": List[DeviationRecord],
        }
    """
    df = _normalize_records(records)

    empty_result: Dict[str, Any] = {
        "yoy": [],
        "ratios": [],
        "forecasts": [],
        "evaluations": [],
        "deviations": [],
    }

    if df.empty:
        return empty_result

    sorted_df = df.sort_values(by=["company", "year"], ascending=[True, True])

    # 1. Year-over-Year (YoY) Analysis
    yoy_results: List[YoYResult] = []
    excluded_yoy = {"year", "expenses", "margin"}
    yoy_metrics = [
        c for c in sorted_df.columns
        if c not in excluded_yoy and pd.api.types.is_numeric_dtype(sorted_df[c])
    ]

    for company, comp_group in sorted_df.groupby("company"):
        years = comp_group["year"].tolist()
        for m in yoy_metrics:
            vals = comp_group[m].tolist()
            if any(pd.notna(v) for v in vals):
                yoy_results.extend(
                    calculate_yoy_for_series(
                        company=str(company),
                        metric=m,
                        years=years,
                        values=vals,
                    )
                )

    # 2. Financial Ratios Analysis
    ratio_analyzer = RatioAnalyzer()
    ratio_results: List[RatioResult] = ratio_analyzer.analyze_ratios(sorted_df)

    # 3. Forecasting & Walk-Forward Evaluations
    forecaster = TrendForecaster()
    candidate_targets = ["revenue", "cost_of_revenue", "net_profit_margin"]
    forecast_results: List[ForecastResult] = []
    eval_results: List[EvaluationResult] = []
    all_backtest_preds: List[BacktestPrediction] = []

    for company, comp_group in sorted_df.groupby("company"):
        years = comp_group["year"].tolist()
        for metric in candidate_targets:
            if metric not in comp_group.columns:
                continue
            vals = comp_group[metric].tolist()
            if not any(pd.notna(v) for v in vals):
                continue

            # Walk-forward backtest evaluation
            eval_res, backtest_preds = forecaster.backtest_series(
                company=str(company),
                metric=metric,
                years=years,
                values=vals,
            )
            eval_results.append(eval_res)
            all_backtest_preds.extend(backtest_preds)

            # Next-year forecast
            forecast_res = forecaster.forecast_next_year(
                company=str(company),
                metric=metric,
                years=years,
                values=vals,
            )
            forecast_results.append(forecast_res)

    # 4. Actual vs Forecast Deviation Analysis
    deviation_analyzer = DeviationAnalyzer(materiality_threshold=materiality)
    mae_lookup: Dict[Tuple[str, str], float] = {
        (str(ev.company), str(ev.metric)): float(ev.mae)
        for ev in eval_results
        if ev.mae is not None and not np.isnan(ev.mae)
    }

    deviation_results: List[DeviationRecord] = []
    for pred in all_backtest_preds:
        mae_val = mae_lookup.get((str(pred.company), str(pred.metric)))
        dev_record = deviation_analyzer.analyze_prediction(pred, walk_forward_mae=mae_val)
        deviation_results.append(dev_record)

    return {
        "yoy": yoy_results,
        "ratios": ratio_results,
        "forecasts": forecast_results,
        "evaluations": eval_results,
        "deviations": deviation_results,
    }
