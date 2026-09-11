"""Year-on-Year (YoY) analysis module for FinSight Trend Agent.

Implements the official HLD contract:
YoYResult defined in analysis/yoy_analysis.py
Produced by: Trend Agent
Consumed by: Evidence Agent and Anomaly Agent

Rules:
- All calculations are deterministic Python (no LLM arithmetic).
- YoY % = ((current - previous) / abs(previous)) * 100 (using abs(previous) preserves intuitive growth direction for negative bases).
- First available year is represented as trend="No previous year", not "Stable".
- Zero previous year is handled safely without ZeroDivisionError.
- Negative previous values are documented with clear data_status flags.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import EXTREME_YOY_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class YoYResult:
    """Official HLD contract for YoY movements per line item."""
    company: str
    year: int
    metric: str
    current_value: float
    previous_value: Optional[float]
    yoy_percent: Optional[float]
    trend: str  # "growth", "decline", "stable", or "No previous year"
    data_status: str  # "normal", "first_year", "zero_previous", "negative_base", "missing_value", "extreme_percentage"
    status: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = self.data_status
        elif self.data_status is None:
            self.data_status = self.status

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("status") is None:
            d["status"] = self.data_status
        return d


def calculate_yoy_for_series(
    company: str,
    metric: str,
    years: List[int],
    values: List[float],
    extreme_threshold: float = EXTREME_YOY_THRESHOLD,
) -> List[YoYResult]:
    """Calculate Year-on-Year changes for a single company-metric series.
    
    Expects years and values to be pre-sorted chronologically ascending.
    Priority order: zero_previous -> negative_base -> extreme_percentage -> normal
    """
    results: List[YoYResult] = []

    for i in range(len(years)):
        curr_yr = int(years[i])
        curr_val = values[i]

        # Check if current value is NaN
        if curr_val is None or (isinstance(curr_val, (float, np.floating)) and np.isnan(curr_val)):
            results.append(
                YoYResult(
                    company=company,
                    year=curr_yr,
                    metric=metric,
                    current_value=np.nan,
                    previous_value=None,
                    yoy_percent=None,
                    trend="No previous year" if i == 0 else "unavailable",
                    data_status="missing_value",
                    status="missing_value",
                )
            )
            continue

        curr_val = float(curr_val)

        # First observation for this company and metric
        if i == 0:
            results.append(
                YoYResult(
                    company=company,
                    year=curr_yr,
                    metric=metric,
                    current_value=curr_val,
                    previous_value=None,
                    yoy_percent=None,
                    trend="No previous year",
                    data_status="first_year",
                    status="first_year",
                )
            )
            continue

        prev_yr = int(years[i - 1])
        prev_val = values[i - 1]

        # If previous value was missing
        if prev_val is None or (isinstance(prev_val, (float, np.floating)) and np.isnan(prev_val)):
            results.append(
                YoYResult(
                    company=company,
                    year=curr_yr,
                    metric=metric,
                    current_value=curr_val,
                    previous_value=None,
                    yoy_percent=None,
                    trend="unavailable",
                    data_status="missing_value",
                    status="missing_value",
                )
            )
            continue

        prev_val = float(prev_val)

        # Priority 1: Zero previous value
        if prev_val == 0.0:
            trend = "growth" if curr_val > 0 else ("decline" if curr_val < 0 else "stable")
            results.append(
                YoYResult(
                    company=company,
                    year=curr_yr,
                    metric=metric,
                    current_value=curr_val,
                    previous_value=0.0,
                    yoy_percent=None,  # Division by zero avoided
                    trend=trend,
                    data_status="zero_previous",
                    status="zero_previous",
                )
            )
            continue

        # Valid non-zero previous value
        diff = curr_val - prev_val
        yoy_pct = round((diff / abs(prev_val)) * 100.0, 4)

        if yoy_pct > 0.0001:
            trend = "growth"
        elif yoy_pct < -0.0001:
            trend = "decline"
        else:
            trend = "stable"

        # Exact priority: zero_previous -> negative_base -> extreme_percentage -> normal
        if prev_val < 0.0:
            status = "negative_base"
        elif abs(yoy_pct) > extreme_threshold:
            status = "extreme_percentage"
        else:
            status = "normal"

        results.append(
            YoYResult(
                company=company,
                year=curr_yr,
                metric=metric,
                current_value=curr_val,
                previous_value=prev_val,
                yoy_percent=yoy_pct,
                trend=trend,
                data_status=status,
                status=status,
            )
        )

    return results


def calculate_yoy_dataframe(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    extreme_threshold: float = EXTREME_YOY_THRESHOLD,
) -> pd.DataFrame:
    """Calculate YoY metrics across all companies and specified numeric metrics in DataFrame."""
    if df.empty:
        return pd.DataFrame()

    if metrics is None:
        # Default to all numeric columns other than 'year'
        exclude = {"year"}
        metrics = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    all_results: List[YoYResult] = []

    # Ensure chronological order per company
    sorted_df = df.sort_values(by=["company", "year"], ascending=[True, True])

    for company, comp_group in sorted_df.groupby("company"):
        years = comp_group["year"].tolist()
        for metric in metrics:
            if metric in comp_group.columns:
                vals = comp_group[metric].tolist()
                series_results = calculate_yoy_for_series(
                    company=str(company),
                    metric=metric,
                    years=years,
                    values=vals,
                    extreme_threshold=extreme_threshold,
                )
                all_results.extend(series_results)

    records = [r.to_dict() for r in all_results]
    return pd.DataFrame(records)
