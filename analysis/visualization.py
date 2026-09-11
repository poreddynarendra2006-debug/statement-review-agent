"""Visualization utilities for FinSight Trend Agent.

Generates Actual vs Forecast line charts using matplotlib.
Non-interactive / headless safe (uses Agg backend).
Completely optional: does not block or impair core pipeline execution.
"""

import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend safe for scripts / servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_actual_vs_forecast(
    company: str,
    metric: str,
    historical_years: List[int],
    historical_actuals: List[float],
    forecast_year: int,
    forecast_value: float,
    output_path: str,
    backtest_years: Optional[List[int]] = None,
    backtest_preds: Optional[List[float]] = None,
) -> str:
    """Plot historical actuals, backtest predictions, and next-year forecast point."""
    plt.figure(figsize=(9, 5))
    
    # 1. Historical Actuals
    plt.plot(
        historical_years,
        historical_actuals,
        marker="o",
        color="#1f77b4",
        linewidth=2,
        label=f"Actual {metric}",
    )

    # 2. Backtest Predictions (if provided)
    if backtest_years and backtest_preds and len(backtest_years) == len(backtest_preds):
        plt.plot(
            backtest_years,
            backtest_preds,
            marker="s",
            linestyle="--",
            color="#ff7f0e",
            linewidth=1.5,
            label="Backtest Model Pred",
        )

    # 3. Next-Year Forecast Point
    if not np.isnan(forecast_value):
        # Connect last actual point to forecast point with dotted line
        plt.plot(
            [historical_years[-1], forecast_year],
            [historical_actuals[-1], forecast_value],
            linestyle=":",
            color="#2ca02c",
            alpha=0.7,
        )
        plt.scatter(
            [forecast_year],
            [forecast_value],
            color="#2ca02c",
            s=90,
            zorder=5,
            label=f"Next-Year Forecast ({forecast_year})",
        )

    plt.title(f"{company} — {metric}: Historical Actual vs. Forecast", fontsize=13, fontweight="bold")
    plt.xlabel("Fiscal Year", fontsize=11)
    plt.ylabel(f"{metric} Value", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
