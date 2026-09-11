"""Trend Agent master orchestrator for FinSight AI (Team 33, Member 4 - Trends).

Orchestrates:
1. Data loading and schema normalization.
2. YoY percentage calculation and HLD YoYResult contract generation.
3. Financial ratio extraction and derivation.
4. Independent time-series forecasting and chronological backtesting.
5. Actual vs Forecast deviation analysis with configurable materiality.
6. Machine-readable export of all CSV and JSON artifacts.
7. Optional visualization chart generation.
"""

from dataclasses import asdict
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_TEST_RATIO,
    DEFAULT_FORECAST_METHOD,
    DEFAULT_MATERIALITY_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
)
from analysis.data_mapping import DataMapper, DataMappingReport
from analysis.deviation_analysis import DeviationAnalyzer
from analysis.forecasting import TrendForecaster
from analysis.ratio_analysis import RatioAnalyzer
from analysis.yoy_analysis import calculate_yoy_dataframe

logger = logging.getLogger(__name__)


class TrendAgent:
    """Core agent responsible for time-series forecasting, YoY, and ratio analysis."""

    def __init__(
        self,
        materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD,
        forecast_method: str = DEFAULT_FORECAST_METHOD,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.materiality_threshold = materiality_threshold
        self.forecast_method = forecast_method
        self.output_dir = output_dir

        self.mapper = DataMapper()
        self.forecaster = TrendForecaster(method=forecast_method, test_ratio=BACKTEST_TEST_RATIO)
        self.deviation_analyzer = DeviationAnalyzer(materiality_threshold=materiality_threshold)
        self.ratio_analyzer = RatioAnalyzer()

    def run(
        self,
        input_path: str,
        generate_charts: bool = False,
    ) -> Dict[str, Any]:
        """Execute full Trend Agent pipeline on input dataset."""
        logger.info(f"Starting Trend Agent on: {input_path}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Load raw dataset
        raw_df = self.mapper.load_dataset(input_path)

        # 2. Normalize and validate schema
        norm_df, map_report = self.mapper.map_and_validate(raw_df, source_file=input_path)

        # 3. Pure in-memory trend analysis (Requirement 11)
        from analysis.trend import run_trend_analysis
        trend_results = run_trend_analysis(norm_df, materiality=self.materiality_threshold)

        yoy_df = pd.DataFrame([r.to_dict() for r in trend_results["yoy"]])
        ratios_df = pd.DataFrame([r.to_dict() for r in trend_results["ratios"]])
        forecast_df = pd.DataFrame([r.to_dict() for r in trend_results["forecasts"]])
        eval_df = pd.DataFrame([r.to_dict() for r in trend_results["evaluations"]])
        deviations_df = pd.DataFrame([r.to_dict() for r in trend_results["deviations"]])

        available_targets = list(forecast_df["metric"].unique()) if not forecast_df.empty else []

        # 4. Export structured outputs
        paths = self._export_outputs(
            forecast_df=forecast_df,
            eval_df=eval_df,
            deviations_df=deviations_df,
            yoy_df=yoy_df,
            ratios_df=ratios_df,
            map_report=map_report,
        )

        # 5. Optional Visualizations
        if generate_charts and not forecast_df.empty:
            charts_dir = os.path.join(self.output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            self._generate_visualizations(norm_df, forecast_df, deviations_df, charts_dir)

        # 6. Compile final summary
        summary = {
            "status": "success",
            "source_file": input_path,
            "companies_processed": map_report.companies_found,
            "companies_count": len(map_report.companies_found),
            "target_metrics_forecasted": available_targets,
            "unsupported_metrics": map_report.unsupported_metrics,
            "derived_metrics": map_report.derived_fields,
            "materiality_threshold": self.materiality_threshold,
            "output_files": paths,
        }

        # Write overall summary JSON
        summary_path = os.path.join(self.output_dir, "trend_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _export_outputs(
        self,
        forecast_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        deviations_df: pd.DataFrame,
        yoy_df: pd.DataFrame,
        ratios_df: pd.DataFrame,
        map_report: DataMappingReport,
    ) -> Dict[str, str]:
        """Save all required artifacts to CSV and JSON."""
        paths = {}

        # 1. forecasts.csv
        p_f = os.path.join(self.output_dir, "forecasts.csv")
        forecast_df.to_csv(p_f, index=False)
        paths["forecasts_csv"] = p_f

        # 2. forecast_evaluation.csv
        p_e = os.path.join(self.output_dir, "forecast_evaluation.csv")
        eval_df.to_csv(p_e, index=False)
        paths["forecast_evaluation_csv"] = p_e

        # 3. forecast_deviations.csv
        p_d = os.path.join(self.output_dir, "forecast_deviations.csv")
        deviations_df.to_csv(p_d, index=False)
        paths["forecast_deviations_csv"] = p_d

        # 4. yoy_results.csv
        p_y = os.path.join(self.output_dir, "yoy_results.csv")
        yoy_df.to_csv(p_y, index=False)
        paths["yoy_results_csv"] = p_y

        # 5. financial_ratios.csv (modular ratio analysis)
        p_r = os.path.join(self.output_dir, "financial_ratios.csv")
        ratios_df.to_csv(p_r, index=False)
        paths["financial_ratios_csv"] = p_r

        # 6. data_mapping_report.json
        p_m = os.path.join(self.output_dir, "data_mapping_report.json")
        with open(p_m, "w", encoding="utf-8") as f:
            json.dump(map_report.to_dict(), f, indent=2)
        paths["data_mapping_report_json"] = p_m

        return paths

    def _generate_visualizations(
        self,
        norm_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        deviations_df: pd.DataFrame,
        charts_dir: str,
    ) -> None:
        """Create Actual vs Forecast charts for companies and metrics."""
        from analysis.visualization import plot_actual_vs_forecast  # only needed for charts

        for _, f_row in forecast_df.iterrows():
            comp = str(f_row["company"])
            metric = str(f_row["metric"])
            f_yr = int(f_row["forecast_year"])
            if pd.isna(f_row["forecast_value"]):
                continue
            f_val = float(f_row["forecast_value"])

            comp_data = norm_df[norm_df["company"] == comp].sort_values("year")
            h_years = comp_data["year"].tolist()
            if metric in comp_data.columns:
                h_actuals = comp_data[metric].tolist()
            elif metric == "cost_of_revenue" and "expenses" in comp_data.columns:
                h_actuals = comp_data["expenses"].tolist()
            elif metric == "net_profit_margin" and "margin" in comp_data.columns:
                h_actuals = comp_data["margin"].tolist()
            else:
                continue

            # Backtest predictions
            b_data = deviations_df[
                (deviations_df["company"] == comp) & (deviations_df["metric"] == metric)
            ].sort_values("year")
            b_years = b_data["year"].tolist() if not b_data.empty else None
            b_preds = b_data["forecast"].tolist() if not b_data.empty else None

            chart_filename = f"forecast_{comp}_{metric}.png"
            chart_path = os.path.join(charts_dir, chart_filename)
            plot_actual_vs_forecast(
                company=comp,
                metric=metric,
                historical_years=h_years,
                historical_actuals=h_actuals,
                forecast_year=f_yr,
                forecast_value=f_val,
                output_path=chart_path,
                backtest_years=b_years,
                backtest_preds=b_preds,
            )
