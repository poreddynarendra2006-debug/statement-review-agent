"""Feature engineering layer for FinSight AI Anomaly Detection Agent.

Supports dynamic feature discovery, ratio computation, temporal panel growth metrics,
and robust feature pruning for any validated financial dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from finsight.analysis.schema_mapper import CANONICAL_FIELDS
from finsight.core.config import FeatureConfig
from finsight.core.models import FeatureMetadata, FinancialRecord

logger = logging.getLogger(__name__)


# Registry of supported financial features grounded in actual financial statement fields
FEATURE_REGISTRY: dict[str, FeatureMetadata] = {
    # 1. Growth Features (YoY relative changes - requires temporal ordering)
    "revenue_growth": FeatureMetadata(
        name="revenue_growth",
        source_columns=["revenue"],
        formula="(revenue_t - revenue_{t-1}) / |revenue_{t-1}|",
        description="Year-on-year relative change in top-line revenue",
        category="growth",
        requires_history=True,
    ),
    "gross_profit_growth": FeatureMetadata(
        name="gross_profit_growth",
        source_columns=["gross_profit"],
        formula="(gross_profit_t - gross_profit_{t-1}) / |gross_profit_{t-1}|",
        description="Year-on-year relative change in gross profit",
        category="growth",
        requires_history=True,
    ),
    "net_income_growth": FeatureMetadata(
        name="net_income_growth",
        source_columns=["net_income"],
        formula="(net_income_t - net_income_{t-1}) / |net_income_{t-1}|",
        description="Year-on-year relative change in bottom-line net income",
        category="growth",
        requires_history=True,
    ),
    "ebitda_growth": FeatureMetadata(
        name="ebitda_growth",
        source_columns=["ebitda"],
        formula="(ebitda_t - ebitda_{t-1}) / |ebitda_{t-1}|",
        description="Year-on-year relative change in EBITDA",
        category="growth",
        requires_history=True,
    ),
    "operating_cash_flow_growth": FeatureMetadata(
        name="operating_cash_flow_growth",
        source_columns=["operating_cash_flow"],
        formula="(ocf_t - ocf_{t-1}) / |ocf_{t-1}|",
        description="Year-on-year relative change in Operating Cash Flow",
        category="growth",
        requires_history=True,
    ),
    # 2. Profitability & Efficiency Ratios (Static / Cross-sectional)
    "gross_margin": FeatureMetadata(
        name="gross_margin",
        source_columns=["gross_profit", "revenue", "gross_margin"],
        formula="gross_profit / revenue (or reported Gross Margin)",
        description="Gross Profit Margin measuring direct cost efficiency",
        category="profitability",
        requires_history=False,
    ),
    "net_profit_margin": FeatureMetadata(
        name="net_profit_margin",
        source_columns=["net_income", "revenue", "net_profit_margin"],
        formula="net_income / revenue (or reported Net Profit Margin)",
        description="Net Profit Margin measuring bottom-line conversion",
        category="profitability",
        requires_history=False,
    ),
    "ebitda_margin": FeatureMetadata(
        name="ebitda_margin",
        source_columns=["ebitda", "revenue"],
        formula="ebitda / revenue",
        description="EBITDA Margin measuring core operational cash profitability",
        category="profitability",
        requires_history=False,
    ),
    "operating_margin": FeatureMetadata(
        name="operating_margin",
        source_columns=["operating_income", "revenue"],
        formula="operating_income / revenue",
        description="Operating Margin measuring operating efficiency",
        category="profitability",
        requires_history=False,
    ),
    "roe": FeatureMetadata(
        name="roe",
        source_columns=["net_income", "shareholder_equity", "roe"],
        formula="net_income / shareholder_equity (or reported ROE)",
        description="Return on Equity measuring net income generated per unit of equity",
        category="profitability",
        requires_history=False,
    ),
    "roa": FeatureMetadata(
        name="roa",
        source_columns=["net_income", "total_assets", "roa"],
        formula="net_income / total_assets (or reported ROA)",
        description="Return on Assets measuring total asset productivity",
        category="profitability",
        requires_history=False,
    ),
    "roi": FeatureMetadata(
        name="roi",
        source_columns=["roi"],
        formula="reported ROI",
        description="Return on Investment",
        category="profitability",
        requires_history=False,
    ),
    "return_on_tangible_equity": FeatureMetadata(
        name="return_on_tangible_equity",
        source_columns=["return_on_tangible_equity"],
        formula="reported Return on Tangible Equity",
        description="Return on Tangible Equity (ROTE)",
        category="profitability",
        requires_history=False,
    ),
    # 3. Cash Flow & Earnings Quality (Static / Cross-sectional)
    "operating_cash_flow_to_net_income": FeatureMetadata(
        name="operating_cash_flow_to_net_income",
        source_columns=["operating_cash_flow", "net_income"],
        formula="operating_cash_flow / net_income",
        description="Earnings Quality ratio comparing cash generation with accrual earnings",
        category="cash_flow",
        requires_history=False,
    ),
    "cash_flow_to_revenue": FeatureMetadata(
        name="cash_flow_to_revenue",
        source_columns=["operating_cash_flow", "revenue"],
        formula="operating_cash_flow / revenue",
        description="Operating Cash Flow Margin",
        category="cash_flow",
        requires_history=False,
    ),
    "free_cash_flow_per_share": FeatureMetadata(
        name="free_cash_flow_per_share",
        source_columns=["free_cash_flow_per_share"],
        formula="reported Free Cash Flow per Share",
        description="Free cash flow available per common share",
        category="cash_flow",
        requires_history=False,
    ),
    # 4. Leverage & Liquidity (Static / Cross-sectional)
    "current_ratio": FeatureMetadata(
        name="current_ratio",
        source_columns=["current_ratio"],
        formula="reported Current Ratio",
        description="Current assets to current liabilities ratio",
        category="liquidity",
        requires_history=False,
    ),
    "quick_ratio": FeatureMetadata(
        name="quick_ratio",
        source_columns=["quick_ratio"],
        formula="reported Quick Ratio",
        description="Quick / Acid-test liquidity ratio",
        category="liquidity",
        requires_history=False,
    ),
    "working_capital_ratio": FeatureMetadata(
        name="working_capital_ratio",
        source_columns=["working_capital_ratio"],
        formula="reported Working Capital Ratio",
        description="Working Capital Ratio",
        category="liquidity",
        requires_history=False,
    ),
    "debt_equity_ratio": FeatureMetadata(
        name="debt_equity_ratio",
        source_columns=["debt_to_equity", "total_liabilities", "shareholder_equity"],
        formula="reported Debt/Equity Ratio (or Total Liabilities / Equity)",
        description="Total debt or liabilities relative to shareholder equity",
        category="leverage",
        requires_history=False,
    ),
    "liabilities_to_assets": FeatureMetadata(
        name="liabilities_to_assets",
        source_columns=["total_liabilities", "total_assets"],
        formula="total_liabilities / total_assets",
        description="Total Liabilities to Total Assets ratio",
        category="leverage",
        requires_history=False,
    ),
    "asset_turnover": FeatureMetadata(
        name="asset_turnover",
        source_columns=["revenue", "total_assets"],
        formula="revenue / total_assets",
        description="Asset Turnover ratio measuring revenue generation per asset dollar",
        category="efficiency",
        requires_history=False,
    ),
    # 5. YoY Change Features (Historical Shifts - requires temporal ordering)
    "gross_margin_change": FeatureMetadata(
        name="gross_margin_change",
        source_columns=["gross_profit", "revenue", "gross_margin"],
        formula="gross_margin_t - gross_margin_{t-1}",
        description="YoY shift in Gross Margin percentage points",
        category="change",
        requires_history=True,
    ),
    "net_margin_change": FeatureMetadata(
        name="net_margin_change",
        source_columns=["net_income", "revenue", "net_profit_margin"],
        formula="net_profit_margin_t - net_profit_margin_{t-1}",
        description="YoY shift in Net Profit Margin percentage points",
        category="change",
        requires_history=True,
    ),
    "ebitda_margin_change": FeatureMetadata(
        name="ebitda_margin_change",
        source_columns=["ebitda", "revenue"],
        formula="ebitda_margin_t - ebitda_margin_{t-1}",
        description="YoY shift in EBITDA Margin percentage points",
        category="change",
        requires_history=True,
    ),
    "roe_change": FeatureMetadata(
        name="roe_change",
        source_columns=["roe", "net_income", "shareholder_equity"],
        formula="roe_t - roe_{t-1}",
        description="YoY shift in Return on Equity",
        category="change",
        requires_history=True,
    ),
    "roa_change": FeatureMetadata(
        name="roa_change",
        source_columns=["roa"],
        formula="roa_t - roa_{t-1}",
        description="YoY shift in Return on Assets",
        category="change",
        requires_history=True,
    ),
    "roi_change": FeatureMetadata(
        name="roi_change",
        source_columns=["roi"],
        formula="roi_t - roi_{t-1}",
        description="YoY shift in Return on Investment",
        category="change",
        requires_history=True,
    ),
    "current_ratio_change": FeatureMetadata(
        name="current_ratio_change",
        source_columns=["current_ratio"],
        formula="current_ratio_t - current_ratio_{t-1}",
        description="YoY shift in Current Ratio",
        category="change",
        requires_history=True,
    ),
    "debt_equity_change": FeatureMetadata(
        name="debt_equity_change",
        source_columns=["debt_to_equity"],
        formula="debt_equity_ratio_t - debt_equity_ratio_{t-1}",
        description="YoY shift in Debt/Equity Ratio",
        category="change",
        requires_history=True,
    ),
    # 6. Cross-Figure Relationships (requires temporal ordering)
    "revenue_profit_growth_spread": FeatureMetadata(
        name="revenue_profit_growth_spread",
        source_columns=["revenue", "net_income"],
        formula="net_income_growth - revenue_growth",
        description="Spread between Net Income growth and Revenue growth",
        category="cross_figure",
        requires_history=True,
    ),
    "ocf_net_income_divergence": FeatureMetadata(
        name="ocf_net_income_divergence",
        source_columns=["operating_cash_flow", "net_income"],
        formula="operating_cash_flow_growth - net_income_growth",
        description="Growth divergence between cash flow from operations and accounting net income",
        category="cross_figure",
        requires_history=True,
    ),
}


def _safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    eps: float = 1e-9,
) -> Union[pd.Series, np.ndarray, float]:
    """Safely divide two series or arrays, handling zero or near-zero denominators deterministically."""
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        is_zero = denominator.abs() < eps
        with np.errstate(divide="ignore", invalid="ignore"):
            res = numerator / denominator
        res[is_zero] = np.nan
        return res
    else:
        num = np.asarray(numerator, dtype=float)
        den = np.asarray(denominator, dtype=float)
        is_zero = np.abs(den) < eps
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(is_zero, np.nan, num / den)
        if np.ndim(numerator) == 0 and np.ndim(denominator) == 0:
            return float(res)
        return res


def compute_financial_features(
    records: Sequence[Union[FinancialRecord, dict]],
    config: Optional[FeatureConfig] = None,
    return_provenance: bool = False,
    standard_features_only: bool = False,
    raw_columns: Optional[list[str]] = None,
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame, list[str]],
    tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]],
]:
    """Compute financial features dynamically based on available columns, ratios, and temporal ordering.

    Strict rules:
    - If company and year exist, groups by Company and sorts by Year ascending.
    - If no year/date exists, NEVER fakes temporal transitions. Only calculates valid static ratios.
    - Discovers all usable numeric columns dynamically.
    - Filters out non-numeric metadata, zero-variance / constant features, and excessive missingness.
    - Zero denominators and missing values are handled safely without fabricating data.
    - Returns (metadata_df, feature_df, ordered_active_feature_names) or with provenance dict if return_provenance=True.
    """
    cfg = config or FeatureConfig()

    empty_provenance = {
        "raw_columns": [],
        "available": [],
        "derived": [],
        "candidates": [],
        "used": [],
        "ignored": {},
    }

    if not records or len(records) == 0:
        empty_meta = pd.DataFrame(columns=["record_id", "company", "year", "category"])
        empty_feat = pd.DataFrame()
        if return_provenance:
            return empty_meta, empty_feat, [], empty_provenance
        return empty_meta, empty_feat, []

    if isinstance(records, pd.DataFrame):
        df_raw = records.copy()
        raw_columns_available = list(df_raw.columns) if raw_columns is None else raw_columns
    elif isinstance(records[0], FinancialRecord):
        all_rows = []
        present_keys = []
        for r in records:
            row_d = {}
            for k, v in r.__dict__.items():
                if k == "extra_fields" and isinstance(v, dict):
                    for ek, ev in v.items():
                        if ev is not None:
                            row_d[ek] = ev
                            if ek not in present_keys:
                                present_keys.append(ek)
                elif v is not None:
                    row_d[k] = v
                    if k not in present_keys:
                        present_keys.append(k)
            all_rows.append(row_d)
        df_raw = pd.DataFrame(all_rows)
        raw_columns_available = present_keys if raw_columns is None else raw_columns
    else:
        present_keys = []
        for d in records:
            if isinstance(d, dict):
                for k, v in d.items():
                    if v is not None and k not in present_keys:
                        present_keys.append(k)
        df_raw = pd.DataFrame(records)
        raw_columns_available = present_keys if raw_columns is None else raw_columns

    # Clean and deduplicate raw columns
    raw_columns_available = list(dict.fromkeys(raw_columns_available))

    # Check temporal ordering availability
    has_year = "year" in df_raw.columns and pd.to_numeric(df_raw["year"], errors="coerce").fillna(0).gt(0).any()
    has_company = "company" in df_raw.columns and df_raw["company"].astype(str).str.strip().ne("None").ne("").ne("nan").any()
    has_temporal = bool(has_year and has_company)

    if has_temporal and len(df_raw) > 1:
        df_raw["company"] = df_raw["company"].astype(str)
        df_raw["year"] = pd.to_numeric(df_raw["year"], errors="coerce").fillna(0).astype(int)
        df_raw = df_raw.sort_values(by=["company", "year"]).reset_index(drop=True)

    metadata_cols = []
    for col in ["record_id", "company", "year", "category", "industry", "sector"]:
        if col in df_raw.columns:
            metadata_cols.append(col)
    if not metadata_cols:
        metadata_df = pd.DataFrame(index=df_raw.index)
        metadata_df["record_id"] = [f"Record #{i+1}" for i in range(len(df_raw))]
    else:
        metadata_df = df_raw[metadata_cols].copy()
        if "record_id" not in metadata_df.columns:
            metadata_df["record_id"] = [f"Record #{i+1}" for i in range(len(df_raw))]

    feature_df = pd.DataFrame(index=df_raw.index)
    derived_features: list[str] = []
    ignored_features: dict[str, str] = {}

    def has_col(col: str) -> bool:
        return col in df_raw.columns and pd.to_numeric(df_raw[col], errors="coerce").notna().sum() > 0

    has_negative_assets = (
        has_col("total_assets")
        and (pd.to_numeric(df_raw["total_assets"], errors="coerce") < 0).any()
    )
    has_negative_revenue = (
        has_col("revenue")
        and (pd.to_numeric(df_raw["revenue"], errors="coerce") < 0).any()
    )
    is_standardized_input = has_negative_assets or has_negative_revenue

    # 1. Base Profitability & Efficiency Ratios (Static / Cross-sectional)
    if has_col("gross_margin"):
        feature_df["gross_margin"] = pd.to_numeric(df_raw["gross_margin"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("gross_margin")
    elif not is_standardized_input and has_col("revenue") and has_col("gross_profit"):
        feature_df["gross_margin"] = _safe_divide(
            pd.to_numeric(df_raw["gross_profit"], errors="coerce"),
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
        ).clip(cfg.clip_margin_bounds[0], cfg.clip_margin_bounds[1])
        derived_features.append("gross_margin")

    if has_col("net_profit_margin"):
        feature_df["net_profit_margin"] = pd.to_numeric(df_raw["net_profit_margin"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("net_profit_margin")
    elif not is_standardized_input and has_col("revenue") and has_col("net_income"):
        feature_df["net_profit_margin"] = _safe_divide(
            pd.to_numeric(df_raw["net_income"], errors="coerce"),
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
        ).clip(cfg.clip_margin_bounds[0], cfg.clip_margin_bounds[1])
        derived_features.append("net_profit_margin")

    if not is_standardized_input and has_col("revenue") and has_col("ebitda"):
        feature_df["ebitda_margin"] = _safe_divide(
            pd.to_numeric(df_raw["ebitda"], errors="coerce"),
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
        ).clip(cfg.clip_margin_bounds[0], cfg.clip_margin_bounds[1])
        derived_features.append("ebitda_margin")

    if not is_standardized_input and has_col("revenue") and has_col("operating_income"):
        feature_df["operating_margin"] = _safe_divide(
            pd.to_numeric(df_raw["operating_income"], errors="coerce"),
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
        ).clip(cfg.clip_margin_bounds[0], cfg.clip_margin_bounds[1])
        derived_features.append("operating_margin")

    if has_col("roe"):
        feature_df["roe"] = pd.to_numeric(df_raw["roe"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("roe")
    elif not is_standardized_input and has_col("net_income") and has_col("shareholder_equity"):
        feature_df["roe"] = _safe_divide(
            pd.to_numeric(df_raw["net_income"], errors="coerce"),
            pd.to_numeric(df_raw["shareholder_equity"], errors="coerce"),
        ).clip(-10.0, 10.0)
        derived_features.append("roe")

    if has_col("roa"):
        feature_df["roa"] = pd.to_numeric(df_raw["roa"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("roa")
    elif not is_standardized_input and has_col("net_income") and has_col("total_assets"):
        feature_df["roa"] = _safe_divide(
            pd.to_numeric(df_raw["net_income"], errors="coerce"),
            pd.to_numeric(df_raw["total_assets"], errors="coerce"),
        ).clip(-5.0, 5.0)
        derived_features.append("roa")

    if has_col("roi"):
        feature_df["roi"] = pd.to_numeric(df_raw["roi"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("roi")

    if has_col("return_on_tangible_equity"):
        feature_df["return_on_tangible_equity"] = pd.to_numeric(df_raw["return_on_tangible_equity"], errors="coerce").clip(-50.0, 50.0)
        derived_features.append("return_on_tangible_equity")

    # 2. Cash Flow & Earnings Quality (Static / Cross-sectional)
    if not is_standardized_input and has_col("operating_cash_flow") and has_col("net_income"):
        feature_df["operating_cash_flow_to_net_income"] = _safe_divide(
            pd.to_numeric(df_raw["operating_cash_flow"], errors="coerce"),
            pd.to_numeric(df_raw["net_income"], errors="coerce"),
        ).clip(-50.0, 50.0)
        derived_features.append("operating_cash_flow_to_net_income")

    if not is_standardized_input and has_col("operating_cash_flow") and has_col("revenue"):
        feature_df["cash_flow_to_revenue"] = _safe_divide(
            pd.to_numeric(df_raw["operating_cash_flow"], errors="coerce"),
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
        ).clip(cfg.clip_margin_bounds[0], cfg.clip_margin_bounds[1])
        derived_features.append("cash_flow_to_revenue")

    if has_col("free_cash_flow_per_share"):
        feature_df["free_cash_flow_per_share"] = pd.to_numeric(df_raw["free_cash_flow_per_share"], errors="coerce").clip(-500.0, 500.0)
        derived_features.append("free_cash_flow_per_share")

    # 3. Leverage & Liquidity (Static / Cross-sectional)
    if has_col("current_ratio"):
        cr_series = pd.to_numeric(df_raw["current_ratio"], errors="coerce")
        lower_bound = -50.0 if (cr_series < 0).any() else 0.0
        feature_df["current_ratio"] = cr_series.clip(lower_bound, 50.0)
        derived_features.append("current_ratio")

    if has_col("quick_ratio"):
        qr_series = pd.to_numeric(df_raw["quick_ratio"], errors="coerce")
        lower_bound = -50.0 if (qr_series < 0).any() else 0.0
        feature_df["quick_ratio"] = qr_series.clip(lower_bound, 50.0)
        derived_features.append("quick_ratio")

    if has_col("working_capital_ratio"):
        wc_series = pd.to_numeric(df_raw["working_capital_ratio"], errors="coerce")
        feature_df["working_capital_ratio"] = wc_series.clip(-50.0, 50.0)
        derived_features.append("working_capital_ratio")

    if has_col("debt_to_equity"):
        de_series = pd.to_numeric(df_raw["debt_to_equity"], errors="coerce")
        feature_df["debt_equity_ratio"] = de_series.clip(-50.0, 50.0)
        derived_features.append("debt_equity_ratio")
    elif not is_standardized_input and has_col("total_liabilities") and has_col("shareholder_equity"):
        feature_df["debt_equity_ratio"] = _safe_divide(
            pd.to_numeric(df_raw["total_liabilities"], errors="coerce"),
            pd.to_numeric(df_raw["shareholder_equity"], errors="coerce"),
        ).clip(-50.0, 50.0)
        derived_features.append("debt_equity_ratio")

    if not is_standardized_input and has_col("total_liabilities") and has_col("total_assets"):
        feature_df["liabilities_to_assets"] = _safe_divide(
            pd.to_numeric(df_raw["total_liabilities"], errors="coerce"),
            pd.to_numeric(df_raw["total_assets"], errors="coerce"),
        ).clip(0.0, 50.0)
        derived_features.append("liabilities_to_assets")

    if not is_standardized_input and has_col("revenue") and has_col("total_assets"):
        feature_df["asset_turnover"] = _safe_divide(
            pd.to_numeric(df_raw["revenue"], errors="coerce"),
            pd.to_numeric(df_raw["total_assets"], errors="coerce"),
        ).clip(0.0, 50.0)
        derived_features.append("asset_turnover")

    # 4. Historical YoY Shifts (Computed ONLY if temporal ordering exists)
    if has_temporal:
        if len(df_raw) == 1:
            growth_candidates = [
                ("revenue", "revenue_growth"),
                ("gross_profit", "gross_profit_growth"),
                ("net_income", "net_income_growth"),
                ("ebitda", "ebitda_growth"),
                ("operating_cash_flow", "operating_cash_flow_growth"),
            ]
            for col_name, feat_name in growth_candidates:
                if has_col(col_name):
                    feature_df[feat_name] = pd.Series([np.nan], index=df_raw.index)
                    derived_features.append(feat_name)

            change_candidates = [
                ("gross_margin", "gross_margin_change"),
                ("net_profit_margin", "net_margin_change"),
                ("ebitda_margin", "ebitda_margin_change"),
                ("roe", "roe_change"),
                ("roa", "roa_change"),
                ("roi", "roi_change"),
                ("current_ratio", "current_ratio_change"),
                ("debt_equity_ratio", "debt_equity_change"),
            ]
            for base_feat, change_feat in change_candidates:
                if base_feat in feature_df.columns:
                    feature_df[change_feat] = pd.Series([np.nan], index=df_raw.index)
                    derived_features.append(change_feat)
        else:
            grouped = df_raw.groupby("company")
            prev_year = grouped["year"].shift(1)
            is_consecutive_year = (df_raw["year"] - prev_year) == 1

            growth_candidates = [
                ("revenue", "revenue_growth"),
                ("gross_profit", "gross_profit_growth"),
                ("net_income", "net_income_growth"),
                ("ebitda", "ebitda_growth"),
                ("operating_cash_flow", "operating_cash_flow_growth"),
            ]

            for col_name, feat_name in growth_candidates:
                if has_col(col_name):
                    cur_val = pd.to_numeric(df_raw[col_name], errors="coerce")
                    prev_val = pd.to_numeric(grouped[col_name].shift(1), errors="coerce")
                    growth = _safe_divide(cur_val - prev_val, prev_val.abs())
                    growth.loc[~is_consecutive_year] = np.nan
                    growth = growth.clip(lower=cfg.clip_growth_bounds[0], upper=cfg.clip_growth_bounds[1])
                    feature_df[feat_name] = growth
                    derived_features.append(feat_name)

            change_candidates = [
                ("gross_margin", "gross_margin_change"),
                ("net_profit_margin", "net_margin_change"),
                ("ebitda_margin", "ebitda_margin_change"),
                ("roe", "roe_change"),
                ("roa", "roa_change"),
                ("roi", "roi_change"),
                ("current_ratio", "current_ratio_change"),
                ("debt_equity_ratio", "debt_equity_change"),
            ]

            for base_feat, change_feat in change_candidates:
                if base_feat in feature_df.columns:
                    cur_val = feature_df[base_feat]
                    prev_val = feature_df.groupby(df_raw["company"])[base_feat].shift(1)
                    margin_shift = cur_val - prev_val
                    margin_shift.loc[~is_consecutive_year] = np.nan
                    margin_shift = margin_shift.clip(lower=-20.0, upper=20.0)
                    feature_df[change_feat] = margin_shift
                    derived_features.append(change_feat)

            if "revenue_growth" in feature_df.columns and "net_income_growth" in feature_df.columns:
                spread = feature_df["net_income_growth"] - feature_df["revenue_growth"]
                spread.loc[~is_consecutive_year] = np.nan
                feature_df["revenue_profit_growth_spread"] = spread.clip(-50.0, 50.0)
                derived_features.append("revenue_profit_growth_spread")

            if "operating_cash_flow_growth" in feature_df.columns and "net_income_growth" in feature_df.columns:
                ocf_spread = feature_df["operating_cash_flow_growth"] - feature_df["net_income_growth"]
                ocf_spread.loc[~is_consecutive_year] = np.nan
                feature_df["ocf_net_income_divergence"] = ocf_spread.clip(-50.0, 50.0)
                derived_features.append("ocf_net_income_divergence")

    # 5. Include raw numeric columns from df_raw for custom/unmapped datasets
    excluded_meta_names = {
        "company", "year", "record_id", "category", "industry", "sector", "extra_fields",
        "entity", "ticker", "date", "description", "notes", "name", "id", "period",
        "market_cap", "market_cap(in_b_usd)", "market_cap_in_b_usd",
        "number_of_employees", "employees", "headcount",
        "inflation_rate", "inflation",
        "ground_truth", "label", "target", "planted", "status", "financial_status",
    }

    if not standard_features_only:
        for col in raw_columns_available:
            col_clean = str(col).lower().replace(" ", "_")
            if col_clean in excluded_meta_names or str(col).lower() in excluded_meta_names:
                ignored_features[col] = "metadata_identifier_field"
                continue

            if col in feature_df.columns:
                continue

            # Alias deduplication guards
            if col_clean in ("debt_to_equity", "debt_equity", "d_e") and "debt_equity_ratio" in feature_df.columns:
                ignored_features[col] = "raw_alias_represented_by_debt_equity_ratio"
                continue
            if col_clean in ("gross_margin_percent", "gross_profit_margin", "gp_margin") and "gross_margin" in feature_df.columns:
                ignored_features[col] = "raw_alias_represented_by_gross_margin"
                continue
            if col_clean in ("net_margin", "net_profit_margin_percent", "npm") and "net_profit_margin" in feature_df.columns:
                ignored_features[col] = "raw_alias_represented_by_net_profit_margin"
                continue
            if col_clean in ("current_ratio_value",) and "current_ratio" in feature_df.columns:
                ignored_features[col] = "raw_alias_represented_by_current_ratio"
                continue

            if col in df_raw.columns:
                numeric_series = pd.to_numeric(df_raw[col], errors="coerce")
                valid_count = numeric_series.notna().sum()
                if valid_count > 0:
                    if len(derived_features) < 15 or (col not in CANONICAL_FIELDS and col not in FEATURE_REGISTRY):
                        feature_df[col] = numeric_series
                    else:
                        ignored_features[col] = "raw_level_represented_by_ratios"
                else:
                    ignored_features[col] = "non_numeric_categorical_field"
            else:
                ignored_features[col] = "non_numeric_categorical_field"
    else:
        for col in raw_columns_available:
            if col not in feature_df.columns:
                ignored_features[col] = "raw_level_metric"

    # 6. Quality Checks & Pruning: Missingness
    candidate_cols = list(dict.fromkeys(feature_df.columns))
    retained_features: list[str] = []

    for col in candidate_cols:
        series = pd.to_numeric(feature_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = series.dropna()

        if len(valid) == 0:
            if len(df_raw) == 1:
                # Keep in feature_df for 1-row datasets
                retained_features.append(col)
            else:
                ignored_features[col] = "all_values_missing_or_nan"
                feature_df.drop(columns=[col], inplace=True, errors="ignore")
            continue

        missing_frac = series.isna().mean()
        if missing_frac > cfg.max_missing_ratio and len(df_raw) > 1:
            ignored_features[col] = f"excessive_missing_values ({missing_frac*100:.1f}% missing)"
            feature_df.drop(columns=[col], inplace=True, errors="ignore")
            continue

        retained_features.append(col)

    active_features = list(dict.fromkeys(retained_features))

    # Deterministic order
    ordered_features = [f for f in FEATURE_REGISTRY.keys() if f in active_features]
    for f in active_features:
        if f not in ordered_features:
            ordered_features.append(f)
    ordered_features = list(dict.fromkeys(ordered_features))

    feature_df = feature_df[ordered_features] if ordered_features else pd.DataFrame(index=df_raw.index)

    derived_features = list(dict.fromkeys(derived_features))
    candidate_cols = list(dict.fromkeys(candidate_cols))

    provenance = {
        "raw_columns": raw_columns_available,
        "available": raw_columns_available,
        "derived": derived_features,
        "candidates": candidate_cols,
        "used": ordered_features,
        "ignored": ignored_features,
    }

    if return_provenance:
        return metadata_df, feature_df, ordered_features, provenance
    return metadata_df, feature_df, ordered_features


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible Transformer that cleans, imputes, and scales financial features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names_: list[str] = []
        self.impute_values_: dict[str, float] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> FeaturePipeline:
        """Fit the preprocessor on raw numeric feature matrix."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame as input to FeaturePipeline.fit")

        self.feature_names_ = list(X.columns)
        self.impute_values_ = {}

        for col in self.feature_names_:
            series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = series.dropna()
            if len(valid) > 0:
                if self.config.impute_strategy == "median":
                    val = float(valid.median())
                elif self.config.impute_strategy == "mean":
                    val = float(valid.mean())
                else:
                    val = 0.0
            else:
                val = 0.0
            self.impute_values_[col] = val

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw feature matrix into a clean, numeric numpy array ready for ML models."""
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform()")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame as input to FeaturePipeline.transform")

        if len(self.feature_names_) == 0:
            return np.empty((len(X), 0), dtype=np.float64)

        aligned_df = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                aligned_df[col] = series.fillna(self.impute_values_[col])
            else:
                aligned_df[col] = self.impute_values_[col]

        matrix = aligned_df[self.feature_names_].to_numpy(dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_metadata(self) -> list[FeatureMetadata]:
        """Return metadata for all active features in this pipeline."""
        return [
            FEATURE_REGISTRY.get(
                feat,
                FeatureMetadata(
                    name=feat,
                    source_columns=[feat],
                    formula="custom",
                    description=f"User-provided financial feature: {feat}",
                    category="custom",
                ),
            )
            for feat in self.feature_names_
        ]
