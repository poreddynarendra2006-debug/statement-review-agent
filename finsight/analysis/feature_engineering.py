"""Dynamic Feature Engineering layer for FinSight AI Anomaly Detection Agent.

Dynamically inspects any validated financial dataset, computes applicable derived
financial metrics ONLY when required source fields exist, calculates temporal
trajectories when multi-period panels exist, prunes uninformative/constant fields,
and maintains full analytical provenance.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from finsight.analysis.dataset_profiler import METADATA_IGNORE_NAMES, profile_dataset
from finsight.core.config import FeatureConfig
from finsight.core.models import DatasetProfile, FeatureMetadata, FinancialRecord

logger = logging.getLogger(__name__)


def _clean_str_num(val: Any) -> Any:
    """Clean string currency/percentage formats into numeric values."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip()
        if cleaned == "" or cleaned.lower() in ("nan", "none", "null", "n/a", "-", "inf", "-inf"):
            return np.nan
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return np.nan


def _safe_divide(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    fill_value: float = np.nan,
    eps: float = 1e-6,
) -> pd.Series | float:
    """Perform division safely handling zeros and infinite results."""
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        if abs(denominator) < eps:
            return fill_value
        return float(numerator / denominator)
    num = pd.Series(numerator, dtype=float)
    den = pd.Series(denominator, dtype=float)

    is_zero = den.abs() < eps
    with np.errstate(divide="ignore", invalid="ignore"):
        res = num / den
        res = res.mask(is_zero, fill_value)
    return res


PERCENT_RATIOS = {"roi", "roe", "roa", "return_on_tangible_equity", "gross_margin", "net_profit_margin", "operating_margin", "ebitda_margin"}

# Raw columns whose names mark them as already scale-free (ratios, margins, rates, growth)
SCALE_FREE_NAME_PARTS = (
    "margin", "ratio", "roe", "roa", "roi", "rate", "percent", "pct", "growth", "yield",
    "turnover", "change", "spread", "divergence", "return", "coverage", "days", "_to_",
)
# Fewest scale-free features needed before raw amounts are left out of a cross-company comparison
MIN_SCALE_FREE_FEATURES = 3
RAW_LEVEL_REASON = "raw_company_size_level (cross-company review compares ratios, margins and growth instead)"


def is_scale_free_name(column: str) -> bool:
    """True when a column name says it holds a ratio, margin, rate or growth rather than an amount."""
    name = re.sub(r"[^a-zA-Z0-9%]+", "_", str(column).strip().lower())
    return "%" in name or any(part in name for part in SCALE_FREE_NAME_PARTS)

# Standard dictionary of canonical financial feature formulas
CANONICAL_RATIO_SPECS = [
    {
        "name": "gross_margin",
        "sources": [("gross_profit", "revenue"), ("gross_margin",)],
        "formula": "gross_profit / revenue",
        "category": "profitability",
    },
    {
        "name": "net_profit_margin",
        "sources": [("net_income", "revenue"), ("net_profit_margin",), ("net_margin",)],
        "formula": "net_income / revenue",
        "category": "profitability",
    },
    {
        "name": "operating_margin",
        "sources": [("operating_income", "revenue"), ("operating_margin",)],
        "formula": "operating_income / revenue",
        "category": "profitability",
    },
    {
        "name": "ebitda_margin",
        "sources": [("ebitda", "revenue"), ("ebitda_margin",)],
        "formula": "ebitda / revenue",
        "category": "profitability",
    },
    {
        "name": "roe",
        "sources": [("net_income", "shareholder_equity"), ("roe",)],
        "formula": "net_income / shareholder_equity",
        "category": "profitability",
    },
    {
        "name": "roa",
        "sources": [("net_income", "total_assets"), ("roa",)],
        "formula": "net_income / total_assets",
        "category": "profitability",
    },
    {
        "name": "roi",
        "sources": [("roi",), ("return_on_investment",)],
        "formula": "reported ROI",
        "category": "profitability",
    },
    {
        "name": "return_on_tangible_equity",
        "sources": [("return_on_tangible_equity",), ("rote",)],
        "formula": "reported Return on Tangible Equity",
        "category": "profitability",
    },
    {
        "name": "debt_equity_ratio",
        "sources": [("total_liabilities", "shareholder_equity"), ("debt", "shareholder_equity"), ("debt_to_equity",), ("debt_equity_ratio",)],
        "formula": "total_debt / shareholder_equity",
        "category": "leverage",
    },
    {
        "name": "liabilities_to_assets",
        "sources": [("total_liabilities", "total_assets")],
        "formula": "total_liabilities / total_assets",
        "category": "leverage",
    },
    {
        "name": "asset_turnover",
        "sources": [("revenue", "total_assets")],
        "formula": "revenue / total_assets",
        "category": "efficiency",
    },
    {
        "name": "current_ratio",
        "sources": [("current_assets", "current_liabilities"), ("current_ratio",)],
        "formula": "current_assets / current_liabilities",
        "category": "liquidity",
    },
    {
        "name": "quick_ratio",
        "sources": [("quick_ratio",)],
        "formula": "quick_assets / current_liabilities",
        "category": "liquidity",
    },
    {
        "name": "operating_cash_flow_to_net_income",
        "sources": [("operating_cash_flow", "net_income")],
        "formula": "operating_cash_flow / net_income",
        "category": "cash_flow",
    },
    {
        "name": "cash_flow_to_revenue",
        "sources": [("operating_cash_flow", "revenue")],
        "formula": "operating_cash_flow / revenue",
        "category": "cash_flow",
    },
    {
        "name": "free_cash_flow_per_share",
        "sources": [("free_cash_flow_per_share",), ("fcf_per_share",)],
        "formula": "reported Free Cash Flow per Share",
        "category": "cash_flow",
    },
]


def extract_dynamic_features(
    data: Union[pd.DataFrame, Sequence[Union[FinancialRecord, dict[str, Any]]]],
    config: Optional[FeatureConfig] = None,
    profile: Optional[DatasetProfile] = None,
    standard_features_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Dynamically engineer financial features from any arbitrary financial dataset.

    Returns
    -------
    metadata_df : pd.DataFrame
        Non-ML context fields (company, year, record_id, category, etc.)
    feature_df : pd.DataFrame
        Numeric feature matrix ready for transformation and modeling.
    active_features : list[str]
        List of feature column names selected for anomaly detection.
    provenance : dict[str, Any]
        Audit dictionary of raw, derived, candidate, used, and ignored features.
    """
    cfg = config or FeatureConfig()

    # 1. Convert to DataFrame
    if isinstance(data, pd.DataFrame):
        df_raw = data.copy()
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return pd.DataFrame(), pd.DataFrame(), [], {
                "raw_columns": [], "available": [], "derived": [], "candidates": [], "used": [], "ignored": {}
            }
        first = data[0]
        if isinstance(first, FinancialRecord):
            df_raw = pd.DataFrame([r.to_dict() for r in data])
        else:
            df_raw = pd.DataFrame(data)
    else:
        df_raw = pd.DataFrame(data)

    if profile is None:
        profile = profile_dataset(df_raw)

    raw_columns = list(df_raw.columns)
    ignored_features: dict[str, str] = {}
    derived_features: list[str] = []

    # Build canonical column lookup (lowercased, stripped, punctuation cleaned)
    canon_map: dict[str, str] = {}
    for col in raw_columns:
        norm = re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower()).strip("_")
        canon_map[norm] = col

    def has_field(f_name: str) -> bool:
        if f_name in df_raw.columns:
            return True
        norm = re.sub(r"[^a-zA-Z0-9]+", "_", f_name.strip().lower()).strip("_")
        return norm in canon_map

    def get_series(f_name: str) -> pd.Series:
        if f_name in df_raw.columns:
            raw_s = df_raw[f_name]
        else:
            norm = re.sub(r"[^a-zA-Z0-9]+", "_", f_name.strip().lower()).strip("_")
            raw_s = df_raw[canon_map[norm]]
        return raw_s.map(_clean_str_num).astype(float)

    # 2. Extract Metadata DataFrame
    metadata_cols = {}
    if profile.entity_column and profile.entity_column in df_raw.columns:
        metadata_cols["company"] = df_raw[profile.entity_column]
    elif "company" in df_raw.columns:
        metadata_cols["company"] = df_raw["company"]
    else:
        metadata_cols["company"] = pd.Series([None] * len(df_raw), index=df_raw.index)

    if profile.year_column and profile.year_column in df_raw.columns:
        metadata_cols["year"] = pd.to_numeric(df_raw[profile.year_column], errors="coerce")
    elif "year" in df_raw.columns:
        metadata_cols["year"] = pd.to_numeric(df_raw["year"], errors="coerce")
    else:
        metadata_cols["year"] = pd.Series([None] * len(df_raw), index=df_raw.index)

    for g_col in profile.grouping_columns:
        if g_col in df_raw.columns and g_col not in metadata_cols:
            metadata_cols[g_col] = df_raw[g_col]

    if "record_id" in df_raw.columns:
        metadata_cols["record_id"] = df_raw["record_id"]
    if "category" in df_raw.columns and "category" not in metadata_cols:
        metadata_cols["category"] = df_raw["category"]

    metadata_df = pd.DataFrame(metadata_cols, index=df_raw.index)

    # 3. Dynamic Derived Ratios
    feature_dict: dict[str, pd.Series] = {}

    for spec in CANONICAL_RATIO_SPECS:
        r_name = spec["name"]
        computed = False

        for option in spec["sources"]:
            if len(option) == 2:
                n_field, d_field = option
                if has_field(n_field) and has_field(d_field):
                    n_s = get_series(n_field)
                    d_s = get_series(d_field)
                    # Check if they have valid numbers
                    if n_s.notna().sum() > 0 and d_s.notna().sum() > 0:
                        ratio_s = _safe_divide(n_s, d_s)
                        feature_dict[r_name] = ratio_s
                        derived_features.append(r_name)
                        computed = True
                        break
            elif len(option) == 1:
                single_field = option[0]
                if has_field(single_field):
                    s = get_series(single_field)
                    if r_name in PERCENT_RATIOS and s.abs().median() > 1.0:
                        s = s / 100.0  # reported in percent -> fraction like the derived ratios
                    if s.notna().sum() > 0:
                        feature_dict[r_name] = s
                        derived_features.append(r_name)
                        computed = True
                        break
        if not computed:
            pass  # Source fields unavailable - will NOT compute or fabricate!

    # 4. Temporal YoY Metrics (Computed ONLY when multi-period panel exists)
    if profile.has_temporal_ordering and len(df_raw) > 1:
        entity_s = metadata_df["company"] if metadata_df["company"].notna().any() else pd.Series(["all"] * len(df_raw), index=df_raw.index)
        year_s = metadata_df["year"]

        order = pd.DataFrame({"entity": entity_s, "year": year_s}, index=df_raw.index).sort_values(["entity", "year"], kind="stable").index

        def previous(series: pd.Series) -> pd.Series:
            ordered = series.loc[order]
            return ordered.groupby(entity_s.loc[order]).shift(1).reindex(series.index)

        prev_year = previous(year_s)
        is_consecutive_period = (year_s - prev_year).abs() <= 2  # consecutive or near-consecutive fiscal period

        # Core growth metrics
        growth_fields = [
            ("revenue", "revenue_growth"),
            ("gross_profit", "gross_profit_growth"),
            ("net_income", "net_income_growth"),
            ("ebitda", "ebitda_growth"),
            ("operating_cash_flow", "operating_cash_flow_growth"),
        ]
        for src_field, g_name in growth_fields:
            if has_field(src_field):
                cur_val = get_series(src_field)
                prev_val = previous(cur_val)
                growth = _safe_divide(cur_val - prev_val, prev_val.abs())
                growth.loc[~is_consecutive_period] = np.nan
                feature_dict[g_name] = growth
                derived_features.append(g_name)

        # YoY Margin & Ratio changes
        change_fields = [
            ("gross_margin", "gross_margin_change"),
            ("net_profit_margin", "net_margin_change"),
            ("ebitda_margin", "ebitda_margin_change"),
            ("operating_margin", "operating_margin_change"),
            ("roe", "roe_change"),
            ("roa", "roa_change"),
            ("roi", "roi_change"),
            ("current_ratio", "current_ratio_change"),
            ("debt_equity_ratio", "debt_equity_change"),
        ]
        for r_field, c_name in change_fields:
            if r_field in feature_dict:
                cur_r = feature_dict[r_field]
                prev_r = previous(cur_r)
                diff = cur_r - prev_r
                diff.loc[~is_consecutive_period] = np.nan
                feature_dict[c_name] = diff
                derived_features.append(c_name)

        # Spreads
        if "revenue_growth" in feature_dict and "net_income_growth" in feature_dict:
            spread = feature_dict["net_income_growth"] - feature_dict["revenue_growth"]
            spread.loc[~is_consecutive_period] = np.nan
            feature_dict["revenue_profit_growth_spread"] = spread
            derived_features.append("revenue_profit_growth_spread")

        if "operating_cash_flow_growth" in feature_dict and "net_income_growth" in feature_dict:
            ocf_spread = feature_dict["operating_cash_flow_growth"] - feature_dict["net_income_growth"]
            ocf_spread.loc[~is_consecutive_period] = np.nan
            feature_dict["ocf_net_income_divergence"] = ocf_spread
            derived_features.append("ocf_net_income_divergence")

    # 5. Include Raw Numerical Features
    # If not standard_features_only, include all meaningful numeric columns
    excluded_meta_keys = {
        "company", "year", "record_id", "category", "industry", "sector",
        "entity", "ticker", "date", "description", "notes", "name", "id", "period",
        "market_cap", "market_cap_in_b_usd", "number_of_employees", "employees",
        "inflation_rate", "inflation", "ground_truth", "label", "target", "planted", "status",
    }

    if not standard_features_only:
        for col in raw_columns:
            c_norm = re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower()).strip("_")
            if c_norm in METADATA_IGNORE_NAMES or c_norm in excluded_meta_keys:
                ignored_features[col] = "metadata_identifier_field"
                continue

            if col in feature_dict or c_norm in feature_dict:
                continue

            # Alias deduplication guards
            if c_norm in ("debt_to_equity", "debt_equity", "d_e") and "debt_equity_ratio" in feature_dict:
                ignored_features[col] = "raw_alias_represented_by_debt_equity_ratio"
                continue
            if c_norm in ("gross_margin_percent", "gross_profit_margin", "gp_margin") and "gross_margin" in feature_dict:
                ignored_features[col] = "raw_alias_represented_by_gross_margin"
                continue
            if c_norm in ("net_margin", "net_profit_margin_percent", "npm") and "net_profit_margin" in feature_dict:
                ignored_features[col] = "raw_alias_represented_by_net_profit_margin"
                continue
            if c_norm in ("current_ratio_value",) and "current_ratio" in feature_dict:
                ignored_features[col] = "raw_alias_represented_by_current_ratio"
                continue

            s = get_series(col)
            valid_count = s.notna().sum()
            if valid_count > 0:
                # Retain raw numerical metric
                feature_dict[col] = s
            else:
                ignored_features[col] = "non_numeric_categorical_field"
    else:
        for col in raw_columns:
            if col not in feature_dict:
                ignored_features[col] = "raw_level_metric"

    # Assemble candidate feature DataFrame
    feature_df = pd.DataFrame(feature_dict, index=df_raw.index)
    candidate_cols = list(feature_df.columns)

    # 6. Quality Filtering & Pruning (Constants, Missingness)
    retained_features: list[str] = []

    for col in candidate_cols:
        series = feature_df[col].replace([np.inf, -np.inf], np.nan)
        valid = series.dropna()

        if len(valid) == 0:
            if len(df_raw) == 1:
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

        # Check for constant zero-variance features on datasets with >= 5 rows
        if len(valid) >= 5 and valid.nunique() == 1:
            ignored_features[col] = "constant_zero_variance"
            feature_df.drop(columns=[col], inplace=True, errors="ignore")
            continue

        retained_features.append(col)

    active_features = list(dict.fromkeys(retained_features))

    # 7. Scale neutrality. Across several companies a large revenue or equity amount
    # only says a company is big, so compare them on ratios, margins and growth.
    # Raw levels stay for single-company data (within-entity review over time) and
    # when too few scale-free features exist to compare on.
    derived_set = set(derived_features)
    raw_levels = [c for c in active_features if c not in derived_set and not is_scale_free_name(c)]
    scale_free = [c for c in active_features if c not in raw_levels]
    if (
        not standard_features_only
        and raw_levels
        and profile.unique_entities_count >= 2
        and len(scale_free) >= MIN_SCALE_FREE_FEATURES
    ):
        for col in raw_levels:
            ignored_features[col] = RAW_LEVEL_REASON
        active_features = scale_free

    feature_df = feature_df[active_features] if active_features else pd.DataFrame(index=df_raw.index)

    provenance = {
        "raw_columns": raw_columns,
        "available": raw_columns,
        "derived": list(dict.fromkeys(derived_features)),
        "candidates": candidate_cols,
        "used": active_features,
        "ignored": ignored_features,
    }

    return metadata_df, feature_df, active_features, provenance


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible Transformer that cleans, imputes, and scales financial features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names_: list[str] = []
        self.impute_values_: dict[str, float] = {}
        self.scale_medians_: dict[str, float] = {}
        self.scale_iqrs_: dict[str, float] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None) -> FeaturePipeline:
        """Fit the preprocessor on raw numeric feature matrix."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame as input to FeaturePipeline.fit")

        self.feature_names_ = list(X.columns)
        self.impute_values_ = {}
        self.scale_medians_ = {}
        self.scale_iqrs_ = {}

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
                q75 = float(valid.quantile(0.75))
                q25 = float(valid.quantile(0.25))
                iqr = q75 - q25
                iqr_val = iqr if iqr >= 1e-6 else 1.0
            else:
                val = 0.0
                iqr_val = 1.0
            self.impute_values_[col] = val
            self.scale_medians_[col] = val
            self.scale_iqrs_[col] = iqr_val

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new feature matrix using fitted imputation values."""
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if len(self.feature_names_) == 0:
            return np.empty((len(X), 0), dtype=np.float64)

        X_out = pd.DataFrame(index=X.index)

        for col in self.feature_names_:
            impute_val = self.impute_values_.get(col, 0.0)
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce")
                series = series.fillna(impute_val)
                series = series.replace(np.inf, 50.0).replace(-np.inf, -50.0)
            else:
                series = pd.Series([impute_val] * len(X), index=X.index)
            X_out[col] = series

        X_arr = np.nan_to_num(X_out.to_numpy(dtype=np.float64), nan=0.0, posinf=50.0, neginf=-50.0)

        # Scaler transformation if enabled
        if self.config.scaler == "robust":
            medians = np.array([self.scale_medians_.get(c, 0.0) for c in self.feature_names_])
            iqrs = np.array([self.scale_iqrs_.get(c, 1.0) for c in self.feature_names_])
            X_arr = (X_arr - medians) / iqrs
            X_arr = np.clip(X_arr, -20.0, 20.0)

        return np.nan_to_num(X_arr, nan=0.0, posinf=20.0, neginf=-20.0)
