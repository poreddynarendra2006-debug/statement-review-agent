"""Data mapping, normalization, and validation layer for FinSight Trend Agent.

This module handles heterogeneous datasets by:
1. Cleaning and normalizing arbitrary column names (e.g., trailing whitespace like 'Company ').
2. Mapping headers to canonical internal keys using alias dictionaries.
3. Deriving proxy metrics (e.g. Cost of Revenue = Revenue - Gross Profit) when explicit fields are missing.
4. Validating data integrity (types, duplicates, missingness, minimum history).
5. Producing a transparent DataMappingReport.
"""

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import COLUMN_ALIASES, MIN_OBSERVATIONS_FOR_FORECAST

logger = logging.getLogger(__name__)


def clean_header_name(header: str) -> str:
    """Clean a raw column header string.
    
    Strips leading/trailing whitespace, converts to lowercase, replaces multiple
    spaces or underscores with single underscores, and strips punctuation.
    """
    if not isinstance(header, str):
        header = str(header)
    cleaned = header.strip().lower()
    # Normalize slashes, dashes, spaces to underscore
    cleaned = re.sub(r"[\s\-\\\/]+", "_", cleaned)
    # Remove any character not alphanumeric or underscore
    cleaned = re.sub(r"[^\w]", "", cleaned)
    # Remove consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


@dataclass
class DataMappingReport:
    """Report detailing column mapping, derivations, and validation findings."""
    source_file: str
    total_rows: int
    total_columns: int
    raw_columns: List[str]
    mapped_fields: Dict[str, str] = field(default_factory=dict)
    unmapped_fields: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    derived_fields: List[Dict[str, str]] = field(default_factory=list)
    unsupported_metrics: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    companies_found: List[str] = field(default_factory=list)
    year_range: Dict[str, Optional[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "raw_columns": self.raw_columns,
            "mapped_fields": self.mapped_fields,
            "unmapped_fields": self.unmapped_fields,
            "missing_required": self.missing_required,
            "derived_fields": self.derived_fields,
            "unsupported_metrics": self.unsupported_metrics,
            "warnings": self.warnings,
            "companies_found": self.companies_found,
            "year_range": self.year_range,
        }


class DataMapper:
    """Normalizes, maps, and validates input financial data."""

    def __init__(self, alias_dict: Optional[Dict[str, List[str]]] = None):
        self.alias_dict = alias_dict or COLUMN_ALIASES

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load financial dataset from CSV or Excel (.xlsx, .xls)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        _, ext = os.path.splitext(file_path.lower())
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext in [".csv", ".txt"]:
            # Try reading with default utf-8, fallback to latin-1
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Expected CSV or Excel (.xlsx/.xls).")

        return df

    def match_column(self, raw_columns: List[str], canonical_key: str) -> Optional[str]:
        """Find best matching raw column for a canonical key using aliases."""
        aliases = self.alias_dict.get(canonical_key, [])
        cleaned_aliases = [clean_header_name(a) for a in aliases]

        for raw_col in raw_columns:
            cleaned_col = clean_header_name(raw_col)
            # Exact match on cleaned name or cleaned alias
            if cleaned_col == canonical_key or cleaned_col in cleaned_aliases:
                return raw_col

        # Fallback substring match for specific compound headers like 'market_cap_in_b_usd'
        for raw_col in raw_columns:
            cleaned_col = clean_header_name(raw_col)
            for alias in cleaned_aliases:
                if alias == cleaned_col or alias in cleaned_col:
                    return raw_col

        return None

    def map_and_validate(self, df: pd.DataFrame, source_file: str = "unknown") -> Tuple[pd.DataFrame, DataMappingReport]:
        """Map raw columns, derive proxy metrics, validate data, and return normalized DataFrame."""
        raw_columns = list(df.columns)
        report = DataMappingReport(
            source_file=source_file,
            total_rows=len(df),
            total_columns=len(raw_columns),
            raw_columns=raw_columns,
        )

        mapped_to_raw: Dict[str, str] = {}
        used_raw_cols = set()

        # Pass 1: exact canonical/alias matches. This prevents a broad alias such as
        # "expenses" from stealing a more specific field such as "operating_expenses".
        for canonical_key in self.alias_dict.keys():
            aliases = self.alias_dict.get(canonical_key, [])
            accepted = {clean_header_name(canonical_key), *(clean_header_name(a) for a in aliases)}
            for raw_col in raw_columns:
                if raw_col in used_raw_cols:
                    continue
                if clean_header_name(raw_col) in accepted:
                    mapped_to_raw[canonical_key] = raw_col
                    used_raw_cols.add(raw_col)
                    break

        # Pass 2: controlled substring fallback for headers such as "Revenue (USD)".
        for canonical_key in self.alias_dict.keys():
            if canonical_key in mapped_to_raw:
                continue
            match = self.match_column([c for c in raw_columns if c not in used_raw_cols], canonical_key)
            if match:
                mapped_to_raw[canonical_key] = match
                used_raw_cols.add(match)

        report.mapped_fields = mapped_to_raw
        report.unmapped_fields = [c for c in raw_columns if c not in used_raw_cols]

        # 1. Validate required fields: company and year
        if "company" not in mapped_to_raw:
            report.missing_required.append("company")
        if "year" not in mapped_to_raw:
            report.missing_required.append("year")

        if report.missing_required:
            raise ValueError(
                f"Dataset missing required primary keys: {report.missing_required}. "
                f"Available columns: {raw_columns}"
            )

        # 2. Construct working normalized DataFrame
        norm_df = pd.DataFrame()
        norm_df["company"] = df[mapped_to_raw["company"]].astype(str).str.strip()
        
        # Clean and convert Year. Accept numeric years, fiscal-year labels, and date columns.
        year_source = df[mapped_to_raw["year"]]
        year_raw = pd.to_numeric(year_source, errors="coerce")
        unresolved = year_raw.isna()
        if unresolved.any():
            # Try datetime parsing for values such as 2024-12-31 / 31-03-2024.
            parsed_dates = pd.to_datetime(year_source, errors="coerce")
            year_raw = year_raw.where(~unresolved, parsed_dates.dt.year)
        unresolved = year_raw.isna()
        if unresolved.any():
            # Finally support labels such as FY2024 / Year Ending 2024.
            extracted = year_source.astype(str).str.extract(r"(\d{4})", expand=False)
            extracted_year = pd.to_numeric(extracted, errors="coerce")
            year_raw = year_raw.where(~unresolved, extracted_year)
        if year_raw.isna().any():
            invalid_count = year_raw.isna().sum()
            report.warnings.append(f"Found {invalid_count} non-numeric/unparseable year values; rows coerced or dropped.")
        norm_df["year"] = year_raw

        # Drop rows where company or year is NaN
        initial_len = len(norm_df)
        valid_mask = norm_df["company"].ne("") & norm_df["company"].ne("nan") & norm_df["year"].notna()
        norm_df = norm_df[valid_mask].copy()
        norm_df["year"] = norm_df["year"].astype(int)

        if len(norm_df) < initial_len:
            dropped = initial_len - len(norm_df)
            report.warnings.append(f"Dropped {dropped} rows with missing company or year.")

        # 3. Handle and normalize financial metrics
        for canonical_key, raw_col in mapped_to_raw.items():
            if canonical_key in ["company", "year"]:
                continue
            # Convert financial figures to numeric float
            norm_df[canonical_key] = pd.to_numeric(df.loc[norm_df.index, raw_col], errors="coerce")

        # 4. Derived metrics logic
        # A. Expenses / Cost of Revenue proxy
        if "expenses" not in norm_df.columns and "cost_of_revenue" not in norm_df.columns:
            if "revenue" in norm_df.columns and "gross_profit" in norm_df.columns:
                # Cost of Revenue = Revenue - Gross Profit
                derived_cor = norm_df["revenue"] - norm_df["gross_profit"]
                norm_df["expenses"] = derived_cor
                norm_df["cost_of_revenue"] = derived_cor
                report.derived_fields.append({
                    "metric": "expenses",
                    "label": "Cost of Revenue (Derived Proxy)",
                    "formula": "Revenue - Gross Profit",
                    "reason": "Explicit 'Expenses' column was absent in dataset. Derived Cost of Revenue as proxy.",
                })
            else:
                report.unsupported_metrics.append({
                    "metric": "expenses",
                    "reason": "Explicit 'Expenses' column absent and cannot derive Cost of Revenue (missing Revenue or Gross Profit).",
                })
        else:
            if "cost_of_revenue" not in norm_df.columns and "expenses" in norm_df.columns:
                norm_df["cost_of_revenue"] = norm_df["expenses"]
            if "expenses" not in norm_df.columns and "cost_of_revenue" in norm_df.columns:
                norm_df["expenses"] = norm_df["cost_of_revenue"]

        # B. Margin / Net Profit Margin proxy
        if "margin" not in norm_df.columns and "net_profit_margin" not in norm_df.columns:
            if "net_income" in norm_df.columns and "revenue" in norm_df.columns:
                # Net Profit Margin (%) = (Net Income / Revenue) * 100
                with np.errstate(divide="ignore", invalid="ignore"):
                    margin_calc = np.where(
                        norm_df["revenue"] != 0,
                        (norm_df["net_income"] / norm_df["revenue"]) * 100.0,
                        np.nan,
                    )
                norm_df["margin"] = margin_calc
                norm_df["net_profit_margin"] = margin_calc
                report.derived_fields.append({
                    "metric": "margin",
                    "label": "Net Profit Margin % (Derived)",
                    "formula": "(Net Income / Revenue) * 100",
                    "reason": "Explicit 'Margin' column was absent in dataset. Derived Net Profit Margin %.",
                })
            else:
                report.unsupported_metrics.append({
                    "metric": "margin",
                    "reason": "Explicit 'Margin' column absent and cannot derive (missing Revenue or Net Income).",
                })
        else:
            if "net_profit_margin" not in norm_df.columns and "margin" in norm_df.columns:
                norm_df["net_profit_margin"] = norm_df["margin"]
            if "margin" not in norm_df.columns and "net_profit_margin" in norm_df.columns:
                norm_df["margin"] = norm_df["net_profit_margin"]

        # 5. Check for duplicate (company, year) pairs
        dup_mask = norm_df.duplicated(subset=["company", "year"], keep=False)
        if dup_mask.any():
            dup_count = dup_mask.sum()
            dup_examples = norm_df[dup_mask][["company", "year"]].drop_duplicates().to_dict(orient="records")
            report.warnings.append(
                f"Detected {dup_count} duplicate company-year records. "
                f"Deterministic handling: keeping the last observed record per (company, year). "
                f"Examples: {dup_examples[:3]}"
            )
            norm_df = norm_df.drop_duplicates(subset=["company", "year"], keep="last")

        # 6. Chronological sorting
        norm_df = norm_df.sort_values(by=["company", "year"], ascending=[True, True]).reset_index(drop=True)

        # 7. Check historical observation count per company
        companies = norm_df["company"].unique().tolist()
        report.companies_found = companies
        if not norm_df.empty:
            report.year_range = {
                "min_year": int(norm_df["year"].min()),
                "max_year": int(norm_df["year"].max()),
            }

        for comp in companies:
            comp_years = norm_df[norm_df["company"] == comp]["year"].count()
            if comp_years < MIN_OBSERVATIONS_FOR_FORECAST:
                report.warnings.append(
                    f"Company '{comp}' has only {comp_years} observation(s); "
                    f"minimum of {MIN_OBSERVATIONS_FOR_FORECAST} required for trend forecasting."
                )

        return norm_df, report
