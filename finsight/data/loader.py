"""Data loader, multi-format file reader, dataset inspector, and preprocessor."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from finsight.analysis.schema_mapper import (
    CANONICAL_FIELDS,
    DataValidationError,
    SchemaMapper,
    SchemaMappingResult,
    normalize_header,
)
from finsight.core.models import DatasetSummary, FinancialRecord

logger = logging.getLogger(__name__)


@dataclass
class LoadedFinancialData:
    """Standardized result returned by load_financial_file."""

    records: list[FinancialRecord]
    dataframe: pd.DataFrame
    mapping: SchemaMappingResult
    file_type: str
    source_name: str
    sheet_name: Optional[str] = None
    has_temporal_ordering: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def available_features(self) -> list[str]:
        return self.mapping.available_features

    @property
    def missing_fields(self) -> list[str]:
        return self.mapping.missing_optional_fields + self.mapping.missing_required_fields

    def summary_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "file_type": self.file_type,
            "sheet_name": self.sheet_name,
            "num_records": len(self.records),
            "num_columns": len(self.dataframe.columns),
            "has_temporal_ordering": self.has_temporal_ordering,
            "mapping": self.mapping.summary_dict(),
            "warnings": self.warnings + self.mapping.warnings,
        }


def clean_numeric_value(val: Any) -> Optional[float]:
    """Clean and parse numeric financial values handling accounting parentheses, currency, and symbols."""
    if val is None or pd.isna(val):
        return None

    if isinstance(val, (int, float, np.integer, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none", "null", "n/a", "-", "--", "nil", "."):
        return None

    is_negative = False
    if val_str.startswith("(") and val_str.endswith(")"):
        is_negative = True
        val_str = val_str[1:-1].strip()

    cleaned = re.sub(r"[\$\€\£\¥\₹\s\,]", "", val_str)

    if cleaned.endswith("-"):
        is_negative = True
        cleaned = cleaned[:-1].strip()

    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()

    try:
        num = float(cleaned)
        return -num if is_negative else num
    except (ValueError, TypeError):
        return None


def _find_best_excel_sheet(
    excel_file: pd.ExcelFile,
    mapper: SchemaMapper,
    preferred_sheet: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Inspect Excel workbook sheets and select the sheet containing financial tabular data."""
    sheet_names = excel_file.sheet_names
    warnings: list[str] = []

    if not sheet_names:
        raise DataValidationError("Excel workbook contains no sheets.")

    if preferred_sheet:
        if preferred_sheet in sheet_names:
            return preferred_sheet, warnings
        raise DataValidationError(
            f"Specified sheet '{preferred_sheet}' not found. Available sheets: {sheet_names}"
        )

    if len(sheet_names) == 1:
        return sheet_names[0], warnings

    candidate_scores: list[tuple[str, int, int]] = []
    ignored_names = {
        "notes", "note", "instruction", "instructions", "cover", "cover page",
        "readme", "about", "metadata", "index", "toc", "legend", "summary",
    }

    for name in sheet_names:
        norm_name = name.strip().lower()
        if norm_name in ignored_names:
            continue

        try:
            df_sample = pd.read_excel(excel_file, sheet_name=name, nrows=50)
            if df_sample.empty or len(df_sample.columns) < 2:
                continue

            mapping = mapper.map_columns([str(c) for c in df_sample.columns])
            score = len(mapping.available_financial_fields)
            if score > 0:
                candidate_scores.append((name, score, len(df_sample)))
        except Exception as exc:
            logger.debug("Failed reading sample from sheet '%s': %s", name, exc)
            continue

    if not candidate_scores:
        return sheet_names[0], [f"No dedicated financial sheet recognized. Defaulted to '{sheet_names[0]}'."]

    if len(candidate_scores) == 1:
        return candidate_scores[0][0], warnings

    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    chosen_sheet = candidate_scores[0][0]
    all_candidate_names = [c[0] for c in candidate_scores]
    warning_msg = (
        f"Multiple plausible financial sheets identified: {all_candidate_names}. "
        f"Automatically selected '{chosen_sheet}' (highest financial field match). "
        f"Use --sheet-name or provide sheet parameter to select another."
    )
    warnings.append(warning_msg)
    return chosen_sheet, warnings


def load_financial_file(
    source: Union[str, Path, pd.DataFrame, io.BytesIO],
    sheet_name: Optional[str] = None,
    mapper: Optional[SchemaMapper] = None,
) -> LoadedFinancialData:
    """Load and normalize user-provided financial statement data from CSV or Excel."""
    schema_mapper = mapper or SchemaMapper()
    warnings: list[str] = []
    file_type = "dataframe"
    source_name = "in_memory_dataframe"
    detected_sheet: Optional[str] = None

    # 1. Read file into raw DataFrame
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        source_name = file_path.name
        if not file_path.exists():
            raise FileNotFoundError(f"Financial dataset file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            file_type = "csv"
            try:
                raw_df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                raise DataValidationError(f"The uploaded CSV file '{source_name}' is empty.")
            except UnicodeDecodeError:
                try:
                    raw_df = pd.read_csv(file_path, encoding="latin1")
                except pd.errors.EmptyDataError:
                    raise DataValidationError(f"The uploaded CSV file '{source_name}' is empty.")
            except Exception as exc:
                raise DataValidationError(f"Error parsing CSV file '{source_name}': {exc}")
        elif suffix in (".xlsx", ".xls"):
            file_type = "excel"
            try:
                xls = pd.ExcelFile(file_path)
                detected_sheet, sheet_warnings = _find_best_excel_sheet(xls, schema_mapper, sheet_name)
                warnings.extend(sheet_warnings)
                raw_df = pd.read_excel(xls, sheet_name=detected_sheet)
            except DataValidationError:
                raise
            except Exception as exc:
                raise DataValidationError(f"Error reading Excel file '{source_name}': {exc}")
        else:
            try:
                raw_df = pd.read_csv(file_path)
                file_type = "csv"
            except pd.errors.EmptyDataError:
                raise DataValidationError(f"The uploaded file '{source_name}' is empty.")
            except Exception:
                try:
                    raw_df = pd.read_excel(file_path)
                    file_type = "excel"
                except Exception as exc:
                    raise DataValidationError(f"Unable to read file '{source_name}' as CSV or Excel: {exc}")
    elif isinstance(source, pd.DataFrame):
        raw_df = source.copy()
        file_type = "dataframe"
    else:
        try:
            raw_df = pd.read_csv(source)
            file_type = "csv"
        except pd.errors.EmptyDataError:
            raise DataValidationError("The uploaded stream/buffer is empty.")
        except Exception:
            try:
                raw_df = pd.read_excel(source)
                file_type = "excel"
            except Exception as exc:
                raise DataValidationError(f"Error reading uploaded buffer: {exc}")

    if raw_df.empty:
        raise DataValidationError(f"The uploaded financial file '{source_name}' is empty.")

    # 2. Clean column headers and map schema
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_columns = list(raw_df.columns)
    mapping_result = schema_mapper.map_columns(raw_columns)
    warnings.extend(mapping_result.warnings)

    # 3. Validate dataset suitability
    schema_mapper.validate_dataset_suitability(mapping_result, raise_on_insufficient=True)

    # 4. Standardize DataFrame with canonical columns and preserve unmapped columns
    canonical_df = pd.DataFrame(index=raw_df.index)

    for raw_col, canonical_name in mapping_result.mapped_columns.items():
        canonical_df[canonical_name] = raw_df[raw_col]

    # Preserve all other raw columns
    for raw_col in raw_df.columns:
        if raw_col not in mapping_result.mapped_columns:
            canonical_df[raw_col] = raw_df[raw_col]

    # Always provide deterministic record_id if not present
    if "record_id" not in canonical_df.columns:
        canonical_df["record_id"] = [f"Record #{i+1}" for i in range(len(canonical_df))]

    # Handle metadata columns (Company & Year) - DO NOT FAKE TEMPORAL ORDERING
    has_company = "company" in canonical_df.columns and canonical_df["company"].notna().any()
    if not has_company:
        canonical_df["company"] = None
    else:
        canonical_df["company"] = canonical_df["company"].astype(str).str.strip()

    has_year = "year" in canonical_df.columns and pd.to_numeric(canonical_df["year"], errors="coerce").fillna(0).gt(0).any()
    if not has_year:
        canonical_df["year"] = None
        warnings.append("Temporal ordering unavailable because no year/date field was provided. Historical YoY growth and change features will not be calculated.")
    else:
        canonical_df["year"] = pd.to_numeric(canonical_df["year"], errors="coerce").fillna(0).astype(int)

    has_temporal = bool(has_company and has_year)

    # Clean numeric financial fields
    for col in canonical_df.columns:
        if col in ("company", "category", "industry", "sector", "record_id", "date", "description", "notes", "name"):
            continue
        # Attempt clean numeric value
        cleaned_col = canonical_df[col].apply(clean_numeric_value)
        if cleaned_col.notna().sum() > 0:
            canonical_df[col] = cleaned_col

    # Convert to FinancialRecord objects
    records: list[FinancialRecord] = []
    for _, row in canonical_df.iterrows():
        row_dict = {k: v for k, v in row.items() if pd.notna(v)}
        record = FinancialRecord.from_dict(row_dict)
        records.append(record)

    # Deduplicate company-year pairs only if temporal panel structure exists
    if has_temporal:
        seen: set[tuple[str, int]] = set()
        unique_records: list[FinancialRecord] = []
        for r in records:
            if r.company and r.year and r.year > 0:
                key = (r.company.strip().upper(), r.year)
                if key not in seen:
                    seen.add(key)
                    unique_records.append(r)
            else:
                unique_records.append(r)
        unique_records.sort(key=lambda r: (str(r.company).upper(), r.year or 0))
    else:
        unique_records = records

    return LoadedFinancialData(
        records=unique_records,
        dataframe=canonical_df,
        mapping=mapping_result,
        file_type=file_type,
        source_name=source_name,
        sheet_name=detected_sheet,
        has_temporal_ordering=has_temporal,
        warnings=warnings,
    )


def inspect_dataset(data: Union[pd.DataFrame, str, Path]) -> DatasetSummary:
    """Inspect dataset structure and schema with automatic multi-format and alias mapping support."""
    if isinstance(data, (str, Path)):
        filepath = Path(data)
        dataset_name = filepath.name
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        if filepath.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    else:
        dataset_name = "in_memory_dataframe"
        df = data.copy()

    df.columns = [str(c).strip() for c in df.columns]
    num_rows, num_columns = df.shape
    columns = list(df.columns)
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    mapper = SchemaMapper()
    mapping = mapper.map_columns(columns)

    company_col = mapping.canonical_to_raw.get("company", "None")
    year_col = mapping.canonical_to_raw.get("year", "None")

    missing_values = {col: int(df[col].isna().sum()) for col in columns}

    if company_col != "None" and year_col != "None":
        duplicate_rows = int(df.duplicated(subset=[company_col, year_col]).sum())
        unique_companies = sorted(df[company_col].dropna().unique().astype(str).tolist())
        years_available = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
        observations_per_company = {
            comp: int((df[company_col].astype(str) == comp).sum()) for comp in unique_companies
        }
    else:
        duplicate_rows = int(df.duplicated().sum())
        unique_companies = []
        years_available = []
        observations_per_company = {}

    numerical_fields = [
        col for col in columns if pd.api.types.is_numeric_dtype(df[col]) and col != year_col
    ]
    categorical_fields = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]

    return DatasetSummary(
        dataset_name=dataset_name,
        num_rows=num_rows,
        num_columns=num_columns,
        columns=columns,
        data_types=data_types,
        company_identifier=company_col,
        year_field=year_col,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        unique_companies=unique_companies,
        years_available=years_available,
        observations_per_company=observations_per_company,
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields,
    )


def load_dataset(
    source: Union[pd.DataFrame, str, Path, Sequence[FinancialRecord], Sequence[dict]],
    sheet_name: Optional[str] = None,
) -> list[FinancialRecord]:
    """Load and normalize financial statement data into structured FinancialRecord objects."""
    if isinstance(source, (str, Path)):
        filepath = Path(source)
        loaded = load_financial_file(filepath, sheet_name=sheet_name)
        return loaded.records
    elif isinstance(source, pd.DataFrame):
        loaded = load_financial_file(source)
        return loaded.records
    elif isinstance(source, Sequence):
        if not source:
            return []
        if isinstance(source[0], FinancialRecord):
            records = list(source)  # type: ignore
            return records
        elif isinstance(source[0], dict):
            df = pd.DataFrame(source)
            loaded = load_financial_file(df)
            return loaded.records
        else:
            raise TypeError(f"Unsupported record sequence type: {type(source[0])}")
    else:
        raise TypeError(f"Unsupported data source type: {type(source)}")


def generate_reference_dataset(
    output_path: Union[str, Path, None] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the complete 161-row reference financial dataset (2009-2023) with all 23 columns."""
    rng = np.random.default_rng(seed)

    companies = [
        {"name": "Apple Inc", "category": "Technology", "base_rev": 65000.0, "gp_m": 0.38, "growth": 0.12, "de": 1.20, "start_yr": 2009, "emp": 164000},
        {"name": "Microsoft Corp", "category": "Technology", "base_rev": 58000.0, "gp_m": 0.65, "growth": 0.11, "de": 0.45, "start_yr": 2009, "emp": 221000},
        {"name": "Amazon.com Inc", "category": "Retail", "base_rev": 34000.0, "gp_m": 0.25, "growth": 0.18, "de": 1.10, "start_yr": 2009, "emp": 1540000},
        {"name": "Alphabet Inc", "category": "Technology", "base_rev": 29000.0, "gp_m": 0.56, "growth": 0.14, "de": 0.10, "start_yr": 2009, "emp": 190000},
        {"name": "Tesla Inc", "category": "Automotive", "base_rev": 110.0, "gp_m": 0.22, "growth": 0.42, "de": 0.85, "start_yr": 2010, "emp": 127000},
        {"name": "Walmart Inc", "category": "Retail", "base_rev": 405000.0, "gp_m": 0.24, "growth": 0.04, "de": 0.75, "start_yr": 2009, "emp": 2100000},
        {"name": "Exxon Mobil Corp", "category": "Energy", "base_rev": 310000.0, "gp_m": 0.30, "growth": 0.03, "de": 0.25, "start_yr": 2009, "emp": 62000},
        {"name": "Johnson & Johnson", "category": "Healthcare", "base_rev": 61000.0, "gp_m": 0.68, "growth": 0.05, "de": 0.40, "start_yr": 2009, "emp": 152000},
        {"name": "JPMorgan Chase & Co", "category": "Financial Services", "base_rev": 115000.0, "gp_m": 0.45, "growth": 0.06, "de": 2.10, "start_yr": 2009, "emp": 293000},
        {"name": "Procter & Gamble Co", "category": "Consumer Goods", "base_rev": 76000.0, "gp_m": 0.50, "growth": 0.04, "de": 0.65, "start_yr": 2009, "emp": 106000},
        {"name": "NVIDIA Corp", "category": "Technology", "base_rev": 3300.0, "gp_m": 0.52, "growth": 0.22, "de": 0.35, "start_yr": 2009, "emp": 26000},
    ]

    macro_inflation = {
        2009: 0.027, 2010: 0.015, 2011: 0.030, 2012: 0.017, 2013: 0.015,
        2014: 0.008, 2015: 0.007, 2016: 0.021, 2017: 0.021, 2018: 0.019,
        2019: 0.023, 2020: 0.014, 2021: 0.070, 2022: 0.065, 2023: 0.034,
    }

    rows = []

    for comp in companies:
        name = comp["name"]
        cat = comp["category"]
        cur_rev = float(comp["base_rev"])
        base_gp_m = float(comp["gp_m"])
        growth_rate = float(comp["growth"])
        base_de = float(comp["de"])
        start_year = int(comp["start_yr"])
        base_emp = int(comp["emp"])

        years = list(range(start_year, 2024))

        for y in years:
            noise = rng.normal(0.0, 0.035)
            if y > start_year:
                cur_rev = max(50.0, cur_rev * (1.0 + growth_rate + noise))

            gp_m = np.clip(base_gp_m + rng.normal(0.0, 0.015), 0.05, 0.95)
            gross_profit = cur_rev * gp_m

            ebitda_m = np.clip(gp_m - rng.uniform(0.08, 0.15), 0.02, 0.80)
            ebitda = cur_rev * ebitda_m

            net_m = np.clip(ebitda_m - rng.uniform(0.04, 0.08), 0.01, 0.60)
            net_income = cur_rev * net_m

            equity = max(100.0, cur_rev * rng.uniform(0.35, 0.75))
            total_assets = equity * (1.0 + base_de)

            ocf = net_income * rng.uniform(0.95, 1.35)
            icf = -abs(cur_rev * rng.uniform(0.04, 0.12))
            fcf = ocf + icf
            fcf_cash_flow = - (cur_rev * rng.uniform(0.02, 0.08))

            shares_m = max(50.0, cur_rev / rng.uniform(15.0, 35.0))
            eps = net_income / shares_m
            fcf_per_share = fcf / shares_m
            pe_multiple = rng.uniform(14.0, 28.0)
            mcap_b = max(1.0, (net_income * pe_multiple) / 1000.0)

            current_ratio = np.clip(rng.uniform(1.10, 2.30), 0.5, 5.0)
            debt_to_equity = np.clip(base_de + rng.normal(0.0, 0.05), 0.05, 5.0)

            roe = net_income / equity
            roa = net_income / total_assets
            roi = net_income / (equity * (1.0 + debt_to_equity * 0.5))
            rote = net_income / (equity * 0.85)

            year_ratio = (y - start_year) / (2023 - start_year if 2023 != start_year else 1)
            emp_count = int(base_emp * (0.6 + 0.4 * year_ratio + rng.normal(0.0, 0.05)))
            inflation = macro_inflation.get(y, 0.025)

            rows.append({
                "Year": int(y),
                "Company": name,
                "Category": cat,
                "Market Cap(in B USD)": round(float(mcap_b), 2),
                "Revenue": round(float(cur_rev), 2),
                "Gross Profit": round(float(gross_profit), 2),
                "Net Income": round(float(net_income), 2),
                "Earning Per Share": round(float(eps), 2),
                "EBITDA": round(float(ebitda), 2),
                "Share Holder Equity": round(float(equity), 2),
                "Operating Cash Flow": round(float(ocf), 2),
                "Investing Cash Flow": round(float(icf), 2),
                "Financing Cash Flow": round(float(fcf_cash_flow), 2),
                "Current Ratio": round(float(current_ratio), 2),
                "Debt/Equity Ratio": round(float(debt_to_equity), 2),
                "ROE": round(float(roe), 4),
                "ROA": round(float(roa), 4),
                "ROI": round(float(roi), 4),
                "Net Profit Margin": round(float(net_m), 4),
                "Free Cash Flow per Share": round(float(fcf_per_share), 2),
                "Return on Tangible Equity": round(float(rote), 4),
                "Number of Employees": int(emp_count),
                "Inflation Rate": round(float(inflation), 4),
            })

    df = pd.DataFrame(rows)
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        logger.info("Saved 23-column reference dataset (%d rows) to: %s", len(df), out_file)

    return df
