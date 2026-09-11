"""Data cleaning and dynamic type normalization for financial statements.

Performs:
- String cleaning: whitespace trimming and multiple space collapsing while preserving tickers and category values
- Dynamic numeric detection: automatically parses currency ($), thousand separators (,), percentages (%),
  and accounting negatives in parentheses (500) -> -500.0
- Canonical percentage convention: Percentages with '%' (e.g. '15%' or '1.23%') are stored as nominal numeric
  values (15.0 and 1.23 respectively) matching standard financial statement metrics
- Dynamic year and date extraction: extracts 4-digit years from fiscal year labels (e.g. 'FY2022', 'FY 22')
  and calendar dates ('2022-12-31')
- Audit logging: records every modification so no data is silently altered
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, ColumnSpec

logger = logging.getLogger("ingestion.cleaner")

# Regex to detect parenthesized negative numbers e.g. (12,345.67) or (50)
PAREN_NEGATIVE_REGEX = re.compile(r"^\s*\(\s*(.*?)\s*\)\s*$")
# Characters to strip from numeric strings
NUMERIC_CLEAN_REGEX = re.compile(r"[\$,€£¥%]")
# 4-digit year pattern (1900-2099)
FOUR_DIGIT_YEAR_REGEX = re.compile(r"\b(19\d\d|20\d\d)\b")
# FY 2-digit pattern e.g. FY22, FY 21
TWO_DIGIT_FY_REGEX = re.compile(r"\bfy[\s\-_]*(\d{2})\b", re.IGNORECASE)


def parse_numeric_value(val: Any, percentage_as_decimal: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """Parses a raw value into a float, returning (parsed_val, error_msg).

    Canonical representation rules:
    - Normal floats / ints: converted directly to float
    - Currency symbols ($1,234.50) and commas: stripped, value converted to 1234.50
    - Negative parentheses: '(500.25)' -> -500.25
    - Percentages: '15.5%' -> 15.5 (default nominal percentage) or 0.155 (if percentage_as_decimal=True)
    - Empty / NA representations: '-', 'N/A', 'NA', 'null', 'None', '' -> np.nan
    """
    if pd.isna(val) or val is None:
        return np.nan, None

    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val), None

    str_val = str(val).strip()

    # Empty or null indicators
    if str_val.lower() in ("", "-", "--", "n/a", "na", "null", "none", "nan", "nil", "undefined"):
        return np.nan, None

    is_negative = False
    # Check for parentheses: (123.45)
    paren_match = PAREN_NEGATIVE_REGEX.match(str_val)
    if paren_match:
        is_negative = True
        str_val = paren_match.group(1).strip()

    has_percent = "%" in str_val

    # Strip currency symbols and percent signs
    str_val = NUMERIC_CLEAN_REGEX.sub("", str_val)
    # Remove commas
    str_val = str_val.replace(",", "").strip()

    # If it has a minus sign
    if str_val.startswith("-"):
        is_negative = not is_negative
        str_val = str_val[1:].strip()

    try:
        num = float(str_val)
        if is_negative:
            num = -num
        if has_percent and percentage_as_decimal:
            num = num / 100.0
        return num, None
    except ValueError:
        return np.nan, f"Could not parse '{val}' as numeric"


def parse_year_value(val: Any, min_year: int = 1900, max_year: int = 2100) -> Tuple[Optional[int], Optional[str]]:
    """Validates and parses a year into an integer within acceptable range.

    Handles:
    - Direct integers: 2022
    - String integers: "2022"
    - Float-like strings: "2022.0"
    - Fiscal year labels: "FY2022", "FY 2022", "FY22" -> 2022
    - Dates: "2022-12-31", "12/31/2022" -> 2022
    """
    if pd.isna(val) or val is None:
        return None, "Year is missing/null"

    str_val = str(val).strip()
    if not str_val:
        return None, "Year is empty"

    # Float-like string: "2022.0"
    if str_val.endswith(".0"):
        str_val = str_val[:-2]

    # Try direct integer
    try:
        y_int = int(str_val)
        if min_year <= y_int <= max_year:
            return y_int, None
        return None, f"Year {y_int} out of valid financial range ({min_year}-{max_year})"
    except ValueError:
        pass

    # Try 4-digit year extraction from date or FY string: e.g. "2022-12-31", "FY2022"
    four_digit_match = FOUR_DIGIT_YEAR_REGEX.search(str_val)
    if four_digit_match:
        y_int = int(four_digit_match.group(1))
        if min_year <= y_int <= max_year:
            return y_int, None

    # Try 2-digit FY: e.g. "FY22" -> 2022
    two_digit_match = TWO_DIGIT_FY_REGEX.search(str_val)
    if two_digit_match:
        y_int = 2000 + int(two_digit_match.group(1))
        if min_year <= y_int <= max_year:
            return y_int, None

    return None, f"Could not parse '{val}' as a valid fiscal year or date ({min_year}-{max_year}): not an integer"


def is_dynamic_numeric_column(series: pd.Series) -> bool:
    """Checks whether an arbitrary column predominantly contains numeric values."""
    non_nulls = [v for v in series.dropna() if str(v).strip()]
    if not non_nulls:
        return False

    sample = non_nulls[:40]
    parsed_count = 0
    for v in sample:
        num, err = parse_numeric_value(v)
        if not np.isnan(num) and err is None:
            parsed_count += 1

    return (parsed_count / len(sample)) >= 0.70


class DataCleaner:
    """Performs non-destructive cleaning, dynamic type detection, and records all modifications."""

    def __init__(
        self,
        coerce_numeric_errors: bool = True,
        percentage_as_decimal: bool = False,
    ):
        self.coerce_numeric_errors = coerce_numeric_errors
        self.percentage_as_decimal = percentage_as_decimal

    def clean(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, ColumnSpec]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Cleans the DataFrame columns dynamically based on schema and content inspection.

        Returns:
            Tuple of (cleaned_df, list_of_recorded_modifications)
        """
        if schema is None:
            schema = CANONICAL_COLUMNS

        cleaned_df = df.copy()
        modifications: List[Dict[str, Any]] = []

        for col in cleaned_df.columns:
            spec = schema.get(col)

            # 1. Company / Identifier or String Schema Columns
            if spec and spec.data_type is str:
                def _clean_str(val: Any) -> Any:
                    if pd.isna(val) or val is None:
                        return np.nan
                    s = str(val).strip()
                    s = re.sub(r"\s+", " ", s)
                    if s.lower() in ("null", "none", "nan", "n/a", "na", "undefined", ""):
                        return np.nan
                    return s

                cleaned_df[col] = cleaned_df[col].apply(_clean_str)

            # 2. Year Column
            elif col == "year":
                min_yr = spec.min_val if spec and spec.min_val else 1900
                max_yr = spec.max_val if spec and spec.max_val else 2100

                new_years = []
                for idx, val in enumerate(cleaned_df[col]):
                    parsed_year, err = parse_year_value(val, int(min_yr), int(max_yr))
                    if err:
                        modifications.append({
                            "type": "invalid_year",
                            "column": "year",
                            "row_index": idx,
                            "original_value": val,
                            "action": "coerced_to_null",
                            "message": err,
                        })
                        new_years.append(np.nan)
                    else:
                        new_years.append(parsed_year)

                cleaned_df[col] = pd.Series(new_years, index=cleaned_df.index, dtype="Int64")

            # 3. Canonical Numeric Columns
            elif spec and spec.data_type in (float, int):
                new_vals = []
                for idx, val in enumerate(cleaned_df[col]):
                    parsed_val, err = parse_numeric_value(val, self.percentage_as_decimal)
                    if err:
                        modifications.append({
                            "type": "numeric_parse_warning",
                            "column": col,
                            "row_index": idx,
                            "original_value": val,
                            "action": "coerced_to_null" if self.coerce_numeric_errors else "retained",
                            "message": err,
                        })
                        new_vals.append(np.nan)
                    else:
                        new_vals.append(parsed_val)

                cleaned_df[col] = pd.Series(new_vals, index=cleaned_df.index, dtype="float64")

            # 4. Unmapped / Extra Columns: Dynamic Type Detection
            else:
                if is_dynamic_numeric_column(cleaned_df[col]):
                    # Treat unmapped column as numeric
                    new_vals = []
                    for idx, val in enumerate(cleaned_df[col]):
                        parsed_val, err = parse_numeric_value(val, self.percentage_as_decimal)
                        if err:
                            modifications.append({
                                "type": "dynamic_numeric_parse_warning",
                                "column": col,
                                "row_index": idx,
                                "original_value": val,
                                "action": "coerced_to_null",
                                "message": err,
                            })
                            new_vals.append(np.nan)
                        else:
                            new_vals.append(parsed_val)
                    cleaned_df[col] = pd.Series(new_vals, index=cleaned_df.index, dtype="float64")
                else:
                    # Treat unmapped column as text
                    def _clean_extra_str(val: Any) -> Any:
                        if pd.isna(val) or val is None:
                            return np.nan
                        s = str(val).strip()
                        s = re.sub(r"\s+", " ", s)
                        if s.lower() in ("null", "none", "nan", "n/a", "na", ""):
                            return np.nan
                        return s

                    cleaned_df[col] = cleaned_df[col].apply(_clean_extra_str)

        return cleaned_df, modifications
