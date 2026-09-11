"""Validation engine for financial statement datasets.

Verifies:
- Financial domain compatibility (accepts any compatible financial statement dataset; rejects non-financial data)
- Flexible core vs. optional schema conformance
- Missing values analysis
- Duplicate row detection
- Range and sanity constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .schema import (
    CANONICAL_COLUMNS,
    CORE_FINANCIAL_METRICS,
    CORE_IDENTIFIERS,
    ColumnSpec,
    evaluate_financial_statement_compatibility,
)

logger = logging.getLogger("ingestion.validator")


@dataclass
class ValidationReport:
    """Detailed report of dataset validation checks."""

    is_valid: bool
    is_financial_dataset: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mapped_canonical_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    missing_values_by_column: Dict[str, int] = field(default_factory=dict)
    duplicate_rows_count: int = 0
    duplicate_entities_count: int = 0


class DataValidator:
    """Performs domain compatibility checks and schema validation for financial datasets."""

    def __init__(self, schema: Optional[Dict[str, ColumnSpec]] = None):
        self.schema = schema or CANONICAL_COLUMNS

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Runs validation checks on the standardized DataFrame.

        Validates whether the dataset contains recognizable financial statement metrics,
        detects null values, duplicate records, and extra columns.
        """
        errors: List[str] = []
        warnings: List[str] = []

        canonical_keys = set(self.schema.keys())
        df_cols = set(df.columns)
        mapped_canonical = df_cols & canonical_keys

        # 1. Financial Dataset Compatibility Check
        is_compat, reason = evaluate_financial_statement_compatibility(mapped_canonical)
        if not is_compat:
            errors.append(reason)
            logger.error(f"Financial validation rejected: {reason}")
            return ValidationReport(
                is_valid=False,
                is_financial_dataset=False,
                errors=errors,
                warnings=warnings,
                mapped_canonical_columns=list(mapped_canonical),
                extra_columns=list(df_cols - canonical_keys),
            )

        # 2. Informational Warnings for Core Fields (does NOT reject if one is absent)
        missing_identifiers = [c for c in CORE_IDENTIFIERS if c not in df_cols]
        if missing_identifiers:
            warnings.append(
                f"Dataset does not contain standard identifier field(s): {', '.join(missing_identifiers)}."
            )

        # 3. Extra / Unmapped Columns (preserved, not rejected)
        extra_cols = list(df_cols - canonical_keys)
        if extra_cols:
            warnings.append(
                f"Dataset contains {len(extra_cols)} additional unmapped column(s): {', '.join(extra_cols)}"
            )

        # 4. Missing Values Analysis
        missing_counts: Dict[str, int] = {}
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                missing_counts[col] = null_count
                if col in mapped_canonical:
                    warnings.append(
                        f"Mapped financial column '{col}' has {null_count} missing (null/NaN) values."
                    )

        # 5. Duplicate Rows
        duplicate_rows_count = int(df.duplicated().sum())
        if duplicate_rows_count > 0:
            warnings.append(f"Found {duplicate_rows_count} completely duplicate row(s).")

        # 6. Entity Key Duplicate Check (if both Year and Company exist)
        duplicate_entities_count = 0
        if "year" in df.columns and "company" in df.columns:
            subset_duplicates = df.duplicated(subset=["year", "company"], keep=False)
            duplicate_entities_count = int(subset_duplicates.sum())
            if duplicate_entities_count > 0:
                warnings.append(
                    f"Found {duplicate_entities_count} records sharing identical (year, company) keys."
                )

        # 7. Constraint Sanity Checks
        for col in mapped_canonical:
            spec = self.schema[col]
            if spec.min_val is not None:
                invalid_min = df[col] < spec.min_val
                count_invalid = int(invalid_min.sum())
                if count_invalid > 0:
                    warnings.append(
                        f"Column '{col}' has {count_invalid} values below minimum threshold {spec.min_val}."
                    )
            if spec.max_val is not None:
                invalid_max = df[col] > spec.max_val
                count_invalid = int(invalid_max.sum())
                if count_invalid > 0:
                    warnings.append(
                        f"Column '{col}' has {count_invalid} values above maximum threshold {spec.max_val}."
                    )

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            is_financial_dataset=is_compat,
            errors=errors,
            warnings=warnings,
            mapped_canonical_columns=list(mapped_canonical),
            extra_columns=extra_cols,
            missing_values_by_column=missing_counts,
            duplicate_rows_count=duplicate_rows_count,
            duplicate_entities_count=duplicate_entities_count,
        )
