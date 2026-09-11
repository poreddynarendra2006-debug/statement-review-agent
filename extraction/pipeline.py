"""Main Orchestrator interface for the Financial Statement Data Ingestion Pipeline.

Provides the primary public API:
    ingest_financial_statement(source, config=None) -> IngestionResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

from .cleaner import DataCleaner
from .config import IngestionConfig
from .mapper import SemanticColumnMapper
from .reader import (
    CSVReadError,
    EmptyCSVError,
    IngestionError,
    InvalidCSVFormatError,
    read_csv_safely,
)
from .reporter import IngestionMetadata
from .schema import (
    CANONICAL_COLUMNS,
    STANDARD_UNITS,
    evaluate_financial_statement_compatibility,
    normalize_column_name,
)
from .validator import DataValidator

# Configure module logger
logger = logging.getLogger("ingestion.pipeline")


@dataclass
class FinancialRecord:
    """Standard financial statement record representation for downstream agents."""

    # 23 Kaggle Fields
    year: Optional[int] = None
    company: Optional[str] = None
    category: Optional[str] = None
    market_cap_b_usd: Optional[float] = None
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    net_income: Optional[float] = None
    earnings_per_share: Optional[float] = None
    ebitda: Optional[float] = None
    shareholder_equity: Optional[float] = None
    cash_flow_operating: Optional[float] = None
    cash_flow_investing: Optional[float] = None
    cash_flow_financing: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_equity_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roi: Optional[float] = None
    net_profit_margin: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None
    return_on_tangible_equity: Optional[float] = None
    number_of_employees: Optional[float] = None
    inflation_rate_us: Optional[float] = None

    # 14 Statement Fields
    currency: Optional[str] = None
    cost_of_revenue: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    pre_tax_income: Optional[float] = None
    taxes: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    cash: Optional[float] = None
    debt: Optional[float] = None
    beginning_cash: Optional[float] = None
    ending_cash: Optional[float] = None

    # Extra unmapped columns
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "extra_fields":
            raise AttributeError(name)  # lets copy and pickle work
        if name in CANONICAL_COLUMNS:
            return None
        if name in self.extra_fields:
            return self.extra_fields[name]
        raise AttributeError(f"'FinancialRecord' object has no attribute '{name}'")

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.extra_fields:
            return self.extra_fields[item]
        raise KeyError(f"Key '{item}' not found in FinancialRecord")

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in CANONICAL_COLUMNS.keys()}
        d["extra_fields"] = dict(self.extra_fields)
        return d


@dataclass
class IngestionResult:
    """Standardized response from the Data Ingestion pipeline.

    Supports both object attribute access (result.data) and dict subscripting (result["data"])
    to ensure seamless integration with the Orchestrator and downstream agents.
    """

    status: str  # "success", "partial_success", "error"
    data: Optional[pd.DataFrame]
    metadata: IngestionMetadata
    message: str = ""

    def __getitem__(self, item: str) -> Any:
        """Allows dictionary subscripting: result['status'], result['data'], result['metadata']."""
        if item == "status":
            return self.status
        elif item == "data":
            return self.data
        elif item == "metadata":
            return self.metadata.to_dict()
        elif item == "message":
            return self.message
        raise KeyError(f"Invalid key '{item}'. Available keys: status, data, metadata, message")

    def __contains__(self, item: str) -> bool:
        return item in ("status", "data", "metadata", "message")

    def keys(self) -> List[str]:
        return ["status", "data", "metadata", "message"]

    def to_dict(self) -> Dict[str, Any]:
        """Returns standard dictionary matching the Orchestrator contract."""
        return {
            "status": self.status,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
            "message": self.message,
        }

    def to_records(self) -> List[FinancialRecord]:
        """Converts standardized DataFrame to a list of typed FinancialRecord objects.

        - One object per row with all 37 canonical fields as attributes
        - Missing or blank fields evaluate to None (never NaN, never 0, never raises)
        - company is a str and year is an int
        - Unmapped extra columns are stored in .extra_fields
        """
        if self.data is None:
            return []

        records: List[FinancialRecord] = []
        canonical_names = set(CANONICAL_COLUMNS.keys())
        df_cols = list(self.data.columns)
        canonical_in_df = [c for c in df_cols if c in canonical_names]
        extra_in_df = [c for c in df_cols if c not in canonical_names]

        for _, row in self.data.iterrows():
            rec_kwargs: Dict[str, Any] = {}
            extra_dict: Dict[str, Any] = {}

            for col in canonical_in_df:
                val = row[col]
                if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "NaN", "None", "null", "<NA>"):
                    rec_kwargs[col] = None
                elif col == "year":
                    try:
                        rec_kwargs[col] = int(val)
                    except (ValueError, TypeError):
                        rec_kwargs[col] = None
                elif col in ("company", "category", "currency"):
                    rec_kwargs[col] = str(val).strip()
                else:
                    try:
                        rec_kwargs[col] = float(val)
                    except (ValueError, TypeError):
                        rec_kwargs[col] = None

            for col in extra_in_df:
                val = row[col]
                if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "NaN", "None", "null", "<NA>"):
                    extra_dict[col] = None
                else:
                    extra_dict[col] = val

            rec_kwargs["extra_fields"] = extra_dict
            records.append(FinancialRecord(**rec_kwargs))

        return records

    def print_summary(self) -> None:
        """Prints formatted summary to console/logs."""
        print(self.metadata.generate_summary_text())


class IngestionPipeline:
    """Core pipeline executing safe read, 3-tier mapping, non-destructive cleaning, and validation."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.mapper = SemanticColumnMapper(
            embedding_threshold=self.config.embedding_threshold,
            ambiguity_margin=self.config.ambiguity_margin,
            llm_resolver=self.config.llm_resolver,
            custom_embedding_fn=self.config.custom_embedding_fn,
        )
        self.cleaner = DataCleaner(
            coerce_numeric_errors=self.config.coerce_numeric_errors,
            percentage_as_decimal=self.config.percentage_as_decimal,
        )
        self.validator = DataValidator()

    def run(
        self,
        source: Any,
    ) -> IngestionResult:
        """Executes the complete ingestion process on the given source.

        Args:
            source: File path, bytes, file stream, or file-like object.

        Returns:
            IngestionResult containing status, standardized DataFrame, and metadata.
        """
        source_repr = str(getattr(source, "name", str(source)[:50]))
        logger.info(f"Starting ingestion for source: {source_repr}")

        # ----------------------------------------------------
        # STEP 1: Safe Dynamic Reading
        # ----------------------------------------------------
        try:
            raw_df, read_meta = read_csv_safely(source)
        except (EmptyCSVError, InvalidCSVFormatError, CSVReadError, FileNotFoundError) as e:
            logger.error(f"Ingestion failed during CSV read: {e}")
            empty_metadata = IngestionMetadata(
                source_name=source_repr,
                rows=0,
                columns=0,
                initial_rows=0,
                initial_columns=0,
                final_rows=0,
                final_columns=0,
                is_financial_dataset=False,
                errors=[str(e)],
            )
            return IngestionResult(
                status="error",
                data=None,
                metadata=empty_metadata,
                message=str(e),
            )
        except Exception as e:
            logger.exception(f"Unexpected error while reading CSV source: {e}")
            empty_metadata = IngestionMetadata(
                source_name=source_repr,
                rows=0,
                columns=0,
                initial_rows=0,
                initial_columns=0,
                final_rows=0,
                final_columns=0,
                is_financial_dataset=False,
                errors=[f"Unexpected error: {str(e)}"],
            )
            return IngestionResult(
                status="error",
                data=None,
                metadata=empty_metadata,
                message=f"Failed to read file: {str(e)}",
            )

        initial_rows = len(raw_df)
        initial_cols = len(raw_df.columns)
        modifications: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # ----------------------------------------------------
        # STEP 2: 3-Tier Semantic Column Mapping & Collision Handling
        # ----------------------------------------------------
        sample_dict = {}
        for col in raw_df.columns:
            non_null_samples = raw_df[col].dropna().tolist()[:3]
            sample_dict[col] = non_null_samples

        column_rename_map, mapping_results, collisions = self.mapper.map_columns(
            raw_columns=list(raw_df.columns),
            sample_data=sample_dict,
        )

        for c in collisions:
            warnings.append(
                f"Column collision: '{c['conflicting_column']}' conflicts with '{c['primary_column']}' "
                f"for '{c['canonical_name']}'. Preserved '{c['conflicting_column']}' as secondary."
            )

        mapped_cols_record: Dict[str, str] = {}
        unmapped_cols: List[str] = []
        ambiguous_cols: List[Dict[str, Any]] = []
        confidence_record: Dict[str, float] = {}
        methods_record: Dict[str, str] = {}

        for result in mapping_results:
            confidence_record[result.raw_name] = result.confidence
            methods_record[result.raw_name] = result.method
            if result.canonical_name:
                mapped_cols_record[result.raw_name] = result.canonical_name
            else:
                unmapped_name = column_rename_map.get(result.raw_name, normalize_column_name(result.raw_name))
                unmapped_cols.append(unmapped_name)
                if result.method == "ambiguous" or result.candidates:
                    ambiguous_cols.append({
                        "column": result.raw_name,
                        "candidates": result.candidates,
                        "reason": result.reason,
                    })

        # Create a standardized copy of data without modifying original input
        standardized_df = raw_df.rename(columns=column_rename_map).copy()

        if self.config.drop_unmapped_columns and unmapped_cols:
            cols_to_drop = [c for c in unmapped_cols if c in standardized_df.columns]
            standardized_df = standardized_df.drop(columns=cols_to_drop)
            warnings.append(
                f"Dropped {len(cols_to_drop)} unmapped column(s): {', '.join(cols_to_drop)}"
            )

        # ----------------------------------------------------
        # STEP 3: Required Fields Validation & Compatibility Check
        # ----------------------------------------------------
        missing_required = [
            req for req in ("year", "company", "revenue")
            if req not in standardized_df.columns
        ]

        is_compat, compat_reason = evaluate_financial_statement_compatibility(set(mapped_cols_record.values()))

        validation_errors: List[str] = []
        if not is_compat:
            validation_errors.append(compat_reason)
        if missing_required:
            validation_errors.append(
                f"Financial statement is missing required column(s): {', '.join(missing_required)}. "
                f"Required fields are 'year', 'company', and 'revenue'."
            )

        if validation_errors:
            err_msg = "; ".join(validation_errors)
            empty_metadata = IngestionMetadata(
                source_name=read_meta["source_name"],
                rows=len(standardized_df),
                columns=len(standardized_df.columns),
                initial_rows=initial_rows,
                initial_columns=initial_cols,
                final_rows=len(standardized_df),
                final_columns=len(standardized_df.columns),
                mapped_columns=mapped_cols_record,
                unmapped_columns=unmapped_cols,
                confidence_scores=confidence_record,
                mapping_methods=methods_record,
                column_collisions=collisions,
                ambiguous_columns=ambiguous_cols,
                units=STANDARD_UNITS,
                is_financial_dataset=is_compat,
                errors=validation_errors,
            )
            return IngestionResult(
                status="error",
                data=None,
                metadata=empty_metadata,
                message=err_msg,
            )

        # ----------------------------------------------------
        # STEP 4: Deduplication (Non-destructive Audit)
        # ----------------------------------------------------
        duplicates_removed = 0
        if self.config.deduplicate_rows:
            dup_mask = standardized_df.duplicated()
            duplicates_removed = int(dup_mask.sum())
            if duplicates_removed > 0:
                dup_indices = standardized_df[dup_mask].index.tolist()
                modifications.append({
                    "type": "duplicate_rows_removed",
                    "count": duplicates_removed,
                    "row_indices": dup_indices,
                    "action": "dropped_identical_rows",
                })
                standardized_df = standardized_df.drop_duplicates().reset_index(drop=True)
                warnings.append(
                    f"Removed {duplicates_removed} duplicate row(s) (recorded in audit trail)."
                )

        # ----------------------------------------------------
        # STEP 5: Dynamic Cleaning & Type Casting
        # ----------------------------------------------------
        cleaned_df, cleaning_mods = self.cleaner.clean(standardized_df)
        modifications.extend(cleaning_mods)

        # ----------------------------------------------------
        # STEP 6: Financial Statement Compatibility & Validation
        # ----------------------------------------------------
        val_report = self.validator.validate(cleaned_df)
        warnings.extend(val_report.warnings)

        total_missing = sum(val_report.missing_values_by_column.values())

        # Determine overall status
        status = "success"
        message = "Dataset successfully ingested and standardized."

        if not val_report.is_valid:
            status = "error"
            message = f"Validation failed: {'; '.join(val_report.errors)}"
        elif warnings:
            status = "partial_success"
            message = f"Ingested with {len(warnings)} warning(s)."

        # Compile metadata
        metadata = IngestionMetadata(
            source_name=read_meta["source_name"],
            rows=len(cleaned_df),
            columns=len(cleaned_df.columns),
            initial_rows=initial_rows,
            initial_columns=initial_cols,
            final_rows=len(cleaned_df),
            final_columns=len(cleaned_df.columns),
            mapped_columns=mapped_cols_record,
            unmapped_columns=unmapped_cols,
            confidence_scores=confidence_record,
            mapping_methods=methods_record,
            column_collisions=collisions,
            ambiguous_columns=ambiguous_cols,
            units=STANDARD_UNITS,
            missing_values=val_report.missing_values_by_column,
            total_missing_values=total_missing,
            duplicates_removed=duplicates_removed,
            is_financial_dataset=val_report.is_financial_dataset,
            warnings=warnings,
            errors=val_report.errors,
            modifications=modifications,
        )

        logger.info(
            f"Ingestion finished: status={status}, rows={len(cleaned_df)}, cols={len(cleaned_df.columns)}"
        )

        return IngestionResult(
            status=status,
            data=cleaned_df if val_report.is_valid else None,
            metadata=metadata,
            message=message,
        )


def ingest_financial_statement(
    source: Any,
    config: Optional[IngestionConfig] = None,
    **kwargs: Any,
) -> IngestionResult:
    """Primary entry point for the Data Ingestion Module.

    Accepts any dynamic file source (path, bytes, stream, or file object) and returns
    a standardized IngestionResult.

    Example:
        result = ingest_financial_statement("path/to/statement.csv")
        if result.status in ("success", "partial_success"):
            records = result.to_records()
            clean_df = result.data
            orchestrator.dispatch(clean_df, result.metadata)
    """
    if config is None:
        config = IngestionConfig(**kwargs)
    pipeline = IngestionPipeline(config=config)
    return pipeline.run(source)
