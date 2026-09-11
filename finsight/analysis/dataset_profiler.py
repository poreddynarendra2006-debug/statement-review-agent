"""Dynamic Dataset Profiling layer for FinSight AI Anomaly Detection Agent.

Automatically inspects any uploaded financial dataset to discover its dimensions,
column types, entity identifiers, temporal structures, amount vs. ratio fields,
and cohort grouping capabilities.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from finsight.core.models import DatasetProfile, FinancialRecord

logger = logging.getLogger(__name__)

# Common entity / company identifier candidate names
ENTITY_CANDIDATE_NAMES = {
    "company", "ticker", "entity", "company_name", "entity_name", "firm",
    "organization", "symbol", "stock", "identifier", "bank", "issuer"
}

# Common temporal / date / period candidate names
PERIOD_CANDIDATE_NAMES = {
    "year", "fiscal_year", "period", "date", "fiscal_period", "year_num",
    "calendar_year", "report_date", "quarter", "fy"
}

# Common metadata / identifier fields that should not be used as anomaly detection features
METADATA_IGNORE_NAMES = {
    "record_id", "id", "row_id", "index", "serial_number", "unnamed: 0",
    "description", "notes", "status", "label", "target", "ground_truth",
    "planted", "source", "url", "comment", "remarks"
}

# Grouping / cohort candidate names for peer analysis
GROUPING_CANDIDATE_NAMES = {
    "category", "industry", "sector", "peer_group", "sub_industry",
    "asset_class", "region", "country", "market_tier", "group"
}


def _clean_col_name(col: str) -> str:
    """Normalize a column name for pattern checking."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower()).strip("_")


def profile_dataset(
    data: Union[pd.DataFrame, Sequence[Union[FinancialRecord, dict[str, Any]]]]
) -> DatasetProfile:
    """Profile an arbitrary financial dataset and return a structured DatasetProfile.

    Parameters
    ----------
    data : pd.DataFrame or sequence of FinancialRecord / dicts
        The validated financial dataset to profile.

    Returns
    -------
    DatasetProfile
        Comprehensive profile describing the dataset characteristics.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return DatasetProfile(
                num_rows=0,
                num_columns=0,
                temporal_reason="Dataset is empty.",
                peer_reason="Dataset is empty.",
            )
        first = data[0]
        if isinstance(first, FinancialRecord):
            df = pd.DataFrame([r.to_dict() for r in data])
        elif isinstance(first, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)

    num_rows, num_columns = df.shape
    raw_columns = list(df.columns)

    if num_rows == 0 or num_columns == 0:
        return DatasetProfile(
            num_rows=num_rows,
            num_columns=num_columns,
            raw_columns=raw_columns,
            temporal_reason="Dataset contains 0 rows or 0 columns.",
            peer_reason="Dataset contains 0 rows or 0 columns.",
        )

    # 1. Discover Entity Identifier Column
    entity_col: str | None = None
    for col in raw_columns:
        c_clean = _clean_col_name(col)
        if c_clean in ENTITY_CANDIDATE_NAMES:
            entity_col = col
            break
    if not entity_col:
        for col in raw_columns:
            c_clean = _clean_col_name(col)
            if any(cand in c_clean for cand in ("company", "ticker", "entity")):
                entity_col = col
                break

    # 2. Discover Temporal / Year Column
    year_col: str | None = None
    for col in raw_columns:
        c_clean = _clean_col_name(col)
        if c_clean in PERIOD_CANDIDATE_NAMES:
            year_col = col
            break
    if not year_col:
        for col in raw_columns:
            c_clean = _clean_col_name(col)
            if any(cand in c_clean for cand in ("year", "period", "date")):
                year_col = col
                break

    # 3. Discover Grouping / Cohort Columns
    grouping_cols: list[str] = []
    for col in raw_columns:
        c_clean = _clean_col_name(col)
        if c_clean in GROUPING_CANDIDATE_NAMES:
            grouping_cols.append(col)
        elif any(cand in c_clean for cand in ("sector", "industry", "category", "peer")):
            grouping_cols.append(col)

    # 4. Classify Numerical, Categorical, Amount, Ratio, and Percentage Columns
    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    amount_cols: list[str] = []
    ratio_cols: list[str] = []
    percentage_cols: list[str] = []

    for col in raw_columns:
        c_clean = _clean_col_name(col)
        if c_clean in METADATA_IGNORE_NAMES:
            continue

        series = df[col]

        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # Try converting to numeric
        num_series = pd.to_numeric(series, errors="coerce")
        valid_num_count = num_series.notna().sum()

        if valid_num_count > 0 and (valid_num_count / max(1, len(series))) >= 0.3:
            # Numerical column
            numerical_cols.append(col)

            # Classify subtype (Amount vs Ratio vs Percentage)
            valid_vals = num_series.dropna().abs()
            is_explicit_pct = "%" in str(col) or "percent" in c_clean or "pct" in c_clean
            is_explicit_ratio = "ratio" in c_clean or "margin" in c_clean or "roe" in c_clean or "roa" in c_clean

            if is_explicit_pct:
                percentage_cols.append(col)
            elif is_explicit_ratio:
                ratio_cols.append(col)
            elif len(valid_vals) > 0 and valid_vals.median() > 1000.0:
                amount_cols.append(col)
            elif len(valid_vals) > 0 and valid_vals.max() <= 100.0 and ("growth" in c_clean or "yield" in c_clean):
                percentage_cols.append(col)
            else:
                amount_cols.append(col)
        else:
            # Non-numeric / Categorical
            if col != entity_col and col != year_col:
                categorical_cols.append(col)

    # 5. Assess Temporal Structure & Multi-Year Availability
    has_temporal = False
    temporal_reason = ""
    unique_periods_count = 0

    if year_col:
        parsed_years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        if len(parsed_years) > 0:
            unique_periods_count = int(parsed_years.nunique())
            if unique_periods_count > 1:
                if entity_col:
                    # Check if at least some entities have multiple observations
                    obs_per_entity = df.groupby(entity_col)[year_col].count()
                    if (obs_per_entity > 1).any():
                        has_temporal = True
                        temporal_reason = f"Temporal panel available: {unique_periods_count} periods across entities."
                    else:
                        has_temporal = False
                        temporal_reason = "Temporal analysis unavailable: each entity has only 1 observation."
                else:
                    has_temporal = True
                    temporal_reason = f"Temporal series available: {unique_periods_count} distinct periods."
            else:
                has_temporal = False
                temporal_reason = "Temporal analysis unavailable: only one period is present."
        else:
            has_temporal = False
            temporal_reason = "Temporal analysis unavailable: year/period column contains no valid numeric periods."
    else:
        has_temporal = False
        temporal_reason = "Temporal analysis unavailable: no temporal or fiscal year column identified."

    # 6. Assess Peer / Grouping Structure
    has_peer = False
    peer_reason = ""
    unique_entities_count = 0

    if entity_col:
        valid_entities = df[entity_col].dropna().astype(str).str.strip()
        valid_entities = valid_entities[~valid_entities.str.lower().isin(["none", "nan", "", "null"])]
        unique_entities_count = int(valid_entities.nunique())

        if unique_entities_count >= 2 and num_rows >= 3:
            has_peer = True
            peer_reason = f"Peer analysis enabled across {unique_entities_count} distinct entities."
        elif unique_entities_count == 1:
            has_peer = False
            peer_reason = "Peer Analysis: NO (Single entity dataset - dataset-wide temporal analysis only)."
        else:
            has_peer = False
            peer_reason = "Peer Analysis: NO (Insufficient unique entities)."
    elif grouping_cols:
        primary_group = grouping_cols[0]
        valid_groups = df[primary_group].dropna().astype(str).str.strip()
        if valid_groups.nunique() >= 2 and num_rows >= 5:
            has_peer = True
            peer_reason = f"Peer cohort analysis enabled across {valid_groups.nunique()} groups ({primary_group})."
        else:
            has_peer = False
            peer_reason = "Peer Analysis: NO (Dataset-Wide Only - insufficient group diversity)."
    else:
        has_peer = False
        peer_reason = "Peer Analysis: NO (Dataset-Wide Only - no entity or peer grouping column)."

    # 7. Classify Dataset Scale
    if num_rows < 50:
        data_scale = "small"
    elif num_rows <= 1000:
        data_scale = "medium"
    elif num_rows <= 50000:
        data_scale = "large"
    else:
        data_scale = "very_large"

    is_single_period = (unique_periods_count <= 1)
    is_single_entity = (unique_entities_count <= 1)

    return DatasetProfile(
        num_rows=num_rows,
        num_columns=num_columns,
        raw_columns=raw_columns,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        entity_column=entity_col,
        year_column=year_col,
        amount_columns=amount_cols,
        ratio_columns=ratio_cols,
        percentage_columns=percentage_cols,
        grouping_columns=grouping_cols,
        has_temporal_ordering=has_temporal,
        has_peer_grouping=has_peer,
        temporal_reason=temporal_reason,
        peer_reason=peer_reason,
        is_single_period=is_single_period,
        is_single_entity=is_single_entity,
        unique_entities_count=unique_entities_count,
        unique_periods_count=unique_periods_count,
        data_scale=data_scale,
    )
