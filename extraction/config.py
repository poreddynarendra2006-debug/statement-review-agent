"""Configuration settings for the Data Ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class IngestionConfig:
    """Pipeline configuration options."""

    # Threshold for Tier 2 embedding-based column matching
    embedding_threshold: float = 0.65

    # Margin between top two embedding candidates to detect ambiguity
    # If (top_score - second_score) < ambiguity_margin and second_score >= 0.35, escalates to Tier 3 LLM
    ambiguity_margin: float = 0.05

    # Whether to represent percentage values as decimals (e.g. 15% -> 0.15)
    # Default False stores nominal percentage points (e.g. 15% -> 15.0) matching standard financial statements
    percentage_as_decimal: bool = False

    # Whether to remove exact duplicate rows while recording them in the audit trail
    deduplicate_rows: bool = True

    # Whether to coerce invalid numeric values to NaN while recording them in the audit trail
    coerce_numeric_errors: bool = True

    # If True, drops columns not present in canonical schema.
    # If False (recommended default), preserves unmapped extra columns with normalized names
    drop_unmapped_columns: bool = False

    # Optional custom embedding function: Callable[[List[str]], np.ndarray]
    custom_embedding_fn: Optional[Callable[[List[str]], Any]] = None

    # Optional custom LLM resolver function: Callable[[str, List[Any], List[str]], Optional[str]]
    llm_resolver: Optional[Callable[[str, List[Any], List[str]], Optional[str]]] = None
