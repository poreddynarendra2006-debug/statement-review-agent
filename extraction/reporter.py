"""Audit trail and report generator for the ingestion module.

Maintains transparency and records all data transformations, deduplications,
missing value detections, confidence scores, and warnings so downstream agents
and users can inspect them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IngestionMetadata:
    """Structured ingestion metadata returned to the Orchestrator and downstream agents."""

    source_name: str
    rows: int
    columns: int
    initial_rows: int
    initial_columns: int
    final_rows: int = 0
    final_columns: int = 0
    mapped_columns: Dict[str, str] = field(default_factory=dict)
    unmapped_columns: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    mapping_methods: Dict[str, str] = field(default_factory=dict)
    column_collisions: List[Dict[str, Any]] = field(default_factory=list)
    ambiguous_columns: List[Dict[str, Any]] = field(default_factory=list)
    units: Dict[str, str] = field(
        default_factory=lambda: {
            "revenue": "millions USD",
            "market_cap": "billions USD",
            "ratios": "percentages",
        }
    )
    missing_values: Dict[str, int] = field(default_factory=dict)
    total_missing_values: int = 0
    duplicates_removed: int = 0
    is_financial_dataset: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.final_rows == 0:
            self.final_rows = self.rows
        if self.final_columns == 0:
            self.final_columns = self.columns

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata to standard dictionary."""
        return asdict(self)

    def generate_summary_text(self) -> str:
        """Generates a human-readable summary text of the ingestion results."""
        lines = [
            "============================================================",
            f" FINANCIAL STATEMENT INGESTION SUMMARY: {self.source_name}",
            "============================================================",
            f" Status:            {'COMPATIBLE' if self.is_financial_dataset and not self.errors else 'REJECTED'}",
            f" Initial Rows/Cols: {self.initial_rows} rows x {self.initial_columns} cols",
            f" Final Output:      {self.final_rows} rows x {self.final_columns} cols",
            f" Duplicates Removed: {self.duplicates_removed}",
            f" Total Missing Values: {self.total_missing_values}",
        ]

        if self.units:
            lines.append(" Standard Units:")
            for k, v in self.units.items():
                lines.append(f"   - {k}: {v}")

        if self.mapped_columns:
            lines.append(f" Mapped Financial Columns ({len(self.mapped_columns)}):")
            for raw, canon in self.mapped_columns.items():
                score = self.confidence_scores.get(raw, 1.0)
                method = self.mapping_methods.get(raw, "unknown")
                lines.append(f"   - '{raw}' -> '{canon}' (Tier: {method}, confidence: {score})")

        if self.unmapped_columns:
            lines.append(f" Unmapped / Additional Columns ({len(self.unmapped_columns)}):")
            for col in self.unmapped_columns:
                lines.append(f"   - {col}")

        if self.ambiguous_columns:
            lines.append(f" Ambiguous Columns ({len(self.ambiguous_columns)}):")
            for amb in self.ambiguous_columns:
                lines.append(f"   ? {amb.get('column', '')}: candidates={amb.get('candidates', [])}")

        if self.column_collisions:
            lines.append(f" Column Collisions Detected ({len(self.column_collisions)}):")
            for c in self.column_collisions:
                lines.append(f"   ! {c.get('reason', '')}")

        if self.missing_values:
            lines.append(" Missing Values by Column:")
            for col, count in sorted(self.missing_values.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"   - {col}: {count} missing")

        if self.warnings:
            lines.append(f" Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"   ! {w}")

        if self.errors:
            lines.append(f" Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"   X {e}")

        if self.modifications:
            lines.append(f" Modifications Recorded: {len(self.modifications)} event(s)")

        lines.append("============================================================")
        return "\n".join(lines)
