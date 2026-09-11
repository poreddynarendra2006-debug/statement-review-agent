"""Financial Statement Data Ingestion Module.

Part of the Financial Review Agent project:
USER -> Upload Statement -> DATA INGESTION -> ORCHESTRATOR
"""

from .config import IngestionConfig
from .pipeline import (
    FinancialRecord,
    IngestionPipeline,
    IngestionResult,
    ingest_financial_statement,
)
from .reader import (
    CSVReadError,
    EmptyCSVError,
    IngestionError,
    InvalidCSVFormatError,
)
from .reporter import IngestionMetadata
from .schema import CANONICAL_COLUMNS, ColumnSpec

__all__ = [
    "ingest_financial_statement",
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
    "FinancialRecord",
    "IngestionMetadata",
    "CANONICAL_COLUMNS",
    "ColumnSpec",
    "IngestionError",
    "EmptyCSVError",
    "InvalidCSVFormatError",
    "CSVReadError",
]
