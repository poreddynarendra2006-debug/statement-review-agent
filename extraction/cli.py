"""Command-Line Interface for the Financial Statement Data Ingestion Module.

Allows users, developers, and orchestrator scripts to pass any financial statement CSV
dynamically without hardcoding file paths.

Usage:
    python -m src.ingestion.cli --file data/samples/sample_financial_statement.csv --summary
    python -m src.ingestion.cli --file "C:/path/to/any/statement.csv" --output standardized.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import IngestionConfig
from .pipeline import ingest_financial_statement


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest, standardize, and validate financial statement CSV datasets dynamically."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to the input financial statement CSV file (dynamic path).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save the standardized CSV output file.",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print detailed ingestion summary report to stdout.",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output ingestion metadata as JSON to stdout (for orchestrator pipes).",
    )
    parser.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="Drop columns not present in canonical schema (default is to keep them).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    config = IngestionConfig(
        drop_unmapped_columns=args.drop_unmapped,
    )

    # Run ingestion dynamically
    result = ingest_financial_statement(input_path, config=config)

    # Save output if requested and data is available
    if args.output and result.data is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_csv(out_path, index=False)
        print(f"Standardized data saved to: {out_path}")

    # Display outputs
    if args.json:
        # JSON output for orchestrator integration
        print(json.dumps(result.to_dict(), default=str, indent=2))
    elif args.summary or not args.output:
        result.print_summary()

    if result.status == "error":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
