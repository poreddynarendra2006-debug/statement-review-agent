"""Safe, dynamic CSV reader supporting file paths, byte buffers, and stream uploads.

Validates that the provided input is indeed a valid CSV file, detects character encoding,
sniffs the delimiter, and prevents security/formatting errors.
"""

from __future__ import annotations

import csv
import io
import os
import pathlib
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import pandas as pd


class IngestionError(Exception):
    """Base exception for data ingestion errors."""

    pass


class EmptyCSVError(IngestionError):
    """Raised when an uploaded file is empty or contains no data rows."""

    pass


class InvalidCSVFormatError(IngestionError):
    """Raised when the uploaded file is not a valid CSV or is unparseable."""

    pass


class CSVReadError(IngestionError):
    """Raised when an error occurs during reading the CSV data."""

    pass


def _detect_encoding(raw_bytes: bytes) -> str:
    """Detects character encoding from raw byte sample.

    Checks BOM headers first, then attempts utf-8, latin-1, cp1252.
    """
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw_bytes.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw_bytes.startswith(b"\xfe\xff"):
        return "utf-16-be"

    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            raw_bytes.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _sniff_delimiter(text_sample: str) -> str:
    """Sniffs delimiter from text sample using Python's csv.Sniffer."""
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Default standard CSV delimiter
        return ","


def read_csv_safely(
    source: Union[str, pathlib.Path, bytes, io.BytesIO, io.StringIO, BinaryIO, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reads a CSV file or buffer safely and returns a DataFrame along with source metadata.

    Args:
        source: Dynamic file source. Can be:
            - str or Path: local path to a CSV file
            - bytes: raw file content
            - io.BytesIO or io.StringIO: in-memory streams
            - File-like object with a .read() method (e.g. FastAPI / Flask / Streamlit UploadedFile)

    Returns:
        Tuple of (DataFrame, metadata_dict)

    Raises:
        EmptyCSVError: If file is empty or contains no rows.
        InvalidCSVFormatError: If file is corrupted, binary non-CSV, or unparseable.
        CSVReadError: If pandas fails to read the source.
    """
    source_name = "in-memory-stream"
    raw_bytes: Optional[bytes] = None

    # Handle string or pathlib.Path
    if isinstance(source, (str, pathlib.Path)):
        # Check if it's an inline string containing CSV text (has newlines) or empty whitespace
        if isinstance(source, str):
            if not source.strip():
                raise EmptyCSVError("Uploaded CSV content is empty or contains only whitespace.")
            if "\n" in source or "\r" in source:
                raw_bytes = source.encode("utf-8")
                source_name = "inline_csv_string"
            else:
                path = pathlib.Path(source)
                source_name = path.name
                if not path.exists():
                    raise FileNotFoundError(f"Financial statement file not found: {source}")
                if path.is_dir():
                    raise InvalidCSVFormatError(f"Provided path is a directory, not a file: {source}")
                if path.stat().st_size == 0:
                    raise EmptyCSVError(f"Uploaded file '{source_name}' is empty (0 bytes).")
                try:
                    with open(path, "rb") as f:
                        raw_bytes = f.read()
                except Exception as e:
                    raise CSVReadError(f"Failed to read file '{source}': {str(e)}") from e
        else:
            path = pathlib.Path(source)
            source_name = path.name
            if not path.exists():
                raise FileNotFoundError(f"Financial statement file not found: {source}")
            if path.is_dir():
                raise InvalidCSVFormatError(f"Provided path is a directory, not a file: {source}")
            if path.stat().st_size == 0:
                raise EmptyCSVError(f"Uploaded file '{source_name}' is empty (0 bytes).")
            try:
                with open(path, "rb") as f:
                    raw_bytes = f.read()
            except Exception as e:
                raise CSVReadError(f"Failed to read file '{source}': {str(e)}") from e

    # Handle raw bytes or bytearray
    elif isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
        source_name = "bytes_buffer"

    # Handle io.BytesIO
    elif isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
        source_name = getattr(source, "name", "bytes_io_stream")

    # Handle io.StringIO
    elif isinstance(source, io.StringIO):
        text_content = source.getvalue()
        raw_bytes = text_content.encode("utf-8")
        source_name = getattr(source, "name", "string_io_stream")

    # Handle file-like objects (e.g. UploadedFile, open file descriptors)
    elif hasattr(source, "read"):
        source_name = getattr(source, "name", getattr(source, "filename", "uploaded_file"))
        try:
            content = source.read()
            # Reset seek position if seekable
            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except Exception:
                    pass
            if isinstance(content, str):
                raw_bytes = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                raw_bytes = bytes(content)
            else:
                raise InvalidCSVFormatError(f"Unsupported content type from stream: {type(content)}")
        except Exception as e:
            raise CSVReadError(f"Failed to read stream from '{source_name}': {str(e)}") from e
    else:
        raise InvalidCSVFormatError(f"Unsupported source type for CSV ingestion: {type(source)}")

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise EmptyCSVError(f"Uploaded CSV file '{source_name}' is empty or contains only whitespace.")

    # Check for binary non-text files (e.g. PDF, ZIP, ELF, EXE, etc.)
    # Null bytes (\x00) within the first 1024 bytes are a classic indicator of binary non-text formats
    sample_header = raw_bytes[:1024]
    if b"\x00" in sample_header:
        raise InvalidCSVFormatError(
            f"The uploaded file '{source_name}' appears to be a binary file, not a valid CSV."
        )

    # Detect encoding
    encoding = _detect_encoding(raw_bytes)

    try:
        text = raw_bytes.decode(encoding)
    except Exception as e:
        # Fallback to latin-1 if decode fails
        try:
            encoding = "latin1"
            text = raw_bytes.decode("latin1")
        except Exception as e2:
            raise InvalidCSVFormatError(
                f"Failed to decode CSV text with detected encoding '{encoding}': {str(e2)}"
            ) from e2

    # Check for empty or header-only content
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise EmptyCSVError(f"Uploaded CSV '{source_name}' contains no readable lines.")

    # Detect delimiter
    sample_text = "\n".join(lines[:10])
    delimiter = _sniff_delimiter(sample_text)

    # Parse with pandas
    try:
        df = pd.read_csv(
            io.StringIO(text),
            delimiter=delimiter,
            dtype=object,  # Read all columns as object initially to allow safe cleaning & validation
            skipinitialspace=True,
            on_bad_lines="warn",
        )
    except Exception as e:
        raise CSVReadError(f"Pandas failed to parse CSV file '{source_name}': {str(e)}") from e

    if df.empty:
        raise EmptyCSVError(
            f"CSV file '{source_name}' has headers but no data rows (0 rows detected)."
        )

    metadata = {
        "source_name": str(source_name),
        "encoding": encoding,
        "delimiter": delimiter,
        "raw_bytes_size": len(raw_bytes),
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
    }

    return df, metadata
