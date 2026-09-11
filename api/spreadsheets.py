"""Excel uploads, turned into the CSV text Data Ingestion reads.

Ingestion reads CSV only. Company workbooks are rarely a bare table: they carry
a title, a units line or blank rows above the header, and sometimes a notes
sheet before the figures. This finds the table and hands Ingestion its CSV
equivalent, so nothing downstream needs to know the file was a workbook.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd

#: A header row fills at least this share of the sheet's widest row with text.
HEADER_FILL = 0.6


class NoTableFound(ValueError):
    """The workbook opened, but no sheet holds a header row with data under it."""


def _table(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The sheet's table under its header row, or None when it has none."""
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    if frame.empty:
        return None

    width = int(frame.notna().sum(axis=1).max())
    for position in range(len(frame)):
        cells = frame.iloc[position].dropna()
        if len(cells) >= max(2, HEADER_FILL * width) and all(isinstance(c, str) for c in cells):
            table = frame.iloc[position + 1:].copy()
            if table.empty:
                return None
            table.columns = [str(h).strip() if pd.notna(h) else f"column_{i + 1}"
                             for i, h in enumerate(frame.iloc[position])]
            return table
    return None


def _numeric_cells(table: pd.DataFrame) -> int:
    values = pd.Series(table.to_numpy().ravel())
    return int(pd.to_numeric(values, errors="coerce").notna().sum())


def excel_to_csv(content: bytes) -> bytes:
    """Return the workbook's table as CSV, with anything above its header dropped.

    When several sheets hold a table, the one with the most figures wins, so a
    notes sheet placed first does not hide the statements behind it.

    Raises:
        NoTableFound: no sheet has a header row followed by at least one row.
        Exception: whatever the Excel reader raises for a file that is not a workbook.
    """
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None,
                           dtype=object, engine="openpyxl")
    tables = [t for t in (_table(frame) for frame in sheets.values()) if t is not None]
    if not tables:
        raise NoTableFound("no sheet holds a table with a header row")
    return max(tables, key=_numeric_cells).to_csv(index=False).encode("utf-8")
