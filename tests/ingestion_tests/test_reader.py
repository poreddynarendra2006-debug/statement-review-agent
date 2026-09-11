"""Unit tests for the safe CSV reader."""

import io
import tempfile
import unittest
from pathlib import Path

from src.ingestion.reader import (
    EmptyCSVError,
    InvalidCSVFormatError,
    read_csv_safely,
)


class TestReader(unittest.TestCase):
    def test_read_valid_csv_filepath(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("year,company,revenue\n2022,AAPL,1000\n")
            f_path = f.name
        try:
            df, meta = read_csv_safely(f_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(meta["initial_columns"], 3)
            self.assertEqual(df.iloc[0]["company"], "AAPL")
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_read_from_bytes(self):
        csv_bytes = b"year,company,revenue\n2022,MSFT,2000\n"
        df, meta = read_csv_safely(csv_bytes)
        self.assertEqual(len(df), 1)
        self.assertEqual(meta["initial_rows"], 1)

    def test_read_from_stringio(self):
        sio = io.StringIO("year,company,revenue\n2022,GOOG,3000\n")
        df, meta = read_csv_safely(sio)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["revenue"], "3000")

    def test_read_from_bytesio(self):
        bio = io.BytesIO(b"year;company;revenue\n2022;NVDA;4000\n")
        df, meta = read_csv_safely(bio)
        self.assertEqual(len(df), 1)
        self.assertEqual(meta["delimiter"], ";")

    def test_empty_file_raises_empty_csv_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("")
            f_path = f.name
        try:
            with self.assertRaises(EmptyCSVError):
                read_csv_safely(f_path)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_whitespace_only_raises_empty_csv_error(self):
        with self.assertRaises(EmptyCSVError):
            read_csv_safely("   \n\t  \n  ")

    def test_header_only_raises_empty_csv_error(self):
        csv_content = "year,company,revenue\n"
        with self.assertRaises(EmptyCSVError):
            read_csv_safely(io.StringIO(csv_content))

    def test_binary_file_raises_invalid_csv_format_error(self):
        binary_data = b"\x00\x01\x02\x03\x00\x00PNG\r\n\x1a\n"
        with self.assertRaises(InvalidCSVFormatError):
            read_csv_safely(binary_data)

    def test_nonexistent_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_csv_safely("non_existent_file_xyz_123.csv")


if __name__ == "__main__":
    unittest.main()
