"""Unit tests for the financial dataset validator module."""

import unittest
import numpy as np
import pandas as pd

from src.ingestion.validator import DataValidator


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()

    def test_valid_financial_dataframe(self):
        df = pd.DataFrame({
            "year": [2022, 2021],
            "company": ["AAPL", "AAPL"],
            "revenue": [394328.0, 365817.0],
            "net_income": [99803.0, 94680.0],
        })
        report = self.validator.validate(df)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.is_financial_dataset)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(report.duplicate_rows_count, 0)

    def test_flexible_subset_financial_dataframe(self):
        # A 2-column financial dataset with company and revenue (missing net_income, year, etc.)
        # Should be accepted as flexible financial statement data!
        df = pd.DataFrame({
            "company": ["AAPL", "MSFT"],
            "revenue": [394328.0, 198270.0],
        })
        report = self.validator.validate(df)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.is_financial_dataset)

    def test_rejection_of_unrelated_non_financial_data(self):
        # Completely unrelated student dataset
        df = pd.DataFrame({
            "student_name": ["Alice", "Bob"],
            "age": [20, 22],
            "marks": [85, 92],
            "department": ["Computer Science", "Physics"],
        })
        report = self.validator.validate(df)
        self.assertFalse(report.is_valid)
        self.assertFalse(report.is_financial_dataset)
        self.assertTrue(any("Dataset could not be identified as a compatible financial statement dataset" in e for e in report.errors))

    def test_rejection_when_missing_core_fields(self):
        # Dataset with only category or unmapped fields without any financial metrics or entity
        df = pd.DataFrame({
            "category": ["IT", "Healthcare"],
            "notes": ["sample note 1", "sample note 2"],
        })
        report = self.validator.validate(df)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Dataset could not be identified" in e for e in report.errors))

    def test_extra_columns_detected(self):
        df = pd.DataFrame({
            "year": [2022],
            "company": ["AAPL"],
            "revenue": [1000.0],
            "net_income": [200.0],
            "custom_metadata_tag": ["val1"],
        })
        report = self.validator.validate(df)
        self.assertTrue(report.is_valid)  # Extra columns are preserved as warnings, not hard failures
        self.assertIn("custom_metadata_tag", report.extra_columns)
        self.assertTrue(any("unmapped column" in w for w in report.warnings))

    def test_duplicate_rows_detection(self):
        df = pd.DataFrame({
            "year": [2022, 2022],
            "company": ["AAPL", "AAPL"],
            "revenue": [1000.0, 1000.0],
            "net_income": [200.0, 200.0],
        })
        report = self.validator.validate(df)
        self.assertEqual(report.duplicate_rows_count, 1)

    def test_missing_values_detection(self):
        df = pd.DataFrame({
            "year": [2022, 2021],
            "company": ["AAPL", None],
            "revenue": [1000.0, np.nan],
            "net_income": [200.0, 150.0],
        })
        report = self.validator.validate(df)
        self.assertEqual(report.missing_values_by_column.get("company"), 1)
        self.assertEqual(report.missing_values_by_column.get("revenue"), 1)
        self.assertNotIn("net_income", report.missing_values_by_column)


if __name__ == "__main__":
    unittest.main()
