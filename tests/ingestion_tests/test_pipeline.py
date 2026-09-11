"""Comprehensive end-to-end integration tests for the Ingestion Pipeline.

Covers all 10 required edge cases and scenarios:
1. valid CSV
2. empty CSV
3. missing required columns
4. extra columns
5. duplicate rows
6. missing values
7. invalid numeric values
8. different column names/aliases
9. invalid year
10. successful standardized output and data contracts
11. real financial statements dataset (if available)
"""

import io
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.config import IngestionConfig
from src.ingestion.pipeline import IngestionPipeline, ingest_financial_statement


class TestIngestionPipeline(unittest.TestCase):
    def test_scenario_1_valid_csv(self):
        """Test standard valid CSV file returns success and cleaned data."""
        csv_data = (
            "Year,Company ,Category,Revenue,Net Income\n"
            "2022,AAPL,IT,394328,99803\n"
            "2021,AAPL,IT,365817,94680\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.data)
        self.assertEqual(len(result.data), 2)
        self.assertIn("revenue", result.data.columns)
        self.assertIn("net_income", result.data.columns)

    def test_scenario_2_empty_csv(self):
        """Test empty CSV file returns error status with meaningful message."""
        result = ingest_financial_statement(io.StringIO(""))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertTrue(len(result.metadata.errors) > 0)
        self.assertIn("empty", result.message.lower())

    def test_scenario_3_missing_required_columns(self):
        """Test CSV missing essential required columns returns error and lists missing fields."""
        # Missing 'net_income' and 'revenue'
        csv_data = (
            "Year,Company,Category\n"
            "2022,AAPL,IT\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertTrue(any("Missing core financial metrics" in err or "compatible financial statement" in err for err in result.metadata.errors))

    def test_scenario_4_extra_columns(self):
        """Test CSV containing unexpected extra columns is handled gracefully."""
        csv_data = (
            "Year,Company,Revenue,Net Income,Auditor Notes,Custom Score\n"
            "2022,AAPL,394328,99803,PwC approved,98.5\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertIn(result.status, ("success", "partial_success"))
        self.assertIsNotNone(result.data)
        self.assertIn("auditor_notes", result.data.columns)
        self.assertIn("custom_score", result.data.columns)
        self.assertIn("auditor_notes", result.metadata.unmapped_columns)

    def test_scenario_5_duplicate_rows(self):
        """Test duplicate rows are detected, removed, and logged in the audit trail."""
        csv_data = (
            "Year,Company,Revenue,Net Income\n"
            "2022,AAPL,394328,99803\n"
            "2022,AAPL,394328,99803\n"  # Exact duplicate
            "2021,AAPL,365817,94680\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(len(result.data), 2)
        self.assertEqual(result.metadata.duplicates_removed, 1)
        # Check audit trail recorded deduplication
        dup_mods = [m for m in result.metadata.modifications if m.get("type") == "duplicate_rows_removed"]
        self.assertEqual(len(dup_mods), 1)

    def test_scenario_6_missing_values(self):
        """Test missing values are quantified per column and reported."""
        csv_data = (
            "Year,Company,Revenue,Net Income,Market Cap(in B USD)\n"
            "2022,AAPL,394328,99803,\n"  # Missing Market Cap
            "2021,AAPL,365817,94680,2913.28\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.metadata.missing_values.get("market_cap_b_usd"), 1)
        self.assertEqual(result.metadata.total_missing_values, 1)
        self.assertTrue(pd.isna(result.data["market_cap_b_usd"].iloc[0]))

    def test_scenario_7_invalid_numeric_values(self):
        """Test dirty numbers ($1,000, negatives with parens, strings) are cleaned without silent loss."""
        csv_data = (
            "Year,Company,Revenue,Net Income\n"
            '2022,AAPL,"$394,328.50",(15000.25)\n'
            "2021,AAPL,N/A,corrupted_val\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.data["revenue"].iloc[0], 394328.50)
        self.assertEqual(result.data["net_income"].iloc[0], -15000.25)
        self.assertTrue(np.isnan(result.data["revenue"].iloc[1]))
        self.assertTrue(np.isnan(result.data["net_income"].iloc[1]))
        # Audit log tracks the corrupted value
        corrupted_mods = [m for m in result.metadata.modifications if m.get("original_value") == "corrupted_val"]
        self.assertEqual(len(corrupted_mods), 1)

    def test_scenario_8_different_column_names_and_aliases(self):
        """Test column name standardization across spaces, casing, and aliases."""
        csv_data = (
            "Year,Company ,Market Cap(in B USD),Total Revenue,Profit/Loss,Share Holder Equity\n"
            "2022,AAPL,2066.94,394328,99803,50672\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIn("market_cap_b_usd", result.data.columns)
        self.assertIn("revenue", result.data.columns)
        self.assertIn("net_income", result.data.columns)
        self.assertIn("shareholder_equity", result.data.columns)

    def test_scenario_9_invalid_year(self):
        """Test invalid year detection (e.g. non-numeric, out-of-range)."""
        csv_data = (
            "Year,Company,Revenue,Net Income\n"
            "1800,AAPL,100,50\n"        # Year too low
            "2022,MSFT,200,80\n"
            "not_a_year,GOOG,300,90\n"  # String
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        # Invalid years are flagged in audit log
        year_mods = [m for m in result.metadata.modifications if m.get("type") == "invalid_year"]
        self.assertEqual(len(year_mods), 2)

    def test_scenario_10_successful_standardized_output_and_dict_access(self):
        """Test dictionary subscripting and typed properties for orchestrator contract."""
        csv_data = (
            "Year,Company,Revenue,Net Income\n"
            "2022,AAPL,394328,99803\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))

        # Test dictionary subscripting
        self.assertEqual(result["status"], "success")
        self.assertIsInstance(result["data"], pd.DataFrame)
        self.assertIsInstance(result["metadata"], dict)
        self.assertEqual(result["metadata"]["rows"], 1)

        # Test typed object attribute access
        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata.rows, 1)

        # Test to_dict conversion
        as_dict = result.to_dict()
        self.assertEqual(as_dict["status"], "success")
        self.assertIn("data", as_dict)
        self.assertIn("metadata", as_dict)

    def test_sample_csv_file(self):
        """Test sample CSV file located in data/samples/."""
        sample_path = Path("data/samples/sample_financial_statement.csv")
        self.assertTrue(sample_path.exists())

        result = ingest_financial_statement(sample_path)
        self.assertIn(result.status, ("success", "partial_success"))
        self.assertIsNotNone(result.data)
        self.assertEqual(len(result.data), 5)
        self.assertEqual(len(result.data.columns), 23)

    def test_real_dataset_if_present(self):
        """Test the real 161-row Kaggle dataset if present on this machine."""
        candidate_paths = [
            Path(r"C:\Users\vpnan\Downloads\archive (4)\Financial Statements.csv"),
            Path("data/raw/Financial Statements(1).csv"),
            Path("data/raw/Financial Statements.csv"),
        ]
        real_path = None
        for p in candidate_paths:
            if p.exists():
                real_path = p
                break

        if real_path:
            result = ingest_financial_statement(real_path)
            self.assertIn(result.status, ("success", "partial_success"))
            self.assertIsNotNone(result.data)
            self.assertEqual(len(result.data), 161)
            self.assertEqual(len(result.data.columns), 23)
            self.assertIn("market_cap_b_usd", result.data.columns)
            self.assertIn("cash_flow_operating", result.data.columns)
            self.assertIn("cash_flow_financing", result.data.columns)
            # The real dataset has 1 missing value in Market Cap
            self.assertEqual(result.metadata.missing_values.get("market_cap_b_usd"), 1)


if __name__ == "__main__":
    unittest.main()
