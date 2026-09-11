"""Test suite specifically covering the 10 multi-dataset scenarios.

Verifies that the Data Ingestion module is completely dataset-agnostic
across varied financial statements, naming conventions, subsets, and formats,
while rejecting non-financial datasets with clear explanations.
"""

import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.config import IngestionConfig
from src.ingestion.pipeline import ingest_financial_statement


class TestMultiDatasets(unittest.TestCase):
    def test_1_current_financial_statements(self):
        """TEST 1: Current Financial Statements dataset -> Expected: SUCCESS."""
        # Use either sample or raw dataset if present
        sample_path = Path("data/samples/sample_financial_statement.csv")
        result = ingest_financial_statement(sample_path)
        self.assertIn(result.status, ("success", "partial_success"))
        self.assertIsNotNone(result.data)
        self.assertEqual(len(result.data), 5)
        self.assertIn("market_cap_b_usd", result.data.columns)
        self.assertIn("cash_flow_operating", result.data.columns)

    def test_2_renamed_columns(self):
        """TEST 2: Same financial data but renamed columns.

        Example:
        Company Name, Fiscal Year, Total Sales, Profit After Tax, EPS
        Expected: SUCCESS
        """
        csv_data = (
            "Company Name,Fiscal Year,Total Sales,Profit After Tax,EPS\n"
            "AAPL,2022,394328,99803,6.11\n"
            "MSFT,2022,198270,72738,9.65\n"
            "GOOG,2022,282836,59972,4.56\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.data)
        # Verify mapped to canonical names
        self.assertIn("company", result.data.columns)
        self.assertIn("year", result.data.columns)
        self.assertIn("revenue", result.data.columns)
        self.assertIn("net_income", result.data.columns)
        self.assertIn("earnings_per_share", result.data.columns)

    def test_3_additional_columns(self):
        """TEST 3: Different financial dataset with additional columns.

        Expected: SUCCESS + additional/unmapped columns reported without deletion.
        """
        csv_data = (
            "Company,Year,Revenue,Net Income,CEO Name,Headquarters,Stock Exchange\n"
            "AAPL,2022,394328,99803,Tim Cook,Cupertino CA,NASDAQ\n"
            "MSFT,2022,198270,72738,Satya Nadella,Redmond WA,NASDAQ\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertIn(result.status, ("success", "partial_success"))
        self.assertIsNotNone(result.data)
        # Mapped columns
        self.assertIn("company", result.data.columns)
        self.assertIn("revenue", result.data.columns)
        self.assertIn("net_income", result.data.columns)
        # Additional columns preserved
        self.assertIn("ceo_name", result.data.columns)
        self.assertIn("headquarters", result.data.columns)
        self.assertIn("stock_exchange", result.data.columns)
        # Reported in metadata
        self.assertIn("ceo_name", result.metadata.unmapped_columns)
        self.assertIn("headquarters", result.metadata.unmapped_columns)

    def test_4_missing_optional_columns(self):
        """TEST 4: Financial dataset missing optional columns (only 4 core columns).

        Expected: SUCCESS if enough core financial information exists.
        """
        csv_data = (
            "Company,Year,Revenue,Net Income\n"
            "AAPL,2022,394328,99803\n"
            "MSFT,2022,198270,72738\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.data)
        self.assertEqual(len(result.data), 2)
        self.assertEqual(len(result.data.columns), 4)

    def test_5_missing_too_many_core_fields(self):
        """TEST 5: Financial dataset missing too many core fields (e.g. only notes or non-financial).

        Expected: REJECT with clear explanation.
        """
        csv_data = (
            "Section,Auditor Notes,Filing Code\n"
            "Item 1,Clean audit,10-K\n"
            "Item 2,Reviewed,10-Q\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertFalse(result.metadata.is_financial_dataset)
        self.assertTrue(
            any("Dataset could not be identified as a compatible financial statement dataset" in e for e in result.metadata.errors)
        )

    def test_6_completely_unrelated_dataset(self):
        """TEST 6: Completely unrelated non-financial dataset.

        Example: Name, Age, Marks, Department
        Expected: REJECT with clear explanation.
        """
        csv_data = (
            "Name,Age,Marks,Department\n"
            "John,21,88,Computer Science\n"
            "Sara,22,94,Electrical Engineering\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertFalse(result.metadata.is_financial_dataset)
        self.assertIn("compatible financial statement dataset", result.message)

    def test_7_different_column_order(self):
        """TEST 7: Different column order.

        Expected: SUCCESS.
        """
        csv_data = (
            "Net Income,EPS,Year,Revenue,Company\n"
            "99803,6.11,2022,394328,AAPL\n"
            "72738,9.65,2022,198270,MSFT\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.data)
        # All columns mapped regardless of order
        self.assertEqual(result.data.iloc[0]["company"], "AAPL")
        self.assertEqual(result.data.iloc[0]["revenue"], 394328.0)
        self.assertEqual(result.data.iloc[0]["net_income"], 99803.0)

    def test_8_different_capitalization_and_spaces(self):
        """TEST 8: Different capitalization and spaces in headers and values.

        Example: "  yEaR  ", "cOmPaNy  nAmE", "  tOtAl  rEvEnUe  ", " nEt  iNcOmE "
        Expected: SUCCESS.
        """
        csv_data = (
            "  yEaR  , cOmPaNy  nAmE ,   tOtAl  rEvEnUe  , nEt  iNcOmE \n"
            " 2022 ,  AAPL  , 394328 , 99803 \n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIn("year", result.data.columns)
        self.assertIn("company", result.data.columns)
        self.assertIn("revenue", result.data.columns)
        self.assertIn("net_income", result.data.columns)
        # Value spaces trimmed
        self.assertEqual(result.data.iloc[0]["company"], "AAPL")

    def test_9_numeric_values_with_currency_commas_percentages_and_parens(self):
        """TEST 9: Numeric values containing commas, currency symbols, percentages, and accounting parens.

        Expected: SUCCESS with correct normalization.
        """
        csv_data = (
            'Company,Year,Revenue,Net Income,ROE,Debt/Equity\n'
            'AAPL,2022,"$1,234,567.80","(50,000.25)","15.5%","1.23"\n'
            'MSFT,2022,$987654.00,$45000,12.0%,0.85\n'
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.data)
        # Check parsed numbers
        self.assertEqual(result.data.iloc[0]["revenue"], 1234567.80)
        self.assertEqual(result.data.iloc[0]["net_income"], -50000.25)
        self.assertEqual(result.data.iloc[0]["roe"], 15.5)
        self.assertEqual(result.data.iloc[0]["debt_equity_ratio"], 1.23)

    def test_10_ambiguous_column_name(self):
        """TEST 10: Ambiguous column name -> Embedding matching first, LLM fallback if necessary."""
        # Case A: Embedding matching handles reworded header without LLM
        csv_data = (
            "Company,Year,Revenue,Net Income,Operating Cashflow Stream\n"
            "AAPL,2022,394328,99803,122151\n"
        )
        result = ingest_financial_statement(io.StringIO(csv_data))
        self.assertEqual(result.status, "success")
        self.assertIn("cash_flow_operating", result.data.columns)
        self.assertEqual(result.metadata.mapping_methods["Operating Cashflow Stream"], "embedding")

        # Case B: Completely opaque name uses LLM fallback when provided
        def mock_llm_resolver(raw_col, samples, candidates):
            if "opaque_firm_sales" in raw_col and "revenue" in candidates:
                return "revenue"
            return None

        custom_config = IngestionConfig(
            embedding_threshold=0.95,
            llm_resolver=mock_llm_resolver,
        )
        opaque_csv = (
            "Company,Year,opaque_firm_sales,Net Income\n"
            "AAPL,2022,394328,99803\n"
        )
        result_llm = ingest_financial_statement(io.StringIO(opaque_csv), config=custom_config)
        self.assertEqual(result_llm.status, "success")
        self.assertIn("revenue", result_llm.data.columns)
        self.assertEqual(result_llm.metadata.mapping_methods["opaque_firm_sales"], "llm")


if __name__ == "__main__":
    unittest.main()
