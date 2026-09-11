import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.config import IngestionConfig
from src.ingestion.mapper import SemanticColumnMapper
from src.ingestion.pipeline import FinancialRecord, IngestionPipeline, ingest_financial_statement
from src.ingestion.schema import (
    CANONICAL_COLUMNS,
    is_amount_to_ratio_violation,
)


class TestMappingCorrectness(unittest.TestCase):
    """Verifies that columns are mapped accurately without misattribution or silent overwrite."""

    def setUp(self):
        self.clean_csv_path = Path("data/samples/dummy_statements_clean.csv")
        self.defective_csv_path = Path("data/samples/dummy_statements_defective.csv")
        self.kaggle_csv_path = Path("data/samples/kaggle_financial_statements.csv")

    def test_dummy_mapping_total_assets_equals_raw(self):
        """On dummy file, total_assets equals raw 'Total Assets' values exactly."""
        raw_df = pd.read_csv(self.clean_csv_path)
        res = ingest_financial_statement(self.clean_csv_path)
        self.assertIn(res.status, ("success", "partial_success"))
        self.assertIn("total_assets", res.data.columns)
        raw_values = [float(v) for v in raw_df["Total Assets"]]
        ingested_values = res.data["total_assets"].tolist()
        self.assertEqual(ingested_values, raw_values)

    def test_dummy_mapping_current_assets_operating_income_ebitda(self):
        """Current Assets, Operating Income, and EBITDA map to distinct canonical fields."""
        raw_df = pd.read_csv(self.clean_csv_path)
        res = ingest_financial_statement(self.clean_csv_path)
        self.assertIn("current_assets", res.data.columns)
        self.assertIn("operating_income", res.data.columns)
        self.assertIn("ebitda", res.data.columns)

        self.assertEqual(
            res.data["operating_income"].tolist(),
            [float(v) for v in raw_df["Operating Income"]],
        )
        self.assertEqual(
            res.data["ebitda"].tolist(),
            [float(v) for v in raw_df["EBITDA"]],
        )
        self.assertEqual(
            res.data["current_assets"].tolist(),
            [float(v) for v in raw_df["Current Assets"]],
        )

    def test_dummy_file_absent_roa_and_current_ratio(self):
        """On dummy file, roa and current_ratio are absent from columns because file has no such columns."""
        res = ingest_financial_statement(self.clean_csv_path)
        self.assertNotIn("roa", res.data.columns)
        self.assertNotIn("current_ratio", res.data.columns)

    def test_no_two_raw_columns_map_to_same_canonical(self):
        """No two raw columns map to the same canonical field; collisions are reported."""
        csv_data = (
            "Year,Company,Revenue,Total Revenue,Net Income\n"
            "2022,AAPL,394328,394328,99803\n"
        )
        res = ingest_financial_statement(io.StringIO(csv_data))
        mapped_targets = list(res.metadata.mapped_columns.values())
        self.assertEqual(len(mapped_targets), len(set(mapped_targets)))
        self.assertTrue(len(res.metadata.column_collisions) > 0)

    def test_deliberately_ambiguous_header_left_unmapped(self):
        """A deliberately ambiguous header with competing candidates is reported and left unmapped, not guessed."""
        mapper = SemanticColumnMapper(
            embedding_threshold=0.50,
            ambiguity_margin=0.99,
            llm_resolver=None,
        )
        raw_cols = ["Year", "Company", "Revenue", "General Corporate Expenses Outflow"]
        rename_map, results, collisions = mapper.map_columns(raw_cols)

        amb_res = next(r for r in results if r.raw_name == "General Corporate Expenses Outflow")
        self.assertIsNone(amb_res.canonical_name)
        self.assertIn(amb_res.method, ("ambiguous", "unmapped"))

    def test_type_guard_amount_to_ratio_rejection(self):
        """Type guard: amount headers are rejected from mapping to ratio fields."""
        amount_headers = [
            "Total Assets",
            "Current Assets",
            "Operating Income",
            "Total Liabilities",
            "Current Liabilities",
            "Cash",
            "Debt",
            "Taxes",
            "Operating Expenses",
        ]
        ratio_targets = [
            "roa",
            "roe",
            "roi",
            "current_ratio",
            "debt_equity_ratio",
            "net_profit_margin",
            "return_on_tangible_equity",
        ]

        for hdr in amount_headers:
            for ratio_field in ratio_targets:
                self.assertTrue(
                    is_amount_to_ratio_violation(hdr, ratio_field),
                    f"Expected violation for header '{hdr}' mapped to ratio '{ratio_field}'",
                )


class TestSchemaCoverageAndValidation(unittest.TestCase):
    """Verifies complete 37 canonical fields support and strict required fields validation."""

    def setUp(self):
        self.clean_csv_path = Path("data/samples/dummy_statements_clean.csv")
        self.kaggle_csv_path = Path("data/samples/kaggle_financial_statements.csv")

    def test_canonical_schema_has_exactly_37_fields(self):
        """CANONICAL_COLUMNS defines exactly 37 fields (23 Kaggle + 14 Statement)."""
        self.assertEqual(len(CANONICAL_COLUMNS), 37)

    def test_dummy_file_produces_all_14_statement_fields(self):
        """Dummy file produces all 14 statement fields under their canonical names."""
        statement_fields = [
            "currency",
            "cost_of_revenue",
            "operating_expenses",
            "operating_income",
            "pre_tax_income",
            "taxes",
            "total_assets",
            "current_assets",
            "total_liabilities",
            "current_liabilities",
            "cash",
            "debt",
            "beginning_cash",
            "ending_cash",
        ]
        res = ingest_financial_statement(self.clean_csv_path)
        self.assertIn(res.status, ("success", "partial_success"))
        for field_name in statement_fields:
            self.assertIn(
                field_name,
                res.data.columns,
                f"Missing statement field '{field_name}' in standardized dummy statement",
            )

    def test_kaggle_file_produces_23_fields_with_statement_fields_absent(self):
        """Kaggle file produces 23 fields and statement fields are absent without error."""
        res = ingest_financial_statement(self.kaggle_csv_path)
        self.assertIn(res.status, ("success", "partial_success"))
        self.assertEqual(len(res.data.columns), 23)

        statement_only_fields = [
            "cost_of_revenue",
            "operating_expenses",
            "operating_income",
            "pre_tax_income",
            "taxes",
            "total_assets",
            "current_assets",
            "total_liabilities",
            "current_liabilities",
            "cash",
            "debt",
            "beginning_cash",
            "ending_cash",
        ]
        for f in statement_only_fields:
            self.assertNotIn(f, res.data.columns)

    def test_missing_required_fields_rejected_with_clear_message(self):
        """Files missing year, company, or revenue are rejected with a clear message."""
        csv_no_year = "Company,Revenue,Net Income\nAAPL,394328,99803\n"
        res1 = ingest_financial_statement(io.StringIO(csv_no_year))
        self.assertEqual(res1.status, "error")
        self.assertIsNone(res1.data)
        self.assertIn("year", res1.message.lower())

        csv_no_company = "Year,Revenue,Net Income\n2022,394328,99803\n"
        res2 = ingest_financial_statement(io.StringIO(csv_no_company))
        self.assertEqual(res2.status, "error")
        self.assertIsNone(res2.data)
        self.assertIn("company", res2.message.lower())

        csv_no_revenue = "Year,Company,Net Income\n2022,AAPL,99803\n"
        res3 = ingest_financial_statement(io.StringIO(csv_no_revenue))
        self.assertEqual(res3.status, "error")
        self.assertIsNone(res3.data)
        self.assertIn("revenue", res3.message.lower())


class TestValuesPreservation(unittest.TestCase):
    """Verifies that data cleaning never alters or recalculates values."""

    def setUp(self):
        self.defective_csv_path = Path("data/samples/dummy_statements_defective.csv")
        self.kaggle_csv_path = Path("data/samples/kaggle_financial_statements.csv")

    def test_defective_file_values_faithful_and_unaltered(self):
        """Every mapped value in the defective file matches raw value without modification."""
        raw_df = pd.read_csv(self.defective_csv_path)
        res = ingest_financial_statement(self.defective_csv_path)
        self.assertIn(res.status, ("success", "partial_success"))

        for raw_col in raw_df.columns:
            canon = res.metadata.mapped_columns.get(raw_col)
            if canon:
                raw_series = raw_df[raw_col].tolist()
                ingested_series = res.data[canon].tolist()
                for idx, (r_val, i_val) in enumerate(zip(raw_series, ingested_series)):
                    if pd.isna(r_val):
                        self.assertTrue(pd.isna(i_val))
                    else:
                        if isinstance(i_val, (int, float)):
                            self.assertEqual(float(r_val), float(i_val), f"Mismatch at row {idx} in {canon}")
                        else:
                            self.assertEqual(str(r_val).strip(), str(i_val).strip())

    def test_kaggle_blank_market_cap_is_none_in_records(self):
        """Blank cell in Kaggle Market Cap evaluates to None in to_records(), not 0 and not NaN."""
        res = ingest_financial_statement(self.kaggle_csv_path)
        records = res.to_records()
        self.assertEqual(len(records), 161)

        pypl_2014 = next(r for r in records if r.company == "PYPL" and r.year == 2014)
        self.assertIsNone(pypl_2014.market_cap_b_usd)
        self.assertNotEqual(pypl_2014.market_cap_b_usd, 0)
        self.assertFalse(isinstance(pypl_2014.market_cap_b_usd, float) and np.isnan(pypl_2014.market_cap_b_usd))

    def test_units_unchanged(self):
        """Units are unchanged: Apple 2022 revenue stays 394328, ROE stays 196.9589."""
        res = ingest_financial_statement(self.kaggle_csv_path)
        records = res.to_records()
        aapl_2022 = next(r for r in records if r.company == "AAPL" and r.year == 2022)
        self.assertEqual(aapl_2022.revenue, 394328.0)
        self.assertAlmostEqual(aapl_2022.roe, 196.9589, places=4)


class TestRecordsInterface(unittest.TestCase):
    """Verifies IngestionResult.to_records() data contracts for downstream agents."""

    def setUp(self):
        self.clean_csv_path = Path("data/samples/dummy_statements_clean.csv")
        self.kaggle_csv_path = Path("data/samples/kaggle_financial_statements.csv")

    def test_to_records_counts(self):
        """to_records() returns 161 records for Kaggle and 480 for dummy file."""
        kaggle_res = ingest_financial_statement(self.kaggle_csv_path)
        dummy_res = ingest_financial_statement(self.clean_csv_path)
        self.assertEqual(len(kaggle_res.to_records()), 161)
        self.assertEqual(len(dummy_res.to_records()), 480)

    def test_record_total_assets_none_on_kaggle_and_does_not_raise(self):
        """record.total_assets is None on a Kaggle record and never raises AttributeError."""
        kaggle_res = ingest_financial_statement(self.kaggle_csv_path)
        records = kaggle_res.to_records()
        first_rec = records[0]

        self.assertIsNone(first_rec.total_assets)
        self.assertIsNone(first_rec.current_assets)
        self.assertIsNone(first_rec.operating_expenses)

    def test_record_types_company_str_year_int(self):
        """record.company is a str and record.year is an int."""
        kaggle_res = ingest_financial_statement(self.kaggle_csv_path)
        first_rec = kaggle_res.to_records()[0]
        self.assertIsInstance(first_rec.company, str)
        self.assertIsInstance(first_rec.year, int)

    def test_extra_fields_dict_on_record(self):
        """Unmapped extra columns are stored in record.extra_fields dict."""
        csv_data = (
            "Year,Company,Revenue,Net Income,Auditor Firm,ESG Score\n"
            "2022,AAPL,394328,99803,Ernst & Young,88.5\n"
        )
        res = ingest_financial_statement(io.StringIO(csv_data))
        records = res.to_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.company, "AAPL")
        self.assertEqual(rec.year, 2022)
        self.assertEqual(rec.revenue, 394328.0)
        self.assertIn("auditor_firm", rec.extra_fields)
        self.assertEqual(rec.extra_fields["auditor_firm"], "Ernst & Young")
        self.assertEqual(rec.extra_fields["esg_score"], 88.5)


class TestPackageImportability(unittest.TestCase):
    """Verifies that the ingestion package can be imported from outside the repo root."""

    def test_subprocess_import_from_temp_directory(self):
        """Package imports successfully from an arbitrary working directory."""
        repo_src = str(Path(__file__).resolve().parent.parent / "src")
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, r'{repo_src}'); "
                    "import ingestion; "
                    "from ingestion import IngestionConfig, IngestionResult, FinancialRecord, ingest_financial_statement; "
                    "print('IMPORT_SUCCESS')"
                ),
            ]
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Subprocess import failed: {result.stderr}")
            self.assertIn("IMPORT_SUCCESS", result.stdout)


if __name__ == "__main__":
    unittest.main()
