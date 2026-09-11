"""Unit tests for the data cleaner and parser."""

import unittest
import numpy as np
import pandas as pd

from extraction.cleaner import DataCleaner, parse_numeric_value, parse_year_value
from extraction.schema import CANONICAL_COLUMNS


class TestCleaner(unittest.TestCase):
    def test_parse_numeric_value_various_formats(self):
        # Plain float
        val, err = parse_numeric_value("123.45")
        self.assertEqual(val, 123.45)
        self.assertIsNone(err)

        # Currency and commas
        val, err = parse_numeric_value(" $1,234,567.89 ")
        self.assertEqual(val, 1234567.89)
        self.assertIsNone(err)

        # Parenthesized negatives
        val, err = parse_numeric_value("(500.50)")
        self.assertEqual(val, -500.50)
        self.assertIsNone(err)

        # Percentages
        val, err = parse_numeric_value(" 25.4% ")
        self.assertEqual(val, 25.4)
        self.assertIsNone(err)

        # N/A and dash strings
        for na_str in ("-", "--", "N/A", "NA", "None", "null", ""):
            val, err = parse_numeric_value(na_str)
            self.assertTrue(np.isnan(val))
            self.assertIsNone(err)

        # Unparseable string
        val, err = parse_numeric_value("invalid_amount_xyz")
        self.assertTrue(np.isnan(val))
        self.assertIsNotNone(err)

    def test_parse_year_value(self):
        # Valid standard year
        val, err = parse_year_value(2022)
        self.assertEqual(val, 2022)
        self.assertIsNone(err)

        # Valid string year
        val, err = parse_year_value(" 2021 ")
        self.assertEqual(val, 2021)
        self.assertIsNone(err)

        # Float-like string
        val, err = parse_year_value("2020.0")
        self.assertEqual(val, 2020)
        self.assertIsNone(err)

        # Out of range year
        val, err = parse_year_value(1850)
        self.assertIsNone(val)
        self.assertIn("out of valid financial range", err)

        val, err = parse_year_value(2200)
        self.assertIsNone(val)
        self.assertIn("out of valid financial range", err)

        # Non-numeric year
        val, err = parse_year_value("twentytwenty")
        self.assertIsNone(val)
        self.assertIn("not an integer", err)

    def test_clean_dataframe_strings_and_numbers(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "year": [" 2022 ", "2021.0", "bad_year"],
            "company": ["  AAPL  ", "MSFT\tCorp", "  "],
            "revenue": ["$1,000", "(200)", "invalid"],
        })

        cleaned_df, mods = cleaner.clean(df, schema=CANONICAL_COLUMNS)

        # Assert year casting and errors
        self.assertEqual(cleaned_df["year"].iloc[0], 2022)
        self.assertEqual(cleaned_df["year"].iloc[1], 2021)
        self.assertTrue(pd.isna(cleaned_df["year"].iloc[2]))

        # Assert company string cleaning
        self.assertEqual(cleaned_df["company"].iloc[0], "AAPL")
        self.assertEqual(cleaned_df["company"].iloc[1], "MSFT Corp")
        self.assertTrue(pd.isna(cleaned_df["company"].iloc[2]))

        # Assert revenue numeric parsing
        self.assertEqual(cleaned_df["revenue"].iloc[0], 1000.0)
        self.assertEqual(cleaned_df["revenue"].iloc[1], -200.0)
        self.assertTrue(np.isnan(cleaned_df["revenue"].iloc[2]))

        # Assert modifications recorded in audit
        mod_types = [m["type"] for m in mods]
        self.assertIn("invalid_year", mod_types)
        self.assertIn("numeric_parse_warning", mod_types)

    def test_percentage_representation_options(self):
        # Nominal percentage default (15.5% -> 15.5)
        cleaner_nominal = DataCleaner(percentage_as_decimal=False)
        df = pd.DataFrame({"roe": ["15.5%", "20%"]})
        cleaned_nominal, _ = cleaner_nominal.clean(df)
        self.assertEqual(cleaned_nominal["roe"].iloc[0], 15.5)
        self.assertEqual(cleaned_nominal["roe"].iloc[1], 20.0)

        # Decimal percentage option (15.5% -> 0.155)
        cleaner_decimal = DataCleaner(percentage_as_decimal=True)
        cleaned_decimal, _ = cleaner_decimal.clean(df)
        self.assertAlmostEqual(cleaned_decimal["roe"].iloc[0], 0.155)
        self.assertAlmostEqual(cleaned_decimal["roe"].iloc[1], 0.20)


if __name__ == "__main__":
    unittest.main()
