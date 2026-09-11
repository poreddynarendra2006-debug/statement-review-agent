"""Unit tests for column mapping, normalization, and validation."""

import pytest
import pandas as pd
from analysis.data_mapping import DataMapper, clean_header_name


def test_clean_header_name():
    assert clean_header_name("Company ") == "company"
    assert clean_header_name("Market Cap(in B USD)") == "market_capin_b_usd"
    assert clean_header_name(" Debt/Equity Ratio ") == "debt_equity_ratio"
    assert clean_header_name("Net-Sales") == "net_sales"
    assert clean_header_name("  Fiscal Year  ") == "fiscal_year"


def test_current_kaggle_schema_mapping(synthetic_kaggle_like_df):
    """Verify mapping of Kaggle dataset format with trailing space in 'Company '."""
    mapper = DataMapper()
    norm_df, report = mapper.map_and_validate(synthetic_kaggle_like_df, source_file="kaggle_test")

    assert "company" in norm_df.columns
    assert "year" in norm_df.columns
    assert "revenue" in norm_df.columns
    assert "gross_profit" in norm_df.columns
    assert "margin" in norm_df.columns
    # Trailing space mapped
    assert report.mapped_fields["company"] == "Company "
    # Cost of Revenue derived as proxy
    assert "expenses" in norm_df.columns
    derived_metrics = [d["metric"] for d in report.derived_fields]
    assert "expenses" in derived_metrics
    # Values check
    assert norm_df["expenses"].iloc[0] == 200.0 - 90.0  # 110.0


def test_alternative_schema_mapping(synthetic_alternative_schema_df):
    """Verify mapping with completely different aliases ('Sales', 'Net Profit', 'Company Name')."""
    mapper = DataMapper()
    norm_df, report = mapper.map_and_validate(synthetic_alternative_schema_df, source_file="alt_test")

    assert "company" in norm_df.columns
    assert "year" in norm_df.columns
    assert "revenue" in norm_df.columns
    assert "net_income" in norm_df.columns
    assert "margin" in norm_df.columns
    assert norm_df["company"].iloc[0] == "ALPHA"
    assert norm_df["revenue"].iloc[0] == 1000.0
    assert report.mapped_fields["revenue"] == "Sales"
    assert report.mapped_fields["company"] == "Company Name"
    assert report.mapped_fields["year"] == "Fiscal Year"


def test_missing_required_column_reporting():
    """Verify that omitting required primary keys (company or year) raises clear error."""
    mapper = DataMapper()
    # Missing year
    bad_df = pd.DataFrame([{"company": "ACME", "revenue": 100}])
    with pytest.raises(ValueError) as excinfo:
        mapper.map_and_validate(bad_df, source_file="bad_test")
    assert "missing required primary keys" in str(excinfo.value)
    assert "year" in str(excinfo.value)


def test_duplicate_company_year_detection(synthetic_duplicate_years_df):
    """Verify duplicate (company, year) records are detected, warned, and resolved deterministically."""
    mapper = DataMapper()
    norm_df, report = mapper.map_and_validate(synthetic_duplicate_years_df, source_file="dup_test")

    # Should have resolved to 2 unique company-year records
    assert len(norm_df) == 2
    # Check warning was recorded
    assert any("duplicate company-year" in w for w in report.warnings)
    # Kept the latest record (revenue 105.0)
    dup_row = norm_df[norm_df["year"] == 2020]
    assert dup_row["revenue"].iloc[0] == 105.0


def test_unsupported_expense_reporting():
    """Verify that if expenses cannot be found or derived, it is safely marked unsupported."""
    mapper = DataMapper()
    # Has revenue but no gross_profit, so Cost of Revenue cannot be derived
    df = pd.DataFrame([
        {"company": "SOLO", "year": 2021, "revenue": 100.0},
        {"company": "SOLO", "year": 2022, "revenue": 120.0},
    ])
    norm_df, report = mapper.map_and_validate(df, source_file="unsupported_test")
    assert "expenses" not in norm_df.columns
    unsupported_metrics = [u["metric"] for u in report.unsupported_metrics]
    assert "expenses" in unsupported_metrics


def test_heterogeneous_financial_schema_with_dates_and_cogs():
    """Verify common alternative headers, date-based periods, and COGS are mapped correctly."""
    df = pd.DataFrame({
        "Company Name": ["ALT_CO"] * 6,
        "Reporting Date": ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"],
        "Net Sales": [100, 110, 120, 130, 140, 150],
        "COGS": [60, 65, 70, 75, 80, 85],
        "Gross Income": [40, 45, 50, 55, 60, 65],
        "Profit After Tax": [10, 12, 14, 16, 18, 20],
        "Unrelated Column": [1, 2, 3, 4, 5, 6],
    })
    mapper = DataMapper()
    norm_df, report = mapper.map_and_validate(df, source_file="heterogeneous")

    assert report.mapped_fields["company"] == "Company Name"
    assert report.mapped_fields["year"] == "Reporting Date"
    assert report.mapped_fields["revenue"] == "Net Sales"
    assert report.mapped_fields["cost_of_revenue"] == "COGS"
    assert report.mapped_fields["net_income"] == "Profit After Tax"
    assert norm_df["year"].tolist() == [2019, 2020, 2021, 2022, 2023, 2024]
    assert "cost_of_revenue" in norm_df.columns


def test_operating_expenses_not_misclassified_as_cost_of_revenue():
    """Operating expenses must not be silently treated as cost of revenue."""
    df = pd.DataFrame({
        "Entity": ["OP_CO"] * 4,
        "Year Ending": [2021, 2022, 2023, 2024],
        "Revenue": [100, 110, 120, 130],
        "Operating Expenses": [20, 22, 24, 26],
        "Net Income": [10, 12, 14, 16],
    })
    mapper = DataMapper()
    norm_df, report = mapper.map_and_validate(df, source_file="operating_expenses")

    assert report.mapped_fields["operating_expenses"] == "Operating Expenses"
    assert "cost_of_revenue" not in norm_df.columns
    assert any(u["metric"] == "expenses" for u in report.unsupported_metrics)
