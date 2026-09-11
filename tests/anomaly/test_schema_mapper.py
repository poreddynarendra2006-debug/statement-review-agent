"""Unit and integration tests for Financial Schema Mapper and normalizer."""

from __future__ import annotations

import pytest

from finsight.analysis.schema_mapper import (
    CANONICAL_FIELDS,
    DataValidationError,
    SchemaMapper,
    SchemaMappingResult,
    normalize_header,
)


def test_normalize_header():
    """Verify header string normalization across spaces, punctuation, symbols, and case."""
    assert normalize_header("  Revenue  ") == "revenue"
    assert normalize_header("NET INCOME") == "net income"
    assert normalize_header("Debt/Equity Ratio") == "debt equity ratio"
    assert normalize_header("Market Cap(in B USD)") == "market cap in b usd"
    assert normalize_header("Cash_Flow_from_Operations") == "cash flow from operations"
    assert normalize_header("Total Revenue ($ in Millions)") == "total revenue in millions"
    assert normalize_header("Net Profit Margin (%)") == "net profit margin"


def test_standard_cognizant_column_mapping():
    """Verify that all 23 standard columns from Financial Statements.csv map cleanly."""
    standard_cols = [
        "Year", "Company", "Category", "Market Cap(in B USD)", "Revenue",
        "Gross Profit", "Net Income", "Earning Per Share", "EBITDA", "Share Holder Equity",
        "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
        "Current Ratio", "Debt/Equity Ratio", "ROE", "ROA", "ROI", "Net Profit Margin",
        "Free Cash Flow per Share", "Return on Tangible Equity", "Number of Employees", "Inflation Rate"
    ]
    mapper = SchemaMapper()
    res = mapper.map_columns(standard_cols)

    assert res.canonical_to_raw["revenue"] == "Revenue"
    assert res.canonical_to_raw["net_income"] == "Net Income"
    assert res.canonical_to_raw["gross_profit"] == "Gross Profit"
    assert res.canonical_to_raw["ebitda"] == "EBITDA"
    assert res.canonical_to_raw["operating_cash_flow"] == "Operating Cash Flow"
    assert res.canonical_to_raw["shareholder_equity"] == "Share Holder Equity"
    assert res.canonical_to_raw["current_ratio"] == "Current Ratio"
    assert res.canonical_to_raw["debt_to_equity"] == "Debt/Equity Ratio"
    assert res.canonical_to_raw["roe"] == "ROE"
    assert res.canonical_to_raw["roa"] == "ROA"
    assert res.canonical_to_raw["roi"] == "ROI"
    assert res.canonical_to_raw["net_profit_margin"] == "Net Profit Margin"
    assert res.canonical_to_raw["free_cash_flow_per_share"] == "Free Cash Flow per Share"
    assert res.canonical_to_raw["return_on_tangible_equity"] == "Return on Tangible Equity"
    assert res.canonical_to_raw["eps"] == "Earning Per Share"
    assert res.canonical_to_raw["market_cap"] == "Market Cap(in B USD)"
    assert res.canonical_to_raw["number_of_employees"] == "Number of Employees"
    assert res.canonical_to_raw["inflation_rate"] == "Inflation Rate"
    assert res.canonical_to_raw["company"] == "Company"
    assert res.canonical_to_raw["year"] == "Year"

    # All 27 standard ML features should be available
    assert len(res.available_features) == 27
    assert res.feature_availability_str == "Available features: 27/27"
    assert len(res.missing_required_fields) == 0


def test_user_renamed_column_aliases():
    """Verify common financial aliases map to canonical fields."""
    user_cols = [
        "Sales",
        "Net Profit After Tax",
        "Gross Earnings",
        "Earnings Before Interest Tax Depreciation and Amortization",
        "Cash from Operating Activities",
        "Stockholders' Equity",
        "D/E Ratio",
        "Return on Equity",
        "Return on Assets",
        "ROIC",
        "FCF per Share",
        "ROTE",
        "Diluted EPS",
        "Market Capitalization",
        "Headcount",
        "Entity",
        "Fiscal Year",
    ]
    mapper = SchemaMapper()
    res = mapper.map_columns(user_cols)

    assert res.canonical_to_raw["revenue"] == "Sales"
    assert res.canonical_to_raw["net_income"] == "Net Profit After Tax"
    assert res.canonical_to_raw["gross_profit"] == "Gross Earnings"
    assert res.canonical_to_raw["ebitda"] == "Earnings Before Interest Tax Depreciation and Amortization"
    assert res.canonical_to_raw["operating_cash_flow"] == "Cash from Operating Activities"
    assert res.canonical_to_raw["shareholder_equity"] == "Stockholders' Equity"
    assert res.canonical_to_raw["debt_to_equity"] == "D/E Ratio"
    assert res.canonical_to_raw["roe"] == "Return on Equity"
    assert res.canonical_to_raw["roa"] == "Return on Assets"
    assert res.canonical_to_raw["roi"] == "ROIC"
    assert res.canonical_to_raw["free_cash_flow_per_share"] == "FCF per Share"
    assert res.canonical_to_raw["return_on_tangible_equity"] == "ROTE"
    assert res.canonical_to_raw["eps"] == "Diluted EPS"
    assert res.canonical_to_raw["market_cap"] == "Market Capitalization"
    assert res.canonical_to_raw["number_of_employees"] == "Headcount"
    assert res.canonical_to_raw["company"] == "Entity"
    assert res.canonical_to_raw["year"] == "Fiscal Year"


def test_conservative_matching_negative_guards():
    """Verify that derivative/growth/margin columns are NOT mistakenly mapped to base metrics."""
    cols = [
        "Revenue Growth",
        "Net Income Margin",
        "Operating Cash Flow Growth",
        "Gross Margin",
        "EBITDA Margin",
        "ROE Change",
    ]
    mapper = SchemaMapper()
    res = mapper.map_columns(cols)

    # "Revenue Growth" must NOT be mapped to "revenue"
    assert "revenue" not in res.canonical_to_raw
    # "Net Income Margin" should map to net_profit_margin NOT net_income
    assert res.canonical_to_raw.get("net_profit_margin") == "Net Income Margin"
    assert "net_income" not in res.canonical_to_raw
    # "Operating Cash Flow Growth" must NOT be mapped to "operating_cash_flow"
    assert "operating_cash_flow" not in res.canonical_to_raw
    # "Gross Margin" must map to "gross_margin", NOT "gross_profit"
    assert res.canonical_to_raw.get("gross_margin") == "Gross Margin"
    assert "gross_profit" not in res.canonical_to_raw


def test_ambiguity_handling():
    """Verify that conflicting column candidates trigger ambiguity warnings rather than silent selection."""
    ambig_cols = ["Sales", "Turnover", "Net Income"]  # Both Sales and Turnover can map to revenue
    mapper = SchemaMapper()
    res = mapper.map_columns(ambig_cols)

    assert "revenue" in res.ambiguous_columns
    assert set(res.ambiguous_columns["revenue"]) == {"Sales", "Turnover"}
    assert "revenue" not in res.canonical_to_raw
    assert len(res.warnings) > 0


def test_partial_features_when_optional_columns_missing():
    """Verify feature availability is accurately computed when optional columns are missing."""
    # Only Revenue and Net Income provided
    cols = ["Company", "Fiscal Year", "Revenue", "Net Income"]
    mapper = SchemaMapper()
    res = mapper.map_columns(cols)

    # Can compute revenue_growth, net_income_growth, net_profit_margin, net_margin_change, revenue_profit_growth_spread
    assert "revenue_growth" in res.available_features
    assert "net_income_growth" in res.available_features
    assert "net_profit_margin" in res.available_features
    assert "revenue_profit_growth_spread" in res.available_features
    # Features requiring gross_profit, ebitda, or ocf must NOT be available
    assert "gross_margin" not in res.available_features
    assert "ebitda_growth" not in res.available_features
    assert "operating_cash_flow_growth" not in res.available_features
    assert "ocf_net_income_divergence" not in res.available_features

    assert len(res.available_features) < 27
    assert "Available features:" in res.feature_availability_str


def test_insufficient_financial_fields_raises_validation_error():
    """Verify that a dataset with zero or only non-financial metadata raises DataValidationError."""
    cols = ["Company Name", "Year", "Category", "Headcount", "Location"]
    mapper = SchemaMapper()
    res = mapper.map_columns(cols)

    with pytest.raises(DataValidationError) as exc_info:
        mapper.validate_dataset_suitability(res)

    assert "does not contain enough financial fields" in str(exc_info.value)
