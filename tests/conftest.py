"""Synthetic test fixtures for FinSight Trend Agent.

All datasets defined here are strictly synthetic fixtures for testing
and are never mixed with real financial statements.
"""

import pandas as pd
import pytest


@pytest.fixture
def synthetic_standard_df() -> pd.DataFrame:
    """Synthetic dataset with standard column names for 2 companies over 5 years."""
    data = [
        # Company A: Steady growth
        {"company": "COMP_A", "year": 2018, "revenue": 100.0, "gross_profit": 40.0, "net_income": 10.0, "net_profit_margin": 10.0},
        {"company": "COMP_A", "year": 2019, "revenue": 120.0, "gross_profit": 48.0, "net_income": 15.0, "net_profit_margin": 12.5},
        {"company": "COMP_A", "year": 2020, "revenue": 140.0, "gross_profit": 56.0, "net_income": 20.0, "net_profit_margin": 14.3},
        {"company": "COMP_A", "year": 2021, "revenue": 160.0, "gross_profit": 64.0, "net_income": 25.0, "net_profit_margin": 15.6},
        {"company": "COMP_A", "year": 2022, "revenue": 180.0, "gross_profit": 72.0, "net_income": 30.0, "net_profit_margin": 16.7},

        # Company B: Fluctuating values
        {"company": "COMP_B", "year": 2019, "revenue": 500.0, "gross_profit": 200.0, "net_income": 50.0, "net_profit_margin": 10.0},
        {"company": "COMP_B", "year": 2020, "revenue": 450.0, "gross_profit": 180.0, "net_income": -20.0, "net_profit_margin": -4.4},
        {"company": "COMP_B", "year": 2021, "revenue": 520.0, "gross_profit": 210.0, "net_income": 40.0, "net_profit_margin": 7.7},
        {"company": "COMP_B", "year": 2022, "revenue": 600.0, "gross_profit": 250.0, "net_income": 70.0, "net_profit_margin": 11.7},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_alternative_schema_df() -> pd.DataFrame:
    """Synthetic dataset using completely different aliases/column names.
    
    Tests that the mapping layer does not hard-code names like 'Revenue' or 'Company'.
    """
    data = [
        {"Company Name": "ALPHA", "Fiscal Year": 2019, "Sales": 1000.0, "Gross Income": 400.0, "Net Profit": 100.0, "Profit Margin": 10.0},
        {"Company Name": "ALPHA", "Fiscal Year": 2020, "Sales": 1100.0, "Gross Income": 450.0, "Net Profit": 120.0, "Profit Margin": 10.9},
        {"Company Name": "ALPHA", "Fiscal Year": 2021, "Sales": 1250.0, "Gross Income": 500.0, "Net Profit": 150.0, "Profit Margin": 12.0},
        {"Company Name": "ALPHA", "Fiscal Year": 2022, "Sales": 1400.0, "Gross Income": 580.0, "Net Profit": 180.0, "Profit Margin": 12.8},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_kaggle_like_df() -> pd.DataFrame:
    """Synthetic dataset replicating the Kaggle schema including trailing whitespace in 'Company '."""
    data = [
        {"Year": 2020, "Company ": "BETA", "Revenue": 200.0, "Gross Profit": 90.0, "Net Income": 30.0, "Net Profit Margin": 15.0, "Current Ratio": 1.2, "Debt/Equity Ratio": 0.8},
        {"Year": 2021, "Company ": "BETA", "Revenue": 220.0, "Gross Profit": 100.0, "Net Income": 35.0, "Net Profit Margin": 15.9, "Current Ratio": 1.3, "Debt/Equity Ratio": 0.75},
        {"Year": 2022, "Company ": "BETA", "Revenue": 250.0, "Gross Profit": 115.0, "Net Income": 42.0, "Net Profit Margin": 16.8, "Current Ratio": 1.4, "Debt/Equity Ratio": 0.7},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_duplicate_years_df() -> pd.DataFrame:
    """Synthetic dataset containing duplicate records for the same company and fiscal year."""
    data = [
        {"company": "DUP_CORP", "year": 2020, "revenue": 100.0, "gross_profit": 50.0, "net_income": 10.0},
        {"company": "DUP_CORP", "year": 2020, "revenue": 105.0, "gross_profit": 52.0, "net_income": 12.0},  # Duplicate
        {"company": "DUP_CORP", "year": 2021, "revenue": 120.0, "gross_profit": 60.0, "net_income": 15.0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_short_history_df() -> pd.DataFrame:
    """Synthetic dataset with a company having only 1 or 2 observations."""
    data = [
        {"company": "SHORT_CORP", "year": 2022, "revenue": 50.0, "gross_profit": 20.0, "net_income": 5.0},
        {"company": "SHORT_CORP", "year": 2023, "revenue": 60.0, "gross_profit": 25.0, "net_income": 7.0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_negatives_and_zeros_df() -> pd.DataFrame:
    """Synthetic dataset with zero and negative base values."""
    data = [
        {"company": "EDGE_CORP", "year": 2019, "revenue": 100.0, "net_income": -50.0},
        {"company": "EDGE_CORP", "year": 2020, "revenue": 0.0,   "net_income": 0.0},
        {"company": "EDGE_CORP", "year": 2021, "revenue": 120.0, "net_income": 20.0},
    ]
    return pd.DataFrame(data)
