"""Configuration settings for FinSight Trend Agent (Team 33, Member 4 - Trends).

All settings are deterministic and configurable.
Materiality is configurable as required by the HLD.
"""

from typing import Dict, List

# ==============================================================================
# MATERIALITY CONFIGURATION
# ==============================================================================
# HLD Requirement: Materiality must be configurable without modifying core logic.
# 0.10 represents 10% deviation.
# NOTE: This 10% value is strictly an implementation demo default and should
# be tuned to the team's agreed audit/review standard.
DEFAULT_MATERIALITY_THRESHOLD: float = 0.10

# ==============================================================================
# YOY ANALYSIS CONFIGURATION
# ==============================================================================
# Threshold for extreme YoY movements (in percent, e.g. 500 represents 500%)
EXTREME_YOY_THRESHOLD: float = 500.0

# ==============================================================================
# FINANCIAL RATIO HEALTH BENCHMARKS
# ==============================================================================
# Thresholds for categorizing ratios into HEALTHY, WARNING, CRITICAL.
RATIO_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "current_ratio": {"healthy_min": 1.5, "warning_min": 1.0},
    "debt_to_equity": {"healthy_max": 1.5, "warning_max": 2.5},
    "roe": {"healthy_min": 15.0, "warning_min": 5.0},
    "roa": {"healthy_min": 5.0, "warning_min": 1.0},
    "roi": {"healthy_min": 8.0, "warning_min": 2.0},
    "margin": {"healthy_min": 10.0, "warning_min": 0.0},
    "return_on_tangible_equity": {"healthy_min": 15.0, "warning_min": 5.0},
}

# ==============================================================================
# FORECASTING CONFIGURATION
# ==============================================================================
# Supported methods: "linear_regression"
DEFAULT_FORECAST_METHOD: str = "linear_regression"

# Chronological backtest split ratio (held out from end of chronological sequence)
# Default is 0.20 (last 20% of historical records per company)
BACKTEST_TEST_RATIO: float = 0.20

# Minimum number of total observations required for backtest evaluation
# Walk-forward evaluation starts from the 4th observation (minimum 3 train + 1 test)
MIN_OBSERVATIONS_FOR_EVAL: int = 4

# Minimum number of observations required to fit a linear trend line
MIN_OBSERVATIONS_FOR_FORECAST: int = 4

# Walk-forward backtesting parameters
WALK_FORWARD_MIN_TRAIN: int = 3
R2_MIN_TEST_POINTS: int = 5

# Output directory for reports and artifacts
DEFAULT_OUTPUT_DIR: str = "data/output"

# ==============================================================================
# DATA MAPPING & NORMALIZATION ALIASES
# ==============================================================================
# Used to map arbitrary column names from heterogeneous datasets into canonical keys.
# Matching is case-insensitive, ignores leading/trailing whitespace, and normalizes
# separators (spaces, underscores, dashes).
COLUMN_ALIASES: Dict[str, List[str]] = {
    "company": [
        "company",
        "company_name",
        "company name",
        "ticker",
        "symbol",
        "corp",
        "corporation",
        "firm",
        "entity",
    ],
    "year": [
        "year",
        "fiscal_year",
        "fiscal year",
        "financial_year",
        "financial year",
        "fy",
        "reporting_year",
        "date",
        "reporting_date",
        "fiscal_date",
        "period_end",
        "period_end_date",
        "year_ended",
        "year_ending",
    ],
    "revenue": [
        "revenue",
        "sales",
        "total_revenue",
        "total revenue",
        "net_sales",
        "net sales",
        "turnover",
        "operating_revenue",
        "operating revenue",
        "net_turnover",
    ],
    "gross_profit": [
        "gross_profit",
        "gross profit",
        "gross_margin_usd",
        "gross income",
        "gross_income",
        "gross_earnings",
        "gross_profit_loss",
    ],
    "net_income": [
        "net_income",
        "net income",
        "profit",
        "net_profit",
        "net profit",
        "earnings",
        "profit_after_tax",
        "pat",
        "profit after tax",
        "net_earnings",
    ],
    "cost_of_revenue": [
        "cost_of_revenue",
        "cost of revenue",
        "cogs",
        "cost_of_goods_sold",
        "cost of goods sold",
        "cost_of_sales",
        "cost of sales",
        "direct_costs",
        "direct costs",
        "total_cost_of_revenue",
        "total_costs",
    ],
    "operating_expenses": [
        "operating_expenses",
        "operating expenses",
        "operating expense",
        "opex",
    ],
    "expenses": [
        "expenses",
        "total_expenses",
        "total expenses",
    ],
    "margin": [
        "net_profit_margin",
        "net profit margin",
        "profit_margin",
        "profit margin",
        "net_margin",
        "net margin",
        "margin",
        "operating_margin",
        "operating margin",
    ],
    # Financial Ratios
    "current_ratio": [
        "current_ratio",
        "current ratio",
        "cr",
        "liquidity_ratio",
    ],
    "debt_to_equity": [
        "debt/equity_ratio",
        "debt/equity ratio",
        "debt_to_equity",
        "debt to equity",
        "debt_equity_ratio",
        "leverage_ratio",
        "d/e",
    ],
    "roe": [
        "roe",
        "return_on_equity",
        "return on equity",
        "return on share holder equity",
    ],
    "roa": [
        "roa",
        "return_on_assets",
        "return on assets",
    ],
    "debt": [
        "debt",
        "total_debt",
        "total debt",
    ],
    "roi": [
        "roi",
        "return_on_investment",
        "return on investment",
    ],
    "return_on_tangible_equity": [
        "return_on_tangible_equity",
        "return on tangible equity",
        "rote",
    ],
    # Balance Sheet & Income Statement Component Fields
    "current_assets": [
        "current_assets",
        "current assets",
        "total_current_assets",
        "total current assets",
    ],
    "current_liabilities": [
        "current_liabilities",
        "current liabilities",
        "total_current_liabilities",
        "total current liabilities",
    ],
    "total_liabilities": [
        "total_liabilities",
        "total liabilities",
        "liabilities",
    ],
    "shareholder_equity": [
        "shareholder_equity",
        "shareholder equity",
        "share holder equity",
        "share_holder_equity",
        "total_shareholders_equity",
        "total shareholders equity",
        "stockholders_equity",
        "stockholders equity",
        "equity",
    ],
    "total_assets": [
        "total_assets",
        "total assets",
        "assets",
    ],
    "tangible_equity": [
        "tangible_equity",
        "tangible equity",
        "tangible_shareholder_equity",
    ],
    "intangible_assets": [
        "intangible_assets",
        "intangible assets",
        "intangibles",
    ],
}
