"""
validation_agent/rules.py

Defines standard schemas, columns, domain sanity constraints, severity levels,
status definitions, and formula rules based on standardized Ingestion agent outputs.
"""

from typing import Dict, List

# Core identifier column names (support both standard lowercase and raw legacy)
COMPANY_COL = "Company "
COMPANY_COL_ALT = "company"
YEAR_COL = "Year"
YEAR_COL_ALT = "year"
REVENUE_COL = "revenue"

# The minimal required columns to perform validation.
# Missing statement fields are normal (e.g. Kaggle datasets lacking balance sheet items).
# Validation blocks ONLY if year, company or revenue are completely absent.
REQUIRED_COLUMNS: List[str] = [
    "year",
    "company",
    "revenue"
]

# Standardized column names produced by the Ingestion agent
STANDARD_COLUMNS: List[str] = [
    "year",
    "company",
    "currency",
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_expenses",
    "operating_income",
    "pre_tax_income",
    "taxes",
    "net_income",
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "shareholder_equity",
    "cash",
    "debt",
    "beginning_cash",
    "ending_cash",
    "cash_flow_operating",
    "cash_flow_investing",
    "cash_flow_financing",
    "roe",
    "roa",
    "net_profit_margin",
    "ebitda",
    "earnings_per_share",
    "market_cap_b_usd"
]

# Legacy raw Kaggle columns for backward-compatibility checks
KAGGLE_RAW_COLUMNS: List[str] = [
    "Year",
    "Company ",
    "Category",
    "Market Cap(in B USD)",
    "Revenue",
    "Gross Profit",
    "Net Income",
    "Earning Per Share",
    "EBITDA",
    "Share Holder Equity",
    "Cash Flow from Operating",
    "Cash Flow from Investing",
    "Cash Flow from Financial Activities",
    "Current Ratio",
    "Debt/Equity Ratio",
    "ROE",
    "ROA",
    "ROI",
    "Net Profit Margin",
    "Free Cash Flow per Share",
    "Return on Tangible Equity",
    "Number of Employees",
    "Inflation Rate(in US)"
]

# Numeric columns that should contain finite floating-point or integer values
NUMERIC_COLUMNS: List[str] = [
    "year",
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_expenses",
    "operating_income",
    "pre_tax_income",
    "taxes",
    "net_income",
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "shareholder_equity",
    "cash",
    "debt",
    "beginning_cash",
    "ending_cash",
    "cash_flow_operating",
    "cash_flow_investing",
    "cash_flow_financing",
    "roe",
    "roa",
    "net_profit_margin",
    "ebitda",
    "earnings_per_share",
    "market_cap_b_usd",
    # Legacy aliases
    "Market Cap(in B USD)",
    "Gross Profit",
    "Net Income",
    "Earning Per Share",
    "Share Holder Equity",
    "Cash Flow from Operating",
    "Cash Flow from Investing",
    "Cash Flow from Financial Activities",
    "Current Ratio",
    "Debt/Equity Ratio",
    "ROE",
    "ROA",
    "ROI",
    "Net Profit Margin",
    "Free Cash Flow per Share",
    "Return on Tangible Equity",
    "Number of Employees",
    "Inflation Rate(in US)"
]

# Domain sanity: columns that CANNOT be negative under normal corporate accounting
NON_NEGATIVE_COLUMNS: List[str] = [
    "revenue",
    "Revenue",
    "number_of_employees",
    "Number of Employees",
    "market_cap_b_usd",
    "Market Cap(in B USD)",
    "current_ratio",
    "Current Ratio"
]

# Legitimate negative columns (allowed by accounting semantics)
LEGITIMATE_NEGATIVE_COLUMNS: List[str] = [
    "net_income",
    "Net Income",
    "earnings_per_share",
    "Earning Per Share",
    "ebitda",
    "EBITDA",
    "shareholder_equity",
    "Share Holder Equity",
    "cash_flow_operating",
    "Cash Flow from Operating",
    "cash_flow_investing",
    "Cash Flow from Investing",
    "cash_flow_financing",
    "Cash Flow from Financial Activities",
    "debt_equity_ratio",
    "Debt/Equity Ratio",
    "roe",
    "ROE",
    "roa",
    "ROA",
    "net_profit_margin",
    "Net Profit Margin"
]

# Temporal sanity boundaries
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2100

# Severity levels
SEVERITY_NONE = "NONE"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_INFO = "INFO"

# Status levels
STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_BLOCK = "BLOCK"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"

# Rule IDs
RULE_BS_01 = "VAL_BS_01"
RULE_GP_02 = "VAL_GP_02"
RULE_OP_03 = "VAL_OP_03"
RULE_NI_04 = "VAL_NI_04"
RULE_CF_05 = "VAL_CF_05"
RULE_RATIO_ROE = "VAL_RATIO_ROE"
RULE_RATIO_NPM = "VAL_RATIO_NPM"

# Accounting rule names
RULE_NAMES: Dict[str, str] = {
    RULE_BS_01: "Balance Sheet",
    RULE_GP_02: "Gross Profit",
    RULE_OP_03: "Operating Income",
    RULE_NI_04: "Net Income",
    RULE_CF_05: "Cash Flow Reconciliation",
    RULE_RATIO_ROE: "Return on Equity",
    RULE_RATIO_NPM: "Net Profit Margin"
}

# Accounting rule formulas
RULE_FORMULAS: Dict[str, str] = {
    RULE_BS_01: "total_assets = total_liabilities + shareholder_equity",
    RULE_GP_02: "gross_profit = revenue - cost_of_revenue",
    RULE_OP_03: "operating_income = gross_profit - operating_expenses",
    RULE_NI_04: "net_income = pre_tax_income - taxes",
    RULE_CF_05: "ending_cash = beginning_cash + cash_flow_operating + cash_flow_investing + cash_flow_financing",
    RULE_RATIO_ROE: "roe = net_income / shareholder_equity * 100",
    RULE_RATIO_NPM: "net_profit_margin = net_income / revenue * 100"
}

# Default formula tolerance for ValidationAgent data quality check (accommodates international reporting variations)
DEFAULT_FORMULA_TOLERANCE = 2.5

# Accounting metrics that CANNOT safely be recalculated from the dataset alone
# due to missing balance sheet or share count inputs
NOT_VALIDATABLE_METRICS: Dict[str, str] = {
    "ROA": "Requires Net Income / Total Assets, but 'Total Assets' is not present in the dataset.",
    "Current Ratio": "Requires Current Assets / Current Liabilities, but neither line item is present in the dataset.",
    "Debt/Equity Ratio": "Requires Total Debt / Shareholder Equity, but 'Total Debt' is not present in the dataset.",
    "ROI": "Requires Net Return / Invested Capital base, which is not present in the dataset.",
    "Free Cash Flow per Share": "Requires Free Cash Flow and Diluted Share count, neither of which is present.",
    "Earning Per Share": "Requires Net Income and Weighted Average Shares, which are not present.",
    "EBITDA": "Requires Operating Income + Depreciation & Amortization, which are not isolated.",
    "Return on Tangible Equity": "Requires Tangible Common Equity (Shareholder Equity - Goodwill - Intangibles), which is not provided."
}

# Confidence scoring penalties per failed finding severity
SEVERITY_PENALTIES: Dict[str, float] = {
    SEVERITY_CRITICAL: 25.0,
    SEVERITY_HIGH: 10.0,
    SEVERITY_MEDIUM: 3.0,
    SEVERITY_LOW: 1.0,
    SEVERITY_INFO: 0.0,
    SEVERITY_NONE: 0.0
}
