"""Canonical Schema and column definitions for the Financial Statement Data Ingestion Module.

Defines the target data contract expected by downstream agents (Validation Agent,
Trend Agent, Anomaly Agent, Evidence Agent, etc.).

Designed to be DATASET-AGNOSTIC:
- Treats canonical fields as a flexible dictionary, not a rigid mandatory 23-column checklist.
- Dynamically validates whether an uploaded CSV contains recognizable financial statement metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type

# Regex to strip units and parentheticals like ($M), (in B USD), (%), ($), etc.
UNIT_PARENTHETICAL_REGEX = re.compile(
    r"\s*[\(\[](?:in\s+)?(?:\$|usd|billions?|millions?|m|b|%|us|eur|gbp)[\)\]]",
    re.IGNORECASE,
)

# Common financial abbreviations mapping directly to canonical columns
COMMON_FINANCIAL_ABBREVIATIONS: Dict[str, str] = {
    "rev": "revenue",
    "pat": "net_income",
    "pbt": "pre_tax_income",
    "ebt": "pre_tax_income",
    "gp": "gross_profit",
    "cfo": "cash_flow_operating",
    "cfi": "cash_flow_investing",
    "cff": "cash_flow_financing",
    "ebit": "operating_income",
    "ebitda": "ebitda",
    "eps": "earnings_per_share",
    "cr": "current_ratio",
    "de": "debt_equity_ratio",
    "d_e": "debt_equity_ratio",
    "rote": "return_on_tangible_equity",
    "npm": "net_profit_margin",
    "fcf": "free_cash_flow_per_share",
    "capex": "cash_flow_investing",
    "fy": "year",
    "co": "company",
    "corp": "company",
    "emp": "number_of_employees",
    "cogs": "cost_of_revenue",
    "opex": "operating_expenses",
}


def clean_header_candidate(name: str) -> str:
    """Cleans a raw header name by removing unit notes in parentheses before normalization.

    Examples:
        'Revenue ($M)' -> 'Revenue'
        'ROE (%)' -> 'ROE'
        'Inflation Rate(in US)' -> 'Inflation Rate'
        'Market Cap (in B USD)' -> 'Market Cap'
    """
    if not isinstance(name, str):
        name = str(name)
    cleaned = UNIT_PARENTHETICAL_REGEX.sub("", name).strip()
    return cleaned if cleaned else name.strip()


def normalize_column_name(name: str) -> str:
    """Standardizes a column name to snake_case.

    Examples:
        'Company ' -> 'company'
        'Total Revenue' -> 'total_revenue'
        'Debt to Equity' -> 'debt_to_equity'
        'Debt/Equity Ratio' -> 'debt_equity_ratio'
        '  yEaR  ' -> 'year'
        'nEt  iNcOmE' -> 'net_income'
        'Cost of Revenue' -> 'cost_of_revenue'
        'Pre-tax Income' -> 'pre_tax_income'
    """
    if not isinstance(name, str):
        name = str(name)
    clean = name.strip()
    # Replace slashes, hyphens, parentheses, commas with underscores
    clean = re.sub(r"[/\-\(\)\[\]\.,]+", "_", clean)
    # Replace non-alphanumeric with underscore
    clean = re.sub(r"[^\w\s]", "_", clean)
    # Convert spaces to underscores
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_").lower()


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a canonical column in the financial dataset."""

    name: str
    data_type: Type[int | float | str]
    is_core: bool
    description: str
    aliases: List[str] = field(default_factory=list)
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    @property
    def type_str(self) -> str:
        if self.data_type is int:
            return "integer"
        elif self.data_type is float:
            return "float"
        return "string"


# Canonical definition of standard financial statement metrics (37 fields)
CANONICAL_COLUMNS: Dict[str, ColumnSpec] = {
    # ----------------------------------------------------
    # Kaggle Fields (23)
    # ----------------------------------------------------
    "year": ColumnSpec(
        name="year",
        data_type=int,
        is_core=True,
        description="Reporting fiscal calendar year (e.g. 2009-2023)",
        aliases=[
            "year",
            "fiscal_year",
            "fy",
            "financial_year",
            "reporting_year",
            "report_year",
            "date",
            "period",
            "reporting_period",
            "fiscal_period",
        ],
        min_val=1900,
        max_val=2100,
    ),
    "company": ColumnSpec(
        name="company",
        data_type=str,
        is_core=True,
        description="Company stock ticker or corporate name (e.g. AAPL, MSFT)",
        aliases=[
            "company",
            "company_name",
            "ticker",
            "symbol",
            "stock_symbol",
            "firm",
            "firm_name",
            "issuer",
            "corporation",
            "organization",
        ],
    ),
    "category": ColumnSpec(
        name="category",
        data_type=str,
        is_core=False,
        description="Industry sector or categorization (e.g. IT, Consumer Discretionary)",
        aliases=["category", "sector", "industry", "business_category", "segment", "industry_sector"],
    ),
    "market_cap_b_usd": ColumnSpec(
        name="market_cap_b_usd",
        data_type=float,
        is_core=False,
        description="Total market capitalization expressed in billions of USD",
        aliases=[
            "market_cap_in_b_usd",
            "market_cap_b_usd",
            "market_cap",
            "market_capitalization",
            "marketcap",
            "mkt_cap",
            "market_cap_billions",
            "market_cap_usd_b",
            "market_cap_b",
        ],
        min_val=0.0,
    ),
    "revenue": ColumnSpec(
        name="revenue",
        data_type=float,
        is_core=True,
        description="Total gross revenue, net sales, or turnover for the fiscal period",
        aliases=[
            "revenue",
            "total_revenue",
            "sales",
            "net_sales",
            "total_sales",
            "turnover",
            "gross_revenue",
            "operating_revenue",
            "gross_sales",
        ],
    ),
    "gross_profit": ColumnSpec(
        name="gross_profit",
        data_type=float,
        is_core=False,
        description="Gross profit calculated as Revenue minus Cost of Goods Sold",
        aliases=["gross_profit", "gross_margin_dollars", "gross_income", "gp", "gross_margin"],
    ),
    "net_income": ColumnSpec(
        name="net_income",
        data_type=float,
        is_core=False,
        description="Net income / net earnings after taxes and all expenses (PAT)",
        aliases=[
            "net_income",
            "net_profit",
            "profit_after_tax",
            "pat",
            "net_earnings",
            "profit_loss",
            "bottom_line",
            "earnings",
            "net_profit_loss",
            "profit_for_the_period",
            "profit_loss_for_the_period",
        ],
    ),
    "earnings_per_share": ColumnSpec(
        name="earnings_per_share",
        data_type=float,
        is_core=False,
        description="Diluted or basic earnings per share (EPS)",
        aliases=["eps", "earning_per_share", "earnings_per_share", "diluted_eps", "basic_eps", "eps_diluted"],
    ),
    "ebitda": ColumnSpec(
        name="ebitda",
        data_type=float,
        is_core=False,
        description="Earnings before interest, taxes, depreciation, and amortization",
        aliases=[
            "ebitda",
            "operating_profit_before_da",
            "operating_profit_before_depreciation_and_amortization",
        ],
    ),
    "shareholder_equity": ColumnSpec(
        name="shareholder_equity",
        data_type=float,
        is_core=False,
        description="Total stockholder or shareholder equity / book value / net worth",
        aliases=[
            "share_holder_equity",
            "shareholder_equity",
            "shareholders_equity",
            "total_shareholder_equity",
            "total_shareholders_equity",
            "stockholders_equity",
            "total_stockholders_equity",
            "total_equity",
            "equity",
            "shareholders_funds",
            "net_worth",
            "book_value",
        ],
    ),
    "cash_flow_operating": ColumnSpec(
        name="cash_flow_operating",
        data_type=float,
        is_core=False,
        description="Net cash provided by or used in operating activities (CFO)",
        aliases=[
            "cash_flow_from_operating",
            "cash_flow_operating",
            "operating_cash_flow",
            "cfo",
            "cash_from_operations",
            "cf_operations",
            "operating_activities_cash_flow",
            "cash_flow_from_operations",
            "cash_flow_from_operating_activities",
        ],
    ),
    "cash_flow_investing": ColumnSpec(
        name="cash_flow_investing",
        data_type=float,
        is_core=False,
        description="Net cash flow used in or provided by investing activities (CapEx, acquisitions)",
        aliases=[
            "cash_flow_from_investing",
            "cash_flow_investing",
            "investing_cash_flow",
            "cfi",
            "cash_from_investing",
            "cf_investing",
            "cash_flow_from_investing_activities",
            "cash_flow_investing_activities",
        ],
    ),
    "cash_flow_financing": ColumnSpec(
        name="cash_flow_financing",
        data_type=float,
        is_core=False,
        description="Net cash flow from financial activities (dividends, share buybacks, debt)",
        aliases=[
            "cash_flow_from_financial_activities",
            "cash_flow_from_financing",
            "cash_flow_financing",
            "financing_cash_flow",
            "cff",
            "cash_from_financing",
            "cash_from_financing_activities",
            "cash_from_financial_activities",
            "cash_flow_financial_activities",
        ],
    ),
    "current_ratio": ColumnSpec(
        name="current_ratio",
        data_type=float,
        is_core=False,
        description="Current ratio percentage measure (Current Assets divided by Current Liabilities)",
        aliases=["current_ratio", "cr", "liquidity_ratio", "working_capital_ratio"],
        min_val=0.0,
    ),
    "debt_equity_ratio": ColumnSpec(
        name="debt_equity_ratio",
        data_type=float,
        is_core=False,
        description="Total debt divided by shareholder equity ratio",
        aliases=[
            "debt/equity_ratio",
            "debt_equity_ratio",
            "debt_equity",
            "debt/equity",
            "debt_to_equity",
            "debt_to_equity_ratio",
            "de_ratio",
            "d_e_ratio",
            "d_e",
            "de",
        ],
    ),
    "roe": ColumnSpec(
        name="roe",
        data_type=float,
        is_core=False,
        description="Return on Equity percentage (Net Income / Shareholder Equity)",
        aliases=["roe", "return_on_equity", "return_on_shareholder_equity", "return_on_shareholders_equity"],
    ),
    "roa": ColumnSpec(
        name="roa",
        data_type=float,
        is_core=False,
        description="Return on Assets percentage (Net Income / Total Assets)",
        aliases=["roa", "return_on_assets", "return_on_total_assets"],
    ),
    "roi": ColumnSpec(
        name="roi",
        data_type=float,
        is_core=False,
        description="Return on Investment percentage",
        aliases=["roi", "return_on_investment", "return_on_invested_capital", "roic"],
    ),
    "net_profit_margin": ColumnSpec(
        name="net_profit_margin",
        data_type=float,
        is_core=False,
        description="Net profit margin percentage (Net Income / Revenue)",
        aliases=["net_profit_margin", "profit_margin", "net_margin", "npm"],
    ),
    "free_cash_flow_per_share": ColumnSpec(
        name="free_cash_flow_per_share",
        data_type=float,
        is_core=False,
        description="Free cash flow per share",
        aliases=["free_cash_flow_per_share", "fcf_per_share", "fcf_share", "free_cash_flow_share", "fcfps"],
    ),
    "return_on_tangible_equity": ColumnSpec(
        name="return_on_tangible_equity",
        data_type=float,
        is_core=False,
        description="Return on Tangible Equity percentage (ROTE)",
        aliases=["return_on_tangible_equity", "rote", "return_tangible_equity"],
    ),
    "number_of_employees": ColumnSpec(
        name="number_of_employees",
        data_type=float,
        is_core=False,
        description="Total company headcount or number of employees",
        aliases=["number_of_employees", "employees", "headcount", "employee_count", "num_employees", "workforce", "total_employees"],
        min_val=0.0,
    ),
    "inflation_rate_us": ColumnSpec(
        name="inflation_rate_us",
        data_type=float,
        is_core=False,
        description="Benchmark annual US inflation rate percentage",
        aliases=[
            "inflation_rate_in_us",
            "inflation_rate_us",
            "us_inflation_rate",
            "inflation_rate",
            "cpi_inflation",
        ],
    ),
    # ----------------------------------------------------
    # Statement Fields (14) - Balance Sheet / P&L
    # ----------------------------------------------------
    "currency": ColumnSpec(
        name="currency",
        data_type=str,
        is_core=False,
        description="Reporting currency code (e.g. USD, EUR, GBP)",
        aliases=["currency", "reporting_currency", "currency_code", "fx"],
    ),
    "cost_of_revenue": ColumnSpec(
        name="cost_of_revenue",
        data_type=float,
        is_core=False,
        description="Cost of revenue / Cost of goods sold (COGS)",
        aliases=["cost_of_revenue", "cost_of_goods_sold", "cogs", "cost_of_sales"],
    ),
    "operating_expenses": ColumnSpec(
        name="operating_expenses",
        data_type=float,
        is_core=False,
        description="Total operating expenses (OpEx) including SG&A and R&D",
        aliases=["operating_expenses", "operating_expense", "total_operating_expenses", "opex"],
    ),
    "operating_income": ColumnSpec(
        name="operating_income",
        data_type=float,
        is_core=False,
        description="Operating income / Operating profit (EBIT)",
        aliases=["operating_income", "operating_profit", "operating_earnings", "ebit"],
    ),
    "pre_tax_income": ColumnSpec(
        name="pre_tax_income",
        data_type=float,
        is_core=False,
        description="Income before income taxes / Profit before tax (EBT / PBT)",
        aliases=[
            "pre_tax_income",
            "pretax_income",
            "income_before_tax",
            "earnings_before_tax",
            "ebt",
            "profit_before_tax",
            "pbt",
            "income_before_income_taxes",
        ],
    ),
    "taxes": ColumnSpec(
        name="taxes",
        data_type=float,
        is_core=False,
        description="Income tax expense or provision for income taxes",
        aliases=[
            "taxes",
            "income_tax",
            "income_taxes",
            "tax_expense",
            "provision_for_income_taxes",
            "income_tax_expense",
            "total_tax",
        ],
    ),
    "total_assets": ColumnSpec(
        name="total_assets",
        data_type=float,
        is_core=False,
        description="Total assets on the balance sheet",
        aliases=["total_assets", "assets", "total_assets_usd"],
    ),
    "current_assets": ColumnSpec(
        name="current_assets",
        data_type=float,
        is_core=False,
        description="Current assets convertible to cash within one year",
        aliases=["current_assets", "total_current_assets"],
    ),
    "total_liabilities": ColumnSpec(
        name="total_liabilities",
        data_type=float,
        is_core=False,
        description="Total liabilities obligations on the balance sheet",
        aliases=["total_liabilities", "liabilities", "total_liabilities_usd"],
    ),
    "current_liabilities": ColumnSpec(
        name="current_liabilities",
        data_type=float,
        is_core=False,
        description="Current liabilities due within one year",
        aliases=["current_liabilities", "total_current_liabilities"],
    ),
    "cash": ColumnSpec(
        name="cash",
        data_type=float,
        is_core=False,
        description="Cash and cash equivalents",
        aliases=["cash", "cash_and_cash_equivalents", "cash_and_equivalents", "cash_equivalents"],
    ),
    "debt": ColumnSpec(
        name="debt",
        data_type=float,
        is_core=False,
        description="Total debt (short-term and long-term borrowings)",
        aliases=["debt", "total_debt", "short_long_term_debt", "financial_debt"],
    ),
    "beginning_cash": ColumnSpec(
        name="beginning_cash",
        data_type=float,
        is_core=False,
        description="Cash position at the beginning of the fiscal period",
        aliases=[
            "beginning_cash",
            "cash_at_beginning_of_period",
            "cash_beginning_of_year",
            "beginning_cash_position",
            "beginning_cash_balance",
        ],
    ),
    "ending_cash": ColumnSpec(
        name="ending_cash",
        data_type=float,
        is_core=False,
        description="Cash position at the end of the fiscal period",
        aliases=[
            "ending_cash",
            "cash_at_end_of_period",
            "cash_end_of_year",
            "ending_cash_position",
            "ending_cash_balance",
        ],
    ),
}

# Pre-computed alias index for deterministic Tier 1 mapping
# Maps normalized string -> canonical name
DETERMINISTIC_ALIAS_INDEX: Dict[str, str] = {}
for canonical_key, spec in CANONICAL_COLUMNS.items():
    DETERMINISTIC_ALIAS_INDEX[canonical_key] = canonical_key
    DETERMINISTIC_ALIAS_INDEX[normalize_column_name(canonical_key)] = canonical_key
    for alias in spec.aliases:
        DETERMINISTIC_ALIAS_INDEX[normalize_column_name(alias)] = canonical_key
        # Also index without punctuation
        cleaned_alias = clean_header_candidate(alias)
        DETERMINISTIC_ALIAS_INDEX[normalize_column_name(cleaned_alias)] = canonical_key

# Also index common financial abbreviations
for abbr, canonical_match in COMMON_FINANCIAL_ABBREVIATIONS.items():
    if canonical_match in CANONICAL_COLUMNS:
        DETERMINISTIC_ALIAS_INDEX[abbr] = canonical_match


# Ratio and amount classification for Type Guard protection
RATIO_PERCENTAGE_FIELDS: Set[str] = {
    "roe",
    "roa",
    "roi",
    "current_ratio",
    "debt_equity_ratio",
    "net_profit_margin",
    "return_on_tangible_equity",
}

AMOUNT_KEYWORDS: Set[str] = {
    "asset",
    "assets",
    "liability",
    "liabilities",
    "income",
    "revenue",
    "cash",
    "debt",
    "expense",
    "expenses",
    "tax",
    "taxes",
    "cost",
    "costs",
    "profit",
    "sales",
    "earnings",
}

RATIO_KEYWORDS: Set[str] = {
    "ratio",
    "return",
    "rate",
    "margin",
    "percentage",
    "percent",
    "pct",
    "%",
    "roe",
    "roa",
    "roi",
    "rote",
    "npm",
    "per_share",
    "pershare",
}


def is_amount_to_ratio_violation(raw_col_name: str, target_canonical: str) -> bool:
    """Type guard: verifies that an amount header is not erroneously mapped to a ratio field.

    A ratio or percentage field (roe, roa, roi, current_ratio, debt_equity_ratio,
    net_profit_margin, return_on_tangible_equity) must never receive a column whose name
    describes an amount (assets, liabilities, income, revenue, cash, debt, expenses, taxes)
    unless the name clearly says it's a ratio or return.
    """
    if target_canonical not in RATIO_PERCENTAGE_FIELDS:
        return False

    norm = normalize_column_name(clean_header_candidate(raw_col_name))
    tokens = set(norm.split("_"))

    # Explicit recognized ratio phrases that describe ratios, not amounts
    if target_canonical == "debt_equity_ratio" and (
        norm in ("debt_equity", "debt_to_equity", "de_ratio", "d_e_ratio", "d_e", "de")
        or "equity" in tokens
    ):
        return False

    has_amount_token = any(tok in AMOUNT_KEYWORDS for tok in tokens)
    has_ratio_token = any(tok in RATIO_KEYWORDS for tok in tokens)

    # If it describes an amount and does NOT clearly state it is a ratio/return, block it
    if has_amount_token and not has_ratio_token:
        return True

    return False


# Standard unit definitions for mapping metadata
STANDARD_UNITS: Dict[str, str] = {
    "revenue": "millions USD",
    "market_cap_b_usd": "billions USD",
    "ratios": "percentages",
}

# Required fields contract: Only year, company and revenue are strictly required
REQUIRED_FIELDS: Set[str] = {"year", "company", "revenue"}

# Identifier and Core Metric categories for Financial Dataset Compatibility Detection
CORE_IDENTIFIERS: Set[str] = {"company", "year"}
CORE_FINANCIAL_METRICS: Set[str] = {
    "revenue",
    "net_income",
    "gross_profit",
    "ebitda",
    "operating_income",
    "shareholder_equity",
    "cash_flow_operating",
    "market_cap_b_usd",
    "earnings_per_share",
    "total_assets",
    "cash",
}


def evaluate_financial_statement_compatibility(mapped_canonical_cols: Set[str]) -> Tuple[bool, str]:
    """Evaluates whether the mapped columns contain recognizable financial statement metrics.

    Returns:
        (is_compatible, reason_message)
    """
    has_identifier = any(col in mapped_canonical_cols for col in CORE_IDENTIFIERS)
    core_metric_matches = [col for col in mapped_canonical_cols if col in CORE_FINANCIAL_METRICS]

    if has_identifier and len(core_metric_matches) >= 1:
        return True, "Valid financial statement structure detected."
    elif len(core_metric_matches) >= 2:
        return True, "Sufficient financial performance metrics detected."

    reason = (
        "Dataset could not be identified as a compatible financial statement dataset. "
        "Missing core financial metrics (such as revenue, net income, or cash flow) "
        "and entity/period identifiers."
    )
    return False, reason
