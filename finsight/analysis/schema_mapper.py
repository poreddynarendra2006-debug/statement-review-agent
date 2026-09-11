"""Financial Schema Normalization Layer for FinSight AI Anomaly Detection Agent.

Maps arbitrary user-provided column headers from CSV and Excel files to standardized
canonical financial fields with collision-resistant matching and dynamic numeric discovery.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when an uploaded financial file fails structural or schema validation."""
    pass


# Definition of all standard canonical financial fields
CANONICAL_FIELDS = {
    # Metadata (Never used as ML features)
    "company": {
        "type": "metadata",
        "required": False,
        "description": "Company name or entity identifier",
        "aliases": [
            "company", "company name", "company identifier", "organization", "organization name",
            "entity", "entity name", "corporation", "corp", "firm", "issuer", "client",
            "ticker", "symbol", "stock symbol", "company ticker", "business",
        ],
        "disallowed": ["sector", "industry", "country", "category", "employee", "officer"],
    },
    "year": {
        "type": "metadata",
        "required": False,
        "description": "Fiscal reporting year or period",
        "aliases": [
            "year", "fiscal year", "financial year", "fy", "reporting year", "period year",
            "year ended", "calendar year", "fyear", "date year", "fiscal period", "period",
            "report year", "date", "reporting period",
        ],
        "disallowed": ["growth", "change", "tenure", "age", "over year", "yoy", "qoq"],
    },
    "category": {
        "type": "metadata",
        "required": False,
        "description": "Industry or sector classification",
        "aliases": [
            "category", "industry", "sector", "industry sector", "industry classification",
            "business segment", "segment", "industry group", "financial status", "status",
        ],
        "disallowed": ["revenue", "profit", "income", "margin"],
    },
    # Core Financial Statement Metrics
    "revenue": {
        "type": "financial_metric",
        "required": False,
        "description": "Total top-line revenue / gross sales",
        "aliases": [
            "revenue", "total revenue", "net revenue", "sales", "total sales",
            "sales revenue", "revenue sales", "revenue from operations", "turnover",
            "operating revenue", "gross revenue", "net sales", "total sales revenue",
            "top line", "top line revenue", "operating sales", "gross sales",
            "gross receipts", "receipts", "annual revenue", "sales and operating revenue",
            "total operating revenue", "revenue $", "total revenue $",
        ],
        "disallowed": [
            "growth", "margin", "percent", "%", "change", "spread", "per share",
            "per employee", "multiple", "cost", "unearned", "deferred", "tax",
        ],
    },
    "gross_profit": {
        "type": "financial_metric",
        "required": False,
        "description": "Gross profit after cost of goods sold",
        "aliases": [
            "gross profit", "gross earnings", "gross income", "total gross profit",
            "gross margin dollar", "gross margin amount", "gross profit loss",
        ],
        "disallowed": ["growth", "margin", "percent", "%", "change", "spread", "per share"],
    },
    "net_income": {
        "type": "financial_metric",
        "required": False,
        "description": "Bottom-line net income / profit after tax",
        "aliases": [
            "net income", "net profit", "net profit after tax", "profit after tax",
            "pat", "net earnings", "bottom line", "net profit loss", "net income loss",
            "profit for the year", "net income after tax", "net earnings loss",
            "profit after taxation", "net operating profit after tax",
            "profit", "earnings", "profit for the period",
        ],
        "disallowed": [
            "growth", "margin", "percent", "%", "change", "spread", "per share",
            "operating income", "ebitda", "ebit", "gross",
        ],
    },
    "operating_expenses": {
        "type": "financial_metric",
        "required": False,
        "description": "Operating Expenses / Opex",
        "aliases": [
            "operating expenses", "operating expense", "opex", "operating costs",
            "total operating expenses", "operating expenditure", "administrative expenses",
            "sga", "sg a", "selling general administrative", "expenses", "total expenses",
            "expenditures", "total expenditure", "cost of sales", "cost of goods sold", "cogs",
        ],
        "disallowed": ["growth", "margin", "percent", "%", "change", "income", "profit"],
    },
    "operating_income": {
        "type": "financial_metric",
        "required": False,
        "description": "Operating Income / EBIT",
        "aliases": [
            "operating income", "ebit", "operating profit", "operating earnings",
        ],
        "disallowed": ["margin", "growth", "cash flow", "net income"],
    },
    "ebitda": {
        "type": "financial_metric",
        "required": False,
        "description": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
        "aliases": [
            "ebitda", "earnings before interest tax depreciation and amortization",
            "earnings before interest taxes depreciation and amortization",
            "adjusted ebitda", "operating ebitda", "total ebitda",
        ],
        "disallowed": ["growth", "margin", "percent", "%", "change", "spread", "per share"],
    },
    "operating_cash_flow": {
        "type": "financial_metric",
        "required": False,
        "description": "Cash Flow from Operating Activities",
        "aliases": [
            "operating cash flow", "cash flow from operations", "cash from operating activities",
            "net cash from operating activities", "cfo", "cash from operations",
            "net cash provided by operating activities", "operating cashflow",
            "cash flow operations", "cash provided by operating activities",
            "cash flow from operating activities", "net cash flow from operating activities",
            "cash flow operating", "cash operating flow", "cash flow", "cashflow",
            "cash flow from operating", "cash from operating",
        ],
        "disallowed": [
            "growth", "margin", "percent", "%", "change", "divergence", "per share",
            "investing", "financing", "free cash", "fcf",
        ],
    },
    "investing_cash_flow": {
        "type": "financial_metric",
        "required": False,
        "description": "Cash Flow from Investing Activities",
        "aliases": [
            "investing cash flow", "cash flow from investing activities", "cash from investing activities",
            "net cash used in investing activities", "cfi", "investing cashflow",
            "cash flow investing", "cash used in investing activities",
            "cash flow from investing",
        ],
        "disallowed": ["operating", "financing", "growth", "margin", "percent", "%"],
    },
    "financing_cash_flow": {
        "type": "financial_metric",
        "required": False,
        "description": "Cash Flow from Financing Activities",
        "aliases": [
            "financing cash flow", "cash flow from financing activities", "cash from financing activities",
            "net cash used in financing activities", "cff", "financing cashflow",
            "cash flow financing", "cash used in financing activities",
            "cash flow from financing", "cash flow from financial activities",
            "cash from financial activities",
        ],
        "disallowed": ["operating", "investing", "growth", "margin", "percent", "%"],
    },
    "shareholder_equity": {
        "type": "financial_metric",
        "required": False,
        "description": "Total Stockholders' / Shareholders' Equity",
        "aliases": [
            "share holder equity", "shareholders equity", "shareholder equity",
            "stockholders equity", "stockholder equity", "total equity",
            "total shareholders equity", "total stockholders equity", "net worth",
            "book value of equity", "common equity", "total common equity",
            "equity", "shareholder funds", "net assets", "share capital and reserves",
        ],
        "disallowed": ["return on", "roe", "change", "growth", "margin", "%", "debt to equity", "debt equity"],
    },
    "total_assets": {
        "type": "financial_metric",
        "required": False,
        "description": "Total Assets from balance sheet",
        "aliases": [
            "total assets", "assets", "balance sheet assets", "total asset",
        ],
        "disallowed": ["return on", "roa", "turnover", "current assets"],
    },
    "total_liabilities": {
        "type": "financial_metric",
        "required": False,
        "description": "Total Liabilities from balance sheet",
        "aliases": [
            "total liabilities", "liabilities", "balance sheet liabilities", "total debt and liabilities",
            "debt", "total debt", "borrowings", "total borrowings",
        ],
        "disallowed": ["current liabilities", "short term debt"],
    },
    "cash": {
        "type": "financial_metric",
        "required": False,
        "description": "Cash and Cash Equivalents",
        "aliases": [
            "cash", "cash and cash equivalents", "cash equivalents", "total cash",
            "cash & cash equivalents", "cash and bank balances", "liquid assets", "liquid cash",
        ],
        "disallowed": ["cash flow", "operating cash", "investing cash", "financing cash"],
    },
    # Financial Ratios & Performance Metrics
    "current_ratio": {
        "type": "financial_ratio",
        "required": False,
        "description": "Liquidity ratio (Current Assets / Current Liabilities)",
        "aliases": [
            "current ratio", "cr", "current assets to liabilities",
        ],
        "disallowed": ["change", "growth", "quick", "cash ratio"],
    },
    "quick_ratio": {
        "type": "financial_ratio",
        "required": False,
        "description": "Quick / Acid-test liquidity ratio",
        "aliases": [
            "quick ratio", "acid test", "acid test ratio", "quick assets to liabilities",
        ],
        "disallowed": ["change", "growth", "current ratio"],
    },
    "working_capital_ratio": {
        "type": "financial_ratio",
        "required": False,
        "description": "Working Capital Ratio",
        "aliases": [
            "working capital ratio", "working capital", "wc ratio", "working capital to assets",
        ],
        "disallowed": ["change", "growth"],
    },
    "debt_to_equity": {
        "type": "financial_ratio",
        "required": False,
        "description": "Leverage ratio (Total Debt / Equity)",
        "aliases": [
            "debt equity ratio", "debt to equity ratio", "debt equity", "d e ratio",
            "total debt equity", "debt to equity", "d e", "leverage ratio",
            "gearing ratio", "total debt to total equity", "d/e", "d/e ratio",
        ],
        "disallowed": ["change", "growth", "short term debt"],
    },
    "roe": {
        "type": "financial_ratio",
        "required": False,
        "description": "Return on Equity (Net Income / Equity)",
        "aliases": [
            "roe", "return on equity", "return on shareholder equity",
            "return on stockholders equity", "return on common equity",
        ],
        "disallowed": ["change", "growth", "tangible", "rote", "assets", "roa", "roi"],
    },
    "roa": {
        "type": "financial_ratio",
        "required": False,
        "description": "Return on Assets (Net Income / Total Assets)",
        "aliases": [
            "roa", "return on assets", "return on total assets",
        ],
        "disallowed": ["change", "growth", "equity", "roe", "roi", "rote"],
    },
    "roi": {
        "type": "financial_ratio",
        "required": False,
        "description": "Return on Investment / Invested Capital",
        "aliases": [
            "roi", "return on investment", "return on invested capital", "roic",
        ],
        "disallowed": ["change", "growth", "equity", "roe", "roa", "rote"],
    },
    "gross_margin": {
        "type": "financial_ratio",
        "required": False,
        "description": "Gross Profit Margin (Gross Profit / Revenue)",
        "aliases": [
            "gross margin", "gross profit margin", "gross margin percent",
            "gross profit margin percent", "gp margin", "gpm",
        ],
        "disallowed": ["change", "growth", "net margin", "ebitda margin"],
    },
    "net_profit_margin": {
        "type": "financial_ratio",
        "required": False,
        "description": "Net Profit Margin (Net Income / Revenue)",
        "aliases": [
            "net profit margin", "net margin", "net profit margin percent",
            "net margin percent", "net profit percentage", "net income margin",
            "npm", "net income margin percent",
        ],
        "disallowed": ["change", "growth", "gross margin", "ebitda margin"],
    },
    "free_cash_flow_per_share": {
        "type": "financial_metric",
        "required": False,
        "description": "Free Cash Flow per common share",
        "aliases": [
            "free cash flow per share", "fcf per share", "fcf share", "fcf per share usd",
            "free cashflow per share", "free cash flow common share",
        ],
        "disallowed": ["growth", "change"],
    },
    "return_on_tangible_equity": {
        "type": "financial_ratio",
        "required": False,
        "description": "Return on Tangible Equity (ROTE)",
        "aliases": [
            "return on tangible equity", "rote", "return on tangible common equity",
            "rotce", "return on tangible net worth",
        ],
        "disallowed": ["change", "growth"],
    },
    "eps": {
        "type": "financial_metric",
        "required": False,
        "description": "Diluted / Basic Earnings Per Share",
        "aliases": [
            "earning per share", "earnings per share", "eps", "diluted eps",
            "diluted earnings per share", "basic eps", "basic earnings per share",
        ],
        "disallowed": ["growth", "change"],
    },
    "market_cap": {
        "type": "financial_metric",
        "required": False,
        "description": "Market Capitalization (in Billion USD or total)",
        "aliases": [
            "market cap in b usd", "market cap in billions", "market cap",
            "market capitalization", "market cap b usd", "market cap usd",
            "market value", "mcap", "total market cap",
        ],
        "disallowed": ["growth", "change", "margin"],
    },
    "number_of_employees": {
        "type": "metadata",
        "required": False,
        "description": "Total headcount / employee count",
        "aliases": [
            "number of employees", "employees", "headcount", "employee count",
            "total employees", "staff count", "num employees",
        ],
        "disallowed": ["growth", "change", "per employee", "cost per employee"],
    },
    "inflation_rate": {
        "type": "metadata",
        "required": False,
        "description": "Macroeconomic inflation rate",
        "aliases": [
            "inflation rate", "inflation", "annual inflation", "cpi rate", "macro inflation",
            "inflation rate in us", "inflation rate us", "inflation in us",
            "inflation rate(in us)", "inflation rate (in us)", "inflation rate in the us",
        ],
        "disallowed": ["growth", "change"],
    },
}


def normalize_header(header: str) -> str:
    """Normalize a column header string into a clean lowercase tokenized sequence.

    Strips punctuation, quotes, replaces symbols, dashes, slashes, underscores, and extra spaces.
    """
    text = str(header).strip().lower()
    text = re.sub(r"[\'\"\`\$\%\,\.\(\)\[\]\{\}\\\/\_\-\:\;\|\*\+\=]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class SchemaMappingResult:
    """Detailed result of schema detection, column mapping, and field availability."""

    mapped_columns: dict[str, str] = field(default_factory=dict)  # raw_header -> canonical_field
    canonical_to_raw: dict[str, str] = field(default_factory=dict)  # canonical_field -> raw_header
    unmapped_columns: list[str] = field(default_factory=list)
    unmapped_numeric_columns: list[str] = field(default_factory=list)
    available_financial_fields: list[str] = field(default_factory=list)
    missing_required_fields: list[str] = field(default_factory=list)
    missing_optional_fields: list[str] = field(default_factory=list)
    ambiguous_columns: dict[str, list[str]] = field(default_factory=dict)
    available_features: list[str] = field(default_factory=list)
    total_features_count: int = 27
    has_temporal_ordering: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def feature_availability_str(self) -> str:
        """Formatted string of available features e.g. 'Available features: 21/27'."""
        return f"Available features: {len(self.available_features)}/{self.total_features_count}"

    def summary_dict(self) -> dict[str, Any]:
        """Convert mapping summary into a dictionary."""
        return {
            "mapped_columns_count": len(self.mapped_columns),
            "mapped_columns": self.mapped_columns,
            "unmapped_columns": self.unmapped_columns,
            "unmapped_numeric_columns": self.unmapped_numeric_columns,
            "available_financial_fields": self.available_financial_fields,
            "missing_required_fields": self.missing_required_fields,
            "missing_optional_fields": self.missing_optional_fields,
            "ambiguous_columns": self.ambiguous_columns,
            "available_features_count": len(self.available_features),
            "total_features_count": self.total_features_count,
            "feature_availability": self.feature_availability_str,
            "available_features": self.available_features,
            "has_temporal_ordering": self.has_temporal_ordering,
            "warnings": self.warnings,
        }


class SchemaMapper:
    """Conservative schema mapper that standardizes raw column headers into canonical schema."""

    def __init__(self, canonical_definitions: Optional[dict[str, dict]] = None):
        self.definitions = canonical_definitions or CANONICAL_FIELDS

    def map_columns(self, raw_columns: Sequence[str]) -> SchemaMappingResult:
        """Map a list of raw column headers to canonical schema fields."""
        mapped: dict[str, str] = {}
        canonical_to_raw: dict[str, str] = {}
        unmapped: list[str] = []
        ambiguities: dict[str, list[str]] = {}
        warnings: list[str] = []

        candidates: dict[str, list[str]] = {k: [] for k in self.definitions.keys()}

        for raw_col in raw_columns:
            raw_str = str(raw_col).strip()
            norm_col = normalize_header(raw_str)

            if not norm_col:
                unmapped.append(raw_str)
                continue

            matched_field: Optional[str] = None
            best_match_exactness: int = 0

            for field_name, meta in self.definitions.items():
                aliases = meta.get("aliases", [])
                disallowed = meta.get("disallowed", [])

                is_disallowed = False
                for dis in disallowed:
                    dis_norm = normalize_header(dis)
                    if (
                        dis_norm in norm_col.split()
                        or f" {dis_norm} " in f" {norm_col} "
                        or norm_col.endswith(f" {dis_norm}")
                        or norm_col.startswith(f"{dis_norm} ")
                    ):
                        is_disallowed = True
                        break

                if is_disallowed:
                    continue

                if norm_col == normalize_header(field_name):
                    matched_field = field_name
                    best_match_exactness = 2
                    break

                for alias in aliases:
                    norm_alias = normalize_header(alias)
                    if norm_col == norm_alias:
                        matched_field = field_name
                        best_match_exactness = 2
                        break

                if best_match_exactness == 2:
                    break

            if not matched_field:
                # Conservative fallback: fuzzy similarity check (threshold >= 0.88)
                best_sim = 0.0
                best_cand = None
                for field_name, meta in self.definitions.items():
                    disallowed = meta.get("disallowed", [])
                    is_dis = any(
                        normalize_header(dis) in norm_col.split()
                        or f" {normalize_header(dis)} " in f" {norm_col} "
                        for dis in disallowed
                    )
                    if is_dis:
                        continue
                    for alias in meta.get("aliases", []):
                        norm_alias = normalize_header(alias)
                        if len(norm_alias) >= 4 and len(norm_col) >= 4:
                            sim = difflib.SequenceMatcher(None, norm_col, norm_alias).ratio()
                            if sim > best_sim and sim >= 0.88:
                                best_sim = sim
                                best_cand = field_name
                if best_cand and best_sim >= 0.88:
                    matched_field = best_cand

            if matched_field:
                candidates[matched_field].append(raw_str)
            else:
                unmapped.append(raw_str)

        for field_name, matched_raw_list in candidates.items():
            if len(matched_raw_list) == 1:
                raw_hdr = matched_raw_list[0]
                mapped[raw_hdr] = field_name
                canonical_to_raw[field_name] = raw_hdr
            elif len(matched_raw_list) > 1:
                exact_matches = [
                    r for r in matched_raw_list if normalize_header(r) == normalize_header(field_name)
                ]
                if len(exact_matches) == 1:
                    chosen = exact_matches[0]
                    mapped[chosen] = field_name
                    canonical_to_raw[field_name] = chosen
                    others = [r for r in matched_raw_list if r != chosen]
                    unmapped.extend(others)
                    warnings.append(
                        f"Disambiguated multiple matches for '{field_name}': selected exact match '{chosen}', ignored {others}."
                    )
                else:
                    ambiguities[field_name] = matched_raw_list
                    warnings.append(
                        f"Ambiguous mapping for field '{field_name}': multiple candidate columns found {matched_raw_list}. "
                        "Skipping automated assignment to prevent data corruption."
                    )
                    unmapped.extend(matched_raw_list)

        # Deduplicate unmapped while preserving order
        unmapped = list(dict.fromkeys(unmapped))

        if unmapped:
            warnings.append(f"Unmapped columns retained as raw: {unmapped}")

        mapped_canonical_fields = set(canonical_to_raw.keys())
        available_financial = [
            f for f in mapped_canonical_fields
            if self.definitions[f].get("type") in ("financial_metric", "financial_ratio")
        ]

        has_temporal = "year" in mapped_canonical_fields

        missing_required: list[str] = []
        missing_optional: list[str] = []

        for field_name, meta in self.definitions.items():
            if field_name not in mapped_canonical_fields:
                if meta.get("required", False):
                    missing_required.append(field_name)
                else:
                    missing_optional.append(field_name)

        available_features = self.determine_available_features(mapped_canonical_fields, has_temporal=has_temporal)

        return SchemaMappingResult(
            mapped_columns=mapped,
            canonical_to_raw=canonical_to_raw,
            unmapped_columns=unmapped,
            unmapped_numeric_columns=[],
            available_financial_fields=available_financial,
            missing_required_fields=missing_required,
            missing_optional_fields=missing_optional,
            ambiguous_columns=ambiguities,
            available_features=available_features,
            total_features_count=27,
            has_temporal_ordering=has_temporal,
            warnings=warnings,
        )

    @staticmethod
    def determine_available_features(available_canonical_fields: set[str], has_temporal: bool = True) -> list[str]:
        """Compute which of the standard features can be computed from the available source fields."""
        from finsight.analysis.anomaly_features import FEATURE_REGISTRY

        available_feats: list[str] = []
        for feat_name, meta in FEATURE_REGISTRY.items():
            if meta.requires_history and not has_temporal:
                continue

            src_cols = meta.source_columns
            if not src_cols:
                continue

            can_compute = False
            if feat_name == "revenue_growth":
                can_compute = "revenue" in available_canonical_fields
            elif feat_name == "gross_profit_growth":
                can_compute = "gross_profit" in available_canonical_fields
            elif feat_name == "net_income_growth":
                can_compute = "net_income" in available_canonical_fields
            elif feat_name == "ebitda_growth":
                can_compute = "ebitda" in available_canonical_fields
            elif feat_name == "operating_cash_flow_growth":
                can_compute = "operating_cash_flow" in available_canonical_fields
            elif feat_name == "gross_margin":
                can_compute = (
                    "revenue" in available_canonical_fields and "gross_profit" in available_canonical_fields
                ) or ("gross_margin" in available_canonical_fields)
            elif feat_name == "net_profit_margin":
                can_compute = ("net_profit_margin" in available_canonical_fields) or (
                    "revenue" in available_canonical_fields and "net_income" in available_canonical_fields
                )
            elif feat_name == "ebitda_margin":
                can_compute = "revenue" in available_canonical_fields and "ebitda" in available_canonical_fields
            elif feat_name == "operating_margin":
                can_compute = "revenue" in available_canonical_fields and "operating_income" in available_canonical_fields
            elif feat_name == "roe":
                can_compute = ("roe" in available_canonical_fields) or (
                    "net_income" in available_canonical_fields and "shareholder_equity" in available_canonical_fields
                )
            elif feat_name == "roa":
                can_compute = ("roa" in available_canonical_fields) or (
                    "net_income" in available_canonical_fields and "total_assets" in available_canonical_fields
                )
            elif feat_name == "roi":
                can_compute = "roi" in available_canonical_fields
            elif feat_name == "return_on_tangible_equity":
                can_compute = "return_on_tangible_equity" in available_canonical_fields
            elif feat_name == "operating_cash_flow_to_net_income":
                can_compute = "operating_cash_flow" in available_canonical_fields and "net_income" in available_canonical_fields
            elif feat_name == "cash_flow_to_revenue":
                can_compute = "operating_cash_flow" in available_canonical_fields and "revenue" in available_canonical_fields
            elif feat_name == "free_cash_flow_per_share":
                can_compute = "free_cash_flow_per_share" in available_canonical_fields
            elif feat_name == "current_ratio":
                can_compute = "current_ratio" in available_canonical_fields
            elif feat_name == "debt_equity_ratio":
                can_compute = ("debt_to_equity" in available_canonical_fields) or (
                    "total_liabilities" in available_canonical_fields and "shareholder_equity" in available_canonical_fields
                )
            elif feat_name == "liabilities_to_assets":
                can_compute = "total_liabilities" in available_canonical_fields and "total_assets" in available_canonical_fields
            elif feat_name == "asset_turnover":
                can_compute = "revenue" in available_canonical_fields and "total_assets" in available_canonical_fields
            elif feat_name == "gross_margin_change":
                can_compute = (
                    "revenue" in available_canonical_fields and "gross_profit" in available_canonical_fields
                ) or ("gross_margin" in available_canonical_fields)
            elif feat_name == "net_margin_change":
                can_compute = ("net_profit_margin" in available_canonical_fields) or (
                    "revenue" in available_canonical_fields and "net_income" in available_canonical_fields
                )
            elif feat_name == "ebitda_margin_change":
                can_compute = "revenue" in available_canonical_fields and "ebitda" in available_canonical_fields
            elif feat_name == "roe_change":
                can_compute = ("roe" in available_canonical_fields) or (
                    "net_income" in available_canonical_fields and "shareholder_equity" in available_canonical_fields
                )
            elif feat_name == "roa_change":
                can_compute = "roa" in available_canonical_fields
            elif feat_name == "roi_change":
                can_compute = "roi" in available_canonical_fields
            elif feat_name == "current_ratio_change":
                can_compute = "current_ratio" in available_canonical_fields
            elif feat_name == "debt_equity_change":
                can_compute = "debt_to_equity" in available_canonical_fields
            elif feat_name == "revenue_profit_growth_spread":
                can_compute = "revenue" in available_canonical_fields and "net_income" in available_canonical_fields
            elif feat_name == "ocf_net_income_divergence":
                can_compute = "operating_cash_flow" in available_canonical_fields and "net_income" in available_canonical_fields
            else:
                can_compute = any(c in available_canonical_fields for c in src_cols)

            if can_compute:
                available_feats.append(feat_name)

        return available_feats

    def validate_dataset_suitability(self, mapping: SchemaMappingResult, raise_on_insufficient: bool = True) -> None:
        """Validate that the mapped columns provide usable financial data for anomaly detection."""
        if mapping.ambiguous_columns:
            ambig_details = "; ".join(f"{k}: {v}" for k, v in mapping.ambiguous_columns.items())
            logger.warning("Dataset contains ambiguous columns: %s", ambig_details)

        num_features = len(mapping.available_features)
        num_financial_fields = len(mapping.available_financial_fields)
        num_unmapped_numeric = len(mapping.unmapped_numeric_columns)

        total_numeric_signals = num_financial_fields + num_features + num_unmapped_numeric

        if total_numeric_signals == 0:
            msg = (
                f"Uploaded dataset does not contain enough financial fields to perform reliable anomaly detection. "
                f"Found 0 usable financial features. "
                f"Detected columns: {list(mapping.mapped_columns.keys()) + mapping.unmapped_columns}"
            )
            if raise_on_insufficient:
                raise DataValidationError(msg)
            else:
                logger.warning(msg)
