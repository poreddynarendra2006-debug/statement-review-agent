"""Data models and data structures for FinSight AI Anomaly Detection Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AnomalyType(str, Enum):
    """Classification of statistical unusualness in financial statements."""

    def __str__(self) -> str:
        return self.value

    UNIVARIATE_OUTLIER = "univariate_outlier"
    MULTIVARIATE_OUTLIER = "multivariate_outlier"
    CROSS_FINANCIAL_PATTERN = "cross_financial_pattern"
    PEER_OUTLIER = "peer_outlier"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    RATIO_ANOMALY = "ratio_anomaly"
    PROFITABILITY_ANOMALY = "profitability_anomaly"
    LIQUIDITY_ANOMALY = "liquidity_anomaly"
    LEVERAGE_ANOMALY = "leverage_anomaly"
    CASH_FLOW_ANOMALY = "cash_flow_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    HISTORICAL_OUTLIER = "historical_outlier"
    CROSS_FIGURE_PATTERN = "cross_figure_pattern"
    EXTREME_FINANCIAL_VALUE = "extreme_financial_value"
    GROWTH_ANOMALY = "growth_anomaly"
    MARGIN_ANOMALY = "margin_anomaly"
    SUDDEN_FINANCIAL_CHANGE = "sudden_financial_change"
    REVENUE_PROFIT_MISMATCH = "revenue_profit_mismatch"
    CASH_FLOW_PROFIT_DIVERGENCE = "cash_flow_profit_divergence"
    RATIO_INCONSISTENCY = "ratio_inconsistency"


class Severity(str, Enum):
    """Degree of statistical unusualness requiring human review."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DatasetProfile:
    """Comprehensive analytical profile of an uploaded financial dataset."""

    num_rows: int = 0
    num_columns: int = 0
    raw_columns: list[str] = field(default_factory=list)
    numerical_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    entity_column: Optional[str] = None
    year_column: Optional[str] = None
    amount_columns: list[str] = field(default_factory=list)
    ratio_columns: list[str] = field(default_factory=list)
    percentage_columns: list[str] = field(default_factory=list)
    grouping_columns: list[str] = field(default_factory=list)
    has_temporal_ordering: bool = False
    has_peer_grouping: bool = False
    temporal_reason: str = ""
    peer_reason: str = ""
    is_single_period: bool = True
    is_single_entity: bool = True
    unique_entities_count: int = 0
    unique_periods_count: int = 0
    data_scale: str = "small"  # "small", "medium", "large", "very_large"


@dataclass
class FinancialRecord:
    """Normalized financial record representing a single entity or company-year observation."""

    company: Optional[str] = None
    year: Optional[int] = None
    record_id: Optional[str] = None
    category: Optional[str] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    operating_expenses: Optional[float] = None
    ebitda: Optional[float] = None
    eps: Optional[float] = None
    market_cap: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholder_equity: Optional[float] = None
    cash: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roi: Optional[float] = None
    gross_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None
    return_on_tangible_equity: Optional[float] = None
    number_of_employees: Optional[float] = None
    inflation_rate: Optional[float] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a dictionary containing standard fields and any extra dynamic fields."""
        base: dict[str, Any] = {
            "company": self.company,
            "year": int(self.year) if self.year is not None and self.year > 0 else None,
            "record_id": self.record_id,
            "category": self.category,
            "market_cap": self.market_cap,
            "revenue": float(self.revenue) if self.revenue is not None else None,
            "gross_profit": float(self.gross_profit) if self.gross_profit is not None else None,
            "net_income": float(self.net_income) if self.net_income is not None else None,
            "operating_income": (
                float(self.operating_income) if self.operating_income is not None else None
            ),
            "operating_expenses": (
                float(self.operating_expenses) if self.operating_expenses is not None else None
            ),
            "eps": float(self.eps) if self.eps is not None else None,
            "ebitda": float(self.ebitda) if self.ebitda is not None else None,
            "shareholder_equity": (
                float(self.shareholder_equity) if self.shareholder_equity is not None else None
            ),
            "total_assets": float(self.total_assets) if self.total_assets is not None else None,
            "total_liabilities": (
                float(self.total_liabilities) if self.total_liabilities is not None else None
            ),
            "cash": float(self.cash) if self.cash is not None else None,
            "operating_cash_flow": (
                float(self.operating_cash_flow) if self.operating_cash_flow is not None else None
            ),
            "investing_cash_flow": (
                float(self.investing_cash_flow) if self.investing_cash_flow is not None else None
            ),
            "financing_cash_flow": (
                float(self.financing_cash_flow) if self.financing_cash_flow is not None else None
            ),
            "current_ratio": float(self.current_ratio) if self.current_ratio is not None else None,
            "debt_to_equity": float(self.debt_to_equity) if self.debt_to_equity is not None else None,
            "roe": float(self.roe) if self.roe is not None else None,
            "roa": float(self.roa) if self.roa is not None else None,
            "roi": float(self.roi) if self.roi is not None else None,
            "gross_margin": float(self.gross_margin) if self.gross_margin is not None else None,
            "net_profit_margin": (
                float(self.net_profit_margin) if self.net_profit_margin is not None else None
            ),
            "free_cash_flow_per_share": (
                float(self.free_cash_flow_per_share)
                if self.free_cash_flow_per_share is not None
                else None
            ),
            "return_on_tangible_equity": (
                float(self.return_on_tangible_equity)
                if self.return_on_tangible_equity is not None
                else None
            ),
            "number_of_employees": (
                float(self.number_of_employees) if self.number_of_employees is not None else None
            ),
            "inflation_rate": float(self.inflation_rate) if self.inflation_rate is not None else None,
        }
        if self.extra_fields:
            for k, v in self.extra_fields.items():
                if k not in base:
                    base[k] = v
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinancialRecord:
        """Create a FinancialRecord from a dictionary with robust key mapping."""
        normalized = {str(k).strip(): v for k, v in data.items()}

        def get_val(*keys: str) -> Any:
            for k in keys:
                if k in normalized and normalized[k] is not None:
                    val = normalized[k]
                    if isinstance(val, str):
                        cleaned = val.replace(",", "").replace("$", "").replace("%", "").strip()
                        if cleaned == "" or cleaned.lower() in ("nan", "none", "null", "n/a", "-"):
                            return None
                        try:
                            return float(cleaned)
                        except ValueError:
                            return cleaned
                    return val
            return None

        company_raw = get_val("Company", "company", "ticker", "Ticker", "Company Name", "entity", "Entity", "Entity Name")
        company = str(company_raw).strip() if company_raw is not None and str(company_raw).strip().lower() not in ("none", "nan", "") else None

        year_raw = get_val("Year", "year", "Fiscal Year", "fiscal_year", "period", "Period", "Fiscal Period")
        try:
            year = int(float(year_raw)) if year_raw is not None and str(year_raw).strip().lower() not in ("none", "nan", "", "0") else None
        except (ValueError, TypeError):
            year = None

        record_id = str(get_val("record_id", "Record_ID", "record", "Record") or "")
        record_id = record_id if record_id else None

        revenue_raw = get_val("Revenue", "revenue", "Total Revenue", "total_revenue", "Sales", "sales", "Sales Revenue", "Turnover")
        revenue = float(revenue_raw) if revenue_raw is not None else None

        net_income_raw = get_val("Net Income", "net_income", "Net_Income", "Net Profit", "net_profit", "Net Earnings", "PAT")
        net_income = float(net_income_raw) if net_income_raw is not None else None

        known_keys = {
            "company", "year", "record_id", "category", "industry", "sector", "market_cap",
            "revenue", "gross_profit", "net_income", "operating_income", "operating_expenses",
            "ebitda", "eps", "shareholder_equity", "total_assets", "total_liabilities",
            "cash", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
            "current_ratio", "debt_to_equity", "roe", "roa", "roi", "gross_margin",
            "net_profit_margin", "free_cash_flow_per_share", "return_on_tangible_equity",
            "number_of_employees", "inflation_rate"
        }

        extra: dict[str, Any] = {}
        for k, v in normalized.items():
            k_low = k.lower().replace(" ", "_")
            if k_low not in known_keys and v is not None:
                if isinstance(v, (int, float)):
                    extra[k] = float(v)
                elif isinstance(v, str):
                    try:
                        extra[k] = float(v.replace(",", "").replace("$", "").replace("%", "").strip())
                    except ValueError:
                        extra[k] = v

        return cls(
            company=company,
            year=year,
            record_id=record_id,
            category=get_val("Category", "category", "Industry", "Sector", "industry", "sector"),
            market_cap=get_val(
                "Market Cap(in B USD)", "Market Cap", "market_cap", "Market Cap (in B USD)"
            ),
            revenue=revenue if revenue is not None else get_val("Turnover", "turnover", "Sales", "sales", "Total Revenue", "total_revenue"),
            gross_profit=get_val("Gross Profit", "gross_profit", "Gross_Profit", "Gross Earnings"),
            net_income=net_income if net_income is not None else get_val("Profit After Tax", "pat", "PAT", "Net Profit", "net_profit", "Profit", "profit", "Net Earnings", "net_earnings"),
            operating_income=get_val("Operating Income", "operating_income", "Operating_Income", "EBIT", "ebit"),
            operating_expenses=get_val("Operating Expenses", "operating_expenses", "Operating_Expenses", "Opex", "opex", "COGS", "cogs", "Cost of Goods Sold", "cost_of_goods_sold"),
            eps=get_val(
                "Earning Per Share", "eps", "EPS", "Earnings Per Share", "Earning_Per_Share"
            ),
            ebitda=get_val("EBITDA", "ebitda", "Ebitda"),
            shareholder_equity=get_val(
                "Share Holder Equity",
                "Shareholder Equity",
                "shareholder_equity",
                "Share_Holder_Equity",
                "Stockholders Equity",
                "Total Stockholders Equity",
                "Equity",
                "equity",
                "Net Assets",
                "net_assets",
                "Total Equity",
                "total_equity",
            ),
            total_assets=get_val("Total Assets", "total_assets", "Assets", "assets", "Total_Assets"),
            total_liabilities=get_val(
                "Total Liabilities", "total_liabilities", "Liabilities", "liabilities", "Total_Liabilities", "Debt", "total_debt", "Borrowings", "borrowings"
            ),
            cash=get_val("Cash", "cash", "Cash and Cash Equivalents", "Cash & Cash Equivalents", "cash_and_cash_equivalents", "Cash and Bank Balances"),
            operating_cash_flow=get_val(
                "Operating Cash Flow", "operating_cash_flow", "Cash from Operations", "OCF", "Cash_Flow_Operating", "Cash Flow from Operations"
            ),
            investing_cash_flow=get_val(
                "Investing Cash Flow", "investing_cash_flow", "Cash from Investing", "Cash_Flow_Investing"
            ),
            financing_cash_flow=get_val(
                "Financing Cash Flow", "financing_cash_flow", "Cash from Financing", "Cash_Flow_Financing"
            ),
            current_ratio=get_val("Current Ratio", "current_ratio", "Current_Ratio"),
            debt_to_equity=get_val(
                "Debt/Equity Ratio", "Debt to Equity Ratio", "debt_to_equity", "debt_equity_ratio", "Debt_to_Equity", "D/E", "D/E Ratio", "d_e_ratio"
            ),
            roe=get_val("ROE", "roe", "Return on Equity", "Return_on_Equity"),
            roa=get_val("ROA", "roa", "Return on Assets", "Return_on_Assets"),
            roi=get_val("ROI", "roi", "Return on Investment"),
            gross_margin=get_val("Gross Margin", "gross_margin", "Gross_Margin"),
            net_profit_margin=get_val(
                "Net Profit Margin", "net_profit_margin", "Net Margin", "net_margin"
            ),
            free_cash_flow_per_share=get_val(
                "Free Cash Flow per Share", "free_cash_flow_per_share", "FCF per share", "fcf_per_share"
            ),
            return_on_tangible_equity=get_val(
                "Return on Tangible Equity", "return_on_tangible_equity", "ROTE"
            ),
            number_of_employees=get_val(
                "Number of Employees", "number_of_employees", "Employees", "employees"
            ),
            inflation_rate=get_val(
                "Inflation Rate",
                "inflation_rate",
                "Inflation",
                "Inflation Rate(in US)",
                "Inflation Rate (in US)",
                "Inflation Rate in US",
                "inflation_rate_in_us",
            ),
            extra_fields=extra,
        )


@dataclass
class AnomalyFinding:
    """Finding produced by the Anomaly Agent for human reviewer and downstream agents."""

    company: Optional[str] = None
    year: Optional[int] = None
    anomaly_type: AnomalyType = AnomalyType.MULTIVARIATE_OUTLIER
    score: float = 0.0
    severity: Severity = Severity.MEDIUM
    confidence: float = 0.85
    record_id: Optional[str] = None
    relevant_features: dict[str, float] = field(default_factory=dict)
    deviations: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommendation: str = ""
    model_name: str = "IsolationForest"
    model_mode: str = "GENERIC_LOCAL"
    percentile: Optional[float] = None

    actual_values: dict[str, Any] = field(default_factory=dict)
    normal_ranges: dict[str, Any] = field(default_factory=dict)

    @property
    def contributing_features(self) -> dict[str, float]:
        """Alias for relevant_features for backward and forward compatibility."""
        return self.relevant_features

    @property
    def type(self) -> str:
        """String representation of anomaly_type value."""
        return self.anomaly_type.value if isinstance(self.anomaly_type, AnomalyType) else str(self.anomaly_type)

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to a serializable dictionary matching the output contract."""
        return {
            "record_id": self.record_id,
            "company": self.company,
            "year": int(self.year) if self.year is not None and self.year > 0 else None,
            "score": round(float(self.score), 4),
            "severity": self.severity.value
            if isinstance(self.severity, Severity)
            else str(self.severity),
            "confidence": round(float(self.confidence), 4),
            "type": self.anomaly_type.value
            if isinstance(self.anomaly_type, AnomalyType)
            else str(self.anomaly_type),
            "anomaly_type": self.anomaly_type.value
            if isinstance(self.anomaly_type, AnomalyType)
            else str(self.anomaly_type),
            "contributing_features": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in self.relevant_features.items()
            },
            "relevant_features": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in self.relevant_features.items()
            },
            "deviations": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in self.deviations.items()
            },
            "actual_values": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in self.actual_values.items()
            },
            "normal_ranges": self.normal_ranges,
            "percentile": round(float(self.percentile), 4) if self.percentile is not None else None,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "model_name": self.model_name,
            "model_mode": self.model_mode,
        }


@dataclass
class DatasetSummaryInfo:
    """Metadata summary of the analyzed dataset."""

    records: int = 0
    raw_columns_count: int = 0
    derived_features_count: int = 0
    candidate_features_count: int = 0
    features_used: int = 0
    features_available: int = 0
    temporal_analysis: bool = False
    peer_analysis: bool = False
    strategy_selected: str = "IsolationForest"
    raw_column_names: list[str] = field(default_factory=list)
    available_feature_names: list[str] = field(default_factory=list)
    derived_feature_names: list[str] = field(default_factory=list)
    candidate_feature_names: list[str] = field(default_factory=list)
    used_feature_names: list[str] = field(default_factory=list)
    ignored_features: dict[str, str] = field(default_factory=dict)
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.raw_column_names:
            self.raw_columns_count = len(self.raw_column_names)
        if self.derived_feature_names:
            self.derived_features_count = len(self.derived_feature_names)
        if self.candidate_feature_names:
            self.candidate_features_count = len(self.candidate_feature_names)
        if self.available_feature_names:
            self.features_available = len(self.available_feature_names)
        if self.used_feature_names:
            self.features_used = len(self.used_feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "raw_columns_count": self.raw_columns_count,
            "derived_features_count": self.derived_features_count,
            "candidate_features_count": self.candidate_features_count,
            "features_used": self.features_used,
            "features_available": self.features_available,
            "temporal_analysis": self.temporal_analysis,
            "peer_analysis": self.peer_analysis,
            "strategy_selected": self.strategy_selected,
            "feature_details": {
                "raw_columns": self.raw_column_names,
                "available": self.available_feature_names,
                "derived": self.derived_feature_names,
                "candidates": self.candidate_feature_names,
                "used": self.used_feature_names,
                "ignored": self.ignored_features,
            },
            "message": self.message,
        }


@dataclass
class AnomalySummaryInfo:
    """Summary of identified anomalies and severity breakdown."""

    total_anomalies: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    anomaly_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_anomalies": self.total_anomalies,
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
            "anomaly_rate": round(float(self.anomaly_rate), 4),
        }


@dataclass
class AnalysisResult:
    """Complete structured output produced by the Anomaly Agent for downstream agents."""

    dataset_summary: DatasetSummaryInfo
    anomaly_summary: AnomalySummaryInfo
    anomalies: list[AnomalyFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_summary": self.dataset_summary.to_dict(),
            "anomaly_summary": self.anomaly_summary.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass
class FeatureMetadata:
    """Metadata describing an engineered financial feature for provenance."""

    name: str
    source_columns: list[str]
    formula: str
    description: str
    category: str
    requires_history: bool = False


@dataclass
class DatasetSummary:
    """Summary of inspected dataset structure and contents."""

    dataset_name: str
    num_rows: int
    num_columns: int
    columns: list[str]
    data_types: dict[str, str]
    company_identifier: str
    year_field: str
    missing_values: dict[str, int]
    duplicate_rows: int
    unique_companies: list[str]
    years_available: list[int]
    observations_per_company: dict[str, int]
    numerical_fields: list[str]
    categorical_fields: list[str]
