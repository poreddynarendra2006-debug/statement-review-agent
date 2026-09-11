"""Financial ratio analysis module for FinSight Trend Agent.

Calculates or extracts:
- Liquidity: Current Ratio
- Leverage: Debt/Equity Ratio
- Profitability: ROE, ROA, ROI, Net Profit Margin
- Return: Return on Tangible Equity

Rules:
- Status must be one of: HEALTHY, WARNING, CRITICAL, UNKNOWN. Legacy "provided" is eliminated.
- If required component fields are available, COMPUTES the ratio with source="computed" and exact formula.
- If component fields are unavailable but reported ratio is present, uses reported value with source="reported".
- If neither is available or denominator is zero/invalid, returns value=None with status="UNKNOWN".
- Fully deterministic arithmetic without LLM.
"""

from dataclasses import asdict, dataclass
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import RATIO_BENCHMARKS

logger = logging.getLogger(__name__)


def classify_ratio_health(canonical_name: str, value: Optional[float]) -> str:
    """Assign health band status (HEALTHY, WARNING, CRITICAL, UNKNOWN) based on benchmarks."""
    if value is None or not np.isfinite(value):
        return "UNKNOWN"

    benchmarks = RATIO_BENCHMARKS.get(canonical_name, {})
    val = float(value)

    if canonical_name == "debt_to_equity":
        # Lower is better; negative equity indicates critical balance sheet distress
        if val < 0.0:
            return "CRITICAL"
        healthy_max = benchmarks.get("healthy_max", 1.5)
        warning_max = benchmarks.get("warning_max", 2.5)
        if val <= healthy_max:
            return "HEALTHY"
        elif val <= warning_max:
            return "WARNING"
        else:
            return "CRITICAL"
    else:
        # Higher is better
        healthy_min = benchmarks.get("healthy_min", 1.5)
        warning_min = benchmarks.get("warning_min", 1.0)
        if val >= healthy_min:
            return "HEALTHY"
        elif val >= warning_min:
            return "WARNING"
        else:
            return "CRITICAL"


@dataclass
class RatioResult:
    company: str
    year: int
    category: str  # "liquidity", "leverage", "profitability", "return"
    ratio_name: str
    value: Optional[float]
    status: str  # "HEALTHY", "WARNING", "CRITICAL", "UNKNOWN"
    formula: Optional[str] = None
    source: str = "reported"  # "reported" or "computed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_numeric(row: Any, candidate_keys: List[str]) -> Optional[float]:
    """Extract float value from row using candidate keys (exact match or cleaned case-insensitive match)."""
    # 1. Direct dictionary/Series check
    for k in candidate_keys:
        if k in row and pd.notna(row[k]):
            try:
                v = float(row[k])
                if np.isfinite(v):
                    return v
            except (ValueError, TypeError):
                continue

    # 2. Case-insensitive and normalized character check across all keys in row
    row_keys = list(row.index) if hasattr(row, "index") else list(row.keys())
    cleaned_candidates = {re.sub(r"[^\w]", "", k.lower()): k for k in candidate_keys}

    for rk in row_keys:
        cleaned_rk = re.sub(r"[^\w]", "", str(rk).lower())
        if cleaned_rk in cleaned_candidates:
            val = row[rk]
            if pd.notna(val):
                try:
                    v = float(val)
                    if np.isfinite(v):
                        return v
                except (ValueError, TypeError):
                    continue
    return None


class RatioAnalyzer:
    """Analyzes, computes, and standardizes financial ratios across companies."""

    def __init__(self, benchmarks: Optional[Dict[str, Dict[str, float]]] = None):
        self.benchmarks = benchmarks or RATIO_BENCHMARKS
        self.supported_ratios = [
            {"name": "Current Ratio", "canonical": "current_ratio", "category": "liquidity"},
            {"name": "Debt/Equity Ratio", "canonical": "debt_to_equity", "category": "leverage"},
            {"name": "ROE", "canonical": "roe", "category": "profitability"},
            {"name": "ROA", "canonical": "roa", "category": "profitability"},
            {"name": "ROI", "canonical": "roi", "category": "profitability"},
            {"name": "Net Profit Margin", "canonical": "margin", "category": "profitability"},
            {"name": "Return on Tangible Equity", "canonical": "return_on_tangible_equity", "category": "return"},
        ]

    def _process_current_ratio(self, row: Any, company: str, year: int) -> RatioResult:
        ca = _find_numeric(row, ["current_assets", "current assets", "total_current_assets", "total current assets"])
        cl = _find_numeric(row, ["current_liabilities", "current liabilities", "total_current_liabilities", "total current liabilities"])

        formula = "Current Assets / Current Liabilities"
        if ca is not None and cl is not None:
            if cl == 0.0:
                return RatioResult(company, year, "liquidity", "Current Ratio", None, "UNKNOWN", formula, "computed")
            val = round(ca / cl, 4)
            status = classify_ratio_health("current_ratio", val)
            return RatioResult(company, year, "liquidity", "Current Ratio", val, status, formula, "computed")

        # Fallback to reported
        rep = _find_numeric(row, ["current_ratio", "current ratio", "cr", "liquidity_ratio"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("current_ratio", val)
            return RatioResult(company, year, "liquidity", "Current Ratio", val, status, None, "reported")

        return RatioResult(company, year, "liquidity", "Current Ratio", None, "UNKNOWN", None, "reported")

    def _process_debt_to_equity(self, row: Any, company: str, year: int) -> RatioResult:
        tl = _find_numeric(row, ["total_liabilities", "total liabilities", "liabilities"])
        se = _find_numeric(row, ["shareholder_equity", "shareholder equity", "share holder equity", "total_shareholders_equity", "stockholders_equity", "equity"])

        formula = "Total Liabilities / Shareholder Equity"
        if tl is not None and se is not None:
            if se == 0.0:
                return RatioResult(company, year, "leverage", "Debt/Equity Ratio", None, "UNKNOWN", formula, "computed")
            val = round(tl / se, 4)
            status = classify_ratio_health("debt_to_equity", val)
            return RatioResult(company, year, "leverage", "Debt/Equity Ratio", val, status, formula, "computed")

        rep = _find_numeric(row, ["debt_to_equity", "debt/equity_ratio", "debt/equity ratio", "debt_equity_ratio", "d/e", "leverage_ratio"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("debt_to_equity", val)
            return RatioResult(company, year, "leverage", "Debt/Equity Ratio", val, status, None, "reported")

        return RatioResult(company, year, "leverage", "Debt/Equity Ratio", None, "UNKNOWN", None, "reported")

    def _process_roe(self, row: Any, company: str, year: int) -> RatioResult:
        ni = _find_numeric(row, ["net_income", "net income", "net_profit", "net profit", "profit", "earnings", "pat"])
        se = _find_numeric(row, ["shareholder_equity", "shareholder equity", "share holder equity", "total_shareholders_equity", "stockholders_equity", "equity"])

        formula = "Net Income / Shareholder Equity * 100"
        if ni is not None and se is not None:
            if se == 0.0:
                return RatioResult(company, year, "profitability", "ROE", None, "UNKNOWN", formula, "computed")
            val = round((ni / se) * 100.0, 4)
            status = classify_ratio_health("roe", val)
            return RatioResult(company, year, "profitability", "ROE", val, status, formula, "computed")

        rep = _find_numeric(row, ["roe", "return_on_equity", "return on equity"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("roe", val)
            return RatioResult(company, year, "profitability", "ROE", val, status, None, "reported")

        return RatioResult(company, year, "profitability", "ROE", None, "UNKNOWN", None, "reported")

    def _process_roa(self, row: Any, company: str, year: int) -> RatioResult:
        ni = _find_numeric(row, ["net_income", "net income", "net_profit", "net profit", "profit", "earnings", "pat"])
        ta = _find_numeric(row, ["total_assets", "total assets", "assets"])

        formula = "Net Income / Total Assets * 100"
        if ni is not None and ta is not None:
            if ta == 0.0:
                return RatioResult(company, year, "profitability", "ROA", None, "UNKNOWN", formula, "computed")
            val = round((ni / ta) * 100.0, 4)
            status = classify_ratio_health("roa", val)
            return RatioResult(company, year, "profitability", "ROA", val, status, formula, "computed")

        rep = _find_numeric(row, ["roa", "return_on_assets", "return on assets"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("roa", val)
            return RatioResult(company, year, "profitability", "ROA", val, status, None, "reported")

        return RatioResult(company, year, "profitability", "ROA", None, "UNKNOWN", None, "reported")

    def _process_roi(self, row: Any, company: str, year: int) -> RatioResult:
        ni = _find_numeric(row, ["net_income", "net income", "net_profit", "net profit", "profit", "earnings", "pat"])
        se = _find_numeric(row, ["shareholder_equity", "shareholder equity", "share holder equity", "total_shareholders_equity", "stockholders_equity", "equity"])
        debt = _find_numeric(row, ["debt", "total_debt", "total debt"])

        formula = "Net Income / (Shareholder Equity + Debt) * 100"
        if ni is not None and se is not None and debt is not None:
            invested = se + debt
            if invested == 0.0:
                return RatioResult(company, year, "profitability", "ROI", None, "UNKNOWN", formula, "computed")
            val = round((ni / invested) * 100.0, 4)
            status = classify_ratio_health("roi", val)
            return RatioResult(company, year, "profitability", "ROI", val, status, formula, "computed")

        rep = _find_numeric(row, ["roi", "return_on_investment", "return on investment"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("roi", val)
            return RatioResult(company, year, "profitability", "ROI", val, status, None, "reported")

        return RatioResult(company, year, "profitability", "ROI", None, "UNKNOWN", None, "reported")

    def _process_margin(self, row: Any, company: str, year: int) -> RatioResult:
        ni = _find_numeric(row, ["net_income", "net income", "net_profit", "net profit", "profit", "earnings", "pat"])
        rev = _find_numeric(row, ["revenue", "sales", "total_revenue", "turnover", "operating_revenue"])

        formula = "Net Income / Revenue * 100"
        if ni is not None and rev is not None:
            if rev == 0.0:
                return RatioResult(company, year, "profitability", "Net Profit Margin", None, "UNKNOWN", formula, "computed")
            val = round((ni / rev) * 100.0, 4)
            status = classify_ratio_health("margin", val)
            return RatioResult(company, year, "profitability", "Net Profit Margin", val, status, formula, "computed")

        rep = _find_numeric(row, ["margin", "net_profit_margin", "net profit margin", "profit_margin", "profit margin", "net_margin"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("margin", val)
            return RatioResult(company, year, "profitability", "Net Profit Margin", val, status, None, "reported")

        return RatioResult(company, year, "profitability", "Net Profit Margin", None, "UNKNOWN", None, "reported")

    def _process_rote(self, row: Any, company: str, year: int) -> RatioResult:
        ni = _find_numeric(row, ["net_income", "net income", "net_profit", "net profit", "profit", "earnings", "pat"])
        te = _find_numeric(row, ["tangible_equity", "tangible equity", "tangible_shareholder_equity"])

        if ni is not None and te is not None:
            formula = "Net Income / Tangible Equity * 100"
            if te == 0.0:
                return RatioResult(company, year, "return", "Return on Tangible Equity", None, "UNKNOWN", formula, "computed")
            val = round((ni / te) * 100.0, 4)
            status = classify_ratio_health("return_on_tangible_equity", val)
            return RatioResult(company, year, "return", "Return on Tangible Equity", val, status, formula, "computed")

        # Check shareholder equity minus intangibles
        se = _find_numeric(row, ["shareholder_equity", "shareholder equity", "share holder equity", "total_shareholders_equity", "stockholders_equity", "equity"])
        ia = _find_numeric(row, ["intangible_assets", "intangible assets", "intangibles"])
        if ni is not None and se is not None and ia is not None:
            formula = "Net Income / (Shareholder Equity - Intangible Assets) * 100"
            derived_te = se - ia
            if derived_te == 0.0:
                return RatioResult(company, year, "return", "Return on Tangible Equity", None, "UNKNOWN", formula, "computed")
            val = round((ni / derived_te) * 100.0, 4)
            status = classify_ratio_health("return_on_tangible_equity", val)
            return RatioResult(company, year, "return", "Return on Tangible Equity", val, status, formula, "computed")

        rep = _find_numeric(row, ["return_on_tangible_equity", "return on tangible equity", "rote"])
        if rep is not None:
            val = round(rep, 4)
            status = classify_ratio_health("return_on_tangible_equity", val)
            return RatioResult(company, year, "return", "Return on Tangible Equity", val, status, None, "reported")

        return RatioResult(company, year, "return", "Return on Tangible Equity", None, "UNKNOWN", None, "reported")

    def analyze_ratios(self, df: pd.DataFrame) -> List[RatioResult]:
        """Extract or compute available ratios per company and year."""
        results: List[RatioResult] = []

        if df.empty or "company" not in df.columns or "year" not in df.columns:
            return results

        sorted_df = df.sort_values(by=["company", "year"], ascending=[True, True])

        for _, row in sorted_df.iterrows():
            company = str(row["company"])
            year = int(row["year"])

            results.append(self._process_current_ratio(row, company, year))
            results.append(self._process_debt_to_equity(row, company, year))
            results.append(self._process_roe(row, company, year))
            results.append(self._process_roa(row, company, year))
            results.append(self._process_roi(row, company, year))
            results.append(self._process_margin(row, company, year))
            results.append(self._process_rote(row, company, year))

        return results

    def get_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ratio results as a tidy DataFrame."""
        results = self.analyze_ratios(df)
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in results])

