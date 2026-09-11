"""Explainability and Recommendation Engine for FinSight AI Anomaly Detection Agent.

Generates grounded, data-backed natural language explanations citing only actual
dataset features and values, categorizes financial anomaly types, and produces
tailored audit recommendations without using non-ASCII characters like 'σ'.

Explanations lead with what happened in financial terms (for example "Revenue
increased 42.1% year over year, while net income increased 756.6%"). The
statistical deviations stay on the finding as supporting evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from finsight.core.models import AnomalyFinding, AnomalyType, Severity

logger = logging.getLogger(__name__)

# A measure is described in the explanation when its robust z-score reaches this.
NOTABLE_DEVIATION = 2.0
MAX_EXPLAINED_MEASURES = 3

_ACRONYMS = {"roe": "ROE", "roa": "ROA", "roi": "ROI", "ebitda": "EBITDA", "eps": "EPS", "ocf": "OCF"}

_PLAIN_NAMES = {
    "revenue": "revenue",
    "gross_profit": "gross profit",
    "net_income": "net income",
    "operating_income": "operating income",
    "operating_expenses": "operating expenses",
    "ebitda": "EBITDA",
    "eps": "earnings per share",
    "operating_cash_flow": "operating cash flow",
    "investing_cash_flow": "investing cash flow",
    "financing_cash_flow": "financing cash flow",
    "shareholder_equity": "shareholder equity",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "gross_margin": "gross margin",
    "net_profit_margin": "net profit margin",
    "operating_margin": "operating margin",
    "ebitda_margin": "EBITDA margin",
    "roe": "return on equity (ROE)",
    "roa": "return on assets (ROA)",
    "roi": "return on investment (ROI)",
    "return_on_tangible_equity": "return on tangible equity",
    "debt_equity_ratio": "debt-to-equity ratio",
    "liabilities_to_assets": "liabilities-to-assets ratio",
    "asset_turnover": "asset turnover",
    "current_ratio": "current ratio",
    "quick_ratio": "quick ratio",
    "operating_cash_flow_to_net_income": "operating cash flow relative to net income",
    "cash_flow_to_revenue": "operating cash flow as a share of revenue",
    "free_cash_flow_per_share": "free cash flow per share",
}

# Level measures expressed as a fraction and shown as a percentage.
_PERCENT_LEVELS = {
    "gross_margin", "net_profit_margin", "operating_margin", "ebitda_margin", "roe", "roa", "roi",
    "return_on_tangible_equity", "liabilities_to_assets", "cash_flow_to_revenue",
}

# Level measures shown as a multiple.
_MULTIPLE_LEVELS = {
    "debt_equity_ratio", "asset_turnover", "current_ratio", "quick_ratio", "operating_cash_flow_to_net_income",
}

# Year-over-year change features: the measure that changed, and whether it is a percentage.
_CHANGE_BASES = {
    "gross_margin_change": ("gross_margin", True),
    "net_margin_change": ("net_profit_margin", True),
    "ebitda_margin_change": ("ebitda_margin", True),
    "operating_margin_change": ("operating_margin", True),
    "roe_change": ("roe", True),
    "roa_change": ("roa", True),
    "roi_change": ("roi", True),
    "current_ratio_change": ("current_ratio", False),
    "debt_equity_change": ("debt_equity_ratio", False),
}

_SPREADS = {"revenue_profit_growth_spread", "ocf_net_income_divergence"}

_GROWTH_STORY_FEATURES = {
    "revenue_profit_growth_spread", "ocf_net_income_divergence",
    "revenue_growth", "net_income_growth", "operating_cash_flow_growth",
}


def format_feature_value(feature_name: str, value: float) -> str:
    """Format numeric feature values appropriately as percentages, ratios, or currency amounts based on feature name."""
    if value is None or not np.isfinite(value):
        return "N/A"

    name_low = str(feature_name).lower()

    # 1. Percentage point change features (YoY margin/ratio shifts)
    is_change = any(kw in name_low for kw in ("change", "shift", "diff"))
    if is_change and any(kw in name_low for kw in ("margin", "growth", "roe", "roa", "roi", "rate")):
        return f"{value * 100.0:+.1f} percentage points"

    # 2. Percentage features (margins, growth rates, returns, spreads, inflation)
    is_percentage_metric = any(
        kw in name_low for kw in ("margin", "growth", "roe", "roa", "roi", "spread", "divergence", "rate", "percent", "%", "pct")
    )
    if is_percentage_metric:
        # Check if already in 0-100 percentage units based on name
        is_already_scaled = any(kw in name_low for kw in ("percent", "%", "_pct", "pct", "(in %)", "in_us", "(in us)", "inflation_rate"))
        val_pct = value if is_already_scaled else value * 100.0
        return f"{val_pct:.1f}%"

    # 3. Ratio / Multiplier features
    is_ratio = any(kw in name_low for kw in ("ratio", "turnover", "per_share", "debt_equity", "debt_to_equity", "debt_to_assets", "liabilities_to_assets", "multiple"))
    if is_ratio:
        return f"{value:.2f}x"

    # 4. Dollar / Currency / Level features
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"{value:.2f}"


def display_name(feature_name: str) -> str:
    """Readable title for a feature name, e.g. 'roa_change' -> 'ROA Change'."""
    return " ".join(_ACRONYMS.get(w.lower(), w.title()) for w in str(feature_name).split("_") if w)


def _plain(feature_name: str) -> str:
    return _PLAIN_NAMES.get(feature_name, str(feature_name).replace("_", " "))


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(np.isfinite(value))


def _pct(fraction: float) -> str:
    pct = fraction * 100.0
    return f"{pct:,.0f}%" if abs(pct) >= 1000 else f"{pct:.1f}%"


def _points(fraction: float) -> str:
    return f"{abs(fraction) * 100.0:.1f} percentage points"


def _amount(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1_000_000 else f"{value:,.2f}"


def _moved(fraction: float) -> str:
    return f"{'increased' if fraction >= 0 else 'decreased'} {_pct(abs(fraction))}"


def format_evidence_value(feature_name: str, value: float) -> str:
    """Format a feature value in its financial unit: %, percentage points, multiple or amount.

    Amounts carry no currency symbol because the unit of an uploaded file is unknown.
    """
    if not _is_finite(value):
        return "N/A"
    name = str(feature_name)
    if name in _CHANGE_BASES:
        return f"{value * 100.0:+.1f} percentage points" if _CHANGE_BASES[name][1] else f"{value:+.2f}"
    if name in _SPREADS:
        return f"{value * 100.0:+.1f} percentage points"
    if name.endswith("_growth") or name in _PERCENT_LEVELS:
        return _pct(value)
    if name in _MULTIPLE_LEVELS:
        return f"{value:.2f}x"
    if any(kw in name.lower() for kw in ("percent", "pct", "margin", "rate", "growth", "%")):
        return format_feature_value(name, value)
    return _amount(value)


def _growth_story(
    selected: Sequence[str],
    values: dict[str, Any],
    anomaly_type: AnomalyType,
) -> tuple[Optional[str], set[str]]:
    """Describe revenue, profit and cash-flow growth together when their gap is the finding."""
    rev = values.get("revenue_growth")
    ni = values.get("net_income_growth")
    ocf = values.get("operating_cash_flow_growth")
    a_type = anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)

    wants_profit = (
        "revenue_profit_growth_spread" in selected
        or ("revenue_growth" in selected and "net_income_growth" in selected)
        or (a_type == AnomalyType.REVENUE_PROFIT_MISMATCH.value and ("revenue_growth" in selected or "net_income_growth" in selected))
    )
    wants_cash = (
        "ocf_net_income_divergence" in selected
        or (a_type == AnomalyType.CASH_FLOW_PROFIT_DIVERGENCE.value and any(n in _GROWTH_STORY_FEATURES for n in selected))
    )

    if wants_profit and _is_finite(rev) and _is_finite(ni):
        others = [f"net income {_moved(ni)}"]
        include_cash = (wants_cash or "operating_cash_flow_growth" in selected) and _is_finite(ocf)
        if include_cash:
            others.append(f"operating cash flow {_moved(ocf)}")
        gap = "profit and cash-flow growth" if include_cash else "profit growth"
        sentence = (
            f"Revenue {_moved(rev)} year over year, while {' and '.join(others)}, "
            f"an unusually large gap between revenue growth and {gap}."
        )
        return sentence, set(_GROWTH_STORY_FEATURES)

    if wants_cash and _is_finite(ni) and _is_finite(ocf):
        sentence = (
            f"Net income {_moved(ni)} year over year, while operating cash flow {_moved(ocf)}, "
            f"an unusually large gap between reported profit and cash generation."
        )
        return sentence, set(_GROWTH_STORY_FEATURES)

    return None, set()


def _describe_measure(name: str, value: float, deviation: float, typical: Any) -> str:
    """One plain-language sentence about a single unusual measure."""
    has_typical = _is_finite(typical)

    if name.endswith("_growth"):
        sentence = f"{_capitalize(_plain(name[: -len('_growth')]))} {_moved(value)} year over year"
        if has_typical:
            sentence += f", compared with a typical change of {_pct(typical)} in this dataset"
        return sentence + "."

    if name in _CHANGE_BASES:
        base, is_percent = _CHANGE_BASES[name]
        size = _points(value) if is_percent else f"{abs(value):.2f}"
        sentence = f"{_capitalize(_plain(base))} {'rose' if value >= 0 else 'fell'} by {size} from the prior year"
        if has_typical:
            sentence += f", compared with a typical change of {format_evidence_value(name, typical)}"
        return sentence + "."

    if name == "revenue_profit_growth_spread":
        return (
            f"Net income growth {'exceeded' if value >= 0 else 'trailed'} revenue growth by {_points(value)}, "
            f"an unusually large revenue-profit growth gap."
        )

    if name == "ocf_net_income_divergence":
        return (
            f"Operating cash flow growth {'exceeded' if value >= 0 else 'trailed'} net income growth by {_points(value)}, "
            f"an unusually large gap between cash generation and reported profit."
        )

    sentence = (
        f"{_capitalize(_plain(name))} was {format_evidence_value(name, value)}, "
        f"unusually {'high' if deviation >= 0 else 'low'}"
    )
    if has_typical:
        sentence += f" compared with a typical {format_evidence_value(name, typical)} in this dataset"
    return sentence + "."


def categorize_anomaly_type(
    top_features: dict[str, float],
    deviations: Optional[dict[str, float]] = None,
    raw_record: Optional[dict[str, Any]] = None,
    is_temporal_shock: bool = False,
    is_peer_deviation: bool = False,
    *args: Any,
    **kwargs: Any,
) -> AnomalyType:
    """Categorize the statistical anomaly type based on dominant contributing features and context."""
    devs = deviations or kwargs.get("top_deviations") or kwargs.get("feature_deviations") or kwargs.get("feature_zscores") or {}
    record = raw_record or kwargs.get("raw_record", {})
    feature_names = set(top_features.keys())

    # Handle flexible positional/keyword args
    if args:
        for arg in args:
            if isinstance(arg, dict):
                if not devs and all(isinstance(v, (int, float, np.floating, np.integer)) for v in arg.values() if v is not None):
                    devs = arg
                elif not record:
                    record = arg

    # 1. Financial Logic Checks: Divergences and Mismatches
    if "ocf_net_income_divergence" in feature_names:
        return AnomalyType.CASH_FLOW_PROFIT_DIVERGENCE

    if "revenue_profit_growth_spread" in feature_names:
        return AnomalyType.REVENUE_PROFIT_MISMATCH

    rev_growth = top_features.get("revenue_growth")
    net_growth = top_features.get("net_income_growth")
    if rev_growth is not None and net_growth is not None:
        if (rev_growth < -0.05 and net_growth > 0.20) or (rev_growth > 0.20 and net_growth < -0.05):
            return AnomalyType.REVENUE_PROFIT_MISMATCH
        if abs(net_growth - rev_growth) > 0.60:
            return AnomalyType.REVENUE_PROFIT_MISMATCH
        if rev_growth > 2.0 or net_growth > 2.0 or rev_growth < -0.7 or net_growth < -0.7:
            return AnomalyType.GROWTH_ANOMALY

    ocf_to_net = top_features.get("operating_cash_flow_to_net_income")
    if ocf_to_net is not None and (ocf_to_net < -0.5 or ocf_to_net > 4.0):
        return AnomalyType.CASH_FLOW_PROFIT_DIVERGENCE

    # 2. Temporal shock
    if is_temporal_shock or any("growth" in f or "change" in f for f in feature_names):
        if any("growth" in f for f in feature_names):
            return AnomalyType.GROWTH_ANOMALY
        if any("margin_change" in f for f in feature_names):
            return AnomalyType.MARGIN_ANOMALY
        return AnomalyType.TEMPORAL_ANOMALY

    # 3. Peer outlier
    if is_peer_deviation:
        return AnomalyType.PEER_OUTLIER

    # 4. Cash flow anomalies
    cash_flow_feats = {
        "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
        "cash_flow_to_revenue", "free_cash_flow_per_share", "cash"
    }
    if feature_names.intersection(cash_flow_feats):
        return AnomalyType.CASH_FLOW_ANOMALY

    # 5. Profitability & Margins
    profit_feats = {
        "gross_margin", "net_profit_margin", "operating_margin", "ebitda_margin",
        "roe", "roa", "roi", "return_on_tangible_equity", "gross_profit", "net_income",
        "operating_income", "ebitda"
    }
    if feature_names.intersection(profit_feats):
        if any("margin" in f for f in feature_names):
            return AnomalyType.MARGIN_ANOMALY
        return AnomalyType.PROFITABILITY_ANOMALY

    # 6. Liquidity anomalies
    liq_feats = {"current_ratio", "quick_ratio", "working_capital_ratio", "current_assets", "current_liabilities"}
    if feature_names.intersection(liq_feats):
        return AnomalyType.LIQUIDITY_ANOMALY

    # 7. Leverage anomalies
    lev_feats = {"debt_equity_ratio", "liabilities_to_assets", "total_liabilities", "debt", "shareholder_equity"}
    if feature_names.intersection(lev_feats):
        return AnomalyType.LEVERAGE_ANOMALY

    # 8. Single Extreme Value vs Multivariate
    if len(feature_names) == 1:
        return AnomalyType.EXTREME_FINANCIAL_VALUE
    if len(feature_names) > 2:
        return AnomalyType.MULTIVARIATE_OUTLIER

    return AnomalyType.STATISTICAL_OUTLIER


def generate_recommendation(anomaly_type: AnomalyType, top_feature_names: Sequence[str]) -> str:
    """Generate tailored, context-specific audit review recommendations."""
    a_type_val = anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)

    if a_type_val in (AnomalyType.LIQUIDITY_ANOMALY.value, "liquidity_anomaly"):
        return "Review short-term debt obligations, working capital movements, and available liquid current assets."

    if a_type_val in (AnomalyType.LEVERAGE_ANOMALY.value, "leverage_anomaly"):
        return "Review debt maturity schedules, covenant compliance, interest coverage, and changes in shareholder equity."

    if a_type_val in (AnomalyType.CASH_FLOW_PROFIT_DIVERGENCE.value, AnomalyType.CASH_FLOW_ANOMALY.value, "cash_flow_anomaly", "cash_flow_profit_divergence"):
        return "Review operating cash flow components, accruals, working-capital changes, and potential revenue recognition timing differences."

    if a_type_val in (AnomalyType.REVENUE_PROFIT_MISMATCH.value, "revenue_profit_mismatch", AnomalyType.CROSS_FIGURE_PATTERN.value, "cross_figure_pattern"):
        return "Review gross margin trends, pricing changes, operating cost structure, one-time write-downs, or unbilled revenue accruals."

    if a_type_val in (AnomalyType.MARGIN_ANOMALY.value, AnomalyType.PROFITABILITY_ANOMALY.value, "profitability_anomaly", "margin_anomaly"):
        return "Review product mix profitability, COGS variance, operating expense shifts, and compare unit economics against baseline periods."

    if a_type_val in (AnomalyType.GROWTH_ANOMALY.value, AnomalyType.TEMPORAL_ANOMALY.value, AnomalyType.HISTORICAL_OUTLIER.value, "growth_anomaly", "temporal_anomaly", "historical_outlier"):
        return "Verify whether sudden YoY movements stem from mergers/acquisitions, restructuring, accounting policy changes, or organic expansion."

    if a_type_val in (AnomalyType.PEER_OUTLIER.value, "peer_outlier"):
        return "Compare company operational metrics and capital structure against industry cohort benchmarks to assess competitive differentiation."

    return "Conduct detailed line-item audit on dominant contributing financial fields to verify ledger accuracy and classification consistency."


def generate_grounded_explanation(
    record_id: Optional[str],
    company: Optional[str],
    year: Optional[int],
    anomaly_type: AnomalyType,
    top_features: dict[str, float],
    deviations: dict[str, float],
    actual_values: dict[str, float],
    normal_ranges: dict[str, Any],
    all_values: Optional[dict[str, float]] = None,
    typical_values: Optional[dict[str, float]] = None,
) -> str:
    """Explain the finding in financial terms, referencing only actual data.

    ``all_values`` holds every feature of the record, so related measures (such as
    revenue growth next to net income growth) can be described together.
    ``typical_values`` holds each feature's dataset median in its own unit.
    """
    if company and str(company).lower() not in ("none", "nan", ""):
        entity_label = str(company)
    elif record_id:
        entity_label = str(record_id) if str(record_id).lower().startswith("record") else f"Record {record_id}"
    else:
        entity_label = "Observation"
    period_label = f" ({year})" if year and str(year).lower() not in ("none", "nan", "0") else ""

    values: dict[str, Any] = {**(all_values or {}), **top_features, **actual_values}
    typical = typical_values or {}

    ranked = sorted(
        [(k, v) for k, v in deviations.items() if _is_finite(v)],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    selected = [k for k, v in ranked if abs(v) >= NOTABLE_DEVIATION][:MAX_EXPLAINED_MEASURES]
    if not selected:
        selected = [k for k, _ in ranked[:1]]

    sentences: list[str] = []
    story, covered = _growth_story(selected, values, anomaly_type)
    if story:
        sentences.append(story)
    for name in selected:
        if name in covered or not _is_finite(values.get(name)):
            continue
        sentences.append(_describe_measure(name, float(values[name]), float(deviations.get(name, 0.0)), typical.get(name)))

    a_type_val = anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)
    type_text = a_type_val.replace("_", " ")

    if not sentences:
        return (
            f"{entity_label}{period_label}: the overall combination of its financial measures is statistically unusual "
            f"for this dataset. This is a potential {type_text} requiring review."
        )

    measures = ", ".join(display_name(n) for n in selected)
    return (
        f"{entity_label}{period_label}: {' '.join(sentences)} "
        f"This is a potential {type_text} requiring review. Statistically unusual measures: {measures}."
    )


def build_anomaly_finding(
    row_idx: int = 0,
    raw_record: Optional[dict[str, Any]] = None,
    feature_row: Optional[dict[str, float]] = None,
    deviations_row: Optional[dict[str, float]] = None,
    score: float = 0.0,
    severity: Optional[Severity] = None,
    confidence: float = 0.85,
    model_name: str = "IsolationForest",
    model_mode: str = "GENERIC_LOCAL",
    percentile: Optional[float] = None,
    is_temporal_shock: bool = False,
    is_peer_deviation: bool = False,
    feature_medians: Optional[dict[str, float]] = None,
    feature_iqrs: Optional[dict[str, float]] = None,
    **kwargs: Any,
) -> AnomalyFinding:
    """Construct a complete, serializable AnomalyFinding dataclass.

    ``feature_medians`` and ``feature_iqrs`` must be in the features' own units
    (not scaled model inputs); they give the typical value and normal range.
    """
    raw_rec = dict(raw_record or kwargs.get("raw_record") or {})
    feats = dict(feature_row or kwargs.get("feature_values") or {})
    devs = dict(deviations_row or kwargs.get("feature_zscores") or {})

    # Extract company / year / record_id from kwargs or raw record
    company = kwargs.get("company", raw_rec.get("company"))
    year = kwargs.get("year", raw_rec.get("year"))
    rec_id = kwargs.get("record_id", raw_rec.get("record_id") or raw_rec.get("id") or str(row_idx))

    if company and "company" not in raw_rec:
        raw_rec["company"] = company
    if year and "year" not in raw_rec:
        raw_rec["year"] = year

    # Rank dominant contributing features by absolute deviation
    sorted_devs = sorted(
        [(k, v) for k, v in devs.items() if np.isfinite(v)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_devs = dict(sorted_devs[:5]) if sorted_devs else {}
    top_feats = {k: feats.get(k, 0.0) for k in top_devs.keys()} if top_devs else dict(list(feats.items())[:5])

    # Compute normal benchmark ranges
    actual_vals: dict[str, float] = {}
    normal_ranges: dict[str, list[float]] = {}

    for k in (top_devs.keys() if top_devs else top_feats.keys()):
        val = feats.get(k, 0.0)
        actual_vals[k] = val
        if feature_medians and k in feature_medians:
            med = feature_medians[k]
            iqr_val = (feature_iqrs.get(k, 1.0) if feature_iqrs else 1.0) or 1.0
            normal_ranges[k] = [round(med - 1.5 * iqr_val, 4), round(med + 1.5 * iqr_val, 4)]

    # Categorize anomaly type
    a_type = categorize_anomaly_type(
        top_features=top_feats,
        deviations=top_devs,
        raw_record=raw_rec,
        is_temporal_shock=is_temporal_shock,
        is_peer_deviation=is_peer_deviation,
    )

    explanation = generate_grounded_explanation(
        record_id=str(rec_id),
        company=str(company) if company else None,
        year=int(year) if year and str(year).isdigit() else None,
        anomaly_type=a_type,
        top_features=top_feats,
        deviations=top_devs,
        actual_values=actual_vals,
        normal_ranges=normal_ranges,
        all_values=feats,
        typical_values=feature_medians,
    )

    recommendation = generate_recommendation(a_type, list(top_devs.keys() if top_devs else top_feats.keys()))

    # Determine severity if not provided
    pct = percentile if percentile is not None else kwargs.get("score_percentile")
    if severity is None:
        if pct is not None:
            pct_val = pct if pct > 1.0 else pct * 100.0
            sev = Severity.HIGH if pct_val >= 90.0 else (Severity.MEDIUM if pct_val >= 70.0 else Severity.LOW)
        else:
            sev = Severity.HIGH if score >= 0.20 else (Severity.MEDIUM if score >= 0.10 else Severity.LOW)
    else:
        sev = severity

    return AnomalyFinding(
        company=str(company) if company and str(company).lower() not in ("none", "nan", "") else None,
        year=int(year) if year and str(year).lower() not in ("none", "nan", "0") else None,
        record_id=str(rec_id),
        anomaly_type=a_type,
        score=float(score),
        severity=sev,
        confidence=float(confidence),
        relevant_features=top_feats,
        deviations=top_devs,
        actual_values=actual_vals,
        normal_ranges=normal_ranges,
        percentile=float(pct) if pct is not None else None,
        explanation=explanation,
        recommendation=recommendation,
        model_name=model_name,
        model_mode=model_mode,
    )
