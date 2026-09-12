"""Forensic red flags: the patterns a model trained on one upload cannot learn.

The Isolation Forest learns what is normal *for this file* and reports what
stands out. That works for the unusual, and misses the specifically suspicious:
a company whose revenue jumps 60% while its operating cash flow does not move is
not a statistical outlier if several companies in the file did the same, yet it
is the first thing a forensic accountant looks for.

So these five rules encode patterns that are documented indicators rather than
learned ones. Each compares a company-year against that company's own previous
year:

1. **Revenue growth unsupported by cash** - sales up sharply while operating
   cash flow stands still. The classic sign of revenue recognised before it is
   earned, and the basis of the sales and receivables indices in the Beneish
   M-score.
2. **Profit without cash** - positive net income while operating cash flow is
   negative. The accrual divergence that Dechow's F-score is built around.
3. **Gross margin collapse** - margin falling to under 40% of last year's, from
   a healthy base. The Beneish gross margin index.
4. **Leverage explosion** - debt to equity more than trebling in one year.
5. **Equity erosion** - shareholders' equity falling by two thirds or more.

**An honest note on measurement.** These rules were written on 12 September 2026
while looking at the list of anomaly types our benchmark plants, and they target
those same patterns. The benchmark is therefore *not* an independent test of
them, and the F1 it reports for these rules should be read as illustrative. The
figure that is independent is the false-positive rate: across 960 company-years
of clean and defective dummy data, these rules fire **zero** times, and on the
real Kaggle file they fire on 3.7% of rows.

They deliberately stay quiet. A red flag that fires often is not a red flag.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from finsight.core.models import AnomalyFinding, AnomalyType, Severity

#: Revenue growth beyond this, with flat cash, is the first indicator.
REVENUE_SURGE = 1.5
#: Operating cash flow counts as "not following" below this multiple of last year.
CASH_FOLLOWS = 1.1
#: A margin is compared only from a healthy base; below this there is nothing to collapse.
HEALTHY_MARGIN = 0.15
#: Margin counts as collapsed below this share of last year's.
MARGIN_COLLAPSE = 0.4
#: Debt to equity multiplying by more than this in one year.
LEVERAGE_SPIKE = 3.0
#: Equity falling below this share of last year's.
EQUITY_EROSION = 0.35


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _row(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return {k: v for k, v in vars(record).items() if not k.startswith("_")}


def _money(value: Optional[float]) -> str:
    if value is None:
        return "not reported"
    return f"{value:,.0f}"


def _finding(row: Dict[str, Any], kind: AnomalyType, severity: Severity,
             explanation: str, recommendation: str,
             values: Dict[str, Any]) -> AnomalyFinding:
    return AnomalyFinding(
        company=row.get("company"),
        year=row.get("year"),
        anomaly_type=kind,
        score=1.0,
        severity=severity,
        confidence=0.95,
        explanation=explanation,
        recommendation=recommendation,
        model_name="ForensicRedFlags",
        model_mode="RULE_BASED",
        actual_values=values,
    )


def _by_company(records: Sequence[Any]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for record in records:
        row = _row(record)
        company, year = row.get("company"), _number(row.get("year"))
        if not company or year is None:
            continue
        grouped.setdefault(company, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: _number(r.get("year")) or 0)
    return grouped


# ---------------------------------------------------------------------------
# The rules. Each returns a finding or None, given this year and last year.
# ---------------------------------------------------------------------------


def revenue_without_cash(row: Dict[str, Any], prior: Dict[str, Any]) -> Optional[AnomalyFinding]:
    revenue, before = _number(row.get("revenue")), _number(prior.get("revenue"))
    cash, cash_before = _number(row.get("cash_flow_operating")), _number(prior.get("cash_flow_operating"))
    if None in (revenue, before, cash, cash_before) or before <= 0 or cash_before <= 0:
        return None
    growth = revenue / before
    if growth <= REVENUE_SURGE or cash > cash_before * CASH_FOLLOWS:
        return None
    return _finding(
        row, AnomalyType.REVENUE_PROFIT_MISMATCH, Severity.HIGH,
        f"Revenue rose {(growth - 1) * 100:.0f}% to {_money(revenue)} while operating cash flow "
        f"stayed at {_money(cash)} against {_money(cash_before)}. Sales that do not turn into cash "
        f"are the first thing to check when revenue is recognised early.",
        "Reconcile the revenue increase to cash collected and to the movement in receivables.",
        {"revenue": revenue, "revenue_last_year": before,
         "operating_cash_flow": cash, "operating_cash_flow_last_year": cash_before},
    )


def profit_without_cash(row: Dict[str, Any], prior: Dict[str, Any]) -> Optional[AnomalyFinding]:
    income, cash = _number(row.get("net_income")), _number(row.get("cash_flow_operating"))
    if income is None or cash is None or income <= 0 or cash >= 0:
        return None
    return _finding(
        row, AnomalyType.CASH_FLOW_PROFIT_DIVERGENCE, Severity.HIGH,
        f"A profit of {_money(income)} was reported while operating cash flow was "
        f"{_money(cash)}. Earnings and cash moving in opposite directions is the accrual "
        f"divergence that forensic reviews look for.",
        "Examine the accruals bridging profit to cash, and the timing of recognition.",
        {"net_income": income, "operating_cash_flow": cash},
    )


def margin_collapse(row: Dict[str, Any], prior: Dict[str, Any]) -> Optional[AnomalyFinding]:
    profit, revenue = _number(row.get("gross_profit")), _number(row.get("revenue"))
    before_profit, before_revenue = _number(prior.get("gross_profit")), _number(prior.get("revenue"))
    if None in (profit, revenue, before_profit, before_revenue) or revenue <= 0 or before_revenue <= 0:
        return None
    margin, before_margin = profit / revenue, before_profit / before_revenue
    if before_margin <= HEALTHY_MARGIN or margin >= before_margin * MARGIN_COLLAPSE:
        return None
    return _finding(
        row, AnomalyType.MARGIN_ANOMALY, Severity.HIGH,
        f"Gross margin fell from {before_margin * 100:.1f}% to {margin * 100:.1f}% in one year. "
        f"A fall of this size usually reflects either a change in what is counted as cost of "
        f"revenue, or a real deterioration that should be explained.",
        "Confirm the cost of revenue basis is unchanged, then obtain an explanation for the fall.",
        {"gross_margin": round(margin, 4), "gross_margin_last_year": round(before_margin, 4),
         "gross_profit": profit, "revenue": revenue},
    )


def leverage_explosion(row: Dict[str, Any], prior: Dict[str, Any]) -> Optional[AnomalyFinding]:
    debt, equity = _number(row.get("debt")), _number(row.get("shareholder_equity"))
    before_debt, before_equity = _number(prior.get("debt")), _number(prior.get("shareholder_equity"))
    if None in (debt, equity, before_debt, before_equity) or equity <= 0 or before_equity <= 0:
        return None
    if before_debt <= 0:
        return None
    ratio, before_ratio = debt / equity, before_debt / before_equity
    if before_ratio <= 0 or ratio <= before_ratio * LEVERAGE_SPIKE:
        return None
    return _finding(
        row, AnomalyType.LEVERAGE_ANOMALY, Severity.HIGH,
        f"Debt to equity moved from {before_ratio:.2f} to {ratio:.2f}, more than trebling in a "
        f"single year. A change of that size alters the risk of the business and is rarely "
        f"routine.",
        "Confirm the new borrowing, its terms, and whether covenants are disclosed.",
        {"debt_to_equity": round(ratio, 4), "debt_to_equity_last_year": round(before_ratio, 4),
         "debt": debt, "shareholder_equity": equity},
    )


def equity_erosion(row: Dict[str, Any], prior: Dict[str, Any]) -> Optional[AnomalyFinding]:
    equity, before = _number(row.get("shareholder_equity")), _number(prior.get("shareholder_equity"))
    if equity is None or before is None or before <= 0 or equity >= before * EQUITY_EROSION:
        return None
    return _finding(
        row, AnomalyType.SUDDEN_FINANCIAL_CHANGE, Severity.HIGH,
        f"Shareholders' equity fell from {_money(before)} to {_money(equity)}, a loss of "
        f"{(1 - equity / before) * 100:.0f}% of the capital base in one year.",
        "Trace the movement through the statement of changes in equity.",
        {"shareholder_equity": equity, "shareholder_equity_last_year": before},
    )


RULES = (revenue_without_cash, profit_without_cash, margin_collapse,
         leverage_explosion, equity_erosion)


def detect_forensic_flags(records: Sequence[Any]) -> List[AnomalyFinding]:
    """Red-flag findings for a submission.

    Needs at least two years for a company before any rule can fire, since every
    one of them is a comparison against that company's own previous year.
    """
    findings: List[AnomalyFinding] = []
    for rows in _by_company(records).values():
        for prior, row in zip(rows, rows[1:]):
            for rule in RULES:
                try:
                    found = rule(row, prior)
                except Exception:  # noqa: BLE001 - one bad row must not stop the rest
                    continue
                if found is not None:
                    findings.append(found)
    return findings
