"""
validation_agent/accounting.py

Core Accounting and Ratio Verification Engine.
Implements the 5 fundamental accounting reconciliation checks and 2 ratio recomputations
as specified by standard accounting frameworks and the orchestrator interface.
"""

from typing import List, Any, Optional, Union
import numpy as np
import pandas as pd

from validation_agent.models import ValidationResult
from validation_agent.rules import (
    RULE_BS_01,
    RULE_GP_02,
    RULE_OP_03,
    RULE_NI_04,
    RULE_CF_05,
    RULE_RATIO_ROE,
    RULE_RATIO_NPM,
    RULE_NAMES,
    RULE_FORMULAS,
    SEVERITY_NONE,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_SKIPPED
)


def _format_num(val: Optional[float]) -> str:
    """Format numeric values for human-readable evidence strings."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return "N/A"
        if f == int(f) and abs(f) < 1e12:
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (ValueError, TypeError):
        return str(val)


class RecordAccessor:
    """
    Adapter to uniformly extract fields by attribute or key from either
    a DataFrame row, a dictionary, or a custom class instance.
    """

    def __init__(self, record: Any):
        self._record = record

    def get(self, field: str) -> Any:
        # 1. Attribute access (record.field)
        if hasattr(self._record, field):
            val = getattr(self._record, field)
        # 2. Dict-like access (record[field])
        elif isinstance(self._record, dict) and field in self._record:
            val = self._record[field]
        # 3. Pandas Series access
        elif isinstance(self._record, pd.Series) and field in self._record.index:
            val = self._record[field]
        else:
            val = None

        if val is None or pd.isna(val):
            return None
        return val

    def get_float(self, field: str) -> Optional[float]:
        val = self.get(field)
        if val is None:
            return None
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def get_int(self, field: str) -> Optional[int]:
        val = self.get(field)
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def get_str(self, field: str) -> Optional[str]:
        val = self.get(field)
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None


def _compute_accounting_fail_severity(
    difference: float,
    revenue: Optional[float],
    materiality: float
) -> str:
    """
    Determine severity for accounting check failures:
    - revenue missing or zero or negative -> MEDIUM
    - |difference| >= 5 * materiality * revenue -> CRITICAL
    - |difference| >= materiality * revenue -> HIGH
    - |difference| < materiality * revenue -> LOW
    """
    if revenue is None or revenue <= 0:
        return SEVERITY_MEDIUM

    diff = abs(difference)
    crit_threshold = 5.0 * materiality * revenue
    high_threshold = materiality * revenue

    if diff >= crit_threshold:
        return SEVERITY_CRITICAL
    elif diff >= high_threshold:
        return SEVERITY_HIGH
    else:
        return SEVERITY_LOW


def _compute_ratio_fail_severity(difference: float) -> str:
    """
    Determine severity for ratio recomputation check failures:
    - under 1 percentage point -> LOW
    - 1 to 5 percentage points -> MEDIUM
    - over 5 percentage points -> HIGH
    """
    diff = abs(difference)
    if diff < 1.0:
        return SEVERITY_LOW
    elif diff <= 5.0:
        return SEVERITY_MEDIUM
    else:
        return SEVERITY_HIGH


def run_all_validations(
    records: Union[pd.DataFrame, List[Any]],
    materiality: float = 0.05,
    tolerance: float = 0.01
) -> List[ValidationResult]:
    """
    Execute all core accounting and ratio recomputation checks across financial records.

    Args:
        records: A pandas DataFrame or a list of record objects (accessed via attribute or dict).
        materiality: Materiality threshold expressed as a fraction of revenue (default 0.05 = 5%).
        tolerance: Numerical tolerance for formula equality (default 0.01 = 0.01 units/points).

    Returns:
        List of ValidationResult objects for all checks across all records.
    """
    results: List[ValidationResult] = []

    # Standardize input iteration
    if isinstance(records, pd.DataFrame):
        row_accessors = [RecordAccessor(row) for _, row in records.iterrows()]
    elif isinstance(records, list):
        row_accessors = [RecordAccessor(r) for r in records]
    else:
        try:
            row_accessors = [RecordAccessor(r) for r in list(records)]
        except Exception:
            row_accessors = []

    for acc in row_accessors:
        company = acc.get_str("company") or "UNKNOWN"
        year = acc.get_int("year") or 0
        revenue = acc.get_float("revenue")

        # ---------------------------------------------------------------------
        # 1. VAL_BS_01: Balance Sheet
        #    total_assets = total_liabilities + shareholder_equity
        # ---------------------------------------------------------------------
        total_assets = acc.get_float("total_assets")
        total_liabilities = acc.get_float("total_liabilities")
        shareholder_equity = acc.get_float("shareholder_equity")

        bs_missing = []
        if total_assets is None:
            bs_missing.append("total_assets")
        if total_liabilities is None:
            bs_missing.append("total_liabilities")
        if shareholder_equity is None:
            bs_missing.append("shareholder_equity")

        if bs_missing:
            results.append(ValidationResult(
                rule_id=RULE_BS_01,
                rule_name=RULE_NAMES[RULE_BS_01],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(bs_missing)}",
                formula=RULE_FORMULAS[RULE_BS_01],
                message=f"Skipped {RULE_NAMES[RULE_BS_01]} check due to missing fields: {', '.join(bs_missing)}"
            ))
        else:
            expected_bs = total_liabilities + shareholder_equity
            actual_bs = total_assets
            diff_bs = actual_bs - expected_bs
            evidence_bs = (
                f"FY{year}: Total Assets {_format_num(actual_bs)} vs "
                f"Liabilities {_format_num(total_liabilities)} + Equity {_format_num(shareholder_equity)} = {_format_num(expected_bs)}"
            )

            if abs(diff_bs) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_BS_01,
                    rule_name=RULE_NAMES[RULE_BS_01],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_bs, 4),
                    actual=round(actual_bs, 4),
                    difference=round(diff_bs, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_bs,
                    formula=RULE_FORMULAS[RULE_BS_01],
                    message=f"{RULE_NAMES[RULE_BS_01]} balanced within tolerance"
                ))
            else:
                sev_bs = _compute_accounting_fail_severity(diff_bs, revenue, materiality)
                results.append(ValidationResult(
                    rule_id=RULE_BS_01,
                    rule_name=RULE_NAMES[RULE_BS_01],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_bs, 4),
                    actual=round(actual_bs, 4),
                    difference=round(diff_bs, 4),
                    severity=sev_bs,
                    evidence=evidence_bs,
                    formula=RULE_FORMULAS[RULE_BS_01],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_BS_01]}: "
                        f"reported {_format_num(actual_bs)} vs expected {_format_num(expected_bs)} "
                        f"(difference {_format_num(diff_bs)})"
                    )
                ))

        # ---------------------------------------------------------------------
        # 2. VAL_GP_02: Gross Profit
        #    gross_profit = revenue - cost_of_revenue
        # ---------------------------------------------------------------------
        gross_profit = acc.get_float("gross_profit")
        cost_of_revenue = acc.get_float("cost_of_revenue")

        gp_missing = []
        if gross_profit is None:
            gp_missing.append("gross_profit")
        if revenue is None:
            gp_missing.append("revenue")
        if cost_of_revenue is None:
            gp_missing.append("cost_of_revenue")

        if gp_missing:
            results.append(ValidationResult(
                rule_id=RULE_GP_02,
                rule_name=RULE_NAMES[RULE_GP_02],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(gp_missing)}",
                formula=RULE_FORMULAS[RULE_GP_02],
                message=f"Skipped {RULE_NAMES[RULE_GP_02]} check due to missing fields: {', '.join(gp_missing)}"
            ))
        else:
            expected_gp = revenue - cost_of_revenue
            actual_gp = gross_profit
            diff_gp = actual_gp - expected_gp
            evidence_gp = (
                f"FY{year}: Gross Profit {_format_num(actual_gp)} vs "
                f"Revenue {_format_num(revenue)} - Cost of Revenue {_format_num(cost_of_revenue)} = {_format_num(expected_gp)}"
            )

            if abs(diff_gp) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_GP_02,
                    rule_name=RULE_NAMES[RULE_GP_02],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_gp, 4),
                    actual=round(actual_gp, 4),
                    difference=round(diff_gp, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_gp,
                    formula=RULE_FORMULAS[RULE_GP_02],
                    message=f"{RULE_NAMES[RULE_GP_02]} verified within tolerance"
                ))
            else:
                sev_gp = _compute_accounting_fail_severity(diff_gp, revenue, materiality)
                results.append(ValidationResult(
                    rule_id=RULE_GP_02,
                    rule_name=RULE_NAMES[RULE_GP_02],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_gp, 4),
                    actual=round(actual_gp, 4),
                    difference=round(diff_gp, 4),
                    severity=sev_gp,
                    evidence=evidence_gp,
                    formula=RULE_FORMULAS[RULE_GP_02],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_GP_02]}: "
                        f"reported {_format_num(actual_gp)} vs expected {_format_num(expected_gp)} "
                        f"(difference {_format_num(diff_gp)})"
                    )
                ))

        # ---------------------------------------------------------------------
        # 3. VAL_OP_03: Operating Income
        #    operating_income = gross_profit - operating_expenses
        # ---------------------------------------------------------------------
        operating_income = acc.get_float("operating_income")
        operating_expenses = acc.get_float("operating_expenses")

        op_missing = []
        if operating_income is None:
            op_missing.append("operating_income")
        if gross_profit is None:
            op_missing.append("gross_profit")
        if operating_expenses is None:
            op_missing.append("operating_expenses")

        if op_missing:
            results.append(ValidationResult(
                rule_id=RULE_OP_03,
                rule_name=RULE_NAMES[RULE_OP_03],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(op_missing)}",
                formula=RULE_FORMULAS[RULE_OP_03],
                message=f"Skipped {RULE_NAMES[RULE_OP_03]} check due to missing fields: {', '.join(op_missing)}"
            ))
        else:
            expected_op = gross_profit - operating_expenses
            actual_op = operating_income
            diff_op = actual_op - expected_op
            evidence_op = (
                f"FY{year}: Operating Income {_format_num(actual_op)} vs "
                f"Gross Profit {_format_num(gross_profit)} - Operating Expenses {_format_num(operating_expenses)} = {_format_num(expected_op)}"
            )

            if abs(diff_op) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_OP_03,
                    rule_name=RULE_NAMES[RULE_OP_03],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_op, 4),
                    actual=round(actual_op, 4),
                    difference=round(diff_op, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_op,
                    formula=RULE_FORMULAS[RULE_OP_03],
                    message=f"{RULE_NAMES[RULE_OP_03]} verified within tolerance"
                ))
            else:
                sev_op = _compute_accounting_fail_severity(diff_op, revenue, materiality)
                results.append(ValidationResult(
                    rule_id=RULE_OP_03,
                    rule_name=RULE_NAMES[RULE_OP_03],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_op, 4),
                    actual=round(actual_op, 4),
                    difference=round(diff_op, 4),
                    severity=sev_op,
                    evidence=evidence_op,
                    formula=RULE_FORMULAS[RULE_OP_03],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_OP_03]}: "
                        f"reported {_format_num(actual_op)} vs expected {_format_num(expected_op)} "
                        f"(difference {_format_num(diff_op)})"
                    )
                ))

        # ---------------------------------------------------------------------
        # 4. VAL_NI_04: Net Income
        #    net_income = pre_tax_income - taxes
        # ---------------------------------------------------------------------
        net_income = acc.get_float("net_income")
        pre_tax_income = acc.get_float("pre_tax_income")
        taxes = acc.get_float("taxes")

        ni_missing = []
        if net_income is None:
            ni_missing.append("net_income")
        if pre_tax_income is None:
            ni_missing.append("pre_tax_income")
        if taxes is None:
            ni_missing.append("taxes")

        if ni_missing:
            results.append(ValidationResult(
                rule_id=RULE_NI_04,
                rule_name=RULE_NAMES[RULE_NI_04],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(ni_missing)}",
                formula=RULE_FORMULAS[RULE_NI_04],
                message=f"Skipped {RULE_NAMES[RULE_NI_04]} check due to missing fields: {', '.join(ni_missing)}"
            ))
        else:
            expected_ni = pre_tax_income - taxes
            actual_ni = net_income
            diff_ni = actual_ni - expected_ni
            evidence_ni = (
                f"FY{year}: Net Income {_format_num(actual_ni)} vs "
                f"Pre-tax Income {_format_num(pre_tax_income)} - Taxes {_format_num(taxes)} = {_format_num(expected_ni)}"
            )

            if abs(diff_ni) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_NI_04,
                    rule_name=RULE_NAMES[RULE_NI_04],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_ni, 4),
                    actual=round(actual_ni, 4),
                    difference=round(diff_ni, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_ni,
                    formula=RULE_FORMULAS[RULE_NI_04],
                    message=f"{RULE_NAMES[RULE_NI_04]} verified within tolerance"
                ))
            else:
                sev_ni = _compute_accounting_fail_severity(diff_ni, revenue, materiality)
                results.append(ValidationResult(
                    rule_id=RULE_NI_04,
                    rule_name=RULE_NAMES[RULE_NI_04],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_ni, 4),
                    actual=round(actual_ni, 4),
                    difference=round(diff_ni, 4),
                    severity=sev_ni,
                    evidence=evidence_ni,
                    formula=RULE_FORMULAS[RULE_NI_04],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_NI_04]}: "
                        f"reported {_format_num(actual_ni)} vs expected {_format_num(expected_ni)} "
                        f"(difference {_format_num(diff_ni)})"
                    )
                ))

        # ---------------------------------------------------------------------
        # 5. VAL_CF_05: Cash Flow Reconciliation
        #    ending_cash = beginning_cash + cf_operating + cf_investing + cf_financing
        # ---------------------------------------------------------------------
        beginning_cash = acc.get_float("beginning_cash")
        ending_cash = acc.get_float("ending_cash")
        cash_flow_operating = acc.get_float("cash_flow_operating")
        cash_flow_investing = acc.get_float("cash_flow_investing")
        cash_flow_financing = acc.get_float("cash_flow_financing")

        cf_missing = []
        if ending_cash is None:
            cf_missing.append("ending_cash")
        if beginning_cash is None:
            cf_missing.append("beginning_cash")
        if cash_flow_operating is None:
            cf_missing.append("cash_flow_operating")
        if cash_flow_investing is None:
            cf_missing.append("cash_flow_investing")
        if cash_flow_financing is None:
            cf_missing.append("cash_flow_financing")

        if cf_missing:
            results.append(ValidationResult(
                rule_id=RULE_CF_05,
                rule_name=RULE_NAMES[RULE_CF_05],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(cf_missing)}",
                formula=RULE_FORMULAS[RULE_CF_05],
                message=f"Skipped {RULE_NAMES[RULE_CF_05]} check due to missing fields: {', '.join(cf_missing)}"
            ))
        else:
            expected_cf = (
                beginning_cash + cash_flow_operating + cash_flow_investing + cash_flow_financing
            )
            actual_cf = ending_cash
            diff_cf = actual_cf - expected_cf
            evidence_cf = (
                f"FY{year}: Ending Cash {_format_num(actual_cf)} vs Beginning Cash {_format_num(beginning_cash)} "
                f"+ Operating CF {_format_num(cash_flow_operating)} + Investing CF {_format_num(cash_flow_investing)} "
                f"+ Financing CF {_format_num(cash_flow_financing)} = {_format_num(expected_cf)}"
            )

            if abs(diff_cf) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_CF_05,
                    rule_name=RULE_NAMES[RULE_CF_05],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_cf, 4),
                    actual=round(actual_cf, 4),
                    difference=round(diff_cf, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_cf,
                    formula=RULE_FORMULAS[RULE_CF_05],
                    message=f"{RULE_NAMES[RULE_CF_05]} verified within tolerance"
                ))
            else:
                sev_cf = _compute_accounting_fail_severity(diff_cf, revenue, materiality)
                results.append(ValidationResult(
                    rule_id=RULE_CF_05,
                    rule_name=RULE_NAMES[RULE_CF_05],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_cf, 4),
                    actual=round(actual_cf, 4),
                    difference=round(diff_cf, 4),
                    severity=sev_cf,
                    evidence=evidence_cf,
                    formula=RULE_FORMULAS[RULE_CF_05],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_CF_05]}: "
                        f"reported {_format_num(actual_cf)} vs expected {_format_num(expected_cf)} "
                        f"(difference {_format_num(diff_cf)})"
                    )
                ))

        # ---------------------------------------------------------------------
        # 6. VAL_RATIO_ROE: Return on Equity
        #    roe = net_income / shareholder_equity * 100
        # ---------------------------------------------------------------------
        roe = acc.get_float("roe")

        roe_missing = []
        if roe is None:
            roe_missing.append("roe")
        if net_income is None:
            roe_missing.append("net_income")
        if shareholder_equity is None:
            roe_missing.append("shareholder_equity")

        if roe_missing:
            results.append(ValidationResult(
                rule_id=RULE_RATIO_ROE,
                rule_name=RULE_NAMES[RULE_RATIO_ROE],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(roe_missing)}",
                formula=RULE_FORMULAS[RULE_RATIO_ROE],
                message=f"Skipped {RULE_NAMES[RULE_RATIO_ROE]} check due to missing fields: {', '.join(roe_missing)}"
            ))
        elif shareholder_equity == 0:
            results.append(ValidationResult(
                rule_id=RULE_RATIO_ROE,
                rule_name=RULE_NAMES[RULE_RATIO_ROE],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Denominator shareholder_equity is zero",
                formula=RULE_FORMULAS[RULE_RATIO_ROE],
                message=f"Skipped {RULE_NAMES[RULE_RATIO_ROE]} check because denominator shareholder_equity is zero"
            ))
        else:
            expected_roe = (net_income / shareholder_equity) * 100.0
            actual_roe = roe
            diff_roe = actual_roe - expected_roe
            evidence_roe = (
                f"FY{year}: Reported ROE {_format_num(actual_roe)}% vs "
                f"Net Income {_format_num(net_income)} / Equity {_format_num(shareholder_equity)} * 100 = {_format_num(expected_roe)}%"
            )

            if abs(diff_roe) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_RATIO_ROE,
                    rule_name=RULE_NAMES[RULE_RATIO_ROE],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_roe, 4),
                    actual=round(actual_roe, 4),
                    difference=round(diff_roe, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_roe,
                    formula=RULE_FORMULAS[RULE_RATIO_ROE],
                    message=f"{RULE_NAMES[RULE_RATIO_ROE]} verified within tolerance"
                ))
            else:
                sev_roe = _compute_ratio_fail_severity(diff_roe)
                results.append(ValidationResult(
                    rule_id=RULE_RATIO_ROE,
                    rule_name=RULE_NAMES[RULE_RATIO_ROE],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_roe, 4),
                    actual=round(actual_roe, 4),
                    difference=round(diff_roe, 4),
                    severity=sev_roe,
                    evidence=evidence_roe,
                    formula=RULE_FORMULAS[RULE_RATIO_ROE],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_RATIO_ROE]}: "
                        f"reported {_format_num(actual_roe)}% vs expected {_format_num(expected_roe)}% "
                        f"(difference {_format_num(diff_roe)} percentage points)"
                    )
                ))

        # ---------------------------------------------------------------------
        # 7. VAL_RATIO_NPM: Net Profit Margin
        #    net_profit_margin = net_income / revenue * 100
        # ---------------------------------------------------------------------
        net_profit_margin = acc.get_float("net_profit_margin")

        npm_missing = []
        if net_profit_margin is None:
            npm_missing.append("net_profit_margin")
        if net_income is None:
            npm_missing.append("net_income")
        if revenue is None:
            npm_missing.append("revenue")

        if npm_missing:
            results.append(ValidationResult(
                rule_id=RULE_RATIO_NPM,
                rule_name=RULE_NAMES[RULE_RATIO_NPM],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Missing {', '.join(npm_missing)}",
                formula=RULE_FORMULAS[RULE_RATIO_NPM],
                message=f"Skipped {RULE_NAMES[RULE_RATIO_NPM]} check due to missing fields: {', '.join(npm_missing)}"
            ))
        elif revenue == 0:
            results.append(ValidationResult(
                rule_id=RULE_RATIO_NPM,
                rule_name=RULE_NAMES[RULE_RATIO_NPM],
                company=company,
                year=year,
                status=STATUS_SKIPPED,
                expected=None,
                actual=None,
                difference=None,
                severity=SEVERITY_NONE,
                evidence=f"FY{year}: Denominator revenue is zero",
                formula=RULE_FORMULAS[RULE_RATIO_NPM],
                message=f"Skipped {RULE_NAMES[RULE_RATIO_NPM]} check because denominator revenue is zero"
            ))
        else:
            expected_npm = (net_income / revenue) * 100.0
            actual_npm = net_profit_margin
            diff_npm = actual_npm - expected_npm
            evidence_npm = (
                f"FY{year}: Reported Net Profit Margin {_format_num(actual_npm)}% vs "
                f"Net Income {_format_num(net_income)} / Revenue {_format_num(revenue)} * 100 = {_format_num(expected_npm)}%"
            )

            if abs(diff_npm) <= tolerance:
                results.append(ValidationResult(
                    rule_id=RULE_RATIO_NPM,
                    rule_name=RULE_NAMES[RULE_RATIO_NPM],
                    company=company,
                    year=year,
                    status=STATUS_PASS,
                    expected=round(expected_npm, 4),
                    actual=round(actual_npm, 4),
                    difference=round(diff_npm, 4),
                    severity=SEVERITY_NONE,
                    evidence=evidence_npm,
                    formula=RULE_FORMULAS[RULE_RATIO_NPM],
                    message=f"{RULE_NAMES[RULE_RATIO_NPM]} verified within tolerance"
                ))
            else:
                sev_npm = _compute_ratio_fail_severity(diff_npm)
                results.append(ValidationResult(
                    rule_id=RULE_RATIO_NPM,
                    rule_name=RULE_NAMES[RULE_RATIO_NPM],
                    company=company,
                    year=year,
                    status=STATUS_FAIL,
                    expected=round(expected_npm, 4),
                    actual=round(actual_npm, 4),
                    difference=round(diff_npm, 4),
                    severity=sev_npm,
                    evidence=evidence_npm,
                    formula=RULE_FORMULAS[RULE_RATIO_NPM],
                    message=(
                        f"Potential inconsistency in {RULE_NAMES[RULE_RATIO_NPM]}: "
                        f"reported {_format_num(actual_npm)}% vs expected {_format_num(expected_npm)}% "
                        f"(difference {_format_num(diff_npm)} percentage points)"
                    )
                ))

    return results
