"""
validation_agent/checks.py

Contains the 5 independent, deterministic data quality and domain sanity validation checks:
1. Schema Validation
2. Data Quality Validation
3. Duplicate Validation
4. Financial Formula Validation (data quality catalog)
5. Domain Sanity Validation

These checks serve as an independent data-quality audit layer and do not replace
or block the core accounting checks.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from validation_agent.rules import (
    COMPANY_COL,
    COMPANY_COL_ALT,
    YEAR_COL,
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    MIN_VALID_YEAR,
    MAX_VALID_YEAR,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_INFO,
    DEFAULT_FORMULA_TOLERANCE,
    NOT_VALIDATABLE_METRICS
)


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name in df (case- and whitespace-insensitive)."""
    if df is None:
        return None
    cols = list(df.columns)
    col_lookup = {str(c).lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in col_lookup:
            return col_lookup[key]
    return None


def _get_company_col(df: pd.DataFrame) -> str:
    """Detect the actual company column name in the DataFrame."""
    found = _find_column(df, [COMPANY_COL, COMPANY_COL_ALT, "Company", "company"])
    return found if found is not None else COMPANY_COL


def _get_year_col(df: pd.DataFrame) -> str:
    """Detect the actual year column name in the DataFrame."""
    found = _find_column(df, [YEAR_COL, "Year", "year"])
    return found if found is not None else YEAR_COL


def _get_row_identifiers(df: pd.DataFrame, idx: Any) -> Tuple[Optional[str], Optional[int]]:
    """Extract Company and Year for a specific row index."""
    company_col = _get_company_col(df)
    year_col = _get_year_col(df)
    company = None
    year = None
    try:
        if company_col in df.columns:
            val = df.at[idx, company_col]
            company = str(val).strip() if pd.notnull(val) else None
        if year_col in df.columns:
            yval = df.at[idx, year_col]
            if pd.notnull(yval):
                try:
                    year = int(float(yval))
                except (ValueError, TypeError):
                    year = None
    except Exception:
        pass
    return company, year


# =========================================================================
# CHECK 1: SCHEMA VALIDATION
# =========================================================================
def validate_schema(
    df: Optional[pd.DataFrame],
    required_columns: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check 1: Verify that all required columns exist in the DataFrame.
    By default, verifies that minimal required columns (year, company, revenue) exist.
    Missing statement fields (e.g. balance sheet line items) do not fail schema.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    findings: List[Dict[str, Any]] = []

    if df is None:
        findings.append({
            "check": "SCHEMA",
            "status": "FAIL",
            "severity": SEVERITY_CRITICAL,
            "field": None,
            "message": "DataFrame is None (missing or failed to load)"
        })
        return {"status": "FAIL", "findings_count": len(findings)}, findings

    if df.empty and len(df.columns) == 0:
        findings.append({
            "check": "SCHEMA",
            "status": "FAIL",
            "severity": SEVERITY_CRITICAL,
            "field": None,
            "message": "Dataset is completely empty with 0 columns"
        })
        return {"status": "FAIL", "findings_count": len(findings)}, findings

    existing_cols_clean = {str(c).lower().strip(): c for c in df.columns}

    for col in required_columns:
        clean_target = col.lower().strip()
        if clean_target not in existing_cols_clean:
            findings.append({
                "check": "SCHEMA",
                "status": "FAIL",
                "severity": SEVERITY_CRITICAL,
                "field": col,
                "message": f"Required column missing: {col}"
            })

    check_status = "PASS" if len(findings) == 0 else "FAIL"
    summary = {
        "status": check_status,
        "findings_count": len(findings),
        "required_columns_count": len(required_columns),
        "present_columns_count": len(df.columns)
    }
    return summary, findings


# =========================================================================
# CHECK 2: DATA QUALITY VALIDATION
# =========================================================================
def validate_data_quality(
    df: Optional[pd.DataFrame],
    numeric_columns: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check 2: Verify cell-level integrity:
    - Missing / NaN / None values in present columns
    - Infinite numbers (+inf, -inf)
    - Invalid types / unparseable numeric representations
    """
    findings: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return {"status": "PASS", "findings_count": 0}, findings

    if numeric_columns is None:
        numeric_columns = NUMERIC_COLUMNS

    # 1. Missing Values check across all columns present
    for col in df.columns:
        null_mask = df[col].isnull()
        if null_mask.any():
            null_indices = df.index[null_mask].tolist()
            for idx in null_indices:
                company, year = _get_row_identifiers(df, idx)
                findings.append({
                    "check": "DATA_QUALITY",
                    "status": "FAIL",
                    "severity": SEVERITY_MEDIUM,
                    "row_index": int(idx),
                    "company": company,
                    "year": year,
                    "field": col,
                    "message": f"Missing value detected in field '{col}'"
                })

    # 2. Check Infinite & Malformed Numeric Values for numeric columns that exist
    numeric_clean = {str(c).lower().strip() for c in numeric_columns}
    cols_to_check = [c for c in df.columns if str(c).lower().strip() in numeric_clean]

    for actual_col in cols_to_check:
        series = df[actual_col]

        for idx, val in series.items():
            if pd.isnull(val):
                continue

            # Check for numeric scalar
            if isinstance(val, (float, int, np.floating, np.integer)):
                if np.isinf(val):
                    company, year = _get_row_identifiers(df, idx)
                    findings.append({
                        "check": "DATA_QUALITY",
                        "status": "FAIL",
                        "severity": SEVERITY_HIGH,
                        "row_index": int(idx),
                        "company": company,
                        "year": year,
                        "field": actual_col,
                        "message": f"Infinite numeric value ({val}) detected in field '{actual_col}'"
                    })
            else:
                try:
                    fval = float(str(val).replace(",", "").strip())
                    if np.isinf(fval):
                        company, year = _get_row_identifiers(df, idx)
                        findings.append({
                            "check": "DATA_QUALITY",
                            "status": "FAIL",
                            "severity": SEVERITY_HIGH,
                            "row_index": int(idx),
                            "company": company,
                            "year": year,
                            "field": actual_col,
                            "message": f"Infinite numeric value ({val}) detected in field '{actual_col}'"
                        })
                except (ValueError, TypeError):
                    company, year = _get_row_identifiers(df, idx)
                    findings.append({
                        "check": "DATA_QUALITY",
                        "status": "FAIL",
                        "severity": SEVERITY_HIGH,
                        "row_index": int(idx),
                        "company": company,
                        "year": year,
                        "field": actual_col,
                        "message": f"Invalid numeric value or malformed type '{val}' in field '{actual_col}'"
                    })

    check_status = "PASS" if len(findings) == 0 else "FAIL"
    summary = {
        "status": check_status,
        "findings_count": len(findings)
    }
    return summary, findings


# =========================================================================
# CHECK 3: DUPLICATE VALIDATION
# =========================================================================
def validate_duplicates(
    df: Optional[pd.DataFrame],
    company_col: Optional[str] = None,
    year_col: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check 3: Verify whether Company + Year combinations are strictly unique.
    Does not flag different companies or different years.
    """
    findings: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return {"status": "PASS", "findings_count": 0}, findings

    if company_col is None:
        company_col = _get_company_col(df)
    if year_col is None:
        year_col = _get_year_col(df)

    if company_col not in df.columns or year_col not in df.columns:
        return {
            "status": "PASS",
            "findings_count": 0,
            "message": "Company or Year column unavailable for duplicate check"
        }, findings

    dup_mask = df.duplicated(subset=[company_col, year_col], keep=False)
    if dup_mask.any():
        dup_indices = df.index[dup_mask].tolist()
        for idx in dup_indices:
            company, year = _get_row_identifiers(df, idx)
            findings.append({
                "check": "DUPLICATE",
                "status": "FAIL",
                "severity": SEVERITY_MEDIUM,
                "row_index": int(idx),
                "company": company,
                "year": year,
                "message": f"Duplicate Company-Year record: Company='{company}', Year={year}"
            })

    check_status = "PASS" if len(findings) == 0 else "FAIL"
    summary = {
        "status": check_status,
        "findings_count": len(findings)
    }
    return summary, findings


# =========================================================================
# CHECK 4: FINANCIAL FORMULA VALIDATION (Data Quality Catalog)
# =========================================================================
def validate_financial_formulas(
    df: Optional[pd.DataFrame],
    tolerance: float = DEFAULT_FORMULA_TOLERANCE,
    include_not_validatable: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check 4: Data-quality check for Net Profit Margin formula recomputation
    and explicit cataloging of NOT_VALIDATABLE metrics once per dataset.
    """
    findings: List[Dict[str, Any]] = []
    not_validatable_findings: List[Dict[str, Any]] = []

    # Catalog non-validatable formulas once per dataset
    if include_not_validatable:
        for metric, reason in NOT_VALIDATABLE_METRICS.items():
            not_validatable_findings.append({
                "check": "FINANCIAL_FORMULA",
                "status": "NOT_VALIDATABLE",
                "severity": SEVERITY_INFO,
                "metric": metric,
                "message": reason
            })

    if df is None or df.empty:
        return {
            "status": "PASS",
            "findings_count": 0,
            "not_validatable_count": len(not_validatable_findings)
        }, not_validatable_findings

    rev_col = _find_column(df, ["Revenue", "revenue"])
    ni_col = _find_column(df, ["Net Income", "net_income"])
    npm_col = _find_column(df, ["Net Profit Margin", "net_profit_margin"])

    if not rev_col or not ni_col or not npm_col:
        return {
            "status": "PASS",
            "findings_count": 0,
            "not_validatable_count": len(not_validatable_findings),
            "message": "Prerequisite columns missing for NPM recalculation"
        }, not_validatable_findings

    for idx, row in df.iterrows():
        rev = row.get(rev_col)
        ni = row.get(ni_col)
        actual_npm = row.get(npm_col)

        try:
            rev_val = float(rev)
            ni_val = float(ni)
            actual_npm_val = float(actual_npm)
        except (ValueError, TypeError):
            continue

        if pd.isnull(rev_val) or pd.isnull(ni_val) or pd.isnull(actual_npm_val):
            continue
        if np.isinf(rev_val) or np.isinf(ni_val) or np.isinf(actual_npm_val):
            continue
        if abs(rev_val) < 1e-9:
            continue

        expected_npm = (ni_val / rev_val) * 100.0
        difference = abs(expected_npm - actual_npm_val)

        if difference > tolerance:
            company, year = _get_row_identifiers(df, idx)
            findings.append({
                "check": "FINANCIAL_FORMULA",
                "status": "FAIL",
                "severity": SEVERITY_HIGH,
                "row_index": int(idx),
                "company": company,
                "year": year,
                "field": npm_col,
                "metric": "Net Profit Margin",
                "formula": "Net Income / Revenue * 100",
                "expected": round(expected_npm, 4),
                "actual": round(actual_npm_val, 4),
                "difference": round(difference, 4),
                "message": (
                    f"Net Profit Margin formula mismatch: expected {expected_npm:.4f}%, "
                    f"got {actual_npm_val:.4f}% (difference {difference:.4f}% > tolerance {tolerance}%)"
                )
            })

    check_status = "PASS" if len(findings) == 0 else "FAIL"
    summary = {
        "status": check_status,
        "findings_count": len(findings),
        "tolerance": tolerance,
        "validated_formulas": ["Net Profit Margin = (Net Income / Revenue) * 100"],
        "not_validatable": list(NOT_VALIDATABLE_METRICS.keys())
    }

    all_findings = findings + not_validatable_findings
    return summary, all_findings


# =========================================================================
# CHECK 5: DOMAIN SANITY VALIDATION
# =========================================================================
def validate_domain_sanity(
    df: Optional[pd.DataFrame],
    company_col: Optional[str] = None,
    year_col: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Check 5: Domain sanity validation based on accounting and physical semantics:
    - Year must be within [1900, 2100].
    - Non-negative fields (Revenue, Number of Employees, Market Cap, Current Ratio) cannot be negative.
    - Explicitly ALLOWS legitimate negative cash flows, net income, etc.
    """
    findings: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return {"status": "PASS", "findings_count": 0}, findings

    if company_col is None:
        company_col = _get_company_col(df)
    if year_col is None:
        year_col = _get_year_col(df)

    non_neg_clean = {str(c).lower().strip() for c in NON_NEGATIVE_COLUMNS}
    cols_to_check = [c for c in df.columns if str(c).lower().strip() in non_neg_clean]

    for idx, row in df.iterrows():
        company, year = _get_row_identifiers(df, idx)

        # 1. Year range sanity
        if year_col in df.columns:
            y_raw = row.get(year_col)
            if pd.notnull(y_raw):
                try:
                    y_val = int(float(y_raw))
                    if y_val < MIN_VALID_YEAR or y_val > MAX_VALID_YEAR:
                        findings.append({
                            "check": "DOMAIN_SANITY",
                            "status": "FAIL",
                            "severity": SEVERITY_HIGH,
                            "row_index": int(idx),
                            "company": company,
                            "year": y_val,
                            "field": year_col,
                            "message": f"Year {y_val} is outside acceptable domain range [{MIN_VALID_YEAR}, {MAX_VALID_YEAR}]"
                        })
                except (ValueError, TypeError):
                    pass

        # 2. Non-negative constraints
        for actual_col in cols_to_check:
            val = row.get(actual_col)
            if pd.notnull(val):
                try:
                    num_val = float(val)
                    if num_val < 0:
                        findings.append({
                            "check": "DOMAIN_SANITY",
                            "status": "FAIL",
                            "severity": SEVERITY_HIGH,
                            "row_index": int(idx),
                            "company": company,
                            "year": year,
                            "field": actual_col,
                            "message": f"{actual_col} cannot be negative in corporate financial domain: {num_val}"
                        })
                except (ValueError, TypeError):
                    pass

    check_status = "PASS" if len(findings) == 0 else "FAIL"
    summary = {
        "status": check_status,
        "findings_count": len(findings)
    }
    return summary, findings
