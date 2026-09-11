"""
validation_agent/validator.py

The core deterministic, model-free Validation Agent module.
Acts as the quality gate before downstream analytical agents (Trend, Anomaly, Evidence).
Provides both the separate data-quality audit report and orchestrator-level accounting validations.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from validation_agent.models import ValidationResult
from validation_agent.rules import (
    REQUIRED_COLUMNS,
    DEFAULT_FORMULA_TOLERANCE,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
    STATUS_PASS,
    STATUS_WARNING,
    STATUS_BLOCK,
    SEVERITY_PENALTIES
)
from validation_agent.checks import (
    validate_schema,
    validate_data_quality,
    validate_duplicates,
    validate_financial_formulas,
    validate_domain_sanity
)
from validation_agent.accounting import run_all_validations


class ValidationAgent:
    """
    Deterministic Financial Statement Validation Agent.

    Provides:
    1. Independent data-quality checks (schema, missing values, duplicates, domain sanity).
    2. Five core accounting reconciliation checks and two ratio recomputations via run_all_validations.

    Returns machine-readable results with PASS/WARNING/BLOCK and explainable confidence scores.
    Never blocks on missing financial statement line items.
    """

    def __init__(
        self,
        tolerance: float = DEFAULT_FORMULA_TOLERANCE,
        materiality: float = 0.05,
        required_columns: Optional[List[str]] = None
    ):
        """
        Initialize the Validation Agent.

        Args:
            tolerance: Numerical tolerance for formula recalculation in percentage points (default 2.5).
            materiality: Materiality threshold fraction of revenue (default 0.05).
            required_columns: Minimal required columns (defaults to ['year', 'company', 'revenue']).
        """
        self.tolerance = tolerance
        self.materiality = materiality
        self.required_columns = required_columns or list(REQUIRED_COLUMNS)

    def validate_accounting(
        self,
        records: Union[pd.DataFrame, List[Any]],
        materiality: Optional[float] = None,
        tolerance: Optional[float] = None
    ) -> List[ValidationResult]:
        """
        Run the 5 core accounting checks and 2 ratio recomputations.
        """
        mat = materiality if materiality is not None else self.materiality
        tol = tolerance if tolerance is not None else 0.01
        return run_all_validations(records, materiality=mat, tolerance=tol)

    def validate(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame across data-quality checks.

        Args:
            df: DataFrame containing financial statements.

        Returns:
            Structured dictionary adhering to the orchestrator JSON schema.
        """
        rows_checked = 0 if (df is None or df.empty) else len(df)
        all_findings: List[Dict[str, Any]] = []

        # 1. Schema Validation (minimal required: year, company, revenue)
        schema_summary, schema_findings = validate_schema(df, self.required_columns)
        all_findings.extend(schema_findings)

        if df is None or df.empty:
            dq_summary = {"status": "FAIL" if df is None else "PASS", "findings_count": 0}
            dup_summary = {"status": "PASS", "findings_count": 0}
            formula_summary = {"status": "PASS", "findings_count": 0}
            sanity_summary = {"status": "PASS", "findings_count": 0}
        else:
            # 2. Data Quality Validation
            dq_summary, dq_findings = validate_data_quality(df)
            all_findings.extend(dq_findings)

            # 3. Duplicate Validation
            dup_summary, dup_findings = validate_duplicates(df)
            all_findings.extend(dup_findings)

            # 4. Financial Formula Validation (data quality catalog)
            formula_summary, formula_findings = validate_financial_formulas(
                df, tolerance=self.tolerance, include_not_validatable=True
            )
            all_findings.extend(formula_findings)

            # 5. Domain Sanity Validation
            sanity_summary, sanity_findings = validate_domain_sanity(df)
            all_findings.extend(sanity_findings)

        # Standardize all findings
        normalized_findings: List[Dict[str, Any]] = []
        for f in all_findings:
            normalized_findings.append({
                "check": f.get("check"),
                "status": f.get("status", "FAIL"),
                "severity": f.get("severity", SEVERITY_MEDIUM),
                "row_index": f.get("row_index"),
                "company": f.get("company"),
                "year": f.get("year"),
                "field": f.get("field"),
                "metric": f.get("metric"),
                "message": f.get("message"),
                "expected": f.get("expected"),
                "actual": f.get("actual"),
                "difference": f.get("difference")
            })

        overall_status = self._compute_status(normalized_findings)
        confidence = self._compute_confidence(normalized_findings, rows_checked)

        result: Dict[str, Any] = {
            "status": overall_status,
            "confidence": confidence,
            "rows_checked": rows_checked,
            "checks": {
                "schema": schema_summary,
                "data_quality": dq_summary,
                "duplicates": dup_summary,
                "financial_formula": formula_summary,
                "domain_sanity": sanity_summary
            },
            "findings": normalized_findings
        }

        return result

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a CSV file directly from disk.
        """
        if not os.path.exists(file_path):
            finding = {
                "check": "SCHEMA",
                "status": "FAIL",
                "severity": SEVERITY_CRITICAL,
                "row_index": None,
                "company": None,
                "year": None,
                "field": None,
                "metric": None,
                "message": f"File not found: {file_path}",
                "expected": None,
                "actual": None,
                "difference": None
            }
            return {
                "status": STATUS_BLOCK,
                "confidence": 0.0,
                "rows_checked": 0,
                "checks": {
                    "schema": {"status": "FAIL", "findings_count": 1, "message": "File not found"},
                    "data_quality": {"status": "SKIPPED", "findings_count": 0},
                    "duplicates": {"status": "SKIPPED", "findings_count": 0},
                    "financial_formula": {"status": "SKIPPED", "findings_count": 0},
                    "domain_sanity": {"status": "SKIPPED", "findings_count": 0}
                },
                "findings": [finding]
            }

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            finding = {
                "check": "SCHEMA",
                "status": "FAIL",
                "severity": SEVERITY_CRITICAL,
                "row_index": None,
                "company": None,
                "year": None,
                "field": None,
                "metric": None,
                "message": f"Failed to parse CSV file ({file_path}): {str(e)}",
                "expected": None,
                "actual": None,
                "difference": None
            }
            return {
                "status": STATUS_BLOCK,
                "confidence": 0.0,
                "rows_checked": 0,
                "checks": {
                    "schema": {"status": "FAIL", "findings_count": 1, "message": "CSV parse failure"},
                    "data_quality": {"status": "SKIPPED", "findings_count": 0},
                    "duplicates": {"status": "SKIPPED", "findings_count": 0},
                    "financial_formula": {"status": "SKIPPED", "findings_count": 0},
                    "domain_sanity": {"status": "SKIPPED", "findings_count": 0}
                },
                "findings": [finding]
            }

        return self.validate(df)

    def _compute_status(self, findings: List[Dict[str, Any]]) -> str:
        """
        Deterministic Status Resolution:
        - PASS: 0 FAIL findings.
        - WARNING: Only medium or low severity issues.
        - BLOCK: Critical schema failures or serious domain sanity / formula breaches.
        """
        fail_findings = [f for f in findings if f.get("status") == "FAIL"]
        if not fail_findings:
            return STATUS_PASS

        severities = {f.get("severity") for f in fail_findings}
        if SEVERITY_CRITICAL in severities or SEVERITY_HIGH in severities:
            return STATUS_BLOCK

        return STATUS_WARNING

    def _compute_confidence(self, findings: List[Dict[str, Any]], rows_checked: int) -> float:
        """
        Deterministic confidence score from 0.0 to 100.0.
        """
        fail_findings = [f for f in findings if f.get("status") == "FAIL"]
        if not fail_findings:
            return 100.0

        penalty = 0.0
        for f in fail_findings:
            sev = f.get("severity", SEVERITY_MEDIUM)
            penalty += SEVERITY_PENALTIES.get(sev, 3.0)

        score = max(0.0, min(100.0, 100.0 - penalty))
        return round(score, 2)
