"""
tests/test_validation.py

Comprehensive test suite for the Validation Agent core accounting checks,
ratio recomputations, standardized Ingestion integration, and behavior invariants.
"""

import os
import json
from typing import Any
import pytest
import pandas as pd

from validation_agent import (
    run_all_validations,
    ValidationResult,
    ValidationAgent,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_SKIPPED,
    STATUS_BLOCK,
    SEVERITY_NONE,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CLEAN_CSV = os.path.join(DATA_DIR, "dummy_statements_clean.csv")
DEFECTIVE_CSV = os.path.join(DATA_DIR, "dummy_statements_defective.csv")
LABELS_JSON = os.path.join(DATA_DIR, "dummy_statements_labels.json")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_financial_statements.csv")


def load_and_standardize_csv(filepath: str, is_kaggle: bool = False) -> pd.DataFrame:
    """
    Standardize CSV column names to match the Ingestion agent output format.
    """
    df = pd.read_csv(filepath)
    if is_kaggle:
        df.columns = [c.strip() for c in df.columns]
        if "Company" in df.columns:
            df["Company"] = df["Company"].astype(str).str.strip()

    rename_map = {
        'Year': 'year',
        'Company': 'company',
        'Revenue': 'revenue',
        'Cost of Revenue': 'cost_of_revenue',
        'Gross Profit': 'gross_profit',
        'Operating Expenses': 'operating_expenses',
        'Operating Income': 'operating_income',
        'Pre-tax Income': 'pre_tax_income',
        'Taxes': 'taxes',
        'Net Income': 'net_income',
        'Total Assets': 'total_assets',
        'Total Liabilities': 'total_liabilities',
        'Share Holder Equity': 'shareholder_equity',
        'Beginning Cash': 'beginning_cash',
        'Ending Cash': 'ending_cash',
        'Cash Flow from Operating': 'cash_flow_operating',
        'Cash Flow from Investing': 'cash_flow_investing',
        'Cash Flow from Financial Activities': 'cash_flow_financing',
        'ROE': 'roe',
        'Net Profit Margin': 'net_profit_margin'
    }
    return df.rename(columns=rename_map)


# =========================================================================
# 1. THE ANSWER KEY & GROUND TRUTH TESTS
# =========================================================================

def test_dummy_clean_zero_fails():
    """dummy_statements_clean.csv: zero FAIL results across all five accounting rules."""
    df_clean = load_and_standardize_csv(CLEAN_CSV)
    results = run_all_validations(df_clean, tolerance=0.01)

    accounting_rules = {"VAL_BS_01", "VAL_GP_02", "VAL_OP_03", "VAL_NI_04", "VAL_CF_05"}
    accounting_fails = [
        r for r in results
        if r.rule_id in accounting_rules and r.status == STATUS_FAIL
    ]
    assert len(accounting_fails) == 0

    # Also verify zero ratio fails on clean
    all_fails = [r for r in results if r.status == STATUS_FAIL]
    assert len(all_fails) == 0


def test_dummy_defective_matches_labels_exactly():
    """
    dummy_statements_defective.csv: the set of (company, year, rule_id)
    for every FAIL matches dummy_statements_labels.json exactly.
    All 147 found (recall 1.0) and nothing extra flagged (precision 1.0).
    """
    df_defective = load_and_standardize_csv(DEFECTIVE_CSV)
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = run_all_validations(df_defective, tolerance=0.01)
    detected_fails = {
        (r.company, r.year, r.rule_id)
        for r in results if r.status == STATUS_FAIL
    }
    expected_fails = {
        (item["company"], item["year"], item["rule_id"])
        for item in ground_truth
    }

    assert len(expected_fails) == 147
    assert detected_fails == expected_fails


def test_defective_rule_breakdown():
    """
    The defective file finds each rule's errors:
    34 VAL_BS_01, 37 VAL_GP_02, 25 VAL_OP_03, 25 VAL_NI_04, 26 VAL_CF_05.
    """
    df_defective = load_and_standardize_csv(DEFECTIVE_CSV)
    results = run_all_validations(df_defective, tolerance=0.01)

    fails_by_rule = {}
    for r in results:
        if r.status == STATUS_FAIL:
            fails_by_rule[r.rule_id] = fails_by_rule.get(r.rule_id, 0) + 1

    assert fails_by_rule.get("VAL_BS_01", 0) == 34
    assert fails_by_rule.get("VAL_GP_02", 0) == 37
    assert fails_by_rule.get("VAL_OP_03", 0) == 25
    assert fails_by_rule.get("VAL_NI_04", 0) == 25
    assert fails_by_rule.get("VAL_CF_05", 0) == 26


# =========================================================================
# 2. REAL DATA TESTS (Kaggle Financial Statements)
# =========================================================================

def test_kaggle_all_five_accounting_checks_skipped():
    """
    Kaggle file: all five accounting checks are SKIPPED on every row (161 rows),
    and none are FAIL.
    """
    df_kaggle = load_and_standardize_csv(KAGGLE_CSV, is_kaggle=True)
    results = run_all_validations(df_kaggle, tolerance=0.01)

    accounting_rules = {"VAL_BS_01", "VAL_GP_02", "VAL_OP_03", "VAL_NI_04", "VAL_CF_05"}
    accounting_results = [r for r in results if r.rule_id in accounting_rules]

    # 161 rows * 5 checks = 805 checks
    assert len(accounting_results) == 161 * 5
    assert all(r.status == STATUS_SKIPPED for r in accounting_results)
    assert not any(r.status == STATUS_FAIL for r in accounting_results)


def test_kaggle_ratio_recomputations():
    """
    Kaggle file, tolerance 0.01:
    54 VAL_RATIO_ROE fails and 14 VAL_RATIO_NPM fails.
    """
    df_kaggle = load_and_standardize_csv(KAGGLE_CSV, is_kaggle=True)
    results = run_all_validations(df_kaggle, tolerance=0.01)

    roe_fails = [r for r in results if r.rule_id == "VAL_RATIO_ROE" and r.status == STATUS_FAIL]
    npm_fails = [r for r in results if r.rule_id == "VAL_RATIO_NPM" and r.status == STATUS_FAIL]

    assert len(roe_fails) == 54
    assert len(npm_fails) == 14


def test_kaggle_never_returns_block():
    """The Kaggle file never returns BLOCK."""
    df_kaggle = load_and_standardize_csv(KAGGLE_CSV, is_kaggle=True)
    agent = ValidationAgent()
    res = agent.validate(df_kaggle)

    assert res["status"] != STATUS_BLOCK


# =========================================================================
# 3. BEHAVIORAL & INVARIANT TESTS
# =========================================================================

def test_missing_single_field_skips_only_that_rule():
    """A record missing one input field gives SKIPPED for that rule only."""
    # Complete valid record
    rec = {
        "year": 2023,
        "company": "ACME",
        "revenue": 1000.0,
        "cost_of_revenue": 600.0,
        "gross_profit": 400.0,
        "operating_expenses": 200.0,
        "operating_income": 200.0,
        "pre_tax_income": 180.0,
        "taxes": 45.0,
        "net_income": 135.0,
        "total_assets": 1500.0,
        "total_liabilities": 700.0,
        "shareholder_equity": 800.0,
        "beginning_cash": 100.0,
        "ending_cash": 150.0,
        "cash_flow_operating": 120.0,
        "cash_flow_investing": -40.0,
        "cash_flow_financing": -30.0,
        "roe": (135.0 / 800.0) * 100.0,
        "net_profit_margin": (135.0 / 1000.0) * 100.0
    }

    # Make only cost_of_revenue missing -> skips VAL_GP_02 only
    rec_missing = dict(rec)
    rec_missing["cost_of_revenue"] = None

    results = run_all_validations([rec_missing], tolerance=0.01)
    res_map = {r.rule_id: r for r in results}

    assert res_map["VAL_GP_02"].status == STATUS_SKIPPED
    assert "cost_of_revenue" in res_map["VAL_GP_02"].message
    # All other rules evaluate to PASS
    assert res_map["VAL_BS_01"].status == STATUS_PASS
    assert res_map["VAL_OP_03"].status == STATUS_PASS
    assert res_map["VAL_NI_04"].status == STATUS_PASS
    assert res_map["VAL_CF_05"].status == STATUS_PASS
    assert res_map["VAL_RATIO_ROE"].status == STATUS_PASS
    assert res_map["VAL_RATIO_NPM"].status == STATUS_PASS


def test_pass_and_skipped_have_severity_none():
    """PASS and SKIPPED always have severity NONE."""
    df_defective = load_and_standardize_csv(DEFECTIVE_CSV)
    results = run_all_validations(df_defective, tolerance=0.01)

    for r in results:
        if r.status in (STATUS_PASS, STATUS_SKIPPED):
            assert r.severity == SEVERITY_NONE, f"Rule {r.rule_id} status {r.status} had severity {r.severity}"


def test_materiality_slider_behavior():
    """
    Raising materiality from 0.05 to 0.50 turns some HIGH/CRITICAL results into LOW
    (verifying that the UI slider dynamically re-grades findings).
    """
    df_defective = load_and_standardize_csv(DEFECTIVE_CSV)

    results_05 = run_all_validations(df_defective, materiality=0.05, tolerance=0.01)
    results_50 = run_all_validations(df_defective, materiality=0.50, tolerance=0.01)

    fails_05 = [r for r in results_05 if r.status == STATUS_FAIL and r.rule_id.startswith("VAL_")]
    fails_50 = [r for r in results_50 if r.status == STATUS_FAIL and r.rule_id.startswith("VAL_")]

    low_05_count = sum(1 for r in fails_05 if r.severity == SEVERITY_LOW)
    low_50_count = sum(1 for r in fails_50 if r.severity == SEVERITY_LOW)

    assert low_50_count > low_05_count, "Raising materiality did not re-grade findings to LOW"


def test_standard_dataframe_does_not_block():
    """A DataFrame with the standard column names doesn't BLOCK."""
    df_clean = load_and_standardize_csv(CLEAN_CSV)
    agent = ValidationAgent()
    res = agent.validate(df_clean)

    assert res["status"] != STATUS_BLOCK
    assert res["status"] == STATUS_PASS


def test_list_of_record_objects_matches_dataframe():
    """A list of record objects gives the same results as the DataFrame."""
    df_clean = load_and_standardize_csv(CLEAN_CSV).head(20)

    class StatementRecord:
        """Class holding financial record accessible via attributes."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    records_list = [StatementRecord(**row.to_dict()) for _, row in df_clean.iterrows()]

    results_from_df = run_all_validations(df_clean, tolerance=0.01)
    results_from_objs = run_all_validations(records_list, tolerance=0.01)

    assert len(results_from_df) == len(results_from_objs)

    for r_df, r_obj in zip(results_from_df, results_from_objs):
        assert r_df.rule_id == r_obj.rule_id
        assert r_df.company == r_obj.company
        assert r_df.year == r_obj.year
        assert r_df.status == r_obj.status
        assert r_df.expected == r_obj.expected
        assert r_df.actual == r_obj.actual
        assert r_df.difference == r_obj.difference
        assert r_df.severity == r_obj.severity
        assert r_df.evidence == r_obj.evidence
        assert r_df.message == r_obj.message


def test_result_serializes_to_dict_and_json():
    """Every result serialises to JSON via to_dict()."""
    df_sample = load_and_standardize_csv(CLEAN_CSV).head(5)
    results = run_all_validations(df_sample, tolerance=0.01)

    for r in results:
        d = r.to_dict()
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        assert json_str is not None
        reloaded = json.loads(json_str)
        assert reloaded["rule_id"] == r.rule_id
        assert reloaded["status"] == r.status


def test_zero_denominator_gives_skipped_not_crash():
    """A zero denominator gives SKIPPED, not a crash."""
    rec = {
        "year": 2023,
        "company": "ZERO_DIV",
        "revenue": 0.0,
        "shareholder_equity": 0.0,
        "net_income": 100.0,
        "roe": 15.0,
        "net_profit_margin": 10.0
    }

    results = run_all_validations([rec], tolerance=0.01)
    res_map = {r.rule_id: r for r in results}

    assert res_map["VAL_RATIO_ROE"].status == STATUS_SKIPPED
    assert res_map["VAL_RATIO_ROE"].severity == SEVERITY_NONE
    assert "zero" in res_map["VAL_RATIO_ROE"].message.lower()

    assert res_map["VAL_RATIO_NPM"].status == STATUS_SKIPPED
    assert res_map["VAL_RATIO_NPM"].severity == SEVERITY_NONE
    assert "zero" in res_map["VAL_RATIO_NPM"].message.lower()


def test_neutral_wording_never_fraud():
    """Evidence and messages stay neutral ('potential inconsistency', never 'fraud')."""
    df_defective = load_and_standardize_csv(DEFECTIVE_CSV)
    results = run_all_validations(df_defective, tolerance=0.01)

    for r in results:
        msg_lower = r.message.lower()
        ev_lower = r.evidence.lower()
        assert "fraud" not in msg_lower
        assert "fraud" not in ev_lower
        if r.status == STATUS_FAIL:
            assert "potential inconsistency" in msg_lower
