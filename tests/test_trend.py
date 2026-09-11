"""Focused integration tests for the in-memory Trend Agent entry point (analysis.trend)."""

import json
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from analysis.data_mapping import DataMapper
from analysis.deviation_analysis import DeviationRecord
from analysis.forecasting import EvaluationResult, ForecastResult
from analysis.ratio_analysis import RatioResult
from analysis.trend import run_trend_analysis
from analysis.yoy_analysis import YoYResult


class SampleRecord:
    """Sample record object accessing fields via attributes."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_synthetic_multiyear_df(num_years: int = 6, company: str = "SYNTH_CORP") -> pd.DataFrame:
    """Helper creating multi-year test data with component and ratio fields."""
    rows = []
    base_year = 2018
    for i in range(num_years):
        yr = base_year + i
        rev = 1000.0 + i * 100.0
        cor = 600.0 + i * 50.0
        gp = rev - cor
        ni = 100.0 + i * 20.0
        rows.append({
            "year": yr,
            "company": company,
            "revenue": rev,
            "cost_of_revenue": cor,
            "gross_profit": gp,
            "net_income": ni,
            "operating_expenses": 200.0,
            "shareholder_equity": 500.0 + i * 30.0,
            "total_assets": 1200.0 + i * 50.0,
            "total_liabilities": 700.0 + i * 20.0,
            "current_assets": 400.0 + i * 20.0,
            "current_liabilities": 200.0 + i * 10.0,
            "roe": (ni / (500.0 + i * 30.0)) * 100.0,
            "roa": (ni / (1200.0 + i * 50.0)) * 100.0,
            "roi": (ni / (1200.0 + i * 50.0)) * 100.0,
            "current_ratio": (400.0 + i * 20.0) / (200.0 + i * 10.0),
            "debt_equity_ratio": (700.0 + i * 20.0) / (500.0 + i * 30.0),
            "net_profit_margin": (ni / rev) * 100.0,
            "return_on_tangible_equity": (ni / 400.0) * 100.0,
        })
    return pd.DataFrame(rows)


def test_run_trend_analysis_importable():
    """Verify run_trend_analysis is importable from analysis.trend."""
    from analysis.trend import run_trend_analysis as rta
    assert callable(rta)


def test_exact_five_return_keys_and_list_types():
    """Verify function returns exactly the five required keys and all are lists."""
    df = create_synthetic_multiyear_df(6)
    result = run_trend_analysis(df)

    expected_keys = {"yoy", "ratios", "forecasts", "evaluations", "deviations"}
    assert set(result.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(result[k], list), f"Key {k} is not a list!"


def test_correct_dataclass_result_types():
    """Verify each returned list contains the exact required dataclass instances."""
    df = create_synthetic_multiyear_df(6)
    result = run_trend_analysis(df)

    assert len(result["yoy"]) > 0
    assert all(isinstance(item, YoYResult) for item in result["yoy"])

    assert len(result["ratios"]) > 0
    assert all(isinstance(item, RatioResult) for item in result["ratios"])

    assert len(result["forecasts"]) > 0
    assert all(isinstance(item, ForecastResult) for item in result["forecasts"])

    assert len(result["evaluations"]) > 0
    assert all(isinstance(item, EvaluationResult) for item in result["evaluations"])

    assert len(result["deviations"]) > 0
    assert all(isinstance(item, DeviationRecord) for item in result["deviations"])


def test_record_object_input():
    """Verify run_trend_analysis accepts a list of record objects accessed by attributes."""
    records = [
        SampleRecord(year=2018, company="OBJ_CO", revenue=500.0, cost_of_revenue=300.0, net_income=50.0),
        SampleRecord(year=2019, company="OBJ_CO", revenue=550.0, cost_of_revenue=320.0, net_income=55.0),
        SampleRecord(year=2020, company="OBJ_CO", revenue=600.0, cost_of_revenue=350.0, net_income=60.0),
        SampleRecord(year=2021, company="OBJ_CO", revenue=650.0, cost_of_revenue=380.0, net_income=70.0),
    ]
    result = run_trend_analysis(records)
    assert len(result["forecasts"]) > 0
    assert result["forecasts"][0].company == "OBJ_CO"
    assert len(result["ratios"]) > 0


def test_dataframe_input():
    """Verify run_trend_analysis accepts a pandas DataFrame."""
    df = create_synthetic_multiyear_df(5, company="DF_CO")
    result = run_trend_analysis(df)
    assert len(result["forecasts"]) > 0
    assert result["forecasts"][0].company == "DF_CO"


def test_missing_and_none_fields_safe():
    """Verify records with missing/None fields process safely without crashing."""
    records = [
        {"year": 2020, "company": "SPARSE_CO", "revenue": 100.0, "net_income": None},
        {"year": 2021, "company": "SPARSE_CO", "revenue": 120.0, "gross_profit": 50.0},
        {"year": 2022, "company": "SPARSE_CO", "revenue": 140.0, "total_assets": None},
        {"year": 2023, "company": "SPARSE_CO", "revenue": 160.0},
    ]
    result = run_trend_analysis(records)
    assert len(result["forecasts"]) > 0
    # Revenue is present and should be forecasted
    rev_fc = next(f for f in result["forecasts"] if f.metric == "revenue")
    assert rev_fc.status == "success"
    # Ratios with missing fields should safely evaluate to UNKNOWN
    assert any(r.status == "UNKNOWN" for r in result["ratios"])


def test_no_filesystem_output(tmp_path, monkeypatch):
    """Verify run_trend_analysis is pure in-memory and produces no filesystem side effects."""
    # List files before
    files_before = set(os.listdir(os.getcwd()))

    df = create_synthetic_multiyear_df(6)
    result = run_trend_analysis(df)

    files_after = set(os.listdir(os.getcwd()))
    assert files_before == files_after, "run_trend_analysis created files in the working directory!"


def test_deterministic_repeated_execution():
    """Verify running run_trend_analysis twice produces identical results."""
    df = create_synthetic_multiyear_df(6)
    res1 = run_trend_analysis(df, materiality=0.10)
    res2 = run_trend_analysis(df, materiality=0.10)

    for k in ["yoy", "ratios", "forecasts", "evaluations", "deviations"]:
        list1 = [item.to_dict() for item in res1[k]]
        list2 = [item.to_dict() for item in res2[k]]
        assert list1 == list2, f"Discrepancy detected in {k} across runs!"


def test_three_year_company_not_forecastable():
    """Requirement: < 4 observations => not_forecastable and insufficient_data."""
    df = create_synthetic_multiyear_df(3, company="SHORT_CO")
    result = run_trend_analysis(df)

    # Forecasts must be not_forecastable
    assert len(result["forecasts"]) > 0
    for fc in result["forecasts"]:
        assert fc.status == "not_forecastable"
        assert pd.isna(fc.forecast_value)

    # Evaluations must be insufficient_data
    assert len(result["evaluations"]) > 0
    for ev in result["evaluations"]:
        assert ev.evaluation_status == "insufficient_data"
        assert ev.mae is None


def test_no_evaluation_with_non_none_r2_when_test_points_below_five():
    """Requirement: calculate R² only when test_points >= 5. Otherwise r2 must be None."""
    # 5 total years => test_points = 5 - 3 = 2 (< 5)
    df = create_synthetic_multiyear_df(5, company="FEW_TEST_CO")
    result = run_trend_analysis(df)

    for ev in result["evaluations"]:
        if ev.test_points < 5:
            assert ev.r2 is None, f"R² was not None ({ev.r2}) for test_points={ev.test_points} < 5!"


def test_no_ratio_status_provided():
    """Requirement: No RatioResult may have status 'provided'. Only HEALTHY, WARNING, CRITICAL, UNKNOWN."""
    df = create_synthetic_multiyear_df(5)
    result = run_trend_analysis(df)

    allowed_statuses = {"HEALTHY", "WARNING", "CRITICAL", "UNKNOWN"}
    for r in result["ratios"]:
        assert r.status != "provided", f"Ratio {r.ratio_name} has forbidden status 'provided'!"
        assert r.status in allowed_statuses, f"Ratio {r.ratio_name} has invalid status {r.status}!"


def test_no_metric_named_expenses_or_margin():
    """Requirement: Ensure there is NO canonical metric named 'expenses' or 'margin'."""
    df = create_synthetic_multiyear_df(5)
    # Add legacy names to input to test aggressive normalization
    df["expenses"] = df["cost_of_revenue"]
    df["margin"] = df["net_profit_margin"]

    result = run_trend_analysis(df)

    forbidden = {"expenses", "margin"}
    for k in ["yoy", "forecasts", "evaluations", "deviations"]:
        for item in result[k]:
            assert item.metric not in forbidden, f"Forbidden metric '{item.metric}' in {k}!"

    for r in result["ratios"]:
        assert r.ratio_name not in forbidden


def test_all_result_dataclasses_are_serializable():
    """Verify all result dataclasses can be serialized to JSON without error."""
    df = create_synthetic_multiyear_df(6)
    result = run_trend_analysis(df)

    for k, items in result.items():
        for item in items:
            d = item.to_dict()
            # Clean NaNs for strict JSON compliance
            cleaned = {k2: (None if isinstance(v2, float) and np.isnan(v2) else v2) for k2, v2 in d.items()}
            serialized = json.dumps(cleaned)
            assert len(serialized) > 0


def test_kaggle_dataset_processes_12_companies():
    """Verify Kaggle Financial Statements.csv processes all 12 companies."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "data", "input", "Financial Statements.csv"),
        os.path.join(os.getcwd(), "data", "kaggle_financial_statements.csv"),
        os.path.join(os.getcwd(), "data", "input", "Financial Statements.csv"),
        "data/input/Financial Statements.csv",
    ]
    kaggle_path = next((c for c in candidates if os.path.exists(c)), None)
    assert kaggle_path is not None and os.path.exists(kaggle_path), "Kaggle Financial Statements.csv not found."

    mapper = DataMapper()
    raw_df = mapper.load_dataset(kaggle_path)
    norm_df, report = mapper.map_and_validate(raw_df, source_file="kaggle")

    result = run_trend_analysis(norm_df)
    companies = {fc.company for fc in result["forecasts"]}

    assert len(companies) == 12, f"Expected 12 companies in Kaggle dataset, got {len(companies)}: {companies}"


def test_dummy_dataset_processes_60_companies_and_deviation_rate():
    """Verify dummy dataset processes 60 companies and clean dummy material deviation rate is < 20%."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "data", "input", "dummy_statements_clean.csv"),
        os.path.join(project_root, "data", "dummy_statements_clean.csv"),
        os.path.join(os.getcwd(), "data", "input", "dummy_statements_clean.csv"),
        os.path.join(os.getcwd(), "data", "dummy_statements_clean.csv"),
        os.environ.get("DUMMY_DATASET_PATH", ""),
        os.path.abspath(os.path.join(project_root, "..", "statement-review-agent", "data", "dummy_statements_clean.csv")),
    ]
    dummy_path = next((c for c in candidates if c and os.path.exists(c)), None)
    assert dummy_path is not None and os.path.exists(dummy_path), "dummy_statements_clean.csv not found in project or relative paths."

    mapper = DataMapper()
    raw_df = mapper.load_dataset(dummy_path)
    norm_df, report = mapper.map_and_validate(raw_df, source_file="dummy")

    result = run_trend_analysis(norm_df, materiality=0.10)
    companies = {fc.company for fc in result["forecasts"]}

    assert len(companies) == 60, f"Expected 60 companies in dummy dataset, got {len(companies)}"

    # Check material deviation rate
    deviations = result["deviations"]
    assert len(deviations) > 0

    material_count = sum(1 for d in deviations if d.material_deviation)
    total_count = len(deviations)
    material_rate = material_count / total_count

    print(f"\n[DUMMY DATASET AUDIT] Total Deviations: {total_count}, Material: {material_count}, Rate: {material_rate:.2%}")
    assert material_rate < 0.20, f"Material deviation rate {material_rate:.2%} exceeded 20% limit!"
