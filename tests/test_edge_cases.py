"""Edge cases and end-to-end integration tests for TrendAgent."""

import json
import os
import pandas as pd
import pytest

from analysis.ratio_analysis import RatioAnalyzer
from analysis.trend_agent import TrendAgent


def test_ratio_analyzer_with_available_and_missing():
    analyzer = RatioAnalyzer()
    df = pd.DataFrame([
        {
            "company": "ALPHA",
            "year": 2021,
            "current_ratio": 1.5,
            "debt_to_equity": 0.8,
            "roe": 15.0,
            # ROA, ROI, Margin, ROTE absent
        }
    ])
    results = analyzer.analyze_ratios(df)
    assert len(results) == 7  # 7 supported ratios

    cr = next(r for r in results if r.ratio_name == "Current Ratio")
    assert cr.status == "HEALTHY"
    assert cr.value == 1.5
    assert cr.source == "reported"

    roa = next(r for r in results if r.ratio_name == "ROA")
    assert roa.status == "UNKNOWN"
    assert roa.value is None
    assert roa.source == "reported"

    # Invariant: No ratio result has legacy status "provided"
    for r in results:
        assert r.status in {"HEALTHY", "WARNING", "CRITICAL", "UNKNOWN"}
        assert r.status != "provided"


def test_ratio_analyzer_computed_from_components():
    """Verify that ratios are computed with source='computed' and mathematical formulas when components exist."""
    analyzer = RatioAnalyzer()
    df = pd.DataFrame([
        {
            "company": "TEST_CO",
            "year": 2023,
            "current_assets": 200.0,
            "current_liabilities": 100.0,
            "total_liabilities": 150.0,
            "shareholder_equity": 100.0,
            "net_income": 20.0,
            "total_assets": 500.0,
            "revenue": 400.0,
            "tangible_equity": 80.0,
            "debt": 50.0,
        }
    ])
    results = analyzer.analyze_ratios(df)
    res_dict = {r.ratio_name: r for r in results}

    # 1. Current Ratio = 200 / 100 = 2.0 -> HEALTHY
    cr = res_dict["Current Ratio"]
    assert cr.value == 2.0
    assert cr.source == "computed"
    assert cr.status == "HEALTHY"
    assert cr.formula == "Current Assets / Current Liabilities"

    # 2. Debt/Equity = 150 / 100 = 1.5 -> HEALTHY
    de = res_dict["Debt/Equity Ratio"]
    assert de.value == 1.5
    assert de.source == "computed"
    assert de.status == "HEALTHY"
    assert de.formula == "Total Liabilities / Shareholder Equity"

    # 3. ROE = 20 / 100 * 100 = 20.0% -> HEALTHY
    roe = res_dict["ROE"]
    assert roe.value == 20.0
    assert roe.source == "computed"
    assert roe.status == "HEALTHY"
    assert roe.formula == "Net Income / Shareholder Equity * 100"

    # 4. ROA = 20 / 500 * 100 = 4.0% -> WARNING (between 1.0 and 5.0)
    roa = res_dict["ROA"]
    assert roa.value == 4.0
    assert roa.source == "computed"
    assert roa.status == "WARNING"
    assert roa.formula == "Net Income / Total Assets * 100"

    # 5. ROI = 20 / (100 + 50) * 100 = 13.3333% -> HEALTHY (8.0 or more)
    roi = res_dict["ROI"]
    assert roi.value == 13.3333
    assert roi.source == "computed"
    assert roi.status == "HEALTHY"
    assert roi.formula == "Net Income / (Shareholder Equity + Debt) * 100"

    # 6. Net Profit Margin = 20 / 400 * 100 = 5.0% -> WARNING (between 0.0 and 10.0)
    margin = res_dict["Net Profit Margin"]
    assert margin.value == 5.0
    assert margin.source == "computed"
    assert margin.status == "WARNING"
    assert margin.formula == "Net Income / Revenue * 100"

    # 7. ROTE = 20 / 80 * 100 = 25.0% -> HEALTHY
    rote = res_dict["Return on Tangible Equity"]
    assert rote.value == 25.0
    assert rote.source == "computed"
    assert rote.status == "HEALTHY"
    assert rote.formula == "Net Income / Tangible Equity * 100"


def test_ratio_analyzer_zero_denominator_safe():
    """Verify safe handling of zero denominators: value=None and status=UNKNOWN."""
    analyzer = RatioAnalyzer()
    df = pd.DataFrame([
        {
            "company": "ZERO_DIV",
            "year": 2023,
            "current_assets": 100.0,
            "current_liabilities": 0.0,
            "total_liabilities": 100.0,
            "shareholder_equity": 0.0,
            "net_income": 10.0,
            "total_assets": 0.0,
            "revenue": 0.0,
        }
    ])
    results = analyzer.analyze_ratios(df)
    for r in results:
        assert r.value is None
        assert r.status == "UNKNOWN"
        assert r.status != "provided"


def test_ratio_analyzer_prefers_computation_over_reported():
    """Requirement: When component fields are available, COMPUTE ratio instead of copying reported."""
    analyzer = RatioAnalyzer()
    df = pd.DataFrame([
        {
            "company": "TEST_CO",
            "year": 2023,
            "current_assets": 300.0,
            "current_liabilities": 100.0,
            "current_ratio": 1.1,  # Conflicting reported ratio
        }
    ])
    results = analyzer.analyze_ratios(df)
    cr = next(r for r in results if r.ratio_name == "Current Ratio")
    assert cr.value == 3.0  # Computed 300 / 100 = 3.0, not reported 1.1
    assert cr.source == "computed"
    assert cr.status == "HEALTHY"


def test_ratio_analyzer_critical_health_bands():
    """Verify CRITICAL health band triggers for distressed ratios."""
    analyzer = RatioAnalyzer()
    df = pd.DataFrame([
        {
            "company": "DISTRESSED",
            "year": 2023,
            "current_assets": 50.0,
            "current_liabilities": 100.0,   # CR = 0.5 < 1.0 -> CRITICAL
            "total_liabilities": 350.0,
            "shareholder_equity": 100.0,   # D/E = 3.5 > 2.5 -> CRITICAL
            "net_income": -20.0,
            "total_assets": 1000.0,        # ROA = -2.0% < 1.0 -> CRITICAL
            "revenue": 200.0,              # Margin = -10.0% < 0 -> CRITICAL
        }
    ])
    results = analyzer.analyze_ratios(df)
    res_dict = {r.ratio_name: r for r in results}

    assert res_dict["Current Ratio"].status == "CRITICAL"
    assert res_dict["Debt/Equity Ratio"].status == "CRITICAL"
    assert res_dict["ROA"].status == "CRITICAL"
    assert res_dict["Net Profit Margin"].status == "CRITICAL"



def test_trend_agent_end_to_end_synthetic(tmp_path, synthetic_standard_df):
    """Test full TrendAgent pipeline on synthetic standard data."""
    csv_path = os.path.join(tmp_path, "synthetic_input.csv")
    synthetic_standard_df.to_csv(csv_path, index=False)

    out_dir = os.path.join(tmp_path, "output")
    agent = TrendAgent(materiality_threshold=0.10, output_dir=out_dir)
    summary = agent.run(input_path=csv_path, generate_charts=True)

    assert summary["status"] == "success"
    assert summary["companies_count"] == 2
    assert "COMP_A" in summary["companies_processed"]
    assert "COMP_B" in summary["companies_processed"]

    # Verify all output files were created
    assert os.path.exists(summary["output_files"]["forecasts_csv"])
    assert os.path.exists(summary["output_files"]["forecast_evaluation_csv"])
    assert os.path.exists(summary["output_files"]["forecast_deviations_csv"])
    assert os.path.exists(summary["output_files"]["yoy_results_csv"])
    assert os.path.exists(summary["output_files"]["financial_ratios_csv"])
    assert os.path.exists(summary["output_files"]["data_mapping_report_json"])

    # Inspect forecasts.csv
    forecasts_df = pd.read_csv(summary["output_files"]["forecasts_csv"])
    assert not forecasts_df.empty
    # COMP_A last actual was 2022, forecast_year should be 2023
    comp_a_fc = forecasts_df[(forecasts_df["company"] == "COMP_A") & (forecasts_df["metric"] == "revenue")].iloc[0]
    assert comp_a_fc["forecast_year"] == 2023
    assert comp_a_fc["last_actual_year"] == 2022


def test_trend_agent_alternative_schema_end_to_end(tmp_path, synthetic_alternative_schema_df):
    """Verify that a second, completely different dataset schema executes cleanly end-to-end."""
    csv_path = os.path.join(tmp_path, "alt_schema_input.csv")
    synthetic_alternative_schema_df.to_csv(csv_path, index=False)

    out_dir = os.path.join(tmp_path, "alt_output")
    agent = TrendAgent(materiality_threshold=0.15, output_dir=out_dir)
    summary = agent.run(input_path=csv_path, generate_charts=False)

    assert summary["status"] == "success"
    assert summary["companies_count"] == 1
    assert "ALPHA" in summary["companies_processed"]

    # Read data mapping report
    with open(summary["output_files"]["data_mapping_report_json"], "r") as f:
        m_report = json.load(f)
    assert m_report["mapped_fields"]["revenue"] == "Sales"
    assert m_report["mapped_fields"]["net_income"] == "Net Profit"
    assert m_report["mapped_fields"]["company"] == "Company Name"
    assert m_report["mapped_fields"]["year"] == "Fiscal Year"


def test_trend_agent_excel_end_to_end(tmp_path, synthetic_standard_df):
    """Verify that TrendAgent can ingest and process an Excel (.xlsx) file."""
    xlsx_path = os.path.join(tmp_path, "financial_data.xlsx")
    synthetic_standard_df.to_excel(xlsx_path, index=False)

    out_dir = os.path.join(tmp_path, "xlsx_output")
    agent = TrendAgent(materiality_threshold=0.10, output_dir=out_dir)
    summary = agent.run(input_path=xlsx_path, generate_charts=False)

    assert summary["companies_count"] == 2
    assert os.path.exists(summary["output_files"]["forecasts_csv"])


def test_trend_agent_alternative_schema_forecasting_pipeline(tmp_path, synthetic_alternative_schema_df):
    """Integration test verifying that the full TrendAgent pipeline ingests alternative column aliases,
    normalizes them, runs revenue and margin forecasting, and outputs results for ALPHA."""
    csv_path = os.path.join(tmp_path, "alt_pipeline_input.csv")
    synthetic_alternative_schema_df.to_csv(csv_path, index=False)

    out_dir = os.path.join(tmp_path, "alt_pipeline_output")
    agent = TrendAgent(materiality_threshold=0.10, output_dir=out_dir)
    summary = agent.run(input_path=csv_path, generate_charts=False)

    # 1. Pipeline execution status
    assert summary["status"] == "success"
    assert "ALPHA" in summary["companies_processed"]

    # 2. Verify alternative column names normalized correctly
    with open(summary["output_files"]["data_mapping_report_json"], "r") as f:
        mapping_report = json.load(f)

    assert mapping_report["mapped_fields"]["company"] == "Company Name"
    assert mapping_report["mapped_fields"]["year"] == "Fiscal Year"
    assert mapping_report["mapped_fields"]["revenue"] == "Sales"
    assert mapping_report["mapped_fields"]["net_income"] == "Net Profit"
    assert mapping_report["mapped_fields"]["margin"] == "Profit Margin"

    # 3. Verify revenue and margin forecasts were generated for ALPHA
    forecasts_df = pd.read_csv(summary["output_files"]["forecasts_csv"])
    assert not forecasts_df.empty

    alpha_forecasts = forecasts_df[forecasts_df["company"] == "ALPHA"]
    assert not alpha_forecasts.empty

    # Revenue forecast verification
    alpha_rev = alpha_forecasts[alpha_forecasts["metric"] == "revenue"].iloc[0]
    assert alpha_rev["forecast_year"] == 2023
    assert alpha_rev["last_actual_year"] == 2022
    assert alpha_rev["last_actual_value"] == 1400.0
    assert pd.notna(alpha_rev["forecast_value"])
    assert alpha_rev["status"] == "success"

    # Margin forecast verification
    alpha_margin = alpha_forecasts[alpha_forecasts["metric"] == "net_profit_margin"].iloc[0]
    assert alpha_margin["forecast_year"] == 2023
    assert alpha_margin["last_actual_year"] == 2022
    assert alpha_margin["status"] == "success"
    assert pd.notna(alpha_margin["forecast_value"])

