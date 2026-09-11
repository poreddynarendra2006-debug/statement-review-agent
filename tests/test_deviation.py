"""Unit tests for Actual vs Forecast deviation analysis and dual-condition materiality."""

import pytest
from analysis.deviation_analysis import DeviationAnalyzer
from analysis.forecasting import BacktestPrediction


def test_deviation_above_forecast():
    """Verify material deviation above forecast when both percent and 2*MAE conditions are met."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=120.0,
        forecast=100.0,
    )
    # dev = 20.0 (20%), walk_forward_mae = 5.0 -> 2 * MAE = 10.0 < 20.0 error
    res = analyzer.analyze_prediction(pred, walk_forward_mae=5.0)

    assert res.deviation == 20.0
    assert res.deviation_percent == 20.0
    assert res.absolute_deviation == 20.0
    assert res.absolute_deviation_percent == 20.0
    assert res.actual_to_forecast_ratio == 1.2
    assert res.material_deviation is True
    assert res.direction == "above_forecast"
    assert res.materiality_threshold == 0.10


def test_deviation_below_forecast():
    """Verify material deviation below forecast when both percent and 2*MAE conditions are met."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=80.0,
        forecast=100.0,
    )
    # dev = -20.0 (-20%), walk_forward_mae = 5.0 -> 2 * MAE = 10.0 < 20.0 error
    res = analyzer.analyze_prediction(pred, walk_forward_mae=5.0)

    assert res.deviation == -20.0
    assert res.deviation_percent == -20.0
    assert res.absolute_deviation == 20.0
    assert res.absolute_deviation_percent == 20.0
    assert res.actual_to_forecast_ratio == 0.8
    assert res.material_deviation is True
    assert res.direction == "below_forecast"


def test_deviation_within_expected_range():
    """Verify non-material deviation when percentage deviation is below threshold."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=105.0,
        forecast=100.0,
    )
    res = analyzer.analyze_prediction(pred, walk_forward_mae=2.0)

    assert res.deviation == 5.0
    assert res.deviation_percent == 5.0
    assert res.material_deviation is False
    assert res.direction == "within_expected_range"


def test_deviation_exceeds_threshold_but_within_mae_band():
    """Verify that deviation percentage exceeding threshold is NOT material if within 2 * MAE."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=115.0,
        forecast=100.0,
    )
    # dev = 15.0 (15% > 10% threshold)
    # but walk_forward_mae = 10.0 -> 2 * MAE = 20.0 > 15.0 error!
    res = analyzer.analyze_prediction(pred, walk_forward_mae=10.0)

    assert res.deviation == 15.0
    assert res.deviation_percent == 15.0
    assert res.material_deviation is False
    assert res.direction == "within_expected_range"


def test_missing_walk_forward_mae_safe():
    """Verify that when walk-forward MAE is None, system handles it safely without crashing."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=150.0,
        forecast=100.0,
    )
    res = analyzer.analyze_prediction(pred, walk_forward_mae=None)

    assert res.material_deviation is False
    assert res.direction == "within_expected_range"
    assert res.notes is not None
    assert "unavailable" in res.notes


def test_zero_forecast_handling():
    """Verify safe zero division when forecast is 0.0."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=50.0,
        forecast=0.0,
    )
    res = analyzer.analyze_prediction(pred, walk_forward_mae=5.0)

    assert res.forecast == 0.0
    assert res.deviation == 50.0
    assert res.deviation_percent is None
    assert res.actual_to_forecast_ratio is None
    assert res.material_deviation is False
    assert res.direction == "within_expected_range"
    assert res.notes is not None
    assert "undefined" in res.notes


def test_configurable_materiality_threshold():
    """Verify that materiality threshold can be modified dynamically."""
    analyzer_25 = DeviationAnalyzer(materiality_threshold=0.25)
    pred = BacktestPrediction(
        company="TEST_CO",
        year=2023,
        metric="cost_of_revenue",
        actual=115.0,
        forecast=100.0,
    )
    # At 25% threshold, 15% deviation is NOT material
    res_25 = analyzer_25.analyze_prediction(pred, walk_forward_mae=5.0)
    assert res_25.material_deviation is False
    assert res_25.direction == "within_expected_range"

    # At 5% threshold, 15% deviation IS material (dev 15 > 2 * 5.0 = 10)
    analyzer_5 = DeviationAnalyzer(materiality_threshold=0.05)
    res_5 = analyzer_5.analyze_prediction(pred, walk_forward_mae=5.0)
    assert res_5.material_deviation is True
    assert res_5.direction == "above_forecast"


def test_analyze_all_with_mae_lookup():
    """Verify batch analysis with MAE lookup table."""
    analyzer = DeviationAnalyzer(materiality_threshold=0.10)
    preds = [
        BacktestPrediction("CO_A", 2022, "revenue", 120.0, 100.0),
        BacktestPrediction("CO_B", 2022, "revenue", 115.0, 100.0),
    ]
    # CO_A: MAE 5.0 -> 2 * MAE = 10.0 < 20.0 error -> material
    # CO_B: MAE 10.0 -> 2 * MAE = 20.0 > 15.0 error -> not material
    mae_lookup = {("CO_A", "revenue"): 5.0, ("CO_B", "revenue"): 10.0}
    df = analyzer.analyze_all(preds, mae_lookup=mae_lookup)

    assert len(df) == 2
    assert df.iloc[0]["material_deviation"] == True
    assert df.iloc[0]["direction"] == "above_forecast"
    assert df.iloc[1]["material_deviation"] == False
    assert df.iloc[1]["direction"] == "within_expected_range"
