"""Unit tests for time-series forecasting, backtesting, and regression metrics."""

import numpy as np
import pytest
from analysis.forecasting import TrendForecaster


def test_linear_regression_perfect_trend():
    """Verify exact fit on synthetic linear series."""
    forecaster = TrendForecaster()
    # Perfect line: y = 10 * year - 20000 (i.e. 2020: 200, 2021: 210, 2022: 220, 2023: 230)
    years = [2020, 2021, 2022, 2023]
    values = [200.0, 210.0, 220.0, 230.0]

    # Next year forecast
    f_res = forecaster.forecast_next_year("TEST_CO", "Revenue", years, values)
    assert f_res.forecast_year == 2024
    assert f_res.last_actual_year == 2023
    assert f_res.last_actual_value == 230.0
    assert pytest.approx(f_res.forecast_value, rel=1e-4) == 240.0
    assert pytest.approx(f_res.slope, rel=1e-4) == 10.0


def test_next_year_calculation_not_hardcoded():
    """Verify that next year is max(year) + 1 regardless of arbitrary era."""
    forecaster = TrendForecaster()
    years = [2012, 2013, 2014, 2015]
    values = [50.0, 60.0, 70.0, 80.0]

    f_res = forecaster.forecast_next_year("OLD_CO", "Revenue", years, values)
    assert f_res.forecast_year == 2016  # NOT 2024
    assert f_res.last_actual_year == 2015


def test_chronological_backtesting_metrics():
    """Verify walk-forward backtesting MAE, baseline MAE, RMSE, and R2 on 4th observation onward."""
    forecaster = TrendForecaster()
    # 8 points: evaluation starts from 4th observation (2018), resulting in 5 test points
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    values = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 162.0, 168.0]

    eval_res, backtest_preds = forecaster.backtest_series("TEST_CO", "cost_of_revenue", years, values)

    assert eval_res.evaluation_status == "success"
    assert eval_res.test_points == 5
    assert eval_res.test_years_count == 5
    assert len(backtest_preds) == 5

    # Verification of error metrics
    assert eval_res.mae is not None and eval_res.mae > 0.0
    assert eval_res.rmse is not None and eval_res.rmse > 0.0
    assert eval_res.baseline_mae is not None and eval_res.baseline_mae > 0.0
    assert eval_res.mape is not None and eval_res.mape > 0.0
    # Model beats naive baseline on steady trend
    assert eval_res.beats_baseline is True
    # R2 is reported because test_points (5) >= 5
    assert eval_res.r2 is not None


def test_r2_null_when_test_points_under_five():
    """Verify R2 is None when test_points < 5 and explains reason in notes."""
    forecaster = TrendForecaster()
    # 5 observations -> walk-forward produces 5 - 3 = 2 test points (< 5)
    years = [2018, 2019, 2020, 2021, 2022]
    values = [100.0, 110.0, 120.0, 130.0, 140.0]

    eval_res, _ = forecaster.backtest_series("SAMPLE_CO", "cost_of_revenue", years, values)
    assert eval_res.evaluation_status == "success"
    assert eval_res.test_points == 2
    assert eval_res.r2 is None
    assert "requires at least 5 test points" in eval_res.notes


def test_negative_r2_handling():
    """Verify that R2 can legitimately be negative when test_points >= 5 and model underperforms variance."""
    forecaster = TrendForecaster()
    # 8 observations (3 train, 5 test points where values crash drastically)
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    values = [100.0, 110.0, 120.0, 20.0, 15.0, 10.0, 5.0, 2.0]

    eval_res, _ = forecaster.backtest_series("CRASH_CO", "cost_of_revenue", years, values)
    assert eval_res.evaluation_status == "success"
    assert eval_res.test_points == 5
    assert eval_res.r2 is not None
    assert eval_res.r2 < 0.0


def test_insufficient_history_handling(synthetic_short_history_df):
    """Verify series with fewer than 4 observations return not_forecastable and insufficient_data."""
    forecaster = TrendForecaster()
    years = synthetic_short_history_df["year"].tolist()  # 2 observations
    values = synthetic_short_history_df["revenue"].tolist()

    eval_res, backtest_preds = forecaster.backtest_series("SHORT_CORP", "revenue", years, values)
    assert eval_res.evaluation_status == "insufficient_data"
    assert eval_res.mae is None
    assert eval_res.baseline_mae is None
    assert eval_res.r2 is None
    assert eval_res.test_points == 0
    assert len(backtest_preds) == 0

    # Next-year forecast should be not_forecastable
    f_res = forecaster.forecast_next_year("SHORT_CORP", "revenue", years, values)
    assert f_res.status == "not_forecastable"
    assert np.isnan(f_res.forecast_value)


def test_three_observations_insufficient_history():
    """Verify exactly 3 observations returns not_forecastable and insufficient_data."""
    forecaster = TrendForecaster()
    years = [2021, 2022, 2023]
    values = [100.0, 110.0, 120.0]

    eval_res, _ = forecaster.backtest_series("THREE_CO", "cost_of_revenue", years, values)
    assert eval_res.evaluation_status == "insufficient_data"
    assert eval_res.mae is None

    f_res = forecaster.forecast_next_year("THREE_CO", "cost_of_revenue", years, values)
    assert f_res.status == "not_forecastable"
    assert np.isnan(f_res.forecast_value)


def test_missing_values_handled_safely():
    """Verify that missing/NaN values inside history are ignored without crashing."""
    forecaster = TrendForecaster()
    # 5 observations, 1 NaN -> 4 valid pairs
    years = [2018, 2019, 2020, 2021, 2022]
    values = [100.0, np.nan, 120.0, 130.0, 140.0]

    f_res = forecaster.forecast_next_year("NAN_CO", "revenue", years, values)
    assert f_res.status == "success"
    assert f_res.forecast_year == 2023


def test_constant_target_not_forecastable():
    """Verify that a target with near-zero variance across history returns not_forecastable with NaN forecast."""
    forecaster = TrendForecaster()
    years = [2018, 2019, 2020, 2021, 2022]
    values = [0.0, 0.0, 0.0, 0.0, 0.0]

    f_res = forecaster.forecast_next_year("CONST_CO", "cost_of_revenue", years, values)
    assert f_res.status == "not_forecastable"
    assert np.isnan(f_res.forecast_value)
    assert f_res.slope == 0.0
    assert f_res.intercept == 0.0
    assert f_res.forecast_year == 2023
    assert "near-zero variance" in f_res.notes

