"""Unit tests for Year-on-Year (YoY) analysis module."""

import numpy as np
import pandas as pd
from analysis.yoy_analysis import calculate_yoy_for_series, calculate_yoy_dataframe, YoYResult


def test_yoy_basic_calculation():
    years = [2020, 2021, 2022]
    values = [100.0, 120.0, 150.0]
    results = calculate_yoy_for_series("TEST_CO", "Revenue", years, values)

    assert len(results) == 3
    # 2021: (120 - 100) / 100 * 100 = 20.0%
    assert results[1].yoy_percent == 20.0
    assert results[1].trend == "growth"
    assert results[1].data_status == "normal"
    # 2022: (150 - 120) / 120 * 100 = 25.0%
    assert results[2].yoy_percent == 25.0
    assert results[2].trend == "growth"


def test_first_year_handling():
    years = [2019, 2020]
    values = [500.0, 450.0]
    results = calculate_yoy_for_series("TEST_CO", "Revenue", years, values)

    first_res = results[0]
    assert first_res.previous_value is None
    assert first_res.yoy_percent is None
    assert first_res.trend == "No previous year"
    assert first_res.data_status == "first_year"


def test_zero_previous_year_handling():
    years = [2020, 2021]
    values = [0.0, 50.0]
    results = calculate_yoy_for_series("TEST_CO", "Net Income", years, values)

    second_res = results[1]
    assert second_res.previous_value == 0.0
    assert second_res.yoy_percent is None
    assert second_res.data_status == "zero_previous"
    assert second_res.trend == "growth"


def test_negative_values_handling():
    # Moving from -50 to +25 is growth
    years = [2020, 2021]
    values = [-50.0, 25.0]
    results = calculate_yoy_for_series("TEST_CO", "Net Income", years, values)

    res = results[1]
    # ((25 - (-50)) / abs(-50)) * 100 = (75 / 50) * 100 = +150.0%
    assert res.yoy_percent == 150.0
    assert res.trend == "growth"
    assert res.data_status == "negative_base"


def test_company_wise_separation(synthetic_standard_df):
    """Verify that YoY does not bleed across company boundaries."""
    yoy_df = calculate_yoy_dataframe(synthetic_standard_df, metrics=["revenue"])

    # Check COMP_A first year is 2018
    comp_a_first = yoy_df[(yoy_df["company"] == "COMP_A") & (yoy_df["year"] == 2018)].iloc[0]
    assert comp_a_first["trend"] == "No previous year"
    assert pd.isna(comp_a_first["yoy_percent"])

    # Check COMP_B first year is 2019 (not compared to COMP_A 2018 or 2022)
    comp_b_first = yoy_df[(yoy_df["company"] == "COMP_B") & (yoy_df["year"] == 2019)].iloc[0]
    assert comp_b_first["trend"] == "No previous year"
    assert pd.isna(comp_b_first["yoy_percent"])


def test_missing_values_in_series():
    years = [2020, 2021, 2022]
    values = [100.0, np.nan, 130.0]
    results = calculate_yoy_for_series("TEST_CO", "Revenue", years, values)

    # 2021 should be missing_value
    assert results[1].data_status == "missing_value"
    # 2022 had a missing previous value, so cannot compute YoY
    assert results[2].data_status == "missing_value"
    assert results[2].yoy_percent is None


def test_extreme_percentage_above_threshold():
    """Verify that YoY > 500% or < -500% results in status extreme_percentage."""
    # Positive jump > 500%
    years = [2020, 2021]
    values = [100.0, 650.0]
    res_pos = calculate_yoy_for_series("TEST_CO", "Revenue", years, values)[1]
    # (650 - 100) / 100 * 100 = 550.0%
    assert res_pos.yoy_percent == 550.0
    assert res_pos.status == "extreme_percentage"
    assert res_pos.data_status == "extreme_percentage"
    assert res_pos.trend == "growth"

    # Large decline < -500%
    values_neg = [100.0, -450.0]
    res_neg = calculate_yoy_for_series("TEST_CO", "Revenue", years, values_neg)[1]
    # (-450 - 100) / 100 * 100 = -550.0%
    assert res_neg.yoy_percent == -550.0
    assert res_neg.status == "extreme_percentage"
    assert res_neg.data_status == "extreme_percentage"
    assert res_neg.trend == "decline"


def test_extreme_percentage_boundary():
    """Verify exact boundary at 500%: 500.0% is normal, 500.01% is extreme_percentage."""
    years = [2020, 2021]
    # Exactly 500.0%
    res_exact = calculate_yoy_for_series("TEST_CO", "Revenue", years, [100.0, 600.0])[1]
    assert res_exact.yoy_percent == 500.0
    assert res_exact.status == "normal"

    # Slightly above 500.0%
    res_above = calculate_yoy_for_series("TEST_CO", "Revenue", years, [100.0, 600.01])[1]
    assert res_above.yoy_percent == 500.01
    assert res_above.status == "extreme_percentage"


def test_priority_order_enforcement():
    """Enforce exact priority: zero_previous -> negative_base -> extreme_percentage -> normal."""
    years = [2020, 2021]

    # 1. zero_previous priority over extreme_percentage
    res_zero = calculate_yoy_for_series("TEST_CO", "Revenue", years, [0.0, 10000.0])[1]
    assert res_zero.yoy_percent is None
    assert res_zero.status == "zero_previous"
    assert res_zero.data_status == "zero_previous"

    # 2. negative_base priority over extreme_percentage (1100% YoY from negative base)
    # diff = 100 - (-10) = 110; YoY = (110 / 10) * 100 = 1100.0%
    res_neg = calculate_yoy_for_series("TEST_CO", "Net Income", years, [-10.0, 100.0])[1]
    assert res_neg.yoy_percent == 1100.0
    assert res_neg.status == "negative_base"
    assert res_neg.data_status == "negative_base"

    # 3. extreme_percentage priority over normal
    res_ext = calculate_yoy_for_series("TEST_CO", "Revenue", years, [10.0, 100.0])[1]
    assert res_ext.yoy_percent == 900.0
    assert res_ext.status == "extreme_percentage"
    assert res_ext.data_status == "extreme_percentage"

    # 4. normal
    res_norm = calculate_yoy_for_series("TEST_CO", "Revenue", years, [100.0, 150.0])[1]
    assert res_norm.yoy_percent == 50.0
    assert res_norm.status == "normal"
    assert res_norm.data_status == "normal"


def test_no_row_with_extreme_percent_has_normal_status():
    """Requirement: No row with abs(yoy_percent) > 500 may have status 'normal'."""
    years = [2020, 2021]
    test_cases = [
        [10.0, 100.0],    # +900% -> extreme_percentage
        [10.0, -70.0],    # -800% -> extreme_percentage
        [-5.0, 100.0],    # +2100% -> negative_base
        [-5.0, -50.0],    # -900% -> negative_base
        [1.0, 10.0],      # +900% -> extreme_percentage
    ]
    for vals in test_cases:
        res = calculate_yoy_for_series("TEST_CO", "Metric", years, vals)[1]
        assert abs(res.yoy_percent) > 500.0
        assert res.status != "normal", f"Row with YoY {res.yoy_percent}% illegally marked normal!"
        assert res.data_status != "normal"
        # Verify to_dict includes status
        d = res.to_dict()
        assert d["status"] != "normal"
        assert d["status"] == res.status


def test_configurable_extreme_yoy_threshold():
    """Verify that extreme_threshold can be configured on series and dataframe functions."""
    years = [2020, 2021]
    values = [100.0, 350.0]  # +250%

    # Default threshold (500): 250% is normal
    res_def = calculate_yoy_for_series("TEST_CO", "Revenue", years, values, extreme_threshold=500.0)[1]
    assert res_def.status == "normal"

    # Custom threshold (200): 250% is extreme_percentage
    res_custom = calculate_yoy_for_series("TEST_CO", "Revenue", years, values, extreme_threshold=200.0)[1]
    assert res_custom.status == "extreme_percentage"

    # Test via DataFrame
    df = pd.DataFrame([
        {"company": "CO_A", "year": 2020, "revenue": 100.0},
        {"company": "CO_A", "year": 2021, "revenue": 350.0},
    ])
    df_yoy = calculate_yoy_dataframe(df, extreme_threshold=200.0)
    assert df_yoy.iloc[1]["status"] == "extreme_percentage"

