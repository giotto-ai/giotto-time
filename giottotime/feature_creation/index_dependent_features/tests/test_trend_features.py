from giottotime.feature_creation import (
    DetrendedFeature,
    RemovePolynomialTrend,
    RemoveExponentialTrend,
)
import pandas as pd

from giottotime.models import PolynomialTrend


def test_correct_index_features():
    output_name = "detrended_feature"
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)
    model = PolynomialTrend()

    detrend_feature = DetrendedFeature(trend_model=model, output_name=output_name)
    ts_t = detrend_feature.transform(ts)
    expected_ts = pd.DataFrame(
        [
            1.22681324e-05,
            8.34525141e-06,
            4.86108426e-06,
            1.81563099e-06,
            -7.91108403e-07,
            -2.95913392e-06,
            -4.68844555e-06,
            -5.97904330e-06,
            -6.83092717e-06,
            -7.24409716e-06,
            -7.21855327e-06,
            -6.75429551e-06,
            -5.85132385e-06,
            -4.50963832e-06,
            -2.72923891e-06,
            -5.10125625e-07,
            2.14770155e-06,
            5.24424260e-06,
            8.77949753e-06,
            1.27534663e-05,
        ],
        columns=[output_name],
        index=time_index,
    )
    pd.testing.assert_frame_equal(ts_t, expected_ts)


def test_correct_polynomial_trend():
    output_name = "detrended_pol_feature"
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)

    detrend_feature = RemovePolynomialTrend(polynomial_order=3, output_name=output_name)
    ts_t = detrend_feature.transform(ts)
    expected_ts = pd.DataFrame(
        [
            -6.59832896e-04,
            -2.39271025e-04,
            6.38171382e-05,
            2.63303644e-04,
            3.73060540e-04,
            4.06959876e-04,
            3.78873701e-04,
            3.02674064e-04,
            1.92233012e-04,
            6.14225966e-05,
            -7.58851350e-05,
            -2.05818134e-04,
            -3.14504350e-04,
            -3.88071736e-04,
            -4.12648241e-04,
            -3.74361819e-04,
            -2.59340418e-04,
            -5.37119912e-05,
            2.56395511e-04,
            6.84854138e-04,
        ],
        columns=[output_name],
        index=time_index,
    )
    pd.testing.assert_frame_equal(ts_t, expected_ts)


def test_correct_exponential_trend():
    output_name = "detrended_exp_feature"
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)

    detrend_feature = RemoveExponentialTrend(output_name=output_name)
    ts_t = detrend_feature.transform(ts)
    expected_ts = pd.DataFrame(
        [
            -1.0,
            -0.18238542,
            0.60196471,
            1.34698345,
            2.04549733,
            2.68902453,
            3.26753629,
            3.76917473,
            4.1799193,
            4.48319226,
            4.65939237,
            4.68534338,
            4.53364205,
            4.17188719,
            3.5617681,
            2.65798675,
            1.40698343,
            -0.25457009,
            -2.40155216,
            -5.1224979,
        ],
        columns=[output_name],
        index=time_index,
    )
    pd.testing.assert_frame_equal(ts_t, expected_ts)
