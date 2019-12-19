from giottotime.feature_creation import (
    DetrendedFeature,
    RemovePolynomialTrend,
    RemoveExponentialTrend,
)
import pandas as pd


def test_correct_index_features():
    output_name = "detrended_feature"
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)

    detrend_feature = DetrendedFeature(output_name=output_name)
    ts_t = detrend_feature.transform(ts)
    expected_ts = pd.DataFrame(
        [
            -2.334165e-07,
            -2.080005e-07,
            -1.825846e-07,
            -1.571686e-07,
            -1.317526e-07,
            -1.063366e-07,
            -8.092063e-08,
            -5.550465e-08,
            -3.008866e-08,
            -4.672680e-09,
            2.074330e-08,
            4.615929e-08,
            7.157527e-08,
            9.699125e-08,
            1.224072e-07,
            1.478232e-07,
            1.732392e-07,
            1.986552e-07,
            2.240712e-07,
            2.494872e-07,
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
            1.226813e-05,
            8.345251e-06,
            4.861084e-06,
            1.815631e-06,
            -7.911084e-07,
            -2.959134e-06,
            -4.688446e-06,
            -5.979043e-06,
            -6.830927e-06,
            -7.244097e-06,
            -7.218553e-06,
            -6.754296e-06,
            -5.851324e-06,
            -4.509638e-06,
            -2.729239e-06,
            -5.101256e-07,
            2.147702e-06,
            5.244243e-06,
            8.779498e-06,
            1.275347e-05,
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
