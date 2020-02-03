import numpy as np

from gtime.feature_extraction import Detrender
import pandas as pd


def test_polynomial_detrend():
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)

    detrend_feature = Detrender(trend="polynomial", trend_x0=np.zeros(3))
    feature_name = detrend_feature.__class__.__name__
    ts_t = detrend_feature.fit_transform(ts)
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
        columns=[f"0__{feature_name}"],
        index=time_index,
    )
    pd.testing.assert_frame_equal(ts_t, expected_ts)


def test_exponential_detrend():
    time_index = pd.date_range(start="2020-01-01", end="2020-01-20")
    ts = pd.DataFrame(range(0, 20), index=time_index)

    detrend_feature = Detrender(trend="exponential", trend_x0=0)
    feature_name = detrend_feature.__class__.__name__
    ts_t = detrend_feature.fit_transform(ts)
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
        columns=[f"0__{feature_name}"],
        index=time_index,
    )
    pd.testing.assert_frame_equal(ts_t, expected_ts)
