import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from gtime.feature_extraction import Calendar
from gtime.utils.hypothesis.time_indexes import giotto_time_series


def test_empty_and_non_finite_kernel_error():
    with pytest.raises(ValueError):
        Calendar(
            start_date="ignored",
            end_date="ignored",
            region="america",
            country="Brazil",
            kernel=np.array([]),
        )

    with pytest.raises(ValueError):
        Calendar(
            start_date="ignored",
            end_date="ignored",
            region="america",
            country="Brazil",
            kernel=np.array([np.nan, 1]),
        )


def test_unevenly_spaced_time_series():
    unevenly_spaced_ts = pd.DataFrame(
        index=[
            pd.Period("2012-01-01"),
            pd.Period("2012-01-03"),
            pd.Period("2012-01-10"),
        ]
    )
    cal_feature = Calendar(
        start_date="ignored",
        end_date="ignored",
        region="america",
        country="Brazil",
        kernel=np.array([0, 1]),
    )

    with pytest.raises(ValueError):
        cal_feature.fit_transform(unevenly_spaced_ts)


@settings(deadline=pd.Timedelta(milliseconds=5000), max_examples=7)
@given(giotto_time_series(min_length=2, max_length=30))
def test_correct_index_random_ts(ts):
    cal_feature = Calendar(
        start_date="ignored",
        end_date="ignored",
        region="america",
        country="Brazil",
        kernel=np.array([1, 2]),
    )
    Xt = cal_feature.fit_transform(ts)
    np.testing.assert_array_equal(Xt.index, ts.index)
