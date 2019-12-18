import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from giottotime.feature_creation import CalendarFeature
from giottotime.utils.hypothesis.time_indexes import giotto_time_series


def test_empty_and_non_finite_kernel_error():
    with pytest.raises(ValueError):
        CalendarFeature(
            start_date="ignored",
            end_date="ignored",
            region="america",
            country="Brazil",
            kernel=np.array([]),
            output_name="cal",
        )

    with pytest.raises(ValueError):
        CalendarFeature(
            start_date="ignored",
            end_date="ignored",
            region="america",
            country="Brazil",
            kernel=np.array([np.nan, 1]),
            output_name="cal",
        )


def test_unevenly_spaced_time_series():
    unevenly_spaced_ts = pd.DataFrame(
        index=[
            pd.Period("2012-01-01"),
            pd.Period("2012-01-03"),
            pd.Period("2012-01-10"),
        ]
    )
    cal_feature = CalendarFeature(
        start_date="ignored",
        end_date="ignored",
        region="america",
        country="Brazil",
        kernel=np.array([0, 1]),
        output_name="cal",
    )

    with pytest.raises(ValueError):
        cal_feature.transform(unevenly_spaced_ts)


def test_correct_calendar_without_input_ts():
    start_date = "2018-01-01"
    end_date = "2019-01-01"

    cal_feature = CalendarFeature(
        start_date=start_date,
        end_date=end_date,
        region="america",
        country="Brazil",
        kernel=np.array([0, 1]),
        output_name="cal",
    )
    Xt = cal_feature.transform()
    expected_index = pd.date_range(start_date, end_date)
    np.testing.assert_array_equal(Xt.index, expected_index)


@settings(deadline=pd.Timedelta(milliseconds=5000), max_examples=7)
@given(giotto_time_series(min_length=2, max_length=30))
def test_correct_index_random_ts(ts):
    output_name = "cal"
    cal_feature = CalendarFeature(
        start_date="ignored",
        end_date="ignored",
        region="america",
        country="Brazil",
        kernel=np.array([1, 2]),
        output_name=output_name,
    )
    Xt = cal_feature.transform(ts)
    np.testing.assert_array_equal(Xt.index, ts.index)
