import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pandas.util import testing as testing

from giottotime.utils.hypothesis.time_indexes import giotto_time_series
from giottotime.feature_generation import (
    Constant,
    PeriodicSeasonal,
)


class TestPeriodicSesonalFeature:
    def test_missing_start_date_or_period(self):
        periodic_feature = PeriodicSeasonal()
        with pytest.raises(ValueError):
            periodic_feature.transform()

        periodic_feature = PeriodicSeasonal(index_period=1)
        with pytest.raises(ValueError):
            periodic_feature.transform()

        periodic_feature = PeriodicSeasonal(start_date="2010-01-01")
        with pytest.raises(ValueError):
            periodic_feature.transform()

    def test_string_period(self):
        testing.N, testing.K = 20, 1
        ts = testing.makeTimeDataFrame(freq="s")
        periodic_feature = PeriodicSeasonal(period="1 days")
        periodic_feature.transform(ts)

        assert type(periodic_feature.period) == pd.Timedelta

    def test_correct_start_date(self):
        testing.N, testing.K = 20, 1
        ts = testing.makeTimeDataFrame(freq="s")
        start_date = "2018-01-01"
        periodic_feature = PeriodicSeasonal(period="1 days", start_date=start_date)
        periodic_feature.transform(ts)

        assert periodic_feature.start_date == ts.index.values[0]

        periodic_feature = PeriodicSeasonal(
            period="3 days", index_period=10, start_date=start_date
        )
        periodic_feature.transform()
        assert periodic_feature.start_date == pd.to_datetime(start_date)

        start_date = pd.to_datetime("2018-01-01")
        periodic_feature = PeriodicSeasonal(
            period="3 days", index_period=10, start_date=start_date
        )
        periodic_feature.transform()
        assert periodic_feature.start_date == start_date

    def test_too_high_sampling_frequency(self):
        start_date = "2018-01-01"
        periodic_feature = PeriodicSeasonal(
            period="2 days",
            start_date=start_date,
            index_period=pd.DatetimeIndex(start=start_date, end="2020-01-01", freq="W"),
        )
        with pytest.raises(ValueError):
            periodic_feature.transform()

    def test_correct_sinusoide(self):
        testing.N, testing.K = 30, 1
        ts = testing.makeTimeDataFrame(freq="MS")
        start_date = "2018-01-01"
        periodic_feature = PeriodicSeasonal(
            period="365 days",
            start_date=start_date,
            index_period=pd.DatetimeIndex(start=start_date, end="2020-01-01", freq="W"),
        )
        output_sin = periodic_feature.transform(ts)
        expected_index = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
                "2000-04-01",
                "2000-05-01",
                "2000-06-01",
                "2000-07-01",
                "2000-08-01",
                "2000-09-01",
                "2000-10-01",
                "2000-11-01",
                "2000-12-01",
                "2001-01-01",
                "2001-02-01",
                "2001-03-01",
                "2001-04-01",
                "2001-05-01",
                "2001-06-01",
                "2001-07-01",
                "2001-08-01",
                "2001-09-01",
                "2001-10-01",
                "2001-11-01",
                "2001-12-01",
                "2002-01-01",
                "2002-02-01",
                "2002-03-01",
                "2002-04-01",
                "2002-05-01",
                "2002-06-01",
            ],
            dtype="datetime64[ns]",
            freq="MS",
        )
        expected_df = pd.DataFrame.from_dict(
            {
                f"0__{periodic_feature.__class__.__name__}": [
                    0.0,
                    0.25433547,
                    0.42938198,
                    0.49999537,
                    0.43585316,
                    0.25062091,
                    0.0043035,
                    -0.25062091,
                    -0.43585316,
                    -0.49999537,
                    -0.42938198,
                    -0.24688778,
                    0.00860668,
                    0.2617078,
                    0.42938198,
                    0.49999537,
                    0.43585316,
                    0.25062091,
                    0.0043035,
                    -0.25062091,
                    -0.43585316,
                    -0.49999537,
                    -0.42938198,
                    -0.24688778,
                    0.00860668,
                    0.2617078,
                    0.42938198,
                    0.49999537,
                    0.43585316,
                    0.25062091,
                ]
            }
        )
        expected_df.index = expected_index
        pd.testing.assert_frame_equal(output_sin, expected_df)


class TestConstantFeature:
    def test_correct_constant_feature(self):
        constant = 12
        df = pd.DataFrame.from_dict({"old_name": [0, 1, 2, 3, 4, 5]})

        constant_feature = Constant(constant=constant)

        df_constant = constant_feature.fit_transform(df)
        expected_df_constant = pd.DataFrame.from_dict(
            {
                f"0__{constant_feature.__class__.__name__}": [
                    constant,
                    constant,
                    constant,
                    constant,
                    constant,
                    constant,
                ]
            }
        )

        testing.assert_frame_equal(expected_df_constant, df_constant, check_dtype=False)

    @given(
        giotto_time_series(
            min_length=1,
            start_date=pd.Timestamp(2000, 1, 1),
            end_date=pd.Timestamp(2010, 1, 1),
        ),
        st.integers(0, 100),
    )
    def test_random_ts_and_constant(self, df: pd.DataFrame, constant: int):

        constant_feature = Constant(constant=constant)
        df_constant = constant_feature.fit_transform(df)

        #  testing.assert_frame_equal(expected_df_constant, df_constant)
