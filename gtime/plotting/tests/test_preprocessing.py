import numpy as np
import re
import hypothesis.strategies as st
from hypothesis import given, settings
from gtime.utils.hypothesis.time_indexes import giotto_time_series, period_indexes
from gtime.plotting.preprocessing import seasonal_split, acf, pacf, _get_cycle_names, _get_season_names, _autocorrelation, _scale, _solve_yw_equation, _week_of_year, yule_walker


class TestSplits:

    @given(t=period_indexes(min_length=1, max_length=1))
    def test_week_of_year(self, t):
        period = t[0]
        res = _week_of_year(period)
        assert re.match(r'\d{4}_\d\d?$', res)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle=st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r'[1-9][DWMQY]', fullmatch=True))
    )
    @settings(deadline=None)
    def test__get_cycle_names_size(self, df, cycle):
        cycle = _get_cycle_names(df, cycle)
        assert len(cycle) == len(df)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle = st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r'[1-9][DWMQY]', fullmatch=True)
        ),
        freq=st.from_regex(r"[1-9]?[DWMQ]", fullmatch=True),
    )
    @settings(deadline=None)
    def test__get_season_names_size(self, df, cycle, freq):
        seasons = _get_season_names(df, cycle, freq)
        assert len(seasons) == len(df)

    @given(
        df=giotto_time_series(min_length=3, max_length=500),
        cycle = st.one_of(
            st.sampled_from(["year", "quarter", "month", "week"]),
            st.from_regex(r'[1-9][DWMQY]', fullmatch=True)
        ),
        freq=st.from_regex(r"[1-9]?[DWMQ]", fullmatch=True),
        agg=st.sampled_from(["mean", "sum", "last"]),
    )
    @settings(deadline=None)
    def test_seasonal_split_shape_named(self, df, cycle, freq, agg):
        split = seasonal_split(df, cycle, freq, agg)
        s = split.notna().mean()
        t = df.notna().mean().mean()
        assert split.stack().shape == df.resample(freq).agg(agg).dropna().shape


class TestAcf:
    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        max_lag=st.integers(min_value=1, max_value=1000),
    )
    def test_acf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = acf(df_array, max_lag)
        assert len(res) == min(max_lag, len(df))

    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        max_lag=st.integers(min_value=1, max_value=100),
    )
    def test_pacf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = pacf(df_array, max_lag)
        assert len(res) == min(max_lag, len(df))
