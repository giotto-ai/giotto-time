import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.plotting.preprocessing import seasonal_split, acf, pacf

class TestSplits:

    @given(df=giotto_time_series(min_length=3, max_length=500),
           cycle=st.sampled_from(['year', 'quarter', 'month', 'week']),
           freq=st.from_regex(r'[1-9][DWMQ]', fullmatch=True),
           agg=st.sampled_from(['mean', 'sum', 'last']))
    @settings(deadline=None)
    def test_seasonal_split_shape_named(self, df, cycle, freq, agg):
        split = seasonal_split(df, cycle, freq, agg)
        s = split.notna().mean()
        t = df.notna().mean().mean()
        assert split.stack().shape == df.resample(freq).agg(agg).dropna().shape

    @given(df=giotto_time_series(min_length=3, max_length=500),
           cycle=st.from_regex(r'[1-9][DWMQY]', fullmatch=True),
           freq=st.from_regex(r'[1-9][DWMQ]', fullmatch=True),
           agg=st.sampled_from(['mean', 'sum', 'last']))
    @settings(deadline=None)
    def test_seasonal_split_shape_freq(self, df, cycle, freq, agg):
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

