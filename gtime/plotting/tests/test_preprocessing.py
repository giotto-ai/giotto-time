import numpy as np
import pandas as pd
import pytest
import hypothesis.strategies as st
from pandas.util import testing as testing
from hypothesis import given, note
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.plotting.preprocessing import seasonal_split, acf, pacf


# class TestSplits:
#     @given(df=giotto_time_series(min_length=3, max_length=500))
#     def test_seasonal_split(self, df):
#         split = seasonal_split(df)
#         s = split.notna().sum().sum()
#         t = df.notna().sum().sum()
#         if t != s:
#             pass
#         assert split.notna().sum().sum() == df.notna().sum().sum()
#
#
# t = TestSplits()
# t.test_seasonal_split()



@given(
    df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
    max_lag=st.integers(min_value=1, max_value=1000),
)
def test_acf_len(df, max_lag):
    df_array = np.ravel(df.values)
    res = acf(df_array, max_lag)
    assert len(res) == min(max_lag, len(df))

@given(
    df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
    max_lag=st.integers(min_value=1, max_value=1000),
)
def test_pacf_len(df, max_lag):
    df_array = np.ravel(df.values)
    res = pacf(df_array, max_lag)
    assert len(res) == min(max_lag, len(df))


test_pacf_len()