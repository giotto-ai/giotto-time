import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.plotting.preprocessing import seasonal_split, acf




class TestSplits:

    @given(df=giotto_time_series(
        min_length=3,
        max_length=500
    ))
    def test_seasonal_split(self, df):
        split = seasonal_split(df)
        s = split.notna().sum().sum()
        t = df.notna().sum().sum()
        if t != s:
            pass
        assert split.notna().sum().sum() == df.notna().sum().sum()

t = TestSplits()
t.test_seasonal_split()