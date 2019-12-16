import numpy as np
import pandas as pd

from hypothesis import given, strategies as st

from pandas.util import testing as testing
from giottotime.models.trend_models.exponential_trend import ExponentialTrend

from random import randint

class TestExponentialTrend:
    def test_exponential_trend(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame( freq="D" )

        df['A'] = df['A'] + 0.0005*pd.Series( index=df.index, data=range(df.shape[0]) ).apply( lambda x : np.exp(0.03*x) )

        print(df)

        tm = ExponentialTrend()
        tm.fit(df)

        detrended = tm.transform(df)
        print(detrended)

        tm = ExponentialTrend()
        tm.fit(detrended)

        assert np.allclose( tm.model_exponent_, 0.0 )

#
