import numpy as np
import pandas as pd

from hypothesis import given, strategies as st

from pandas.util import testing as testing
from giottotime.models.trend_models.polynomial_trend import PolynomialTrend

from random import randint

class TestPolynomialTrend:
    def test_polynomial_trend(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame( freq="D" )

        df['A'] = df['A'] + 0.0005*pd.Series( index=df.index, data=range(df.shape[0]) )*pd.Series( index=df.index, data=range(df.shape[0]) )

        tm = PolynomialTrend(order=3)
        tm.fit(df)

        detrended = tm.transform(df)

        tm = PolynomialTrend(order=3)
        tm.fit(detrended)

        assert np.allclose( tm.model_weights_,  [0.0]*len(tm.model_weights_) )




#
