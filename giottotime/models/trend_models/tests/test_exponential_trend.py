import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from pandas.util import testing as testing

from giottotime.models.trend_models.exponential_trend import ExponentialTrend


class TestExponentialTrend:
    def test_exponential_trend(self):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        df["A"] = df["A"] + 0.0005 * pd.Series(
            index=df.index, data=range(df.shape[0])
        ).apply(lambda x: np.exp(0.03 * x))

        tm = ExponentialTrend()
        tm.fit(df)

        detrended = tm.transform(df)

        tm = ExponentialTrend()
        tm.fit(detrended)

        assert np.allclose(tm.model_exponent_, 0.0)

    @settings(deadline=None, max_examples=2)
    @given(arrays(dtype=float, shape=(500,), elements=floats(-10, 10)))
    def test_exponential_trend_multiple(self, random_array):
        testing.N, testing.K = 500, 1
        df = testing.makeTimeDataFrame(freq="D")

        df["A"] = random_array + 0.0005 * pd.Series(
            index=df.index, data=range(df.shape[0])
        ).apply(lambda x: np.exp(0.03 * x))

        tm = ExponentialTrend()
        tm.fit(df)

        detrended = tm.transform(df)

        tm = ExponentialTrend()
        tm.fit(detrended)

        assert np.allclose(tm.model_exponent_, 0.0)


#
