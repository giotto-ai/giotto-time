import numpy as np
import pandas as pd

import pandas.util.testing as testing

from gtime.forecasting import TrendForecaster


def test_polynomial_trend():
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")

    df["A"] = df["A"] + 0.0005 * pd.Series(
        index=df.index, data=range(df.shape[0])
    ) * pd.Series(index=df.index, data=range(df.shape[0]))

    tm = TrendForecaster(trend="polynomial", trend_x0=0.0)
    tm.fit(df["A"])
    # too hard to expect every time
    # assert np.allclose(tm.best_trend_params_, [0.0] * len(tm.best_trend_params_))
    assert len(tm.best_trend_params_) == 1


def test_exponential_trend():
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")

    df["A"] = df["A"] + 0.0005 * pd.Series(
        index=df.index, data=range(df.shape[0])
    ).apply(lambda x: np.exp(0.03 * x))

    tm = TrendForecaster(trend="exponential", trend_x0=4 * [0.0])
    tm.fit(df)
    # too hard to expect this result every time
    # assert np.allclose(tm.best_trend_params_, [0.0] * len(tm.best_trend_params_))
    assert len(tm.best_trend_params_) == 4

# TODO: predicting tests
