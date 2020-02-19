import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing

from gtime.forecasting import NaiveForecaster, SeasonalNaiveForecaster



@pytest.fixture()
def generate_ts():
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")

    df["A"] = df["A"] + 0.0005 * pd.Series(
        index=df.index, data=range(df.shape[0])
    )
    return df


def test_naive_fit(generate_ts):

    train = generate_ts.iloc[:-1]

    tm = NaiveForecaster()
    tm.fit(train)
    assert tm.next_value_['A'] == train.iloc[-1]['A']


def test_naive_predict(generate_ts):

    train = generate_ts.iloc[:-1]
    test = generate_ts.iloc[-1]

    tm = NaiveForecaster()
    tm.fit(train)
    expected = pd.DataFrame(data=train.iloc[-1], index=test.index)
    assert all(tm.predict(test) == expected)


def test_snaive_fit(generate_ts):

    s = 12
    train = generate_ts.iloc[:-1]

    tm = SeasonalNaiveForecaster(seasonal_length=s)
    tm.fit(train)

    assert all(tm.next_value_['A'] == train.iloc[-s:]['A'])


def test_snaive_predict(generate_ts):

    s = 12
    train = generate_ts.iloc[:-1]
    test = generate_ts.iloc[-1:]

    tm = SeasonalNaiveForecaster(seasonal_length=s)

    tm.fit(train)
    expected = pd.DataFrame(train.iloc[-s].values, index=test.index)
    assert all(tm.predict(test).values == expected.values)


# TODO add multi-period tests