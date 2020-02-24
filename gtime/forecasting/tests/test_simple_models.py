import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing

from gtime.forecasting import (
    NaiveModel,
    SeasonalNaiveModel,
    DriftModel,
)


@pytest.fixture()
def generate_ts():
    testing.N, testing.K = 500, 1
    df = testing.makeTimeDataFrame(freq="D")
    train = df.iloc[:-3]
    test = df.iloc[-3:]
    return train, test


class TestNaive:
    def test_fit(self, generate_ts):
        train, _ = generate_ts
        tm = NaiveModel()
        tm.fit(train)
        assert tm.last_value_["A"] == train.iloc[-1]["A"]

    def test_predict(self, generate_ts):
        train, test = generate_ts
        print(train, test)
        tm = NaiveModel()
        tm.fit(train)
        expected = pd.DataFrame(data=train.iloc[[-1]], index=test.index)
        assert all(tm.predict(test) == expected)


class TestSeasonalNaive:
    def test_fit(self, generate_ts):
        train, _ = generate_ts
        tm = SeasonalNaiveModel(seasonal_length=4)
        tm.fit(train)
        assert all(tm.last_value_["A"] == train.iloc[-tm.lag_:]["A"])

    def test_predict(self, generate_ts):
        train, test = generate_ts
        print(train, test)
        tm = SeasonalNaiveModel(seasonal_length=4)
        tm.fit(train)
        expected = pd.DataFrame(data=train.iloc[-4:-1], index=test.index)
        assert all(tm.predict(test) == expected)


class TestDriftModel:
    def test_fit(self, generate_ts):
        train, _ = generate_ts
        tm = DriftModel()
        tm.fit(train)
        assert all(tm.drift_ == (train.iloc[-1] - train.iloc[0]) / len(train))
        assert all(tm.last_value_ == train.iloc[-1])

    def test_predict(self, generate_ts):
        train, test = generate_ts
        print(train, test)
        tm = DriftModel()
        tm.fit(train)
        drift = tm.drift_
        last = train.iloc[-1]
        y_pred = tm.predict(test)
        assert all(y_pred.iloc[2] == last + 2 * drift)
