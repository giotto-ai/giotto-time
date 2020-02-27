import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series

from gtime.forecasting import (
    NaiveModel,
    SeasonalNaiveModel,
    DriftModel,
)

# @st.composite
# def forecast_input(draw, max_lenth):
#     df = draw(giotto_time_series(
#                 min_length=3,
#                 max_length=max_lenth,
#                 allow_nan=False,
#                 allow_infinity=False
#             ))
#     split = draw(st.integers(min_value=1, max_value=len(df)-1))
#     train = df.iloc[:split]
#     test = df.iloc[split:]
#     return train, test
#
#
# class TestNaive:
#     @given(x=forecast_input(50))
#     def test_fit(self, x):
#         train, test = x
#         tm = NaiveModel()
#         tm.fit(train, test)
#         assert tm._y_columns == train.columns
#
#     @given(x=forecast_input(50))
#     def test_predict(self, x):
#         train, test = x
#         tm = NaiveModel()
#         tm.fit(train, test)
#         y_pred = tm.predict(test)
#         assert y_pred.shape == (len(test), len(test))
#
#
# t = TestNaive()
# t.test_predict()


# class TestDriftModel:
#     def test_fit(self, generate_ts):
#         train, _ = generate_ts
#         tm = DriftModel()
#         tm.fit(train)
#         assert all(tm.drift_ == (train.iloc[-1] - train.iloc[0]) / len(train))
#         assert all(tm.last_value_ == train.iloc[-1])
#
#     def test_predict(self, generate_ts):
#         train, test = generate_ts
#         print(train, test)
#         tm = DriftModel()
#         tm.fit(train)
#         drift = tm.drift_
#         last = train.iloc[-1]
#         y_pred = tm.predict(test)
#         assert all(y_pred.iloc[2] == last + 2 * drift)
#
#
# class TestSeasonalNaive:
#     def test_fit(self, generate_ts):
#         train, _ = generate_ts
#         tm = SeasonalNaiveModel(seasonal_length=4)
#         tm.fit(train)
#         assert all(tm.last_value_["A"] == train.iloc[-tm.lag_:]["A"])
#
#     def test_predict(self, generate_ts):
#         train, test = generate_ts
#         print(train, test)
#         tm = SeasonalNaiveModel(seasonal_length=4)
#         tm.fit(train)
#         expected = pd.DataFrame(data=train.iloc[-4:-1], index=test.index)
#         assert all(tm.predict(test) == expected)