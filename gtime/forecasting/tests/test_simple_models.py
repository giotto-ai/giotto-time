import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.model_selection import horizon_shift, FeatureSplitter

from gtime.forecasting import (
    NaiveModel,
    SeasonalNaiveModel,
    DriftModel,
    AverageModel
)


@st.composite
def forecast_input(draw, max_lenth):
    length = draw(st.integers(min_value=2, max_value=max_lenth))
    horizon = draw(st.integers(min_value=1, max_value=length - 1))
    X = draw(giotto_time_series(
        min_length=length,
        max_length=max_lenth,
        allow_nan=False,
        allow_infinity=False
    ))
    y = horizon_shift(X, horizon=horizon)
    X_train, y_train, X_test, y_test = FeatureSplitter().transform(X, y)
    return X_train, y_train, X_test


class TestNaiveModel:

    @given(data=forecast_input(50))
    def setup(self, data):
        X_train, y_train, X_test = data
        self.model = NaiveModel()
        self.model.fit(X_train, y_train)
        self.X_test = X_test

    def test_fit(self):
        assert self.model._horizon == len(self.X_test)

    def test_predict(self):
        horizon = len(self.X_test)
        y_pred = self.model.predict(self.X_test)
        y_cols = ['y_'+str(x+1) for x in range(len(self.X_test))]
        res = np.broadcast_to(self.X_test, (horizon, horizon))
        expected_df = pd.DataFrame(res, index=self.X_test.index, columns=y_cols)
        testing.assert_frame_equal(y_pred, expected_df)

