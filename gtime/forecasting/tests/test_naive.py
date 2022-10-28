import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as testing
from hypothesis import given, note
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.model_selection import horizon_shift, FeatureSplitter

from gtime.forecasting import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    DriftForecaster,
    AverageForecaster,
)


@st.composite
def forecast_input(draw, max_lenth):
    length = draw(st.integers(min_value=2, max_value=max_lenth))
    horizon = draw(st.integers(min_value=1, max_value=length - 1))
    X = draw(
        giotto_time_series(
            min_length=length,
            max_length=max_lenth,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    y = horizon_shift(X, horizon=horizon)
    X_train, y_train, X_test, y_test = FeatureSplitter().transform(X, y)
    return X_train, y_train, X_test


class SimplePipelineTest:
    def setup(self, data, Model):
        X_train, y_train, X_test = data
        self.model = Model
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_pred = self.model.predict(X_test)

    def test_fit_horizon(self):
        assert self.model.horizon_ == len(self.X_test)

    def test_predict_shape(self):
        assert self.y_pred.shape == (self.model.horizon_, self.model.horizon_)


class TestNaiveModel(SimplePipelineTest):
    @given(data=forecast_input(50))
    def setup(self, data):
        super().setup(data, NaiveForecaster())

    def test_predict_df(self):
        horizon = len(self.X_test)
        y_cols = ["y_" + str(x + 1) for x in range(len(self.X_test))]
        res = np.broadcast_to(self.X_test, (horizon, horizon))
        expected_df = pd.DataFrame(res, index=self.X_test.index, columns=y_cols)
        testing.assert_frame_equal(self.y_pred, expected_df)


class TestSeasonalNaiveModel(SimplePipelineTest):
    @given(data=forecast_input(50), season_length=st.data())
    def setup(self, data, season_length):
        season_length = season_length.draw(
            st.integers(min_value=1, max_value=len(data[0]))
        )
        self.season_length = season_length
        super().setup(data, SeasonalNaiveForecaster(seasonal_length=season_length))

    def test_predict_seasonality(self):
        if self.season_length < self.model.horizon_:
            assert all(
                self.y_pred.iloc[:, 0] == self.y_pred.iloc[:, self.season_length]
            )


class TestDriftModel(SimplePipelineTest):
    @given(data=forecast_input(50))
    def setup(self, data):
        super().setup(data, DriftForecaster())

    def test_predict_drift(self):
        pytest.approx(self.y_pred.diff().diff().sum().sum())
        # assert pytest.approx(self.y_pred.diff().diff().sum().sum()) == 0


class TestAverageModel(SimplePipelineTest):
    @given(data=forecast_input(50))
    def setup(self, data):
        super().setup(data, AverageForecaster())

    def test_predict_difference(self):
        assert pytest.approx(self.y_pred.diff(axis=1).sum().sum()) == 0
