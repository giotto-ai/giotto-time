from hypothesis import given, settings
import hypothesis.strategies as st
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.time_series_models.arima import ARIMA


class TestARIMA:
    @given(x=giotto_time_series(
            min_length=20,
            max_length=100,
            allow_nan=False,
            allow_infinity=False,
            ),
           order=st.tuples(st.integers(min_value=0, max_value=4),
                           st.integers(min_value=0, max_value=2),
                           st.integers(min_value=0, max_value=4)),
            horizon=st.integers(min_value=1, max_value=10)
        )
    @settings(deadline=None)
    def test_arima_oos_forecast(self, x, order, horizon):
        model = ARIMA(order=order, method='css', horizon=horizon)
        model.fit(x)
        x_test_oos = x.iloc[[-1]]
        y_pred = model.predict(x_test_oos)
        assert y_pred.shape == (1, horizon)



