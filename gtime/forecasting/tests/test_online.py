import pandas as pd
import numpy as np
from gtime.forecasting.online import HedgeForecaster


def test_hedge_fit_predict():
    time_index = pd.date_range("2020-01-01", "2020-01-20")
    X_np = np.concatenate((np.random.randint(4, size=(20, 2)), np.array([100] * 20).reshape(-1, 1)), axis=1)
    X = pd.DataFrame(X_np, index=time_index)
    y = pd.DataFrame(np.random.randint(4, size=(20, 1)), index=time_index, columns=["y_1"])
    hr = HedgeForecaster(random_state=42)

    preds = hr.fit_predict(X, y)
    np.testing.assert_equal(preds.shape, y.shape)
    np.testing.assert_almost_equal(hr.weights_[0], hr.weights_[1], decimal=2)
    assert hr.weights_[2] < hr.weights_[0]
    assert hr.weights_[2] < hr.weights_[1]
