from gtime.forecasting import ARIMAForecaster
from sklearn.compose import make_column_selector
from gtime.feature_extraction import Shift
from gtime.time_series_models import TimeSeriesForecastingModel


class ARIMA(TimeSeriesForecastingModel):

    def __init__(self, horizon: int, order, method):
        features = [
            ("s1", Shift(0), make_column_selector()),
        ]
        super().__init__(features=features, horizon=horizon, model=ARIMAForecaster(order, method))