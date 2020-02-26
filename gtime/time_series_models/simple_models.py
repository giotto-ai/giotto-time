import pandas as pd
from gtime.compose import FeatureCreation
from sklearn.compose import make_column_selector
from gtime.feature_extraction import Shift, MovingAverage, MovingCustomFunction
from gtime.time_series_models import TimeSeriesForecastingModel
from gtime.forecasting import NaiveModel, SeasonalNaiveModel, DriftModel, AverageModel


class NaiveForecastModel(TimeSeriesForecastingModel):

    def __init__(self, horizon: int):
        features = FeatureCreation(
            [('s1', Shift(0), make_column_selector()),
            ])
        super().__init__(features=features, horizon=horizon, model=NaiveModel())


class AverageForecastModel(TimeSeriesForecastingModel):

    def __init__(self, horizon: int):
        features = FeatureCreation(
            [('s1', Shift(0), make_column_selector()),
            ])
        super().__init__(features=features, horizon=horizon, model=AverageModel())


class SeasonalNaiveForecastModel(TimeSeriesForecastingModel):

    def __init__(self, horizon: int, seasonal_length: pd.Timedelta):
        features = FeatureCreation(
            [('s1', Shift(0), make_column_selector()),
            ])
        super().__init__(features=features, horizon=horizon, model=SeasonalNaiveModel(seasonal_length))


class DriftForecastModel(TimeSeriesForecastingModel):

    def __init__(self, horizon: int):
        features = FeatureCreation(
            [('s1', Shift(0), make_column_selector()),
            ])
        super().__init__(features=features, horizon=horizon, model=DriftModel())