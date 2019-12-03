from sklearn.metrics import mean_squared_error

from giottotime.feature_creation.base import TimeSeriesFeature
from giottotime.models.trend_models.polynomial_trend import PolynomialTrend
from giottotime.models.trend_models.exponential_trend import ExponentialTrend


class DetrendedFeature(TimeSeriesFeature):
    def __init__(self, trend_model, output_name: str):
        self.trend_model = trend_model
        self.output_name = output_name

    def transform(self, time_series):
        self.trend_model.fit(time_series)
        return self.trend_model.transform(time_series)


class RemovePolynomialTrend(DetrendedFeature):
    def __init__(self, polynomial_order=1, loss=mean_squared_error):
        self.trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=self.trend_model)


class RemoveExponentialTrend(DetrendedFeature):
    def __init__(self, loss=mean_squared_error):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model)


class RemoveFunctionTrend(DetrendedFeature):
    def __init__(self, loss=mean_squared_error):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=self.trend_model)

#
