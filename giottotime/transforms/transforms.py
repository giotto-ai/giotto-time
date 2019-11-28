import pandas as pd

from sklearn.metrics import mean_squared_error

from .base import TimeSeriesTransform
from ..trend_model.polynomial_trend import PolynomialTrend
from ..trend_model.polynomial_trend import ExponentialTrend

class TimeSeriesTransform(ABCMeta):

    def __init__(self):
        pass

    def transform(self, time_series):
        return 1

    @abstractmethod
    def _transform(self, time_series):
        pass

#

class DetrendedFeature(TimeSeriesFeature):
    def __init__(self, trend_model):
        self.trend_model = trend_model

    def transform(self, time_series):
        self.trend_model.fit(time_series)
        return self.trend_model.transform(time_series)

class RemovePolynomialTrend(DetrendedFeature):
    def __init__( self, polynomial_order=1, loss=mean_squared_error ):
        self.trend_model = PolynomialTrend(order=polynomial_order, loss=loss)
        super().__init__(trend_model=trend_model)

class RemoveExponentialTrend(DetrendedFeature):
    def __init__( self, loss=mean_squared_error ):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=trend_model)

class RemoveFunctionTrend(DetrendedFeature):
    def __init__( self, loss=mean_squared_error ):
        self.trend_model = ExponentialTrend(loss=loss)
        super().__init__(trend_model=trend_model)

#
