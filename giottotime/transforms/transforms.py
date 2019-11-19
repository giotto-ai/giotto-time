import pandas as pd

from .base import TimeSeriesTransform

class TimeSeriesTransform(ABCMeta):

    def __init__(self):
        pass

    def transform(self, time_series):
        return 1

    @abstractmethod
    def _transform(self, time_series):
        pass

#

class RemoveLinearTrend(TimeSeriesFeature):
    def __init__(self, a, b):
        pass

    def transform(self, time_series):
        return time_series

class RemovePolynomialTrend(TimeSeriesFeature):
    def __init__(self, coeffs=[]):
        pass

    def transform(self, time_series):
        return time_series

class RemoveExponentialTrend(TimeSeriesFeature):
    def __init__(self, exponent):
        pass

    def transform(self, time_series):
        return time_series


#
