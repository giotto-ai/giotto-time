import pandas as pd

from giottotime.features.features_creation.base import TimeSeriesFeature


class ShiftFeature(TimeSeriesFeature):
    """
    Tentative Docstring
    """
    def __init__(self, shift):
        self.shift = shift

    def transform(self, time_series):
        return time_series.shift(self.shift)


class MovingAverageFeature(TimeSeriesFeature):
    def __init__(self, window_size):
        self.window_size = window_size

    def transform(self, time_series):
        return time_series.rolling(self.window_size).mean().shift(1)


class ConstantFeature(TimeSeriesFeature):
    def __init__(self, constant=1):
        self.constant = constant

    def transform(self, time_series):
        return pd.Series(data=self.constant, index=time_series.index)


class PolynomialFeature(TimeSeriesFeature):
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, time_series):
        pass


class ExogenousFeature(TimeSeriesFeature):
    def __init__(self, exogenous_time_series, name):
        self.exogenous_time_series = exogenous_time_series
        self.name = name

    def __repr__(self):
        return "{class_name}({name})".format(class_name=self.__class__.__name__,
                                             name=self.name)

    def transform(self, time_series):
        return self.exogenous_time_series.reindex(index=time_series.index)


class CustomFeature(TimeSeriesFeature):
    def __init__(self, custom_feature_function, **kwargs):
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def __repr__(self):
        return "{class_name}({function_name})".format(class_name=self.__class__.__name__,
                                                      function_name=self.custom_feature_function.__name__)

    def transform(self, time_series):
        return self.custom_feature_function(time_series, **self.kwargs)
