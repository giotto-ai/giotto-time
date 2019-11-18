import pandas as pd

from .base import TimeSeriesFeature


class TimeSeriesFeature(ABCMeta):

    def __init__(self, polinomial_features):
        self.polinomial_features = polinomial_features

    def fit_transform(self, time_series):
        transformed_time_series = self._fit_transform(time_series)
        time_series_with_polinomial_features = self.add_polinomial_features_to(transformed_time_series)
        return time_series_with_polinomial_features

    @abstractmethod
    def _fit_transform(self, time_series):
        pass

    def add_polinomial_features_to(self, time_series):
        # do something ..
        return time_series


class ShiftFeature(TimeSeriesFeature):
    def __init__(self, shift):
        self.shift = shift

    def fit_transform(self, time_series):
        return time_series.shift(self.shift)


class MovingAverageFeature(TimeSeriesFeature):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit_transform(self, time_series):
        return time_series.rolling(self.window_size).mean().shift(1)


class ConstantFeature(TimeSeriesFeature):
    def __init__(self, constant=1):
        self.constant = constant

    def fit_transform(self, time_series):
        return pd.Series(data=self.constant, index=time_series.index)


class ExogenousFeature(TimeSeriesFeature):
    def __init__(self, exogenous_time_series, name):
        self.exogenous_time_series = exogenous_time_series
        self.name = name

    def __repr__(self):
        return "{class_name}({name})".format(class_name=self.__class__.__name__,
                                             name=self.name)

    def fit_transform(self, time_series):
        return self.exogenous_time_series.reindex(index=time_series.index)


class CustomFeature(TimeSeriesFeature):
    def __init__(self, custom_feature_function, **kwargs):
        self.custom_feature_function = custom_feature_function
        self.kwargs = kwargs

    def __repr__(self):
        return "{class_name}({function_name})".format(class_name=self.__class__.__name__,
                                                      function_name=self.custom_feature_function.__name__)

    def fit_transform(self, time_series):
        return self.custom_feature_function(time_series, **self.kwargs)


polinomial_features = PolynomialFeatures(first_order_features=[ShiftFeature(-1), ShiftFeature(-2), MovingAverageFeature(10)],
                                         order=2,
                                         interaction_only=True)

feature_combination = PipelineFeature(feature=[ShiftFeature(-1), CustomFeature(lambda x: np.atan(x))])

raw_time_series = ...
time_series_preparation = TimeSeriesPreparation()
time_series = time_series_preparation.fit_transform(raw_time_series)

shift_feature = ShiftFeature(-1)
custom_feature = CustomFeature(lambda_function)

feature_creation = FeaturesCreation(horizon=3, time_series_features=[shift_feature, custom_feature])
X, y = feature_creation.fit_transform(time_series)

X_train, y_train, X_test, y_test = split_train_test(X, y)

model = RegressorChain()
model.fit(X_train, y_train)
