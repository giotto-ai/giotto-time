from copy import deepcopy

import pandas as pd
import pandas.util.testing as testing
from sklearn.linear_model import LinearRegression

from ..feature_creation.feature_creation import FeaturesCreation
from ..feature_creation.time_series_features import ShiftFeature, MovingAverageFeature, ExogenousFeature, CustomFeature
from ..feature_creation.utils import split_train_test


class AR:
    def __init__(self, horizon, base_model):
        self.horizon = horizon
        self.base_model = base_model
        self.model_per_timestep = [deepcopy(base_model) for _ in range(horizon)]

    def fit(self, X, y, **kwargs):
        self.model_per_timestep = [self.model_per_timestep[k].fit(X.values, y.iloc[:, k].values, **kwargs)
                                   for k in range(self.horizon)]
        return self

    def predict(self, X):
        predictions = pd.DataFrame(index=X.index)
        for k, model in enumerate(self.model_per_timestep):
            col_name = "{model_name}_{index}".format(model_name=model.__class__.__name__,
                                                     index=k)
            predictions[col_name] = model.predict(X)
        return predictions


def my_custom_function(time_series, window_size):
    return time_series.rolling(window_size).mean().shift(1)


if __name__ == "__main__":
    base_model = LinearRegression()
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_1 = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_2 = testing.makeTimeDataFrame(freq='MS')

    time_series_features = [MovingAverageFeature(2),
                            MovingAverageFeature(4),
                            ShiftFeature(-1),
                            CustomFeature(my_custom_function, window_size=4),
                            ExogenousFeature(x_exogenous_1, "test_ex1"),
                            ExogenousFeature(x_exogenous_2, "test_ex2")]

    horizon = 4
    feature_creation = FeaturesCreation(horizon, time_series_features)
    X, y = feature_creation.fit_transform(ts)

    X_train, y_train, X_test, y_test = split_train_test(X, y)
    ar = AR(horizon, base_model)
    ar.fit(X_train, y_train)

    predictions = ar.predict(X_test)
    print(predictions)
