from copy import deepcopy

import pandas as pd
import pandas.util.testing as testing
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import MovingAverageFeature, ShiftFeature, ExogenousFeature
from giottotime.feature_creation.utils import split_train_test


class RegressorChain:
    def __init__(self, base_model, horizon, feed_forward=False):
        self.horizon = horizon
        self.model_per_predstep = [deepcopy(base_model) for _ in range(horizon)]
        self.feed_forward = feed_forward

    def fit(self, X, y, **kwargs):
        features = X

        for pred_step in range(self.horizon):
            target_y = y.iloc[:, pred_step]
            self.model_per_predstep[pred_step].fit(features, target_y, **kwargs)
            if self.feed_forward:
                predictions = self.model_per_predstep[pred_step].predict(features)
                dataframe_pred = pd.DataFrame(index=features.index)
                dataframe_pred["y_pred_" + str(pred_step)] = predictions
                features = pd.concat([features, dataframe_pred], axis=1)

        return self

    def predict(self, X, **kwargs):
        """
        predictions = pd.DataFrame(index=X.index)

        for pred_step in range(len(self.model_per_predstep)):
            model_predictions = self.model_per_predstep[pred_step].predict(X, **kwargs)
            predictions[f"pred_{pred_step}"] = model_predictions

        return predictions
        """
        pass


if __name__ == "__main__":
    base_model = LinearRegression()
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_1 = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_2 = testing.makeTimeDataFrame(freq='MS')

    time_series_features = [MovingAverageFeature(2),
                            MovingAverageFeature(4),
                            ShiftFeature(-1),
                            ExogenousFeature(x_exogenous_1, "test_ex1"),
                            ExogenousFeature(x_exogenous_2, "test_ex2")]

    horizon_main = 4
    feature_creation = FeaturesCreation(horizon_main, time_series_features)
    X, y = feature_creation.fit_transform(ts)
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    reg_chain = RegressorChain(base_model, horizon_main)
    reg_chain.fit(X_train, y_train)
    pred = reg_chain.predict(X_test)
    print(pred)