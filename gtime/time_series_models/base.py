from abc import ABCMeta
from sklearn.base import BaseEstimator, RegressorMixin

from gtime.compose import FeatureCreation
from gtime.model_selection import horizon_shift, FeatureSplitter


class TimeSeriesForecastingModel(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    def __init__(self, features: FeatureCreation, horizon: int, model: RegressorMixin):
        self.features = features
        self.horizon = horizon
        self.feature_splitter = FeatureSplitter()
        self.model = model

    def fit(self, X, y=None):
        X, y = self._create_X_y_feature_matrices(X, y)
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)
        self._fit_model(X_train, y_train)

        self.X_test_ = X_test

    def predict(self, X=None):
        if X is None:
            return self.model.predict(self.X_test_)
        else:
            raise NotImplementedError

    def _create_X_y_feature_matrices(self, X, y=None):
        feature_X = self.features.fit_transform(X, y)
        feature_y = horizon_shift(X, horizon=self.horizon)
        return feature_X, feature_y

    def _split_train_test(self, X, y):
        return self.feature_splitter.transform(X, y)

    def _fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
