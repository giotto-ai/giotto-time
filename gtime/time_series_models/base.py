from typing import List, Tuple

from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from gtime.compose import FeatureCreation
from gtime.model_selection import horizon_shift, FeatureSplitter


class TimeSeriesForecastingModel(BaseEstimator, RegressorMixin):
    """ Base class for a generic time series forecasting model.

    Internally this class follows our approach for time series forecasting:
    - feature creation
    - train and test split
    - forecasting model training
    - prediction

    Parameters
    ----------

    features : List[Tuple]], required
        input of class FeatureCreation, which inherits from
        ``sklearn.compose.ColumnTransformer``. It is
        used internally to instantiate FeatureCreation
    horizon : int, required
        how many steps to predict in the future
    model: RegressorMixin, required
        forecasting model used for predictions

    Examples
    --------
    >>> import pandas._testing as testing
    >>> from sklearn.linear_model import LinearRegression
    >>> from gtime.feature_extraction import Shift, MovingAverage
    >>> from gtime.forecasting import GAR
    >>> from gtime.time_series_models import TimeSeriesForecastingModel
    >>>
    >>> testing.N, testing.K = 20, 1
    >>> data = testing.makeTimeDataFrame(freq="s")
    >>> features = [('s1', Shift(1), ['A']), ('ma3', MovingAverage(window_size=3), ['A'])]
    >>> lr = LinearRegression()
    >>> time_series_pipeline = TimeSeriesForecastingModel(features=features, horizon=3, model=GAR(lr))
    >>>
    >>> time_series_pipeline.fit(data)
    >>> time_series_pipeline.predict()
                              y_1       y_2       y_3
    2000-01-01 00:00:17  0.574204 -0.147355  0.449696
    2000-01-01 00:00:18  0.034620  0.308283 -0.113223
    2000-01-01 00:00:19  0.801922  0.178843  0.518739

    """
    def __init__(self, features: List[Tuple], horizon: int, model: RegressorMixin):
        self.feature_creation = FeatureCreation(features)
        self.horizon = horizon
        self.feature_splitter = FeatureSplitter()
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """ Fit function for a time series forecasting model.

        It does the following:
        - creates the X and y feature matrices
        - splits them into train and test
        - train the forecasting model on the train

        Parameters
        ----------
        X : pd.DataFrame, required
            input time series
        y : pd.DataFrame, optional, default: ``None``
            added for compatibility reasons with ``sklearn.compose.ColumnTransformer``

        Returns
        -------
        self
        """
        X, y = self._create_X_y_feature_matrices(X, y)
        X_train, y_train, X_test, y_test = self._split_train_test(X, y)

        self.model_ = self._fit_model(X_train, y_train)
        self.X_test_ = X_test

        return self

    def predict(self, X=None):
        """ Predict

        Parameters
        ----------
        X : pd.DataFrame, optional, default: ``None``
            time series to predict, optional. If not present, it predicts
            on the time series given as input in ``self.fit()``

        Returns
        -------
        predictions: pd.DataFrame
        """
        check_is_fitted(self)

        if X is None:
            return self.model_.predict(self.X_test_)
        else:
            X_test = self.feature_creation.transform(X)
            return self.model_.predict(X_test)

    def _create_X_y_feature_matrices(self, X, y=None):
        feature_X = self.feature_creation.fit_transform(X, y)
        feature_y = horizon_shift(X, horizon=self.horizon)
        return feature_X, feature_y

    def _split_train_test(self, X, y):
        return self.feature_splitter.transform(X, y)

    def _fit_model(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
