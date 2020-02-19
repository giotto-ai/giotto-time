import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted



class NaiveForecaster(BaseEstimator, RegressorMixin):
    """Naive forecasting model.
    TDB

    """

    def __init__(

        self,
    ):
        """
        Not needed here really but let it be for compatibility
        """
        self.next_value_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "NaiveForecaster":
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self.next_value_ = X.iloc[-1]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, 1)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """
        check_is_fitted(self)

        predictions = pd.DataFrame(data=self.next_value_, index=X.index)
        return predictions


class SeasonalNaiveForecaster(NaiveForecaster):

    """Seasonal naive forecasting model.
    TDB

    """

    def __init__(

        self,
        seasonal_length = 1,
    ):
        """
        Not needed here really but let it be for compatibility
        """
        super().__init__()
        self.lag = seasonal_length

    def fit(self, X: pd.DataFrame, y=None) -> "SeasonalNaiveForecaster":
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self.next_value_ = X.iloc[-self.lag:]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, 1)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """
        check_is_fitted(self)

        len_x = len(X)
        y_pred = list(self.next_value_.values) * (len_x // self.lag) + list(self.next_value_.values)[:(len_x % self.lag)]

        predictions = pd.DataFrame(data=y_pred, index=X.index)

        return predictions


class MovingAverageForecaster(BaseEstimator, RegressorMixin):
    """Naive forecasting model.
    TDB

    """

    def __init__(

        self,
        window: int,
    ):
        """
        Not needed here really but let it be for compatibility
        """

        self.window = window
        self.next_value_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "MovingAverageForecaster":
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self.next_value_ = X.iloc[-self.window:].mean()

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Using the fitted polynomial, predict the values starting from ``X``.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, 1)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)

        predictions = pd.DataFrame(data=self.next_value_.values, index=X.index, columns=X.columns)
        return predictions