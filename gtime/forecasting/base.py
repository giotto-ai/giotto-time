import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class BaseForecaster(BaseEstimator, RegressorMixin, metaclass=ABCMeta):

    """Base abstract class for simple models """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), train sample, required for compatibility, not used for a naive model.

        y : pd.DataFrame, Used to store the predict feature names and prediction horizon.

        Returns
        -------
        self : BaseForecaster
            Returns self.
        """

        self.y_columns_ = y.columns
        self.horizon_ = len(y.columns)

        return self

    @abstractmethod
    def _predict(self, X: pd.DataFrame):

        """Create a numpy array of predictions. A virtual method to be implemented in child classes.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.


        """

        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        """A wrapper to convert the result of ``_predict`` to a pd.DataFrame with appropriate indices.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, 1), required
            The time series on which to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, self._horizon)
            The output predictions.

        Raises
        ------
        NotFittedError
            Raised if the model is not fitted yet.

        """

        check_is_fitted(self)
        np_prediction = self._predict(X)
        predictions_df = pd.DataFrame(
            np_prediction, columns=self.y_columns_, index=X.index
        )
        return predictions_df
