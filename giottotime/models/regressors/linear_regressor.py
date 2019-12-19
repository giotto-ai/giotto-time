import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from sklearn.utils.validation import check_is_fitted


class LinearRegressor:
    """Implementation of a LinearRegressor that takes a custom lossfunction and is able
     to fit the model over it.

    Parameters
    ----------
    loss : Callable, optional, default: ``mean_squared_error``
        The loss function to use when fitting the model. The loss function must accept
        y_true, y_pred and return a single real number.

    Examples
    --------
    >>> from giottotime.models.regressors.linear_regressor import LinearRegressor
    >>> from giottotime.loss_functions import max_error
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.random.random((100, 10))
    >>> y = np.random.random(100)
    >>> lr = LinearRegressor(loss=max_error)
    >>> X_train, y_train = X[:90], y[:90]
    >>> X_test, y_test = X[90:], y[90:]
    >>> x0 = [0]*11
    >>> lr.fit(X_train, y_train, x0=x0)
    >>> y_pred = lr.predict(X_test)

    """

    def __init__(self, loss=mean_squared_error):
        self.loss = loss

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> "LinearRegressor":
        """Fit the linear model on ``X`` and ``y`` on the given loss function.To do the
        minimization, the ``scipy.optimize.minimize`` function is used. To have more
        details and check which kind of options are available, please refer to the scipy
        `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The X matrix used as features in the fitting procedure.

        y : pd.DataFrame, shape (n_samples, 1), required
            The y matrix to use as target values in the fitting procedure.

        kwargs: dict, optional.
            Optional arguments to pass to the ``minimize`` function of scipy.

        Returns
        -------
        self: LinearRegressor
            The fitted model.

        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.DataFrame):
            y = y.values

        def prediction_error(model_weights):
            predictions = [
                model_weights[0] + np.dot(model_weights[1:], row) for row in X
            ]
            return self.loss(y, predictions)

        res = minimize(prediction_error, **kwargs)

        self.model_weights_ = res["x"]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the y values associated to the features ``X``.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            The features used to predict.

        Returns
        -------
        predictions : pd.DataFrame, shape (n_samples, 1)
            The predictions of the model

        """
        check_is_fitted(self)

        predictions = self.model_weights_[0] + np.dot(X, self.model_weights_[1:])
        return predictions
