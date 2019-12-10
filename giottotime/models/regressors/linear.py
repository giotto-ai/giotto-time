import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted


class LinearRegressor(BaseEstimator):
    """This class implements a LinearRegressor that takes a custom loss
    function and is able to fit the model over it.

    Parameters
    ----------
    loss : ``Callable``, optional, (default=``mean_squared_error``).
        The loss function to use when fitting the model. The loss function must
        accept y_true, y_pred and return a single real number.

    """

    def __init__(
        self, loss=mean_squared_error
    ):  # weight_initialization_rule = lambda X, y: np.zeros(X.shape[1]) ):
        self.loss = loss

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, disp: bool = False, **kwargs):
        """Fit the linear model on ``X`` and ``y`` on the given loss function.
        To do the minimization, the ``scipy.optimize.minimize`` function is
        used. To have more details and check which kind of options are
        available, please refer to the scipy `documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The X matrix used as features in the fitting procedure.

        y : ``pd.DataFrame``, required.
            The y matrix to use as target values in the fitting procedure.

        disp : ``bool``, optional, (default=``False``).
            Set to True to print convergence messages.

        kwargs: ``dict``, optional.
            Optional arguments to pass to the ``minimize`` function of scipy.

        Returns
        -------
        self:
            The fitted model.

        """

        def prediction_error(model_weights):
            predictions = [
                model_weights[0] + np.dot(model_weights[1:], row) for row in X.values
            ]
            return self.loss(y, predictions)

        if "r2_seed" in kwargs:
            lm = LinearRegression(fit_intercept=True).fit(X, y)
            self.r2_seed = [lm.intercept_] + list(lm.coef_)
            kwargs["x0"] = self.r2_seed
            print(kwargs["x0"])
            del kwargs["r2_seed"]

        if "x0" not in kwargs:
            kwargs["x0"] = np.zeros(
                X.shape[1] + 1
            )  # weight_initialization_rule(X, y) np.zeros(X.shape[1]+1)
        else:
            kwargs["x0"] = kwargs["x0"] + [0] * (X.shape[1] + 1 - len(kwargs["x0"]))

        res = minimize(prediction_error, options={"disp": disp}, **kwargs)

        self.model_weights_ = res["x"]

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the y values associated to the features ``X``.

        Parameters
        ----------
        X : ``pd.DataFrame``, required.
            The features used to predict.

        Returns
        -------
        predictions : ``pd.DataFrame``
            The predictions of the model

        """
        check_is_fitted(self, attributes=["model_weights_"])

        return self.model_weights_[0] + np.dot(X, self.model_weights_[1:])
