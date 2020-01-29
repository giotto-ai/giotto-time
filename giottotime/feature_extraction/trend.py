from abc import ABC
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from giottotime.base import FeatureMixin

__all__ = "Detrender"

from giottotime.utils.trends import TRENDS


# TODO: improve doc, trends params specifications
class Detrender(BaseEstimator, TransformerMixin, FeatureMixin):
    """Apply a de-trend transformation to a time series.

    Parameters
    ----------
    trend : string,
        The kind of trend removal to apply.
        Supported trends: ['polynomial', 'exponential']

    trend_x0 : np.array,
        Initialisation parameters passed to the trend function

    loss : Callable,
        Loss function

    method : string,
        Loss function optimisation method

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giottotime.feature_extraction import Detrender
    >>> detrender = Detrender(trend='polynomial', trend_x0=np.zeros(2))
    >>> time_index = pd.date_range("2020-01-01", "2020-01-10")
    >>> X = pd.DataFrame(range(0, 10), index=time_index)
    >>> detrender.fit_transform(X)
                0__Detrender
    2020-01-01  9.180937e-07
    2020-01-02  8.020709e-07
    2020-01-03  6.860481e-07
    2020-01-04  5.700253e-07
    2020-01-05  4.540024e-07
    2020-01-06  3.379796e-07
    2020-01-07  2.219568e-07
    2020-01-08  1.059340e-07
    2020-01-09 -1.008878e-08
    2020-01-10 -1.261116e-07

    """

    def __init__(
        self,
        trend: str,
        trend_x0: np.array,
        loss: Callable = mean_squared_error,
        method: str = "BFGS",
    ):
        self.trend = trend
        self.trend_x0 = trend_x0
        self.loss = loss
        self.method = method

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """

        # TODO: create validation function
        if self.trend not in TRENDS:
            raise ValueError(
                "The trend '%s' is not supported. Supported "
                "trends are %s." % (self.trend, list(sorted(TRENDS)))
            )

        self.best_trend_params_ = minimize(
            lambda opt: self.loss(
                X.values, [TRENDS[self.trend](t, opt) for t in range(0, X.shape[0])]
            ),
            self.trend_x0,
            method=self.method,
            options={"disp": False},
        )["x"]

        self.t0_ = X.index[0]
        freq = X.index.freq
        if freq is not None:
            self.period_ = freq
        else:
            self.period_ = X.index[1] - X.index[0]

        return self

    def transform(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Transform the ``time_series`` by removing the trend.

        Parameters
        ----------
        ts: pd.DataFrame, shape (n_samples, 1), required
            The time series to transform.

        Returns
        -------
        ts_t : pd.DataFrame, shape (n_samples, n_features)
            The transformed time series, without the trend.

        """
        check_is_fitted(self)

        time_steps = (ts.index - self.t0_) / self.period_

        predictions = pd.Series(
            index=ts.index,
            data=np.array(
                [TRENDS[self.trend](t, self.best_trend_params_) for t in time_steps]
            ).flatten(),
        )

        return ts.sub(predictions, axis=0).add_suffix("__" + self.__class__.__name__)
