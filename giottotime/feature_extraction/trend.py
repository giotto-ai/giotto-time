from abc import ABC
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from giottotime.base import FeatureMixin

__all__ = [
    "Detrender",
]

from giottotime.utils.trends import TRENDS


class Detrender(BaseEstimator, TransformerMixin, FeatureMixin):
    """Apply a de-trend transformation to a time series.

    Parameters
    ----------
    trend_model : TrendModel, optional, default: ``PolynomialTrend()``
        The kind of trend removal to apply.

    output_name : str, optional, default: ``'DetrendedFeature'``
        The name of the output column.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Detrender
    >>> from giottotime.models import PolynomialTrend
    >>> model = PolynomialTrend(order=2)
    >>> detrend_feature = Detrender(trend_model=model)
    >>> time_index = pd.date_range("2020-01-01", "2020-01-10")
    >>> X = pd.DataFrame(range(0, 10), index=time_index)
    >>> detrend_feature.transform(X)
                DetrendedFeature
    2020-01-01      2.092234e-06
    2020-01-02      6.590209e-07
    2020-01-03     -4.104701e-07
    2020-01-04     -1.116238e-06
    2020-01-05     -1.458284e-06
    2020-01-06     -1.436607e-06
    2020-01-07     -1.051207e-06
    2020-01-08     -3.020852e-07
    2020-01-09      8.107597e-07
    2020-01-10      2.287327e-06

    """

    def __init__(self, trend, trend_init, loss: Callable = mean_squared_error, method: str = "BFGS"):
        self.trend = trend
        self.trend_init = trend_init
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
            raise ValueError("The trend '%s' is not supported. Supported "
                             "trends are %s."
                             % (self.trend, list(sorted(TRENDS))))

        self.best_trend_params_ = minimize(
            lambda opt: self.loss(X.values, [TRENDS[self.trend](t, opt) for t in range(0, X.shape[0])]),
            self.trend_init,
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

        predictions = pd.Series(index=ts.index,
                                data=np.array([TRENDS[self.trend](t, self.best_trend_params_) for t in time_steps])
                                .flatten())

        return ts.sub(predictions, axis=0).add_suffix('__' + self.__class__.__name__)
