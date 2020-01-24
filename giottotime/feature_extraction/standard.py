from typing import Optional

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "Shift",
    "MovingAverage",
    "Polynomial",
    "Exogenous",
]

from sklearn.utils.validation import check_is_fitted

from ..base import FeatureMixin


# TODO: retest examples
class Shift(BaseEstimator, TransformerMixin, FeatureMixin):
    """Perform a shift of a DataFrame of size equal to ``shift``.

    Parameters
    ----------
    shift : int, optional, default: ``1``
        How much to shift.

    Notes
    -----
    The ``shift`` parameter can also accept negative values. However, this should be
    used carefully, since if the resulting feature is used for training or testing it
    might generate a leak from the feature.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Shift
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> shift_feature = Shift(shift=3)
    >>> shift_feature.fit_transform(ts)
       ShiftFeature
    0           NaN
    1           NaN
    2           NaN
    3           0.0
    4           1.0
    5           2.0

    """

    def __init__(self, shift: int = 1):
        super().__init__()
        self.shift = shift

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
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Create a shifted version of ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame to shift.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            The shifted version of the original ``time_series``.

        """
        check_is_fitted(self)

        time_series_shifted = time_series.shift(self.shift).add_suffix('__' + self.__class__.__name__)
        return time_series_shifted


class MovingAverage(BaseEstimator, TransformerMixin, FeatureMixin):
    """For each row in ``time_series``, compute the moving average of the previous
     ``window_size`` rows. If there are not enough rows, the value is Nan.

    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the moving average

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import MovingAverage
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> mv_avg_feature = MovingAverage(window_size=2)
    >>> mv_avg_feature.fit_transform(ts)
       MovingAverageFeature
    0                   NaN
    1                   0.5
    2                   1.5
    3                   2.5
    4                   3.5
    5                   4.5

    """

    def __init__(self, window_size: int = 1):
        super().__init__()
        self.window_size = window_size

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
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Compute the moving average, for every row of ``time_series``, of the previous
        ``window_size`` elements.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving average

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``time_series``, containing the rolling
            moving average for each element.

        """
        check_is_fitted(self)

        time_series_mvg_avg = time_series.rolling(self.window_size).mean().add_suffix('__' + self.__class__.__name__)
        return time_series_mvg_avg


# TODO: use make_column_transformer instead
class Polynomial(BaseEstimator, TransformerMixin, FeatureMixin):
    """Compute the polynomial feature_extraction, of a degree equal to the input
    ``degree``.

    Parameters
    ----------
    degree: int, optional, default: ``2``
        The degree of the polynomial feature_extraction.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Polynomial
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> pol_feature = Polynomial(degree=3)
    >>> pol_feature.fit_transform(ts)
       pol_0  pol_1  pol_2  pol_3
    0    1.0    0.0    0.0    0.0
    1    1.0    1.0    1.0    1.0
    2    1.0    2.0    4.0    8.0
    3    1.0    3.0    9.0   27.0
    4    1.0    4.0   16.0   64.0
    5    1.0    5.0   25.0  125.0

    """

    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree

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
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Compute the polynomial feature_extraction of ``time_series``, up to a degree
        equal to ``degree``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            The computed polynomial feature_extraction.

        """
        check_is_fitted(self)

        pol_feature = PolynomialFeatures(self.degree)
        pol_of_X_array = pol_feature.fit_transform(time_series)
        pol_of_X = pd.DataFrame(pol_of_X_array, index=time_series.index).add_suffix('__' + self.__class__.__name__)

        return pol_of_X


class Exogenous(BaseEstimator, TransformerMixin, FeatureMixin):
    """Reindex ``exogenous_time_series`` with the index of ``time_series``. To check the
    documentation of ``pandas.DataFrame.reindex`` and to see which type of
    ``method`` are available, please refer to the pandas `documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html>`_.

    Parameters
    ----------
    exogenous_time_series : pd.DataFrame, shape (n_samples, 1), required
        The time series to reindex

    method : str, optional, default: ``None``
        The method used to re-index. This must be a method used by the
        ``pandas.DataFrame.reindex`` method.

    Examples
    --------
    >>> import pandas as pd
    >>> from giottotime.feature_extraction import Exogenous
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5], index=[3, 4, 5, 6, 7, 8])
    >>> exog_ts = pd.DataFrame([10, 8, 1, 3, 2, 7])
    >>> exog_feature = Exogenous(exog_ts)
    >>> exog_feature.fit_transform(ts)
       ExogenousFeature
    3               3.0
    4               2.0
    5               7.0
    6               NaN
    7               NaN
    8               NaN

    >>> exog_feature = Exogenous(exog_ts, method="nearest")
    >>> exog_feature.transform(ts)
       ExogenousFeature
    3                 3
    4                 2
    5                 7
    6                 7
    7                 7
    8                 7
    """

    def __init__(
            self,
            exogenous_time_series: pd.DataFrame,
            method: Optional[str] = None,
    ):
        super().__init__()
        self.method = method
        self.exogenous_time_series = exogenous_time_series

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
        self.columns_ = X.columns.values
        return self

    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Reindex the ``exogenous_time_series`` with the index of ``time_series``.

        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The input DataFrame. Used only for its index.

        Returns
        -------
        time_series_t :  pd.DataFrame, shape (n_samples, 1)
            The original ``exogenous_time_series``, re-indexed with the index
            of ``time_series``.

        """
        check_is_fitted(self)

        exog_feature = self.exogenous_time_series.reindex(
            index=time_series.index, method=self.method
        ).add_suffix('__' + self.__class__.__name__)

        return exog_feature
