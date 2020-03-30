from copy import deepcopy
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from gtime.hierarchical.base import HierarchicalBase


class HierarchicalNaive(HierarchicalBase):
    """ Simplest hierarchical model possible.
    It does not perform any aggregation of the results.
    Each time series is fitted and predicted independently.

    Parameters
    ----------
    model: BaseEstimator, required
        time series forecasting model that is applied to each of the time series. A cross validation model
        can also be passed.
    Examples
    --------
    >>> import pandas._testing as testing
    >>> from gtime.time_series_models import AR
    >>> from gtime.hierarchical import HierarchicalNaive
    >>>
    >>> testing.N, testing.K = 20, 1
    >>> data1 = testing.makeTimeDataFrame(freq="s")
    >>> data2 = testing.makeTimeDataFrame(freq="s")
    >>> data = {'data1': data1, 'data2': data2}
    >>> time_series_model = AR(p=2, horizon=3)
    >>>
    >>> hierarchical_model = HierarchicalNaive(model=time_series_model)
    >>> hierarchical_model.fit(data)
    >>> hierarchical_model.predict()
    {'data1':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.475903  0.834633  0.649467
    2000-01-01 00:00:18  0.644168  0.610287  0.383904
    2000-01-01 00:00:19  0.180920  0.596606  0.696133, 'data2':                           y_1       y_2       y_3
    2000-01-01 00:00:17 -0.117342  0.006594 -0.638133
    2000-01-01 00:00:18 -0.394193 -0.607146  0.323875
    2000-01-01 00:00:19 -0.381479  0.088210 -0.356775}
    """

    def __init__(self, model: BaseEstimator):
        super().__init__(model=model, hierarchy_tree="infer")

    def fit(self, X: Dict[str, pd.DataFrame], y: pd.DataFrame = None):
        """ Fit method

        Parameters
        ----------
        X : Dict[str, pd.DataFrame], required
            A dictionary of time series. Each is fitted independently
        y : pd.DataFrame, optional, default = ``None``
            only for compatibility

        Returns
        -------
        self
        """
        self._check_is_dict_of_dataframes_with_str_key(X)
        self._infer_hierarchy_tree(X)
        self._initialize_models(X)
        for key, time_series in X.items():
            self.models_[key].fit(time_series)
        return self

    def predict(self, X: Dict[str, pd.DataFrame] = None):
        """ Predict method

        Parameters
        ----------
        X : Dict[str, pd.DataFrame], optional, default = ``None``
            time series to predict. If ``None`` all the fitted time series are predicted.
            The keys in ``X`` have to match the ones used to fit.

        Returns
        -------
        predictions : Dict[str, pd.DataFrame]
        """
        check_is_fitted(self)
        if X is None:
            return self._predict_fitted_time_series()
        else:
            return self._predict_new_time_series(X)

    def _initialize_models(self, X: Dict[str, pd.DataFrame]):
        print(self.model)
        self.models_ = {key: deepcopy(self.model) for key in X}

    def _infer_hierarchy_tree(self, X: Dict[str, pd.DataFrame]):
        self.hierarchy_tree_ = set(
            X.keys()
        )  # No need of a proper hierarchy tree for HierarchicalNaive

    def _predict_fitted_time_series(self) -> Dict[str, pd.DataFrame]:
        return {key: model.predict() for key, model in self.models_.items()}

    def _predict_new_time_series(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {
            key: self.models_[key].predict(time_series)
            for key, time_series in X.items()
        }
