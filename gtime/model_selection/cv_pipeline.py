from typing import Dict, Callable

import pandas as pd
from time import time
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, RegressorMixin
from gtime.metrics import mse
from gtime.model_selection.cross_validation import (
    time_series_split,
    blocking_time_series_split,
)
from sklearn.utils.validation import check_is_fitted


def _default_selection(results: pd.DataFrame) -> RegressorMixin:
    """
    Selects a model with lowest test score according to the first of the provided metrics

    Parameters
    ----------
    results: pd.DataFrame - cross-validation results

    Returns
    -------
    best_model: RegressorMixin - selected model

    """
    first_metric = results.index.levels[1][0]
    scores = results.loc[pd.IndexSlice[:, first_metric], "Test score"]
    best_model = scores.argmin()[0]
    return best_model


class CVPipeline(BaseEstimator, RegressorMixin):
    """
    Cross-validation for models of ``time_series_models`` classes

    Parameters
    ----------
    models_sets: Dict, a dictionary with models as keys and model parameter grid dictionaries as values
    n_splits: int, number of intervals for cross-validation
    blocking: bool, whether to perform a basic time series split or a blocking one
    metrics: Dict, a dictionary with metric names as keys and metric functions as values
    selection: Callable, a function to select the best model given score table

    """

    def __init__(
        self,
        models_sets: Dict,
        n_splits: int = 4,
        blocking: bool = True,
        metrics: Dict = None,
        selection: Callable = None,
    ):

        self.models_sets = models_sets
        model_list = []
        for model, param_grid in models_sets.items():
            param_iterator = ParameterGrid(param_grid)
            for params in param_iterator:
                model_list.append(model(**params))
        self.model_list = model_list
        self.metrics = mse if metrics is None else metrics
        self.selection = _default_selection if selection is None else selection
        self.cv = blocking_time_series_split if blocking else time_series_split
        self.n_splits = n_splits
        result_idx = pd.MultiIndex.from_product([self.model_list, self.metrics.keys()])
        self.cv_results_ = pd.DataFrame(
            0.0, index=result_idx, columns=["Fit time", "Train score", "Test score"]
        )

    def _cv_fit_one(self, X_split: pd.DataFrame) -> pd.DataFrame:
        """
        Fits all models from ``self.model_list`` on a provided time series, splitting it to train and test and calculating fir time

        Parameters
        ----------
        X_split: pd.DataFrame, input time series

        Returns
        -------
        results: pd.DataFrame, results table

        """
        results = self.cv_results_.copy()
        for model in self.model_list:
            start_time = time()
            model.cache_features = True
            model.fit(X_split)
            fit_time = time() - start_time
            scores = model.score(metrics=self.metrics)
            results.loc[
                (model, self.metrics), ["Train score", "Test score"]
            ] = scores.values
            results.loc[(model, self.metrics), "Fit time"] = fit_time
        return results

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Performs cross-validation, selecting the best model from ``self.model_list`` according to ``self.selection``
        and refits all the models on all available data.

        Parameters
        ----------
        X: pd.DataFrame, input time series
        y: pd.DataFrame, left for compatibility, not used

        Returns
        -------
        self: CVPipeline

        """
        for idx in self.cv(X, self.n_splits):
            X_split = X.loc[idx]
            self.cv_results_ += self._cv_fit_one(X_split)

        self.cv_results_ = self.cv_results_ / self.n_splits
        self.best_model_ = self.selection(self.cv_results_)
        for model in self.model_list:
            model.fit(X)
        return self

    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Predicting with selected ``self.best_model_``

        Parameters
        ----------
        X: pd.DataFrame, optional, default: ``None``
            time series to predict, optional. If not present, it predicts
            on the time series given as input in ``self.fit()``

        Returns
        -------
        predictions: pd.DataFrame
        """
        check_is_fitted(self)
        return self.best_model_.predict(X)
