from typing import Dict, Callable, Union, List

import pandas as pd
from time import time
from sklearn.model_selection import ParameterGrid
from gtime.time_series_models import TimeSeriesForecastingModel
from sklearn.base import BaseEstimator, RegressorMixin
from gtime.metrics import mse
from gtime.model_selection.cross_validation import (
    time_series_split,
    blocking_time_series_split,
)
from sklearn.utils.validation import check_is_fitted


result_cols = ["Fit time", "Train score", "Test score"]


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
        model_list = {}
        for model, param_grid in models_sets.items():
            param_iterator = ParameterGrid(param_grid)
            for params in param_iterator:
                model_index = model.__name__ + ': ' + str(params)
                model_list[model_index] = model(**params)
        self.model_list = model_list
        self.metrics = mse if metrics is None else metrics
        self.selection = self._default_selection if selection is None else selection
        self.cv = blocking_time_series_split if blocking else time_series_split
        self.n_splits = n_splits

    @staticmethod
    def _default_selection(results: pd.DataFrame) -> str:
        """
        Selects a model with lowest test score according to the first of the provided metrics

        Parameters
        ----------
        results: pd.DataFrame - cross-validation results

        Returns
        -------
        best_model_index: str, model index

        """
        if len(results) == 0:
            return None
        first_metric = results['Metric'].iloc[0]
        scores = results[results['Metric'] == first_metric]['Test score']
        best_model_index = scores.idxmin()
        return best_model_index

    def _models_are_equal(self, target: TimeSeriesForecastingModel) -> str:
        """
        Finds a model in ``self.model_list`` based on its horizon and features and returns its index

        Parameters
        ----------
        target: BaseEstimator, a target model

        Returns
        -------
        idx: str, model index in ``self.model_list``
        """
        for idx, model in self.model_list.items():
            if (model.model == target.model) & (model.features == target.features) & (model.horizon == target.horizon):
                return idx
        return None

    def _fit_one_model(self, model: BaseEstimator, X_split: pd.DataFrame, results: pd.DataFrame, only_model: bool = False) -> pd.DataFrame:
        """
        Fits one model on a split and calculates its score and fit time

        Parameters
        ----------
        model: BaseEstimator, model to fit
        X_split: pd.DataFrame, subset of training data to fit on
        results: pd.DataFrame, results dataframe to add score results
        only_model: bool, use ``only_model`` property to reuse the fitted features

        Returns
        -------

        """
        start_time = time()
        model_index = self._models_are_equal(model)
        model.cache_features = True
        model.fit(X_split, only_model=only_model)
        scores = model.score(metrics=self.metrics)
        results.loc[model_index, ["Train score", "Test score"]] = scores.values
        fit_time = time() - start_time
        results.loc[model_index, "Fit time"] = fit_time
        return results

    def _fit_ts_forecaster_model(self, model: TimeSeriesForecastingModel, params: Dict, X_split: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and score for a TimeSeriesForecastingModel model with different parameters

        Parameters
        ----------
        model: BaseEstimator, model to fit
        params: Dict, model parameters dictionary
        X_split: pd.DataFrame, subset of training data to fit on
        results: pd.DataFrame, results dataframe to add score results

        Returns
        -------
        results: pd.DataFrame

        """
        for feature in params['features']:
            for horizon in params['horizon']:
                submodel = model(features=feature, horizon=horizon, model=params['model'][0])
                results = self._fit_one_model(submodel, X_split, results)
                for next_model in params['model'][1:]:
                    submodel.set_model(next_model)
                    results = self._fit_one_model(submodel, X_split, results, only_model=True)
        return results

    def _fit_other_models(self, model: BaseEstimator, X_split: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and score for a model with pre-defined features to a split.

        Parameters
        ----------
        model: BaseEstimator, model to fit
        X_split: pd.DataFrame, subset of training data to fit on
        results: pd.DataFrame, results dataframe to add score results

        Returns
        -------
        results: pd.DataFrame

        """
        model_list = list(filter(lambda x: isinstance(x, model), self.model_list.values()))
        for submodel in model_list:
            results = self._fit_one_model(submodel, X_split, results)
        return results

    def _cv_fit_one_split(self, X_split: pd.DataFrame) -> pd.DataFrame:
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
        for model, params in self.models_sets.items():
            if model == TimeSeriesForecastingModel:
                results = self._fit_ts_forecaster_model(model, params, X_split, results)
            else:
                results = self._fit_other_models(model, X_split, results)
        return results

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, refit: Union[str, List] = 'best'):
        """
        Performs cross-validation, selecting the best model from ``self.model_list`` according to ``self.selection``
        and refits all the models on all available data.

        Parameters
        ----------
        X: pd.DataFrame, input time series
        y: pd.DataFrame, left for compatibility, not used
        refit: Union[str, List], models to refit on whole train data, ``all``, ``best`` or a list of model keys

        Returns
        -------
        self: CVPipeline

        """

        result_idx = pd.MultiIndex.from_product([self.model_list, self.metrics.keys()])
        result_idx.names = ['Model', 'Metric']
        self.cv_results_ = pd.DataFrame(
            0.0, index=result_idx, columns=result_cols
        ).reset_index().set_index('Model')

        for idx in self.cv(X, self.n_splits):
            X_split = X.loc[idx]
            self.cv_results_[result_cols] += self._cv_fit_one_split(X_split)[result_cols]

        self.cv_results_[result_cols] /= self.n_splits
        self.best_model_ = self.model_list[self.selection(self.cv_results_.dropna())]

        if refit == 'all':
            for model in self.model_list.values():
                model.fit(X)
        elif refit == 'best':
            self.best_model_.fit(X)
        else:
            for idx in refit:
                self.model_list[idx].fit(X)
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

