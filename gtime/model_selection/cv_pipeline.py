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

result_cols = ["Fit time", "Train score", "Test score"]


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
    if len(results) == 0:
        return None
    first_metric = results['Metric'].iloc[0]
    scores = results[results['Metric'] == first_metric]['Test score']
    best_model_index = scores.idxmin()
    return results.loc[best_model_index, 'Model']

def _models_are_equal(models, target):
    return [(model.model == target.model) & (model.features == target.features) & (model.horizon == target.horizon) for model in models]


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
        self.models = models_sets
        self.metrics = mse if metrics is None else metrics
        self.selection = _default_selection if selection is None else selection
        self.cv = blocking_time_series_split if blocking else time_series_split
        self.n_splits = n_splits
        result_idx = pd.MultiIndex.from_product([self.model_list, self.metrics.keys()])
        result_idx.names = ['Model', 'Metric']
        self.cv_results_ = pd.DataFrame(
            0.0, index=result_idx, columns=result_cols
        ).reset_index()


    def _fit_one_model(self, X_split: pd.DataFrame, model, results, only_model=False):
        start_time = time()
        model_index = results[_models_are_equal(results['Model'], model)].index
        model.cache_features = True
        model.fit(X_split, only_model=only_model)
        scores = model.score(metrics=self.metrics)
        results.loc[model_index, ["Train score", "Test score"]] = scores.values
        fit_time = time() - start_time
        results.loc[model_index, "Fit time"] = fit_time
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
        from gtime.time_series_models import TimeSeriesForecastingModel
        results = self.cv_results_.copy()
        for model, params in self.models_sets.items():
            if model == TimeSeriesForecastingModel:
                for feature in params['features']:
                    for horizon in params['horizon']:
                        submodel = model(features=feature, horizon=horizon, model=params['model'][0])
                        results = self._fit_one_model(X_split, submodel, results)
                        for next_model in params['model'][1:]:
                            submodel.set_model(next_model)
                            results = self._fit_one_model(X_split, submodel, results, only_model=True)
            else:
                model_list = list(filter(lambda x: isinstance(x, model), self.model_list))
                for submodel in model_list:
                    results = self._fit_one_model(X_split, submodel, results)
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
            self.cv_results_[result_cols] += self._cv_fit_one_split(X_split)[result_cols]

        self.cv_results_[result_cols] /= self.n_splits
        self.best_model_ = self.selection(self.cv_results_.dropna())
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


if __name__ == '__main__':
    from gtime.preprocessing import TimeSeriesPreparation
    # from gtime.model_selection import CVPipeline
    from gtime.metrics import rmse, mape
    from gtime.time_series_models import Naive, AR, TimeSeriesForecastingModel
    from gtime.forecasting import NaiveForecaster, DriftForecaster
    from gtime.feature_extraction import MovingAverage, Polynomial, Shift
    from sklearn.model_selection import ParameterGrid

    df_sp = pd.read_csv('https://storage.googleapis.com/l2f-open-models/giotto-time/examples/data/^GSPC.csv')
    df_sp.head()
    df_close = df_sp.set_index('Date')['Close']
    df_close.index = pd.to_datetime(df_close.index)
    time_series_preparation = TimeSeriesPreparation()
    period_index_time_series = time_series_preparation.transform(df_close)

    shift_feature = [('s3', Shift(1), ['time_series'])]
    ma_feature = [('ma10', MovingAverage(10), ['time_series'])]
    scoring = {'RMSE': rmse,
               'MAPE': mape}
    models = {
        TimeSeriesForecastingModel: {'features': [shift_feature, ma_feature],
                                     'horizon': [3, 5],
                                     'model': [NaiveForecaster(), DriftForecaster()]},
        Naive: {'horizon': [3, 5, 9]},
        AR: {'horizon': [3, 5, 7],
             'p': [2, 3, 4]}
    }

    c = CVPipeline(models_sets=models, metrics=scoring)
    c.fit(period_index_time_series)