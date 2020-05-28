from typing import Dict, Callable, Union, List, Any

import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import ParameterGrid, KFold
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

    Examples
    --------
    >>> from gtime.preprocessing import TimeSeriesPreparation
    >>> from gtime.time_series_models import CVPipeline
    >>> from gtime.metrics import rmse, mape
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gtime.time_series_models import Naive, AR, TimeSeriesForecastingModel
    >>> from gtime.forecasting import NaiveForecaster, DriftForecaster
    >>> from gtime.feature_extraction import MovingAverage, Polynomial, Shift
    >>> from sklearn.model_selection import ParameterGrid
    >>> idx = pd.period_range(start="2011-01-01", end="2012-01-01")
    >>> np.random.seed(5)
    >>> df = pd.DataFrame(np.random.standard_normal((len(idx), 1)), index=idx, columns=["time_series"])
    >>> shift_feature = [('s3', Shift(1), ['time_series'])]
    >>> ma_feature = [('ma10', MovingAverage(10), ['time_series'])]
    >>> scoring = {'RMSE': rmse, 'MAPE': mape}
    >>> models = {
    ...     TimeSeriesForecastingModel: {'features': [shift_feature, ma_feature],
    ...                                  'horizon': [3, 5],
    ...                                  'model': [NaiveForecaster(), DriftForecaster()]},
    ...     Naive: {'horizon': [3, 5, 9]},
    ...     AR: {'horizon': [3, 5],
    ...          'p': [2, 3]}
    ... }
    >>> c = CVPipeline(models_sets=models, metrics=scoring)
    >>> c.fit(df).predict()
                     y_1       y_2       y_3       y_4       y_5
    2011-12-28  0.025198  0.005753  0.041398  0.008531 -0.053772
    2011-12-29  0.024587  0.004619 -0.021253 -0.086931 -0.012732
    2011-12-30  0.000045  0.011903 -0.055153  0.007690  0.151219
    2011-12-31  0.025556  0.006280  0.071624  0.054207 -0.073940
    2012-01-01 -0.017229  0.018712 -0.000043  0.199268  0.219392

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
                model_index = model.__name__ + ": " + str(params)
                model_list[model_index] = model(**params)
        self.model_list = model_list
        self.metrics = {"MSE": mse} if metrics is None else metrics
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
        first_metric = results["Metric"].iloc[0]
        scores = results[results["Metric"] == first_metric]["Test score"]
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
            if (
                (model.model == target.model)
                & (model.features == target.features)
                & (model.horizon == target.horizon)
            ):
                return idx

    def _fit_one_model(
        self,
        model: TimeSeriesForecastingModel,
        X_split: pd.DataFrame,
        results: pd.DataFrame,
        only_model: bool = False,
    ) -> pd.DataFrame:
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
        results.loc[[model_index], ["Train score", "Test score"]] = scores.values
        fit_time = time() - start_time
        results.loc[model_index, "Fit time"] = fit_time
        return results

    def _fit_ts_forecaster_model(
        self,
        model: TimeSeriesForecastingModel,
        params: Dict,
        X_split: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
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
        for feature in params["features"]:
            for horizon in params["horizon"]:
                submodel = model(
                    features=feature, horizon=horizon, model=params["model"][0]
                )
                results = self._fit_one_model(submodel, X_split, results)
                for next_model in params["model"][1:]:
                    submodel.set_model(next_model)
                    results = self._fit_one_model(
                        submodel, X_split, results, only_model=True
                    )
        return results

    def _fit_other_models(
        self,
        model: TimeSeriesForecastingModel,
        X_split: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
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
        model_list = list(
            filter(lambda x: isinstance(x, model), self.model_list.values())
        )
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

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame = None, refit: Union[str, List] = "best"
    ):
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
        result_idx.names = ["Model", "Metric"]
        self.cv_results_ = (
            pd.DataFrame(0.0, index=result_idx, columns=result_cols)
            .reset_index()
            .set_index("Model")
        )

        for idx in self.cv(X, self.n_splits):
            X_split = X.loc[idx]
            self.cv_results_[result_cols] += self._cv_fit_one_split(X_split)[
                result_cols
            ]

        self.cv_results_[result_cols] /= self.n_splits
        self.best_model_ = self.model_list[self.selection(self.cv_results_.dropna())]

        if refit == "all":
            for model in self.model_list.values():
                model.fit(X)
        elif refit == "best":
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


CVSplitter = Union[Callable, KFold]


# {
#     'AR': {
#         'features': [('s1', Shift(1), ['NumOfContainers']), ('s2', Shift(2), ['NumOfContainers'])],
#         'models': [GAR(LinearRegression()), MultiFeatureGAR(RandomForest()), ...],
#         'horizon': 12
#     }
# }


class CrossValidationModel(TimeSeriesForecastingModel):
    results_index_names = ["Model", "Regressor", "Metric"]

    def __init__(
        self, time_series_models: Dict, cv: CVSplitter = None, metrics: Dict = None,
    ):
        super().__init__(features=None, horizon=None, model=None, cache_features=False)
        self.time_series_models = time_series_models
        self.cv = cv
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        results = []
        for model_name, model_params in self.time_series_models.items():
            self._check_and_set_model_params(model_params)
            model_results = self._cross_validate_time_series_model(
                X, y, model_params, model_name
            )
            results.append(model_results)
        self.results_ = pd.concat(results)

    def _cross_validate_time_series_model(
        self, X: pd.DataFrame, y: pd.DataFrame, model_params: Dict, model_name: str
    ) -> pd.DataFrame:
        model_results = []

        X, y, X_test, y_test = self._compute_train_test_matrices(X, y)
        for i, (train_index, validation_index) in enumerate(self.cv.split(X)):
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index, :]
            X_validation, y_validation = (
                X.iloc[validation_index, :],
                y.iloc[validation_index, :],
            )
            split_results = self._cross_validate_forecasters_on(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                model_params=model_params,
                model_name=model_name,
                split_num=i,
            )

            model_results.append(split_results)
        return pd.concat(model_results, axis=1)

    def _cross_validate_forecasters_on(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_validation: pd.DataFrame,
        y_validation: pd.DataFrame,
        model_params: Dict,
        model_name: str,
        split_num: int,
    ) -> pd.DataFrame:
        split_results = []
        for forecaster_name, forecaster in model_params["models"].items():
            forecaster.fit(X_train, y_train)
            y_predictions = forecaster.predict(X_validation)

            for metric_name, metric in self.metrics.items():
                error = metric(y_validation, y_predictions)
                if isinstance(error, np.ndarray):
                    error = np.mean(error)
                split_results.append(
                    self._result_row(
                        error=error,
                        model_name=model_name,
                        forecaster_name=forecaster_name,
                        metric_name=metric_name,
                        split_num=split_num,
                    )
                )
        return pd.concat(split_results)

    def _check_and_set_model_params(self, model_params: Dict[str, Any]):
        if "features" not in model_params:
            raise KeyError("features must be a key of model_params")
        if "horizon" not in model_params:
            raise KeyError("horizon must be a key of model_params")
        if "models" not in model_params:
            raise KeyError("models must be a key of model_params")
        self.features = model_params["features"]
        self.horizon = model_params["horizon"]

    def _result_row(
        self,
        error: float,
        model_name: str,
        forecaster_name: str,
        metric_name: str,
        split_num: int,
    ):
        index = pd.MultiIndex.from_tuples([(model_name, forecaster_name, metric_name)])
        index.names = self.results_index_names
        return pd.DataFrame(index=index, data={f"Split {split_num}": [error]})

    def predict(self, X: pd.DataFrame = None, **kwargs):
        pass
