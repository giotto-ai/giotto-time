import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from gtime.compose import FeatureCreation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer
from gtime.metrics import mse, max_error
from gtime.model_selection.cross_validation import time_series_split, blocking_time_series_split
from joblib import Parallel, delayed

from sklearn.utils.validation import check_is_fitted

from gtime.feature_extraction import Shift
from gtime.time_series_models import Naive, AR


# def _fit_one(model, X, y, horizon, model_params=None):
#     start_time = time()
#     model = model(horizon=horizon) if model_params == None else model(model_params)
#     # model.
#     score = model.score(X.loc[test], y.loc[test])
#     fit_time = time() - start_time
#     return model, score, fit_time

def _default_selection(results):
    first_metric = results.index.levels[1][0]
    scores = results.loc[pd.IndexSlice[:, first_metric], 'Test score']
    best_model = scores.argmin()[0]
    return best_model


class CVPipeline(BaseEstimator, RegressorMixin):

    def __init__(self, models_sets, features=None, n_splits=4, blocking=True, metrics=None, selection=None, n_jobs=-1):

        self.models_sets = models_sets
        model_list = []
        for model, param_grid in models_sets.items():
            param_iterator = ParameterGrid(param_grid)
            for params in param_iterator:
                model_list.append(model(**params))
        self.model_list = model_list
        self.n_jobs = n_jobs
        # self.features = None if features is None else FeatureCreation(features)
        self.metrics = mse if metrics is None else metrics
        self.selection = _default_selection if selection is None else selection
        self.cv = blocking_time_series_split if blocking else time_series_split
        self.n_splits = n_splits


    def fit(self, X, y=None):


        result_idx = pd.MultiIndex.from_product([self.model_list, self.metrics.keys()])
        results = pd.DataFrame(np.nan, index=result_idx, columns=['Fit time', 'Train score', 'Test score'])
        # self.models_ = dict(zip(self.model_list, [np.nan] * len(self.model_list)))
        for idx in self.cv(X, self.n_splits):
            X_split = X.loc[idx]
            for model in self.model_list:
                start_time = time()
                model.cache_features = True
                model.fit(X_split)
                fit_time = time() - start_time
                scores = model.score(metrics=self.metrics)
                results.loc[(model, self.metrics), ['Train score', 'Test score']] = scores.values
                results.loc[(model, self.metrics), 'Fit time'] = fit_time

        self.cv_results_ = results
        self.best_model_ = self.selection(results)
        print('AAA')

    def predict(self, X=None):
        check_is_fitted(self)
        return self.best_model_.predict(X)


if __name__ == '__main__':
    idx = pd.period_range(start='2011-01-01', end='2012-01-01')
    np.random.seed(2)
    df = pd.DataFrame(np.random.random((len(idx), 1)), index=idx, columns=['1'])
    scoring = {'MSE': mse,
               'Max': max_error}
    feat = [('s1', Shift(1), [0])]
    models = {
        Naive: {'horizon': [3, 5, 9]},
        AR: {'horizon': [3, 5, 7],
             'p': [2, 3, 4]}
    }
    c = CVPipeline(models_sets=models, metrics=scoring, features=feat)
    c.fit(df)
    print('AAA')