from itertools import chain, combinations

import numpy as np

import pandas as pd
import pandas.util.testing as testing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import random

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import ExogenousFeature, MovingAverageFeature, ShiftFeature
from giottotime.feature_creation.utils import split_train_test


def sliding_window_cv(time_series, train_size, test_size, random_index=True):

    random.randint(0, len(time_series))
    return []

def shuffle_cv():
    return []


split_functions = {"sliding": sliding_window_cv,
                   "shuffle": shuffle_cv}



def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def construct_model_param_dictionary(parameters):
    tuple_dictionary = []

    for key in parameters.keys():
        for value in parameters[key]:
            tuple_dictionary.append((key, value))

    valid_combinations = []
    for i, combo in enumerate(_powerset(tuple_dictionary), 1):
        if len(combo) == len(parameters):
            if _check_no_repetitions(combo):
                print(combo)
                valid_combinations.append(combo)

    return valid_combinations


def _check_no_repetitions(tuple_list):
    elems = [x[0] for x in tuple_list]
    return len(np.unique(elems)) == len(tuple_list)


class AutoRegressorChain:
    def __init__(self, feed_forward=False):
        self.feed_forward = feed_forward

    def cv_model_selection(self, X, y, model_dictionary, split_type='sliding', n_splits=5, **kwargs):
        features = X
        pred_steps = y.shape[1]

        best_models = []
        best_dictionaries = []
        results = []

        for pred_step in range(pred_steps):
            print("Prediction step {} of {}".format(pred_step + 1, pred_steps))
            target_y = y.iloc[:, pred_step]

            cv_index = 0
            for x_train_cv, y_train_cv, x_test_cv, y_test_cv in split_functions[split_type](features, target_y):
                for model_dict in model_dictionary:
                    base_model = model_dict["model"]
                    model_params = model_dict["params"]
                    valid_combinations = construct_model_param_dictionary(model_params)
                    for combination in valid_combinations:
                        dictionary = {key: value for key, value in combination}
                        model = base_model(**dictionary)
                        model.fit(x_train_cv, y_train_cv, **kwargs)
                        model_score = model.score(x_test_cv, y_test_cv)
                        results.append({"score": model_score,
                                        "model_name": model.__class__.__name__,
                                        "n_split": cv_index,
                                        "params": dictionary})
                cv_index += 1
            dataframe_with_result = pd.DataFrame(results)
            print(dataframe_with_result)

            if self.feed_forward:
                predictions = best_model_per_step.predict(features)
                dataframe_pred = pd.DataFrame(index=features.index)
                dataframe_pred["y_pred_" + str(pred_step)] = predictions
                features = pd.concat([features, dataframe_pred], axis=1)


        self.best_models = best_models
        print(best_dictionaries)


if __name__ == "__main__":
    testing.N, testing.K = 200, 1

    ts = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_1 = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_2 = testing.makeTimeDataFrame(freq='MS')

    time_series_features = [MovingAverageFeature(2),
                            MovingAverageFeature(4),
                            ShiftFeature(-1),
                            ExogenousFeature(x_exogenous_1, "test_ex1"),
                            ExogenousFeature(x_exogenous_2, "test_ex2")]

    horizon_main = 4
    feature_creation = FeaturesCreation(horizon_main, time_series_features)
    X, y = feature_creation.fit_transform(ts)

    X_t, y_t, X_te, y_te = split_train_test(X, y)

    model_dictionary = [{"model": RandomForestRegressor, "params": {"n_estimators": [1000],
                                                                    "max_depth": [None, 10],
                                                                    "max_features": [1/3, 1/2]}}
                        ]

    a_chain = AutoRegressorChain(model_dictionary, feed_forward=False)
    a_chain.cv_model_selection(X_t, y_t)
