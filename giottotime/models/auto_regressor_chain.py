import random
from itertools import chain, combinations

import numpy as np
import pandas as pd
import pandas.util.testing as testing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import ExogenousFeature, MovingAverageFeature, ShiftFeature
from giottotime.feature_creation.utils import split_train_test


def sliding_window_cv(X, y, train_size=15, test_size=10, n_splits=5):
    for split in range(n_splits):
        start_train_index = random.randint(0, len(X) - (train_size + test_size))
        start_test_index = start_train_index + train_size
        
        X_train_window = X.iloc[start_train_index: start_test_index, :]
        y_train_window = y.iloc[start_train_index: start_test_index]

        X_test_window = X.iloc[start_test_index: start_test_index + test_size, :]
        y_test_window = y.iloc[start_test_index: start_test_index + test_size]

        yield X_train_window, y_train_window, X_test_window, y_test_window
        

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


def create_model_name_from_dict(base_model, dictionary):
    model_name = base_model().__class__.__name__
    parameter_strings = model_name + "-" + "-".join(["_".join([str(key), str(dictionary[key])])
                                                     for key in dictionary.keys()])
    return parameter_strings


def extract_mean_std(cv_dataframe, pred_step):
    cv_dataframe_at_pred_step = cv_dataframe[cv_dataframe["pred_step"] == pred_step]
    grouped_by_model_name = cv_dataframe_at_pred_step.groupby("model_name")\
        .agg({'score': ['mean', 'std']})\
        .sort_values(by=[('score', 'mean')], ascending=False)
    grouped_by_model_name["pred_step"] = pred_step

    return grouped_by_model_name


def extract_best_model_from_cv(dataframe_with_mean_std, model_names):
    best_model_name = dataframe_with_mean_std.index[0]
    best_model_configuration = model_names[best_model_name]

    best_model = best_model_configuration[0]
    best_model_params = best_model_configuration[1]

    return best_model, best_model_params


class AutoRegressorChain:
    def __init__(self, feed_forward=False):
        self.feed_forward = feed_forward
        self.cv_matrix = None

    @property
    def cv_matrix(self):
        if self.__cv_matrix is None:
            raise ValueError("Not set")
        return self.__cv_matrix

    @cv_matrix.setter
    def cv_matrix(self, cv_matrix):
        self.__cv_matrix = cv_matrix

    def cv_model_selection(self, X, y, model_dictionary, split_type='sliding', n_splits=5, **kwargs):
        features = X
        pred_steps = y.shape[1]
        print(self.cv_matrix)
        best_models = []
        best_dictionaries = {}
        all_results = []

        for pred_step in range(pred_steps):
            print("Prediction step {} of {}".format(pred_step + 1, pred_steps))
            target_y = y.iloc[:, pred_step]

            cv_index = 0
            model_names = {}
            results_per_pred_step = []
            for x_train_cv, y_train_cv, x_test_cv, y_test_cv in sliding_window_cv(features, target_y):
                for model_dict in model_dictionary:
                    base_model = model_dict["model"]
                    model_params = model_dict["params"]
                    valid_combinations = construct_model_param_dictionary(model_params)
                    for combination in valid_combinations:
                        dictionary = {key: value for key, value in combination}
                        model_name = create_model_name_from_dict(base_model, dictionary)
                        model_names[model_name] = [base_model, dictionary]
                        model = base_model(**dictionary)
                        model.fit(x_train_cv, y_train_cv, **kwargs)
                        model_score = model.score(x_test_cv, y_test_cv)
                        results_per_pred_step.append({"score": model_score,
                                                      "model_name": model_name,
                                                      "n_split": cv_index,
                                                      "pred_step": pred_step})
                cv_index += 1
            dataframe_with_result = pd.DataFrame(results_per_pred_step)
            results_with_mean_std = extract_mean_std(dataframe_with_result, pred_step)
            all_results.append(results_with_mean_std)
            best_model, best_model_params = extract_best_model_from_cv(results_with_mean_std, model_names)
            print(results_with_mean_std)
            # best_model_fit = best_model(**best_model_params)
            best_dictionaries[pred_step] = (best_model, best_model_params)

            """
            if self.feed_forward:
                predictions = best_model_per_step.predict(features)
                dataframe_pred = pd.DataFrame(index=features.index)
                dataframe_pred["y_pred_" + str(pred_step)] = predictions
                features = pd.concat([features, dataframe_pred], axis=1)
            """
        self.cv_matrix = pd.concat(all_results, axis=0)
        print(self.cv_matrix)
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
                                                                    "max_features": [1/3, 1/2]}
                         },
                        {"model": SVR, "params": {"kernel": ["linear", "poly", "rbf"],
                                                  "gamma": ["auto"]}
                         }
                        ]

    a_chain = AutoRegressorChain(feed_forward=False)
    a_chain.cv_model_selection(X_t, y_t, model_dictionary)
