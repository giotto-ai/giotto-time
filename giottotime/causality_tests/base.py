from abc import ABCMeta, abstractmethod

import pandas as pd
from scipy import stats


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests.

    """

    def __init__(self, bootstrap_iterations, bootstrap_samples):
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_samples = bootstrap_samples

    @abstractmethod
    def fit(self, data_matrix):
        raise NotImplementedError  # to exclude from pytest coverage

    @abstractmethod
    def transform(self, time_series):
        raise NotImplementedError  # to exclude from pytest coverage

    def _compute_is_test_significant(self, data, x, y, best_shift):
        bootstrap_matrix = data.copy()
        bootstrap_matrix[y] = bootstrap_matrix.shift(best_shift)[y]
        bootstrap_matrix.dropna(axis=0, inplace=True)
        rhos = []

        for k in range(self.bootstrap_iterations):
            bootstraps = bootstrap_matrix.sample(n=len(data), replace=True)
            rhos.append(stats.pearsonr(bootstraps[x], bootstraps[y])[0])
        rhos = pd.DataFrame(rhos)

        percentile = stats.percentileofscore(rhos, 0) / 100
        p_value = 2 * (percentile if percentile < 0.5 else 1 - percentile)

        return p_value
