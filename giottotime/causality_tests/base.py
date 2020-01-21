from abc import ABCMeta, abstractmethod

import pandas as pd
from scipy.stats import pearsonr


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests.

    """

    def __init__(self, bootstrap_iterations, bootstrap_samples, threshold):
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_samples = bootstrap_samples
        self.threshold = threshold

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
            bootstraps = bootstrap_matrix.sample(n=self.bootstrap_samples, replace=True)
            rhos.append(pearsonr(bootstraps[x], bootstraps[y])[0])
        rhos = pd.DataFrame(rhos)
        threshold = self.threshold / 2

        lq = rhos.quantile(threshold).iloc[0]
        uq = rhos.quantile(1 - threshold).iloc[0]

        return (0 < lq) or (0 > uq)
