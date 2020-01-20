from abc import ABCMeta, abstractmethod

import pandas as pd
from scipy.stats import pearsonr


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests.

    """

    def __init__(self, bootstrap_iterations, bootstrap_samples, confidence):
        self.confidence = confidence
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_iterations = bootstrap_iterations

    @abstractmethod
    def fit(self, data_matrix):
        raise NotImplementedError  # to exclude from pytest coverage

    @abstractmethod
    def transform(self, time_series):
        raise NotImplementedError  # to exclude from pytest coverage

    def _compute_is_test_significant(self, data, x, y):
        rhos = []
        for k in range(self.bootstrap_iterations):
            bootstraps = data.sample(n=self.bootstrap_samples, replace=True)
            rhos.append(pearsonr(bootstraps[x], bootstraps[y])[0])
        rhos = pd.DataFrame(rhos)
        lq = rhos.quantile(self.confidence / 2).iloc[0]
        uq = rhos.quantile(self.confidence / 2).iloc[0]
        return (0 < lq) or (0 > uq)
