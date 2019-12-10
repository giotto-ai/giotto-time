from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class CausalityTest(BaseEstimator, TransformerMixin):
    """ Base class for causality tests. The children classes must implement
    the fit and transform methods.

    """

    @abstractmethod
    def fit(self, data_matrix):
        pass

    @abstractmethod
    def transform(self, time_series):
        pass
