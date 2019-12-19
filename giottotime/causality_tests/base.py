from abc import ABCMeta, abstractmethod


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests.

    """

    @abstractmethod
    def fit(self, data_matrix):
        pass

    @abstractmethod
    def transform(self, time_series):
        pass
