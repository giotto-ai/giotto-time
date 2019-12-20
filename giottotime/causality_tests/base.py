from abc import ABCMeta, abstractmethod


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests.

    """

    @abstractmethod
    def fit(self, data_matrix):
        raise NotImplementedError  # to exclude from pytest coverage

    @abstractmethod
    def transform(self, time_series):
        raise NotImplementedError  # to exclude from pytest coverage
