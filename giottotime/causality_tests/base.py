import inspect
from abc import ABCMeta, abstractmethod


class CausalityTest(metaclass=ABCMeta):
    """ Base class for causality tests. The children classes must implement
    the fit and transform methods.

    Both the data_matrix dataframe must be passed in the fit method.
    Both the time_series dataframe must be passed in the transform method.
    """
    @abstractmethod
    def fit(self, data_matrix):
        pass

    @abstractmethod
    def transform(self, time_series):
        pass

    def __repr__(self):
        constructor_attributes = inspect.getfullargspec(self.__init__).args
        attributes_to_print = [str(getattr(self, attribute))
                               for attribute in constructor_attributes
                               if attribute in self.__dict__]
        return f"{self.__class__.__name__}({', '.join(attributes_to_print)})"
