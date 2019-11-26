import inspect
from abc import ABCMeta, abstractmethod


class TimeSeriesFeature(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, output_name):
        self.output_name = output_name

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def __repr__(self):
        constructor_attributes = inspect.getfullargspec(self.__init__).args
        attributes_to_print = [str(getattr(self, attribute)) for attribute in constructor_attributes
                               if attribute in self.__dict__]
        return "{class_name}({attributes})".format(class_name=self.__class__.__name__,
                                                   attributes=", ".join(attributes_to_print))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
