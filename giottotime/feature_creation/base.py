import inspect
from abc import ABCMeta, abstractmethod


class TimeSeriesFeature(metaclass=ABCMeta):
    @abstractmethod
    def fit_transform(self, time_series):
        pass

    def __repr__(self):
        constructor_attributes = inspect.getfullargspec(self.__init__).args
        attributes_to_print = [str(getattr(self, attribute)) for attribute in constructor_attributes
                               if attribute in self.__dict__]
        return "{class_name}({attributes})".format(class_name=self.__class__.__name__,
                                                   attributes=", ".join(attributes_to_print))

