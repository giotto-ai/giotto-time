import inspect
from abc import ABCMeta, abstractmethod
from typing import List, Union


StringOrList = Union[List[str], str]


def convert_to_list(variable: StringOrList) -> List[str]:
    return variable if isinstance(variable, list) else [variable]


class TimeSeriesTransformer(metaclass=ABCMeta):
    def __init__(
            self,
            input_columns: StringOrList,
            output_columns: StringOrList,
            drop_input_columns: bool = False
    ):
        self.input_columns = convert_to_list(input_columns)
        self.output_columns = convert_to_list(output_columns)
        self.drop_input_columns = drop_input_columns

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self):
        constructor_attributes = inspect.getfullargspec(self.__init__).args
        attributes_to_print = [str(getattr(self, attribute))
                               for attribute in constructor_attributes
                               if attribute in self.__dict__]
        return "{class_name}({attributes})".format(class_name=self.__class__.__name__,
                                                   attributes=", ".join(attributes_to_print))
