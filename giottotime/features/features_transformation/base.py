import inspect
from abc import ABCMeta, abstractmethod
from typing import List, Union


ColumnNames = Union[List[str], str]

class TimeSeriesTransformer(metaclass=ABCMeta):
    def __init__(self, input_columns: ColumnNames, output_columns: ColumnNames,
                 drop_input_columns: bool = False):
        self._check_input(input_columns, output_columns, drop_input_columns)
        self.input_columns = input_columns
        if isinstance(input_columns, str):
            self.input_columns = [input_columns]
        self.output_columns = output_columns
        if isinstance(input_columns, str):
            self.input_columns = [input_columns]
        self.drop_input_columns = drop_input_columns

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _check_input(self, input_columns: ColumnNames,
                     output_columns: ColumnNames, drop_input_columns: bool):
        self._check_columns(input_columns, 'input_columns')
        self._check_columns(output_columns, 'output_columns')
        self._check_drop_input_columns(drop_input_columns)

    def _check_columns(self, columns: ColumnNames, attribute_name: str):
        if isinstance(columns, str):
            return
        elif isinstance(columns, list):
            if len(columns) == 0:
                raise ValueError(f'{attribute_name} is empty')
        else:
            raise TypeError(f'{attribute_name} must be a string or a list')

    def _check_drop_input_columns(self, drop_input_columns: bool):
        if not isinstance(drop_input_columns, bool):
            raise TypeError('drop_input_columns must be a boolean')

    def __repr__(self):
        constructor_attributes = inspect.getfullargspec(self.__init__).args
        attributes_to_print = [str(getattr(self, attribute))
                               for attribute in constructor_attributes
                               if attribute in self.__dict__]
        return "{class_name}({attributes})".format(class_name=self.__class__.__name__,
                                                   attributes=", ".join(attributes_to_print))
