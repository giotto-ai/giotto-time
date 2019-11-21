from typing import List, Union

from .base import TimeSeriesTransformer


ColumnNames = Union[List[str], str]


class ColumnsProduct(TimeSeriesTransformer):

    def __init__(
            self,
            input_columns: ColumnNames,
            output_columns: ColumnNames
    ):
        super().__init__(input_columns, output_columns, False)
        self.output_column = output_columns[0]

    def transform(self, X):
        return X[self.input_columns].prod(axis=1)

    def _check_input(self, input_columns: List[str], output_columns: List[str],
                     drop_input_columns: bool):
        super(self)._check_input(input_columns, output_columns,
                                 self.drop_input_columns)
        if len(output_columns) != 1:
            raise ValueError('output_columns contains more than one column')


