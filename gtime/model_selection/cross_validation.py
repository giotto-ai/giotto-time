import numpy as np
import pandas as pd
from typing import Union, List

class TsCrossValidation:
   """Cross Validation for X and y by TimeSeries Split method.

   X and y are independent and dependent variable columns in array-like shape.

   """

   def __init__(self):
      super().__init__()

   def check_input(self, X, y, num_splits=3):
      if len(X) != len(y):
         raise ValueError(
            f"The arrays must have the same length, but they "
            f"have length {len(X)} and {len(y)}."
         )
      if np.isnan(X).any() or np.isnan(y).any():
         raise ValueError(
            "The two arrays should not contain Nan values, but they are "
            f"{X}, {y}."
         )

      if np.isinf(X).any() or np.isinf(y).any():
         raise ValueError(
            "The two arrays should not contain Inf values, but they are "
            f"{X}, {y}."
         )

      if num_splits >= len(X):
         raise ValueError(
            "The number of splits is greater of equal to number of samples"
      ) 

      if num_splits <= 0:
         raise ValueError(
            "The number of splits can not be zero or less than zero"
      )


   def convert_ndarray(self, X, y):
      if isinstance(X, pd.DataFrame):
         X = X.values
      elif isinstance(X, List):
         X = np.array(X)

      if isinstance(y, pd.DataFrame):
         y = y.values
      elif isinstance(y, List):
         y = np.array(y)

      return X, y


   def ts_split(self, X, y, num_splits=3):
      """
      Yet to edit the docstrings
      Examples
      --------
      >>> import pandas as pd
      >>> import numpy as np
      >>> from gtime.model_selection import FeatureSplitter
      >>> X = pd.DataFrame.from_dict({"feature_0": [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8],
      ...                             "feature_1": [np.nan, np.nan, 0.5, 1.5, 2.5, 3.5,
      ...                                            4.5, 5.5, 6.5, 7.5, ]
      ...                            })
      >>> y = pd.DataFrame.from_dict({"y_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      ...                             "y_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],
      ...                             "y_2": [2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan]
      ...                            })
      >>> feature_splitter = FeatureSplitter()
      >>> X_train, y_train, X_test, y_test = feature_splitter.transform(X, y)
      >>> X_train
         feature_0  feature_1
      2        1.0        0.5
      3        2.0        1.5
      4        3.0        2.5
      5        4.0        3.5
      6        5.0        4.5
      7        6.0        5.5
      >>> y_train
         y_0  y_1  y_2
      2    2  3.0  4.0
      3    3  4.0  5.0
      4    4  5.0  6.0
      5    5  6.0  7.0
      6    6  7.0  8.0
      7    7  8.0  9.0
      >>> X_test
         feature_0  feature_1
      8        7.0        6.5
      9        8.0        7.5
      >>> y_test
         y_0  y_1  y_2
      8    8  9.0  NaN
      9    9  NaN  NaN
      """
      # TODO : function to check the input parameters
      # in the same function check the integrity of the parameters such as num_splits is not > n_samples
      # TODO : function to format input into np arrays

      X, y = self.convert_ndarray(X, y)
      #self.check_input(X, y, num_splits=num_splits)
      num_parts = num_splits + 1 
      num_samples = len(X)
      num_range = np.arange(num_samples)
      chunk = num_samples // num_parts
      left_over_chunk = num_samples % (chunk)

      start = 0
      end = chunk
      test_start = chunk
      for itr in range(0, num_splits):
         train, test = X[start:end], y[test_start:(end+chunk)]
         test_start = end + chunk
         end += chunk   



   