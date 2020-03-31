import numpy as np
import pandas as pd
import math


def time_series_split(time_series, n_splits=4, split_on='index'):
   """
   Split the input DataFrame into n_splits. If the data is not a timeries then the split
   is based on the number of samples.
   If the data is is a timeseries then divide the time series into months, days or years.

   Note: The split is based on the index, if split_on is timeseries then the data will be split 
   based on time

   Parameters
    ----------
    time_series : pandas DataFrame, shape (n_samples,), required
    The dataframe should have datetime as index if it is a timeseries data

    n_splits : int, default = 4, required
    The number of splits/folds on the dataset

    split_on : 'index', default parameter
    If the index is a datetime then the dataset will be split based on time

    Yields
    -------
    fold.index : list/lists of pandas indexes as folds

   Examples
   --------
   
   """
   # TODO : Write Tests
   # TODO : Examples in Doctrings
   
   n_samples = len(time_series)
   if n_samples < n_splits:
      raise ValueError(
         "The number of splits is greater than number of samples"
      )

   month_count = time_series.resample('M').first().dropna().shape[0]
   day_count = time_series.resample('D').first().dropna().shape[0]
   hour_count = time_series.resample('H').first().dropna().shape[0]

   n_set = n_samples // n_splits
   start = 0
   end = n_set
   if isinstance(time_series.index, pd.DatetimeIndex):
      n_set = month_count // n_splits
      end = n_set 
      if isinstance((month_count / n_splits), float) and (month_count < n_splits):
         n_set = day_count // n_splits
         end = n_set
         if (n_set == 0) and (hour_count > 1):
            n_set = hour_count // n_splits
            end = n_set
      else:
         n_set = math.floor(month_count / n_splits)
         end = n_set

   for itr in range(n_splits - 1):
      fold = time_series[start:end]
      yield fold.index
      end += n_set
   last_fold = time_series[start:]
   


   