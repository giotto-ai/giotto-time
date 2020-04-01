import numpy as np
import pandas as pd
import math
from datetime import date, datetime, timedelta


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
    fold.index : list/lists of pandas indexes of folds
    time_fold.index : list/lists of pandas indexes of folds

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
   
   if split_on == 'index':
      n_set = n_samples // n_splits
      start = 0
      end = n_set
      for itr in range(n_splits - 1):
         fold = time_series[start:end]
         yield fold.index
         end += n_set
      last_fold = time_series[start:]

   elif split_on == 'time':
      if isinstance(time_series.index, pd.DatetimeIndex):
         start_date = time_series.index[0] 
         end_date = time_series.index[-1]
         n_days = len(time_series.resample('D'))
         fold_length = n_days // n_splits
         next_date = start_date + pd.Timedelta(days=fold_length)

         for split in range(n_splits - 1):
            time_fold = time_series[(time_series.index >= start_date) & (time_series.index < next_date)]
            yield time_fold.index
            next_date += pd.Timedelta(days=fold_length)
         last_time_fold = time_series[0:]
      else:
         raise ValueError(
            "The input parameter split_on is 'time' but the data does not have time index"
      )

   
   


   