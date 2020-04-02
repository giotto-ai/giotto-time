import pandas as pd


def time_series_split(time_series: pd.DataFrame, n_splits=4, split_on='index'):
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
   Example 1)
   >>> date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
   >>> time_series = pd.DataFrame(date_rng, columns=['date'])
   >>> time_series.set_index('date', inplace=True)
   >>> time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))
   >>> for fold in (time_series_split(time_series, n_splits=4, split_on='time')):
   ...      print(fold)
   DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='date', freq=None)
   DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], 
                  dtype='datetime64[ns]', name='date', freq=None)
   DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                  '2018-01-05', '2018-01-06'],
                   dtype='datetime64[ns]', name='date', freq=None)
   DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                  '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                   dtype='datetime64[ns]', name='date', freq=None)

   Example 2)
   >>> df = pd.DataFrame(np.random.randint(0,100,size=(16, 4)), columns=list('ABCD'))
   >>> for fold in (time_series_split(df, n_splits=4, split_on='index')):
   ...      print(fold)
   RangeIndex(start=0, stop=4, step=1)
   RangeIndex(start=0, stop=8, step=1)
   RangeIndex(start=0, stop=12, step=1)
   RangeIndex(start=0, stop=16, step=1)

   """
   
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
      yield last_fold.index

   elif split_on == 'time':
      if isinstance(time_series.index, pd.DatetimeIndex):
         start_date = time_series.index[0] 
         end_date = time_series.index[-1]
         split_length = ((end_date - start_date) / n_splits)
         next_date = start_date + pd.Timedelta(split_length)

         for split in range(n_splits - 1):
            time_fold = time_series[(time_series.index >= start_date) & (time_series.index < next_date)]
            yield time_fold.index
            next_date += pd.Timedelta(split_length)
         last_time_fold = time_series[0:]
         yield last_time_fold.index

      else:
         raise ValueError(
            "The input parameter split_on is 'time' but the data does not have time index"
      )

   
   


   