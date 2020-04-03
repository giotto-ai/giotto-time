import pandas as pd 
import numpy as np
import pytest 
from gtime.model_selection import time_series_split


class TestTimeSeriesSplit:
    def _correct_split_on_time_record_length(self):
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        time_series = pd.DataFrame(date_rng, columns=['date'])
        time_series.set_index('date', inplace=True)
        time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))
        n_splits = 5
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

    def test_too_many_folds(self):
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        time_series = pd.DataFrame(date_rng, columns=['date'])
        time_series.set_index('date', inplace=True)
        time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))

        with pytest.raises(ValueError):
            for element in time_series_split(time_series, n_splits=(len(time_series)+1), split_on='index'):
                pass

    def test_split_on_time_with_non_time_indexed_dataframe(self):
        # If the split_on is set to 'time' but index is not DateTime
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        time_series = pd.DataFrame(date_rng, columns=['date'])
        time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))

        with pytest.raises(ValueError):
            for element in time_series_split(time_series, n_splits=5, split_on='time'):
                pass

    def test_split_on_index(self):
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        time_series = pd.DataFrame(date_rng, columns=['date'])
        time_series.set_index('date', inplace=True)
        time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))

        splits = 5
        split_length = len(time_series) // splits
        fold_length = split_length
        length_list = []
        
        for fold_index in time_series_split(time_series, n_splits=splits, split_on='index'):
            length_list.append(len(fold_index))
        for index_length in range(len(length_list)-1):
            assert fold_length == length_list[index_length]
            fold_length += split_length 
        last_fold_length = len(time_series)

        assert last_fold_length == length_list[-1]

    def test_split_on_time(self):
        date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='D')
        time_series = pd.DataFrame(date_rng, columns=['date'])
        time_series.set_index('date', inplace=True)
        time_series['data'] = np.random.randint(0,100,size=(len(date_rng)))

        record_count = []
        for fold_index in time_series_split(time_series, n_splits=5, split_on='time'):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_split_on_time_record_length():
            correct_record_count.append(len(fold_index))
        
        assert all([a == b for a, b in zip(correct_record_count, record_count)])
    