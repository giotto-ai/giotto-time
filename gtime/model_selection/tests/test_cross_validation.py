import pandas as pd
import numpy as np
import pytest
from pytest import fixture
from gtime.model_selection import time_series_split, blocking_time_series_split


@pytest.fixture
def time_series():
    date_rng = pd.date_range(start="1/1/2018", end="1/08/2018", freq="D")
    time_series = pd.DataFrame(date_rng, columns=["date"])
    time_series.set_index("date", inplace=True)
    time_series["data"] = np.random.randint(0, 100, size=(len(date_rng)))
    return time_series


@pytest.fixture
def non_time_index():
    time_series = pd.DataFrame()
    time_series["data"] = np.random.randint(0, 100, size=10)
    return time_series


@pytest.fixture
def period_index():
    pidx = pd.period_range(start="2005-12-21 08:45", end="2005-12-31 11:55", freq="D")
    time_series = pd.DataFrame(index=pidx)
    time_series["data"] = np.random.randint(0, 100, size=(len(pidx)))
    return time_series


@pytest.fixture
def timedelta_index():
    tidx = pd.timedelta_range(start="1 days 02:00:12.001124", periods=25, freq="D")
    time_series = pd.DataFrame(index=tidx)
    time_series["data"] = np.random.randint(0, 100, size=(len(tidx)))
    return time_series


class TestTimeSeriesSplit:
    def _correct_split_on_time_record_length(self, time_series):
        n_splits = 5
        time_series = time_series.index
        start_date = time_series[0]
        end_date = time_series[-1]
        split_length = (end_date - start_date) / n_splits
        next_date = start_date + pd.Timedelta(split_length)

        for split in range(n_splits - 1):
            time_fold = time_series[
                (time_series >= start_date) & (time_series < next_date)
            ]
            yield time_fold
            next_date += pd.Timedelta(split_length)
        last_time_fold = time_series[0:]
        yield last_time_fold

    def test_too_many_folds(self, time_series):

        with pytest.raises(ValueError):
            for element in time_series_split(
                time_series, n_splits=(len(time_series) + 1), split_on="index"
            ):
                pass

    def test_split_on_time_with_non_time_indexed_dataframe(self, non_time_index):
        # If the split_on is set to 'time' but index is not DateTime

        with pytest.raises(ValueError):
            for element in time_series_split(
                non_time_index, n_splits=5, split_on="time"
            ):
                pass

    def test_split_on_index(self, time_series):
        splits = 5
        split_length = len(time_series) // splits
        fold_length = split_length
        length_list = []

        for fold_index in time_series_split(
            time_series, n_splits=splits, split_on="index"
        ):
            length_list.append(len(fold_index))
        for index_length in range(len(length_list) - 1):
            assert fold_length == length_list[index_length]
            fold_length += split_length
        last_fold_length = len(time_series)

        assert last_fold_length == length_list[-1]

    def test_split_on_time(self, time_series):
        record_count = []
        for fold_index in time_series_split(time_series, n_splits=5, split_on="time"):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_split_on_time_record_length(time_series):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])

    def test_split_on_neither_time_nor_index(self, time_series):

        with pytest.raises(ValueError):
            for element in time_series_split(time_series, n_splits=5, split_on="abc"):
                pass

    def test_period_index(self, period_index):
        record_count = []
        for fold_index in time_series_split(period_index, n_splits=5, split_on="time"):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_split_on_time_record_length(period_index):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])

    def test_timedelta_index(self, timedelta_index):
        record_count = []
        for fold_index in time_series_split(
            timedelta_index, n_splits=5, split_on="time"
        ):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_split_on_time_record_length(timedelta_index):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])


class TestBlockingTimeSeriesSplit:
    def _correct_btss_split_on_time_record_length(self, time_series):
        n_splits = 5
        time_series = time_series.index
        start_date = time_series[0]
        end_date = time_series[-1]
        split_length = (end_date - start_date) / n_splits
        next_date = start_date + pd.Timedelta(split_length)

        for split in range(n_splits - 1):
            time_fold = time_series[
                (time_series >= start_date) & (time_series < next_date)
            ]
            yield time_fold
            start_date = next_date
            next_date += pd.Timedelta(split_length)
        last_time_fold = time_series[(time_series >= start_date)]
        yield last_time_fold

    def test_btss_too_many_folds(self, time_series):

        with pytest.raises(ValueError):
            for element in blocking_time_series_split(
                time_series, n_splits=(len(time_series) + 1), split_on="index"
            ):
                pass

    def test_btss_split_on_time_with_non_time_indexed_dataframe(self, non_time_index):
        # If the split_on is set to 'time' but index is not DateTime

        with pytest.raises(ValueError):
            for element in blocking_time_series_split(
                non_time_index, n_splits=5, split_on="time"
            ):
                pass

    def test_btss_split_on_index(self, time_series):
        splits = 5
        split_length = len(time_series) // splits
        fold_length = split_length
        length_list = []

        for fold_index in blocking_time_series_split(
            time_series, n_splits=splits, split_on="index"
        ):
            length_list.append(len(fold_index))
        for index_length in range(len(length_list) - 1):
            assert fold_length == length_list[index_length]
        last_fold_length = split_length + (len(time_series) % splits)

        assert last_fold_length == length_list[-1]

    def test_btss_split_on_time(self, time_series):
        record_count = []
        for fold_index in blocking_time_series_split(
            time_series, n_splits=5, split_on="time"
        ):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_btss_split_on_time_record_length(time_series):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])

    def test_btss_split_on_neither_time_nor_index(self, time_series):

        with pytest.raises(ValueError):
            for element in blocking_time_series_split(
                time_series, n_splits=5, split_on="abc"
            ):
                pass

    def test_btss_period_index(self, period_index):
        record_count = []
        for fold_index in blocking_time_series_split(
            period_index, n_splits=5, split_on="time"
        ):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_btss_split_on_time_record_length(period_index):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])

    def test_btss_timedelta_index(self, timedelta_index):
        record_count = []
        for fold_index in blocking_time_series_split(
            timedelta_index, n_splits=5, split_on="time"
        ):
            record_count.append(len(fold_index))
        correct_record_count = []
        for fold_index in self._correct_btss_split_on_time_record_length(
            timedelta_index
        ):
            correct_record_count.append(len(fold_index))

        assert all([a == b for a, b in zip(correct_record_count, record_count)])
