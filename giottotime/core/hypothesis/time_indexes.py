from typing import Optional, Tuple, Dict, Union

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis._strategies import defines_strategy
from hypothesis.extra.numpy import arrays

from .utils import freq_to_timedelta, initialize_start_date_end_date, \
    order_pair, initialize_start_timedelta_end_timedelta, \
    expected_index_length_from, expected_start_date_from, \
    expected_end_date_from

IndexRangeArgs = Dict[str, Union[pd.datetime, int, pd.Timedelta]]

string_freqs = ['B', 'D', 'W', 'M', 'Q', 'A', 'Y', 'H', 'T', 'S']
pandas_range_params = ['start', 'end', 'periods', 'freq']


@defines_strategy
@st.composite
def series_with_period_index(draw,
                             start: Optional[pd.datetime] = None,
                             end: Optional[pd.datetime] = None,
                             max_length: int = 1000):
    """ Returns a strategy to generate a Pandas Series with PeriodIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    max_length : int, optional, (default=None)

    Returns
    -------
    pd.Series with PeriodIndex
    """
    index = draw(period_indexes(start, end, max_length))
    values = draw(arrays(dtype=np.float64, shape=index.shape[0]))
    return pd.Series(index=index, data=values)


@defines_strategy
@st.composite
def series_with_datetime_index(draw,
                               start: Optional[pd.datetime] = None,
                               end: Optional[pd.datetime] = None,
                               max_length: int = 1000):
    """ Returns a strategy to generate a Pandas Series with DatetimeIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    max_length : int, optional, (default=None)

    Returns
    -------
    pd.Series with DatetimeIndex
    """
    index = draw(datetime_indexes(start, end, max_length))
    values = draw(arrays(dtype=np.float64, shape=index.shape[0]))
    return pd.Series(index=index, data=values)


@defines_strategy
@st.composite
def series_with_timedelta_index(draw,
                                start: Optional[pd.Timedelta] = None,
                                end: Optional[pd.Timedelta] = None,
                                max_length: int = 1000):
    """ Returns a strategy to generate a Pandas Series with Timedelta index

    Parameters
    ----------
    draw
    start : ``pd.Timedelta``, optional, (default=None)
    end : ``pd.Timedelta``, optional, (default=None)
    max_length : int, optional, (default=None)

    Returns
    -------
    pd.Series with TimedeltaIndex
    """
    index = draw(timedelta_indexes(start, end, max_length))
    values = draw(arrays(dtype=np.float64, shape=index.shape[0]))
    return pd.Series(index=index, data=values)


@defines_strategy
@st.composite
def period_indexes(draw,
                   start: Optional[pd.datetime] = None,
                   end: Optional[pd.datetime] = None,
                   max_length: int = 1000):
    """ Returns a strategy to generate Pandas PeriodIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    max_length : int, optional, (default=1000)

    Returns
    -------
    LazyStrategy that generates pandas PeriodIndex
    """
    period_range_args = draw(period_range_args_from(start, end, max_length))
    assume(period_range_args_are_correct(period_range_args))
    return compute_period_range(period_range_args)


@defines_strategy
@st.composite
def datetime_indexes(draw,
                     start: Optional[pd.datetime] = None,
                     end: Optional[pd.datetime] = None,
                     max_length: int = 1000):
    """ Returns a strategy to generate Pandas DatetimeIndex.

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    max_length : int, default: ``1000``

    Returns
    -------
    LazyStrategy that generates pd.DatetimeIndex
    """
    date_range_args = draw(date_range_args_from(start, end, max_length))
    assume(date_range_args_are_correct(date_range_args))
    return compute_date_range(date_range_args)


@defines_strategy
@st.composite
def timedelta_indexes(draw,
                      start: Optional[pd.Timedelta] = None,
                      end: Optional[pd.Timedelta] = None,
                      max_length: int = 1000):
    """ Returns a strategy to generate Pandas TimedeltaIndex

    Parameters
    ----------
    draw
    start : ``pd.Timedelta``, optional, (default=None)
    end : ``pd.Timedelta``, optional, (default=None)
    max_length : int, (default=1000)

    Returns
    -------
    LazyStrategy that generates pd.TimedeltaIndex
    """
    timedelta_range_args = draw(timedelta_range_args_from(start, end, max_length))
    assume(timedelta_range_args_are_correct(timedelta_range_args))
    return compute_timedelta_range(timedelta_range_args)


@defines_strategy
def pair_of_ordered_dates(start: Optional[pd.datetime] = None,
                          end: Optional[pd.datetime] = None):
    """ Returns an hypothesis strategy that generates a pair of ordered
    pd.datetime. Useful to create a Pandas index

    Parameters
    ----------
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)

    Returns
    -------
    LazyStrategy that generates Tuple[pd.datetime, pd.datetime]
    """
    start, end = initialize_start_date_end_date(start, end)

    date1 = st.datetimes(start, end)
    date2 = st.datetimes(start, end)

    return order_pair(date1, date2)


@defines_strategy
def pair_of_ordered_timedeltas(start: Optional[pd.Timedelta] = None,
                               end: Optional[pd.Timedelta] = None):
    """ Returns an hypothesis strategy that generates a pair of ordered
    pd.Timedelta. Useful to create a Pandas TimedeltaIndex

    Parameters
    ----------
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)

    Returns
    -------
    LazyStrategy that generates Tuple[pd.Timedelta, pd.Timedelta]
    """
    start, end = initialize_start_timedelta_end_timedelta(start, end)

    timedelta1 = st.timedeltas(start, end)
    timedelta2 = st.timedeltas(start, end)

    return order_pair(timedelta1, timedelta2)


@defines_strategy
def positive_bounded_integers(max_length):
    return st.integers(min_value=0, max_value=max_length)


@defines_strategy
def samples_from(iterable):
    return st.builds(lambda index: iterable[index],
                     st.integers(0, len(iterable) - 1))


@defines_strategy
@st.composite
def available_freqs(draw):
    return freq_to_timedelta(draw(samples_from(string_freqs)))


@defines_strategy
@st.composite
def period_range_args_from(draw,
                           start: pd.datetime,
                           end: pd.datetime,
                           max_length: int,
                           arg_to_remove: Optional[str] = None) -> IndexRangeArgs:
    start, end = draw(pair_of_ordered_dates(start, end))
    periods = draw(positive_bounded_integers(max_length))
    freq = draw(available_freqs())
    element_to_exclude = draw(samples_from(pandas_range_params)) \
        if arg_to_remove is None else arg_to_remove
    assume(element_to_exclude != 'freq')

    args = {'start': start, 'end': end, 'periods': periods, 'freq': freq}
    del args[element_to_exclude]

    return args


def period_range_args_are_correct(period_range_args: IndexRangeArgs,
                                  max_length: int = 1000) -> bool:
    if 'periods' not in period_range_args:
        return expected_index_length_from(**period_range_args) < max_length
    else:
        return True


def compute_period_range(period_range_args: IndexRangeArgs):
    return pd.period_range(**period_range_args)


@defines_strategy
@st.composite
def date_range_args_from(draw,
                         start: pd.datetime,
                         end: pd.datetime,
                         max_length: int,
                         arg_to_remove: Optional[str] = None) -> IndexRangeArgs:
    start, end = draw(pair_of_ordered_dates(start, end))
    periods = draw(positive_bounded_integers(max_length))
    freq = draw(available_freqs())
    element_to_exclude = draw(samples_from(pandas_range_params)) \
        if arg_to_remove is None else arg_to_remove

    args = {'start': start, 'end': end, 'periods': periods, 'freq': freq}
    del args[element_to_exclude]

    return args


def date_range_args_are_correct(date_range_args: IndexRangeArgs,
                                max_length: int = 1000,
                                min_start: pd.Timestamp = None,
                                max_end: pd.Timestamp = None, ) -> bool:
    min_start = min_start if min_start is not None else pd.Timestamp('1980-01-01')
    max_end = max_end if max_end is not None else pd.Timestamp('2020-01-01')
    try:
        if 'periods' not in date_range_args:
            return expected_index_length_from(**date_range_args) < max_length
        elif 'start' not in date_range_args:
            return expected_start_date_from(**date_range_args) >= min_start
        elif 'end' not in date_range_args:
            return expected_end_date_from(**date_range_args) <= max_end
        else:
            return True
    except OverflowError:
        return False


def compute_date_range(date_range_args: IndexRangeArgs):
    return pd.date_range(**date_range_args)


@defines_strategy
@st.composite
def timedelta_range_args_from(draw,
                              start: pd.Timedelta,
                              end: pd.Timedelta,
                              max_length: int,
                              arg_to_remove: Optional[str] = None) -> IndexRangeArgs:
    start, end = draw(pair_of_ordered_timedeltas(start, end))
    periods = draw(positive_bounded_integers(max_length))
    freq = draw(available_freqs())
    element_to_exclude = draw(samples_from(pandas_range_params)) \
        if arg_to_remove is None else arg_to_remove

    args = {'start': start, 'end': end, 'periods': periods, 'freq': freq}
    del args[element_to_exclude]

    return args


def timedelta_range_args_are_correct(timedelta_range_args: IndexRangeArgs,
                                     max_length: int = 1000,
                                     min_start: pd.Timedelta = None,
                                     max_end: pd.Timedelta = None) -> bool:
    min_start = min_start if min_start is not None else pd.Timedelta(0)
    max_end = max_end if max_end is not None else pd.Timedelta('40Y')
    try:
        if 'periods' not in timedelta_range_args:
            return expected_index_length_from(**timedelta_range_args) < max_length
        elif 'start' not in timedelta_range_args:
            return expected_start_date_from(**timedelta_range_args) >= min_start
        elif 'end' not in timedelta_range_args:
            return expected_end_date_from(**timedelta_range_args) <= max_end
        else:
            return True
    except OverflowError:
        return False


def compute_timedelta_range(date_range_args: IndexRangeArgs):
    return pd.timedelta_range(**date_range_args)
