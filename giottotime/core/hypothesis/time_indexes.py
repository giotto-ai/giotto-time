from typing import Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis._strategies import defines_strategy
from hypothesis.extra.numpy import arrays
from hypothesis.searchstrategy.lazy import LazyStrategy

available_freq = ['B', 'D', 'W', 'M', 'Q', 'A', 'Y', 'H', 'T', 'S']
_pandas_range_params = ['start', 'end', 'periods', 'freq']


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
    start, end, periods, freq = draw(
        _start_dates_end_dates_periods_freqs(start,
                                             end,
                                             max_length)
    )
    element_to_exclude = draw(samples_from(_pandas_range_params))
    assume(element_to_exclude != 'freq')

    index = _build_period_range_from(start,
                                     end,
                                     periods,
                                     freq,
                                     element_to_exclude)
    return index


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
    start, end, periods, freq = draw(
        _start_dates_end_dates_periods_freqs(start, end, max_length)
    )
    element_to_exclude = draw(samples_from(_pandas_range_params))

    index = _build_date_range_from(start,
                                   end,
                                   periods,
                                   freq,
                                   element_to_exclude,
                                   max_length)
    return index


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
    start, end, periods, freq = draw(
        _start_timedelta_end_timedelta_periods_freq(start, end, max_length)
    )
    element_to_exclude = draw(samples_from(_pandas_range_params))

    index = _build_timedelta_range_from(start,
                                        end,
                                        periods,
                                        freq,
                                        element_to_exclude,
                                        max_length)
    return index


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
    start, end = _initialize_start_date_end_date(start, end)

    date1 = st.datetimes(start, end)
    date2 = st.datetimes(start, end)

    return _order_pair(date1, date2)


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
    start, end = _initialize_start_timedelta_end_timedelta(start, end)

    timedelta1 = st.timedeltas(start, end)
    timedelta2 = st.timedeltas(start, end)

    return _order_pair(timedelta1, timedelta2)


@defines_strategy
def positive_bounded_integers(max_length):
    return st.integers(min_value=0, max_value=max_length)


@defines_strategy
def samples_from(iterable):
    return st.builds(lambda index: iterable[index],
                     st.integers(0, len(iterable) - 1))


@defines_strategy
def available_freqs():
    return samples_from(available_freq)


@st.composite
def _start_dates_end_dates_periods_freqs(
        draw,
        start: Optional[pd.datetime] = None,
        end: Optional[pd.datetime] = None,
        max_length: int = 1000
) -> Tuple[pd.datetime, pd.datetime, int, pd.Timedelta]:
    start_end_date_pair = draw(pair_of_ordered_dates(start, end))
    periods = draw(positive_bounded_integers(max_length))
    freq = _freq_to_timedelta(draw(samples_from(available_freq)))
    return start_end_date_pair[0], start_end_date_pair[1], periods, freq


@st.composite
def _start_timedelta_end_timedelta_periods_freq(
        draw,
        start: Optional[pd.Timedelta] = None,
        end: Optional[pd.datetime] = None,
        max_length: int = 1000
) -> Tuple[pd.datetime, pd.datetime, int, pd.Timedelta]:
    start_end_date_pair = draw(pair_of_ordered_timedeltas(start, end))
    periods = draw(positive_bounded_integers(max_length))
    freq = _freq_to_timedelta(draw(samples_from(available_freq)))
    return start_end_date_pair[0], start_end_date_pair[1], periods, freq


def _build_period_range_from(start: pd.datetime,
                             end: pd.datetime,
                             periods: int,
                             freq: pd.Timedelta,
                             element_to_exclude: str,
                             max_length: int = 1000):
    period_range_kwargs = _get_pandas_range_kwargs_from(start,
                                                        end,
                                                        periods,
                                                        freq,
                                                        element_to_exclude)
    if 'periods' not in period_range_kwargs:
        assume(_expected_index_length_from(**period_range_kwargs) < max_length)
    return pd.period_range(**period_range_kwargs)


def _build_date_range_from(start: pd.datetime,
                           end: pd.datetime,
                           periods: int,
                           freq: pd.Timedelta,
                           element_to_exclude: str,
                           max_length: int = 1000):
    date_range_kwargs = _get_pandas_range_kwargs_from(start,
                                                      end,
                                                      periods,
                                                      freq,
                                                      element_to_exclude)
    try:
        if 'periods' not in date_range_kwargs:
            assume(_expected_index_length_from(**date_range_kwargs) < max_length)
        elif 'start' not in date_range_kwargs:
            assume(_expected_start_date_from(**date_range_kwargs) >= start)
        elif 'end' not in date_range_kwargs:
            assume(_expected_end_date_from(**date_range_kwargs) <= end)
    except OverflowError:
        _reject_test_case()
    return pd.date_range(**date_range_kwargs)


def _build_timedelta_range_from(start: pd.Timedelta,
                                end: pd.Timedelta,
                                periods: int,
                                freq: pd.Timedelta,
                                element_to_exclude: str,
                                max_length: int = 1000):
    timedelta_range_kwargs = _get_pandas_range_kwargs_from(start,
                                                           end,
                                                           periods,
                                                           freq,
                                                           element_to_exclude)
    try:
        if 'periods' not in timedelta_range_kwargs:
            assume(_expected_index_length_from(**timedelta_range_kwargs) < max_length)
        elif 'start' not in timedelta_range_kwargs:
            assume(_expected_start_date_from(**timedelta_range_kwargs) >= start)
        elif 'end' not in timedelta_range_kwargs:
            assume(_expected_end_date_from(**timedelta_range_kwargs) <= end)
    except OverflowError:
        _reject_test_case()
    return pd.timedelta_range(**timedelta_range_kwargs)


def _get_pandas_range_kwargs_from(start: pd.datetime,
                                  end: pd.datetime,
                                  periods: int,
                                  freq: pd.Timedelta,
                                  element_to_exclude: str):
    range_kwargs = {
        'start': start,
        'end': end,
        'periods': periods,
        'freq': freq,
    }
    del range_kwargs[element_to_exclude]
    return range_kwargs


def _initialize_start_date_end_date(
        start: pd.datetime,
        end: pd.datetime
) -> Tuple[pd.datetime, pd.datetime]:
    start = start if start is not None \
        else pd.Timestamp('1980-01-01')
    end = end if end is not None else pd.Timestamp('2020-01-01')
    return start, end


def _initialize_start_timedelta_end_timedelta(start: pd.Timedelta,
                                              end: pd.Timedelta):
    start = start if start is not None \
        else pd.Timedelta(0)
    end = end if end is not None else pd.Timedelta('40Y')
    return start, end


def _order_pair(element1: LazyStrategy,
                element2: LazyStrategy):
    return st.builds(lambda start, end: (start, end),
                     start=element1,
                     end=element2).filter(lambda x: x[0] < x[1])


def _expected_start_date_from(end: pd.datetime,
                              periods: int,
                              freq: str) -> pd.datetime:
    return end - periods * _freq_to_timedelta(freq)


def _expected_end_date_from(start: pd.datetime,
                            periods: int,
                            freq: str) -> pd.datetime:
    return start + periods * _freq_to_timedelta(freq)


def _expected_index_length_from(start: pd.datetime,
                                end: pd.datetime,
                                freq: pd.Timedelta) -> int:
    expected_index_length = (end - start) // freq
    return expected_index_length


def _freq_to_timedelta(freq: str,
                       approximate_if_non_uniform: bool = True) -> pd.Timedelta:
    try:
        return pd.to_timedelta(f'1{freq}')
    except ValueError as e:
        if approximate_if_non_uniform:
            correspondences = {
                'B': pd.Timedelta(1, unit='D'),
                'Q': pd.Timedelta(90, unit='D'),
                'A': pd.Timedelta(365, unit='D'),
            }
            return correspondences[freq]
        else:
            raise e


def _reject_test_case():
    assume(False)
