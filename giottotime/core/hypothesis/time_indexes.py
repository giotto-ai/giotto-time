import random
from typing import Optional, Tuple, Iterable

import pandas as pd
import hypothesis.strategies as st
from hypothesis import assume
from hypothesis._strategies import defines_strategy

available_freq = ['B', 'D', 'W', 'M', 'Q', 'A', 'Y', 'H', 'T', 'S']
_pandas_range_params = ['start', 'end', 'periods', 'freq']


@defines_strategy
@st.composite
def period_indexes(draw,
                   start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None,
                   max_length: int = 1000):
    """ Returns a strategy to generate Pandas PeriodIndex

    Parameters
    ----------
    draw
    start_date: ``pd.Timestamp``, optional, (default=None)
    end_date: ``pd.Timestamp``, optional, (default=None)
    max_length: int, optional, (default=1000)

    Returns
    -------
    LazyStrategy that generates pandas PeriodIndex
    """
    start_date, end_date, periods, freq = draw(
        _start_dates_end_dates_periods_freqs(start_date,
                                             end_date,
                                             max_length)
    )
    element_to_exclude = draw(samples_from(_pandas_range_params))
    assume(element_to_exclude != 'freq')

    index = _build_period_range_from(start_date,
                                     end_date,
                                     periods,
                                     freq,
                                     element_to_exclude)
    return index


@defines_strategy
@st.composite
def datetime_indexes(draw,
                     start_date: Optional[pd.Timestamp] = None,
                     end_date: Optional[pd.Timestamp] = None,
                     max_length: int = 1000):
    """ Returns a strategy to generate Pandas DatetimeIndex.

    Parameters
    ----------
    draw
    start_date: pd.Timestamp, optional
    end_date: pd.Timestamp, optional
    max_length: int, default: ``1000``

    Returns
    -------
    LazyStrategy that generates pd.DatetimeIndex
    """
    start_date, end_date, periods, freq = draw(
        _start_dates_end_dates_periods_freqs(start_date,
                                             end_date,
                                             max_length)
    )
    element_to_exclude = draw(samples_from(_pandas_range_params))

    index = _build_date_range_from(start_date,
                                   end_date,
                                   periods,
                                   freq,
                                   element_to_exclude)
    return index


@defines_strategy
def pair_of_ordered_dates(start_date: Optional[pd.Timestamp] = None,
                          end_date: Optional[pd.Timestamp] = None):
    """ Returns an hypothesis strategy that generates a pair of ordered
    pd.Timestamp. Useful to create a Pandas index

    Parameters
    ----------
    start_date: pd.Timestamp, optional
    end_date: pd.Timestamp, optional

    Returns
    -------
    LazyStrategy that generates Tuple[pd.Timestamp, pd.Timestamp]
    """
    start_date, end_date = _check_start_date_end_date(start_date, end_date)

    start_dates = st.datetimes(start_date, end_date)
    end_dates = st.datetimes(start_date, end_date)

    return _compute_ordered_date_pair_strategy_from(start_dates, end_dates)


@defines_strategy
def positive_bounded_integers(max_length):
    return st.integers(min_value=0, max_value=max_length)


@defines_strategy
def samples_from(iterable):
    return st.builds(lambda index: iterable[index],
                     st.integers(0, len(iterable) - 1))


@st.composite
def _start_dates_end_dates_periods_freqs(draw,
                                         start_date: pd.Timestamp = None,
                                         end_date: pd.Timestamp = None,
                                         max_length: int = 1000):
    start_end_date_pair = draw(pair_of_ordered_dates(start_date, end_date))
    periods = draw(positive_bounded_integers(max_length))
    freq = draw(samples_from(available_freq))
    return start_end_date_pair[0], start_end_date_pair[1], periods, freq


def _build_period_range_from(start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             periods: int,
                             freq: str,
                             element_to_exclude: str,
                             max_length: int = 1000):
    period_range_kwargs = _get_pandas_range_kwargs_from(start_date,
                                                      end_date,
                                                      periods,
                                                      freq,
                                                      element_to_exclude)
    if 'periods' not in period_range_kwargs:
        assume(_expected_index_length_from(**period_range_kwargs) < max_length)
    return pd.period_range(**period_range_kwargs)


def _build_date_range_from(start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           periods: int,
                           freq: str,
                           element_to_exclude: str):
    date_range_kwargs = _get_pandas_range_kwargs_from(start_date,
                                                      end_date,
                                                      periods,
                                                      freq,
                                                      element_to_exclude)
    try:
        return pd.date_range(**date_range_kwargs)
    except ValueError:
        _reject_test_case()


def _get_pandas_range_kwargs_from(start_date: pd.Timestamp,
                                  end_date: pd.Timestamp,
                                  periods: int,
                                  freq: str,
                                  element_to_exclude: str):
    range_kwargs = {
        'start': start_date,
        'end': end_date,
        'periods': periods,
        'freq': freq,
    }
    del range_kwargs[element_to_exclude]
    return range_kwargs


def _check_start_date_end_date(start_date: pd.Timestamp,
                               end_date: pd.Timestamp):
    start_date = start_date if start_date is not None \
        else pd.Timestamp('1980-01-01')
    end_date = end_date if end_date is not None else pd.Timestamp('2020-01-01')
    return start_date, end_date


def _compute_ordered_date_pair_strategy_from(start_dates: st.datetimes,
                                             end_dates: st.datetimes):
    return st.builds(lambda start, end: (start, end),
                     start=start_dates,
                     end=end_dates).filter(lambda x: x[0] < x[1])


def _expected_index_length_from(start: pd.Timestamp,
                                end: pd.Timestamp,
                                freq: str):
    timedelta_freq = _freq_to_timedelta(freq)
    expected_index_length = (end - start) // timedelta_freq
    return expected_index_length


def _freq_to_timedelta(freq: str, approximate_if_non_uniform: bool = True):
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
