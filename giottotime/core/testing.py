import random
from typing import Optional, Tuple

import pandas as pd
import hypothesis.strategies as st
from hypothesis import assume
from hypothesis._strategies import defines_strategy


@defines_strategy
@st.composite
def datetime_index(draw,
                   start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None,
                   max_length: int = 1000):
    start_end_date_pair = draw(pair_of_ordered_dates(start_date, end_date))
    periods = draw(positive_bounded_integers(max_length))
    freq = draw(date_frequencies())
    element_to_exclude = draw(st.integers(0, 3))

    index = _build_date_range_from(start_end_date_pair,
                                   periods,
                                   freq,
                                   element_to_exclude)
    return index


@defines_strategy
def pair_of_ordered_dates(start_date: Optional[pd.Timestamp] = None,
                    end_date: Optional[pd.Timestamp] = None):
    start_date, end_date = _check_start_date_end_date(start_date, end_date)

    start_dates = st.datetimes(start_date, end_date)
    end_dates = st.datetimes(start_date, end_date)

    return _compute_ordered_date_pair_strategy_from(start_dates, end_dates)


@defines_strategy
def positive_bounded_integers(max_length):
    return st.integers(min_value=0, max_value=max_length)


@defines_strategy
def date_frequencies():
    available_freq = ['B', 'D', 'W', 'M', 'SM', 'BM', 'MS', 'SMS', 'BMS', 'Q',
                      'BQ', 'QS' , 'BQS', 'A', 'Y', 'BA', 'BY', 'AS', 'YS',
                      'BAS', 'BYS', 'BH', 'H', 'T', 'S']
    return st.builds(lambda index: available_freq[index],
                     st.integers(0, len(available_freq)-1))


def _build_date_range_from(start_end_date_pair: Tuple[pd.Timestamp],
                           periods: int,
                           freq: str,
                           element_to_exclude: int,
                           assume_if_fails: bool = True):
    date_range_args = [
        start_end_date_pair[0],
        start_end_date_pair[1],
        periods,
        freq
    ]
    date_range_args[element_to_exclude] = None
    try:
        return pd.date_range(*date_range_args)
    except Exception as e:
        if assume_if_fails:
            assume(False)
        else:
            raise e


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


