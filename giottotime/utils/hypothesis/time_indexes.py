from typing import Optional, Dict, Union, Iterable

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis.extra.numpy import arrays

from .utils import (
    freq_to_timedelta,
    initialize_start_date_end_date,
    order_pair,
    initialize_start_timedelta_end_timedelta,
    expected_index_length_from,
    expected_start_date_from,
    expected_end_date_from,
)

IndexRangeArgs = Dict[str, Union[pd.datetime, int, pd.Timedelta]]

string_freqs = ["B", "D", "W", "M", "Q", "A", "Y", "H", "T", "S"]
pandas_range_params = ["start", "end", "periods", "freq"]


@st.composite
def giotto_time_series(
    draw,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    min_length: int = 0,
    max_length: int = 1000,
    name: str = "time_series",
    allow_nan: bool = True,
    allow_infinity: bool = True,
):
    period_index_series = draw(
        series_with_period_index(
            start_date,
            end_date,
            min_length,
            max_length,
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
        )
    )
    return pd.DataFrame({name: period_index_series})


@st.composite
def series_with_period_index(
    draw,
    start: Optional[pd.datetime] = None,
    end: Optional[pd.datetime] = None,
    min_length: int = 0,
    max_length: int = 1000,
    allow_nan: bool = True,
    allow_infinity: bool = True,
):
    """ Returns a strategy to generate a Pandas Series with PeriodIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=None)
    allow_nan : ``bool``, optional, (default=True)
    allow_infinity : ``bool``, optional, (default=True)

    Returns
    -------
    pd.Series with PeriodIndex
    """
    index = draw(period_indexes(start, end, min_length, max_length))
    values = draw(
        arrays(
            dtype=np.float64,
            shape=index.shape[0],
            elements=st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity),
        )
    )
    return pd.Series(index=index, data=values)


@st.composite
def series_with_datetime_index(
    draw,
    start: Optional[pd.datetime] = None,
    end: Optional[pd.datetime] = None,
    min_length: int = 0,
    max_length: int = 1000,
    allow_nan: bool = True,
    allow_infinity: bool = True,
):
    """ Returns a strategy to generate a Pandas Series with DatetimeIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)
    allow_nan : ``bool``, optional, (default=True)
    allow_infinity : ``bool``, optional, (default=True)

    Returns
    -------
    pd.Series with DatetimeIndex
    """
    index = draw(datetime_indexes(start, end, min_length, max_length))
    values = draw(
        arrays(
            dtype=np.float64,
            shape=index.shape[0],
            elements=st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity),
        )
    )
    return pd.Series(index=index, data=values)


@st.composite
def series_with_timedelta_index(
    draw,
    start: Optional[pd.Timedelta] = None,
    end: Optional[pd.Timedelta] = None,
    min_length: int = 0,
    max_length: int = 1000,
    allow_nan: bool = True,
    allow_infinity: bool = True,
):
    """ Returns a strategy to generate a Pandas Series with Timedelta index

    Parameters
    ----------
    draw
    start : ``pd.Timedelta``, optional, (default=None)
    end : ``pd.Timedelta``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=None)
    allow_nan : ``bool``, optional, (default=True)
    allow_infinity : ``bool``, optional, (default=True)

    Returns
    -------
    pd.Series with TimedeltaIndex
    """
    index = draw(timedelta_indexes(start, end, min_length, max_length))
    values = draw(
        arrays(
            dtype=np.float64,
            shape=index.shape[0],
            elements=st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity),
        )
    )
    return pd.Series(index=index, data=values)


@st.composite
def period_indexes(
    draw,
    start: Optional[pd.datetime] = None,
    end: Optional[pd.datetime] = None,
    min_length: int = 0,
    max_length: int = 1000,
):
    """ Returns a strategy to generate Pandas PeriodIndex

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)

    Returns
    -------
    LazyStrategy that generates pandas PeriodIndex
    """
    period_range_args = draw(period_range_args_from(start, end, min_length, max_length))
    assume(
        period_range_args_are_correct(
            period_range_args, min_length, max_length, start, end
        )
    )
    return compute_period_range(period_range_args)


@st.composite
def datetime_indexes(
    draw,
    start: Optional[pd.datetime] = None,
    end: Optional[pd.datetime] = None,
    min_length: int = 0,
    max_length: int = 1000,
):
    """ Returns a strategy to generate Pandas DatetimeIndex.

    Parameters
    ----------
    draw
    start : ``pd.datetime``, optional, (default=None)
    end : ``pd.datetime``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, default: ``1000``

    Returns
    -------
    LazyStrategy that generates pd.DatetimeIndex
    """
    date_range_args = draw(date_range_args_from(start, end, min_length, max_length))
    assume(
        date_range_args_are_correct(date_range_args, min_length, max_length, start, end)
    )
    return compute_date_range(date_range_args)


@st.composite
def timedelta_indexes(
    draw,
    start: Optional[pd.Timedelta] = None,
    end: Optional[pd.Timedelta] = None,
    min_length: int = 0,
    max_length: int = 1000,
):
    """ Returns a strategy to generate Pandas TimedeltaIndex

    Parameters
    ----------
    draw
    start : ``pd.Timedelta``, optional, (default=None)
    end : ``pd.Timedelta``, optional, (default=None)
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, (default=1000)

    Returns
    -------
    LazyStrategy that generates pd.TimedeltaIndex
    """
    timedelta_range_args = draw(
        timedelta_range_args_from(start, end, min_length, max_length)
    )
    assume(
        timedelta_range_args_are_correct(
            timedelta_range_args,
            min_length=min_length,
            max_length=max_length,
            min_start=start,
            max_end=end,
        )
    )
    return compute_timedelta_range(timedelta_range_args)


def pair_of_ordered_dates(
    start: Optional[pd.datetime] = None, end: Optional[pd.datetime] = None
):
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


def pair_of_ordered_timedeltas(
    start: Optional[pd.Timedelta] = None, end: Optional[pd.Timedelta] = None
):
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


def positive_bounded_integers(max_value):
    """ Returns a strategy that generates a positive bounded integer

    Parameters
    ----------
    max_value : ``int``, required
        maximum value of the integer

    Returns
    -------
    LazyStrategy
    """
    return st.integers(min_value=0, max_value=max_value)


def samples_from(iterable: Iterable):
    """ Strategy that samples from an iterable.

    Parameters
    ----------
    iterable : ``Iterable[Generic]``, required

    Returns
    -------
    LazyStrategy
    """
    return st.builds(lambda index: iterable[index], st.integers(0, len(iterable) - 1))


@st.composite
def available_freqs(draw) -> pd.Timedelta:
    """ Strategy that samples from the available frequencies and returns a
    corresponding pd.Timedelta

    Parameters
    ----------
    draw

    Returns
    -------
    pd.Timedelta
    """
    return freq_to_timedelta(draw(samples_from(string_freqs)))


@st.composite
def period_range_args_from(
    draw,
    start: pd.datetime,
    end: pd.datetime,
    min_length: int,
    max_length: int,
    arg_to_remove: Optional[str] = None,
) -> IndexRangeArgs:
    """ Strategy that computes arguments to call pd.period_range() from
    start, end and max length of the resulting period
    It returns a dictionary with 3 out of the 3 keys 'start', 'end', 'periods'
    and 'freq'. One of them is automatically removed since pd.period_range()
    accepts only 3 of the 4 inputs.

    Parameters
    ----------
    draw
    start : ``pd.datetime``, required
    end : ``pd.datetime``, required
    min_length : ``int``, required
    max_length : ``int``, required
    arg_to_remove : ``str``, optional, (default=None)

    Returns
    -------
    IndexRangeArgs
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.
    """
    start, end = draw(pair_of_ordered_dates(start, end))
    periods = draw(st.integers(min_value=min_length, max_value=max_length))
    freq = draw(available_freqs())
    element_to_exclude = (
        draw(samples_from(pandas_range_params))
        if arg_to_remove is None
        else arg_to_remove
    )
    assume(element_to_exclude != "freq")

    args = {"start": start, "end": end, "periods": periods, "freq": freq}
    del args[element_to_exclude]

    return args


def period_range_args_are_correct(
    period_range_args: IndexRangeArgs,
    min_length: int = 0,
    max_length: int = 1000,
    min_start: Optional[pd.Period] = None,
    max_end: Optional[pd.Period] = None,
) -> bool:
    """ Returns True if the period range arguments are correct. i.e. if
    'periods' is not a key it checks that the length of the resulting period
    index is below ``max_length``.

    Parameters
    ----------
    period_range_args : ``IndexRangeArgs``, required
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.

    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)
    min_start : ``pd.Period``, optional, (default=1000)
    max_end : ``pd.Period``, optional, (default=1000


    Returns
    -------
    bool
    """
    min_start = min_start if min_start is not None else pd.Timestamp("1980-01-01")
    max_end = max_end if max_end is not None else pd.Timestamp("2020-01-01")
    return _range_args_are_correct(
        period_range_args, min_length, max_length, min_start, max_end
    )


def compute_period_range(period_range_args: IndexRangeArgs) -> pd.PeriodIndex:
    """ Returns a Pandas PeriodIndex from the given ``period_range_args``.

    Parameters
    ----------
    period_range_args : ``IndexRangeArgs``, required

    Returns
    -------
    pd.PeriodIndex
    """
    return pd.period_range(**period_range_args)


@st.composite
def date_range_args_from(
    draw,
    start: pd.datetime,
    end: pd.datetime,
    min_length: int,
    max_length: int,
    arg_to_remove: Optional[str] = None,
) -> IndexRangeArgs:
    """ Strategy that computes arguments to call pd.date_range() from
    start, end and max length of the resulting period
    It returns a dictionary with 3 out of the 3 keys 'start', 'end', 'periods'
    and 'freq'. One of them is automatically removed since pd.date_range()
    accepts only 3 of the 4 inputs.

    Parameters
    ----------
    draw
    start : ``pd.datetime``, required
    end : ``pd.datetime``, required
    min_length: ``int``, required
    max_length : ``int``, required
    arg_to_remove: ``str``, optional, (default=None)

    Returns
    -------
    IndexRangeArgs
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.
    """
    start, end = draw(pair_of_ordered_dates(start, end))
    periods = draw(st.integers(min_value=min_length, max_value=max_length))
    freq = draw(available_freqs())
    element_to_exclude = (
        draw(samples_from(pandas_range_params))
        if arg_to_remove is None
        else arg_to_remove
    )

    args = {"start": start, "end": end, "periods": periods, "freq": freq}
    del args[element_to_exclude]

    return args


def date_range_args_are_correct(
    date_range_args: IndexRangeArgs,
    min_length: int = 0,
    max_length: int = 1000,
    min_start: pd.Timestamp = None,
    max_end: pd.Timestamp = None,
) -> bool:
    """ Returns True if the date range arguments are correct.
    It checks 3 things:

    - if 'periods' is not a key it checks that the length of the resulting
    period index is below ``max_length``.
    - if 'start' is not a keyword it checks that the expected start date is
    after ``min_start``
    - if 'end' is not a keyword it checks that the expected end date is
    before ``max_end``

    Parameters
    ----------
    date_range_args : ``IndexRangeArgs``, required
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.

    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)
    min_start : ``pd.Timestamp``, optional, (default=None)
    max_end : ``pd.Timestamp``, optional, (default=None)

    Returns
    -------
    bool
    """
    min_start = min_start if min_start is not None else pd.Timestamp("1980-01-01")
    max_end = max_end if max_end is not None else pd.Timestamp("2020-01-01")
    return _range_args_are_correct(
        date_range_args, min_length, max_length, min_start, max_end
    )


def compute_date_range(date_range_args: IndexRangeArgs) -> pd.DatetimeIndex:
    """ Returns a Pandas DatetimeIndex from the given ``date_range_args``.

    Parameters
    ----------
    date_range_args: ``IndexRangeArgs``, required

    Returns
    -------
    pd.DatetimeIndex
    """
    return pd.date_range(**date_range_args)


@st.composite
def timedelta_range_args_from(
    draw,
    start: pd.Timedelta,
    end: pd.Timedelta,
    min_length: int,
    max_length: int,
    arg_to_remove: Optional[str] = None,
) -> IndexRangeArgs:
    """ Strategy that computes arguments to call pd.timedelta_range() from
    start, end and max length of the resulting period
    It returns a dictionary with 3 out of the 3 keys 'start', 'end', 'periods'
    and 'freq'. One of them is automatically removed since pd.timedelta_range()
    accepts only 3 of the 4 inputs.

    Parameters
    ----------
    draw
    start : pd.datetime, required
    end : pd.datetime, required
    min_length : int, required
    max_length : int, required
    arg_to_remove: str, optional, (default=None)

    Returns
    -------
    IndexRangeArgs
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.
    """
    start, end = draw(pair_of_ordered_timedeltas(start, end))
    periods = draw(st.integers(min_value=min_length, max_value=max_length))
    freq = draw(available_freqs())
    element_to_exclude = (
        draw(samples_from(pandas_range_params))
        if arg_to_remove is None
        else arg_to_remove
    )

    args = {"start": start, "end": end, "periods": periods, "freq": freq}
    del args[element_to_exclude]

    return args


def timedelta_range_args_are_correct(
    timedelta_range_args: IndexRangeArgs,
    min_length: int = 0,
    max_length: int = 1000,
    min_start: pd.Timedelta = None,
    max_end: pd.Timedelta = None,
) -> bool:
    """ Returns True if the timedelta range arguments are correct.
    It checks 3 things:

    - if 'periods' is not a key it checks that the length of the resulting
    period index is below ``max_length``.
    - if 'start' is not a keyword it checks that the expected start timedelta
    is after ``min_start``
    - if 'end' is not a keyword it checks that the expected end timedelta is
    before ``max_end``

    Parameters
    ----------
    timedelta_range_args : ``IndexRangeArgs``, required
        dictionary with 3 out of the 4 keys 'start', 'end', 'periods' and
        'freq'.

    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)
    min_start : ``pd.Timestamp``, optional, (default=None)
    max_end : ``pd.Timestamp``, optional, (default=None)

    Returns
    -------
    bool
    """
    min_start = min_start if min_start is not None else pd.Timedelta(0)
    max_end = max_end if max_end is not None else pd.Timedelta("40Y")
    return _range_args_are_correct(
        timedelta_range_args, min_length, max_length, min_start, max_end
    )


def compute_timedelta_range(timedelta_range_args: IndexRangeArgs):
    """ Returns a Pandas TimedeltaIndex from the given 
    ``timedelta_range_args``.

    Parameters
    ----------
    timedelta_range_args: ``IndexRangeArgs``, required

    Returns
    -------
    pd.DatetimeIndex
    """
    return pd.timedelta_range(**timedelta_range_args)


def _range_args_are_correct(
    range_args: IndexRangeArgs,
    min_length: int = 0,
    max_length: int = 1000,
    min_start: Union[pd.Timedelta, pd.datetime] = None,
    max_end: Union[pd.Timedelta, pd.datetime] = None,
) -> bool:
    """Checks if range args are correct.

    Parameters
    ----------
    range_args: ``IndexRangeArgs``, required
    min_length : ``int``, optional, (default=0)
    max_length : ``int``, optional, (default=1000)
    min_start : ``Union[pd.Timedelta, pd.datetime]``, optional, (default=None)
    max_end : ``Union[pd.Timedelta, pd.datetime]``, optional, (default=None)

    Returns
    -------
    bool
    """
    try:
        if "periods" not in range_args:
            return min_length < expected_index_length_from(**range_args) < max_length
        elif "start" not in range_args:
            return expected_start_date_from(**range_args) >= min_start
        elif "end" not in range_args:
            return expected_end_date_from(**range_args) <= max_end
        else:
            return True
    except OverflowError:
        return False
