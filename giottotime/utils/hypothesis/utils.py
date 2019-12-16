from typing import Union, Tuple

import hypothesis.strategies as st
import pandas as pd


def initialize_start_date_end_date(
    start: pd.datetime, end: pd.datetime
) -> Tuple[pd.datetime, pd.datetime]:
    start = start if start is not None else pd.Timestamp("1980-01-01")
    end = end if end is not None else pd.Timestamp("2020-01-01")
    return start, end


def initialize_start_timedelta_end_timedelta(start: pd.Timedelta, end: pd.Timedelta):
    start = start if start is not None else pd.Timedelta(0)
    end = end if end is not None else pd.Timedelta("40Y")
    return start, end


def order_pair(element1, element2):
    return st.builds(
        lambda start, end: (start, end), start=element1, end=element2
    ).filter(lambda x: x[0] < x[1])


def expected_start_date_from(
    end: Union[pd.datetime, pd.Period], periods: int, freq: pd.Timedelta
) -> Union[pd.datetime, pd.Period]:
    return end - periods * freq


def expected_end_date_from(
    start: Union[pd.datetime, pd.Period], periods: int, freq: pd.Timedelta
) -> Union[pd.datetime, pd.Period]:
    return start + periods * freq


def expected_index_length_from(
    start: Union[pd.datetime, pd.Period],
    end: Union[pd.datetime, pd.Period],
    freq: pd.Timedelta,
) -> int:
    expected_index_length = (end - start) // freq
    return expected_index_length


def freq_to_timedelta(
    freq: str, approximate_if_non_uniform: bool = True
) -> pd.Timedelta:
    try:
        return pd.to_timedelta(f"1{freq}")
    except ValueError as e:
        if approximate_if_non_uniform:
            correspondences = {
                "B": pd.Timedelta(1, unit="D"),
                "Q": pd.Timedelta(90, unit="D"),
                "A": pd.Timedelta(365, unit="D"),
            }
            return correspondences[freq]
        else:
            raise e
