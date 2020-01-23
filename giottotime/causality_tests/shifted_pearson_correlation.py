from itertools import product

import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from giottotime.causality_tests.base import CausalityTest


class ShiftedPearsonCorrelation(CausalityTest):
    """Class responsible for assessing the shifted Pearson correlations (PPMCC) between
    two or more series.

    Parameters
    ----------
    max_shift : int, optional, default: ``10``

    target_col : str, optional, default: ``'y'``
            The column to use as the a reference (i.e., the columns which is not
            shifted).

    dropna : bool, optional, default: ``False``
        Determines if the Nan values created by shifting are retained or dropped.

    Examples
    --------
    >>> from giottotime.causality_tests.shifted_pearson_correlation import ShiftedPearsonCorrelation
    >>> import pandas.util.testing as testing
    >>> data = testing.makeTimeDataFrame(freq="s")
    >>> spc = ShiftedPearsonCorrelation(target_col="A")
    >>> spc.fit(data)
    >>> spc.best_shifts_
    y  A  B  C  D
    x
    A  8  9  6  5
    B  7  4  4  6
    C  3  4  9  9
    D  7  1  9  1
    >>> spc.max_corrs_
    y         A         B         C         D
    x
    A  0.383800  0.260627  0.343628  0.360151
    B  0.311608  0.307203  0.255969  0.298523
    C  0.373613  0.267335  0.211913  0.140034
    D  0.496535  0.204770  0.402473  0.310065
    """

    def __init__(
        self,
        max_shift: int = 10,
        target_col: str = "y",
        dropna: bool = False,
        bootstrap_iterations=1000,
        bootstrap_samples=100,
    ):
        super().__init__(bootstrap_iterations, bootstrap_samples)
        self.max_shift = max_shift
        self.target_col = target_col
        self.dropna = dropna

    def fit(self, data: pd.DataFrame) -> "ShiftedPearsonCorrelation":
        """Create the dataframe of shifts of each time series which maximize the
         Pearson correlation (PPMCC).

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The DataFrame containing the time series on which to compute the shifted
            correlations.

        Returns
        -------
        self : ``ShiftedPearsonCorrelation``

        """
        best_shifts = pd.DataFrame(columns=["x", "y", "shift", "max_corr", "p_values"])
        best_shifts = best_shifts.astype(
            {
                "x": np.float64,
                "y": np.float64,
                "shift": np.int64,
                "max_corr": np.int64,
                "p_values": np.float64,
            }
        )

        for x, y in product(data.columns, repeat=2):
            res = self._get_max_corr_shift(data, self.max_shift, x=x, y=y)
            best_shift = res[1]
            max_corr = res[0]
            p_value = self._compute_is_test_significant(data, x, y, best_shift)
            # N = data.shape[0] - max_shift

            best_shifts = best_shifts.append(
                {
                    "x": x,
                    "y": y,
                    "shift": best_shift,
                    "max_corr": max_corr,
                    "p_values": p_value,
                },
                ignore_index=True,
            )

        pivot_best_shifts = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="max_corr"
        )
        p_values = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="p_values"
        )

        self.best_shifts_ = pivot_best_shifts
        self.max_corrs_ = max_corrs
        self.p_values_ = p_values

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shifts each input time series by the amount which optimizes correlation with
        the selected 'y' column.

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The DataFrame containing the time series on which to perform the
            transformation.

        Returns
        -------
        data_t : pd.DataFrame, shape (n_samples, n_time_series)
            The DataFrame (Pivot table) of the shifts which maximize the correlation
            between each time series The shift is indicated in rows.

        """
        check_is_fitted(self)
        data_t = data.copy()

        for col in data_t:
            if col != self.target_col:
                data_t[col] = data_t[col].shift(
                    -self.best_shifts_[col][self.target_col]
                )

        if self.dropna:
            data_t = data_t.dropna()

        return data_t

    def _get_max_corr_shift(
        self, data: pd.DataFrame, max_shift: int, x: str = "x", y: str = "y"
    ):
        shifts = pd.DataFrame()

        for shift in range(1, max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(data[y])

        q = self.shifted_corrs.max(), self.shifted_corrs.idxmax()
        return q
