from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from giottotime.causality_tests.base import CausalityTest


class ShiftedLinearCoefficient(CausalityTest):
    """Test the shifted linear fit coefficients between two or more time series.

    Parameters
    ----------
    max_shift : int, optional, default: ``10``
        The maximum number of shifts to check for.

    target_col : str, optional, default: ``'y'``
        The column to use as the a reference (i.e., the column which is not
        shifted).

    dropna : bool, optional, default: ``False``
        Determines if the Nan values created by shifting are retained or dropped.

    Examples
    --------

    >>> from giottotime.causality_tests.shifted_linear_coefficient import ShiftedLinearCoefficient
    >>> import pandas.util.testing as testing
    >>> data = testing.makeTimeDataFrame(freq="s")
    >>> slc = ShiftedLinearCoefficient(target_col="A")
    >>> slc.fit(data)
    >>> slc.best_shifts_
    y  A  B  C  D
    x
    A  3  6  8  5
    B  9  9  4  1
    C  8  2  4  9
    D  3  9  4  3
    >>> slc.max_corrs_
    y         A         B         C         D
    x
    A  0.460236  0.420005  0.339370  0.267143
    B  0.177856  0.300350  0.367150  0.550490
    C  0.484860  0.263036  0.456046  0.251342
    D  0.580068  0.344688  0.253626  0.256220
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

    def fit(self, data: pd.DataFrame) -> "ShiftedLinearCoefficient":
        """Create the DataFrame of shifts of each time series which maximize the shifted
         linear fit coefficients.

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The DataFrame containing the time-series on which to compute the shifted
            linear fit coefficients.

        Returns
        -------
        self : ``ShiftedLinearCoefficient``

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
            res = self._get_max_coeff_shift(data, self.max_shift, x=x, y=y)
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
        """Shifts each input time series by the amount which maximizes shifted linear
        fit coefficients with the selected 'y' column.

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The DataFrame containing the time series on which to perform the
            transformation.

        Returns
        -------
        data_t : pd.DataFrame, shape (n_samples, n_time_series)
            The DataFrame (Pivot table) of the shifts which maximize the shifted linear
            fit coefficients between each time series. The shift is indicated in rows.

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

    def _get_max_coeff_shift(
        self, data: pd.DataFrame, max_shift: int, x: str = "x", y: str = "y"
    ) -> (float, int):
        shifts = pd.DataFrame()

        shifts[x] = data[x]
        shifts[y] = data[y]

        for shift in range(1, max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        lf = LinearRegression().fit(
            shifts[range(1, max_shift)].values, shifts[y].values
        )

        q = lf.coef_.max(), np.argmax(lf.coef_) + 1
        return q
