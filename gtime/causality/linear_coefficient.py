import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from gtime.causality.base import CausalityMixin


class ShiftedLinearCoefficient(BaseEstimator, TransformerMixin, CausalityMixin):
    """Test the shifted linear fit coefficients between two or more time series.

    Parameters
    ----------
    min_shift : int, optional, default: ``1``
        The minimum number of shifts to check for.

    max_shift : int, optional, default: ``10``
        The maximum number of shifts to check for.

    target_col : str, optional, default: ``None``
        The column to use as the a reference (i.e., the column which is not
        shifted).

    dropna : bool, optional, default: ``False``
        Determines if the Nan values created by shifting are retained or dropped.

    bootstrap_iterations : int, optional, default: ``None``
        If not None, compute the p_values of the test, by performing bootstrapping of
        the original data (sampling with replacement).

    permutation_iterations : int, optional, default: ``None``
        If not None, compute the p_values of the test, by performing permutations of
        the original data.

    Examples
    --------

    >>> from gtime.causality.linear_coefficient import ShiftedLinearCoefficient
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
        min_shift: int = 1,
        max_shift: int = 10,
        target_col: str = None,
        dropna: bool = False,
        bootstrap_iterations: int = None,
        permutation_iterations: int = None,
    ):
        super().__init__(
            bootstrap_iterations=bootstrap_iterations,
            permutation_iterations=permutation_iterations,
        )
        self.min_shift = min_shift
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
        best_shifts = self._compute_best_shifts(data, self._get_max_coeff_shift)
        pivot_tables = self._create_pivot_tables(best_shifts)

        self.best_shifts_ = pivot_tables["best_shifts"]
        self.max_corrs_ = pivot_tables["max_corrs"]

        if self.bootstrap_iterations:
            self.bootstrap_p_values_ = pivot_tables["bootstrap_p_values"]

        if self.permutation_iterations:
            self.permutation_p_values_ = pivot_tables["permutation_p_values"]

        return self

    def _get_max_coeff_shift(self, data: pd.DataFrame, x, y):
        shifts = pd.DataFrame()
        shifts[x] = data[x]
        shifts[y] = data[y]
        # print("shifts:", shifts)
        # print("data:", data)
        for shift in range(self.min_shift, self.max_shift + 1):
            # print("data", shift, ":", data[x].shift(shift))
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()

        lf = LinearRegression().fit(
            shifts[range(self.min_shift, self.max_shift + 1)].values, shifts[y].values
        )

        q = lf.coef_.max(), np.argmax(lf.coef_) + (self.min_shift - 0)
        return q
