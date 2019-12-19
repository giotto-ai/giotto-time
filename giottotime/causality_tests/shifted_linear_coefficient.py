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

    >>> from giottotime.causality_tests.shifted_pearson_correlation import ShiftedPearsonCorrelation
    >>> import pandas.util.testing as testing
    >>> data = testing.makeTimeDataFrame(freq="s")
    >>> slc = ShiftedPearsonCorrelation(target_col="A")
    >>> slc.fit(data)
    >>> slc.best_shifts_
    y  A  B  C  D  E  F  G  H  I  J
    x
    A  6  5  5  9  3  8  5  8  3  2
    B  7  1  6  1  2  5  5  5  9  4
    C  1  1  7  4  4  1  2  3  1  9
    D  8  1  6  5  2  1  7  5  9  5
    E  4  9  3  9  5  1  2  6  7  8
    F  2  2  1  2  5  4  4  7  6  9
    G  5  9  3  1  9  1  8  1  9  4
    H  5  3  1  4  8  8  4  3  7  2
    I  5  4  3  2  1  6  6  1  1  3
    J  8  8  8  8  5  9  1  4  5  4
    >>> slc.max_corrs_
    y         A         B         C         D         E         F         G         H         I         J
    x
    A  0.107887  0.146828  0.117625  0.021599  0.172352  0.124256  0.106914  0.116788  0.201324  0.096791
    B  0.115584  0.087271  0.128803  0.083926  0.240002  0.085621  0.119507  0.105164  0.099835  0.172130
    C  0.123297  0.084162  0.135995  0.195333  0.130223  0.144253  0.185256  0.096939  0.112538  0.114886
    D  0.052654 -0.004008  0.106089  0.127790  0.151123  0.095424  0.127398  0.130226  0.087748  0.093426
    E  0.091095  0.174533  0.073424  0.046163  0.070084  0.118448  0.110653  0.160709  0.126197  0.101954
    F  0.071367  0.014495  0.050906  0.133828  0.139331  0.044114  0.109927  0.073282  0.104515  0.127253
    G  0.083635  0.081576  0.193632  0.040063  0.097557  0.090042  0.061074  0.101688  0.145634  0.101889
    H  0.083066  0.135847  0.098878  0.059478  0.108100  0.116164  0.149509  0.149741  0.132864  0.213872
    I  0.140539  0.086271  0.129208  0.080950  0.034588  0.137265  0.085493  0.050050  0.034805  0.095459
    J  0.157655  0.167399  0.089095  0.113485  0.189612  0.067281  0.078631  0.038612  0.157670  0.159494
    """

    def __init__(
        self, max_shift: int = 10, target_col: str = "y", dropna: bool = False
    ):
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
        best_shifts = pd.DataFrame(columns=["x", "y", "shift", "max_corr"])
        best_shifts = best_shifts.astype(
            {"x": np.float64, "y": np.float64, "shift": np.int64, "max_corr": np.int64}
        )

        for x, y in product(data.columns, repeat=2):
            res = self._get_max_coeff_shift(data, self.max_shift, x=x, y=y)

            best_shift = res[1]
            max_corr = res[0]
            # N = data.shape[0] - max_shift

            best_shifts = best_shifts.append(
                {"x": x, "y": y, "shift": best_shift, "max_corr": max_corr},
                ignore_index=True,
            )

        pivot_best_shifts = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="shift"
        )
        max_corrs = pd.pivot_table(
            best_shifts, index=["x"], columns=["y"], values="max_corr"
        )

        self.best_shifts_ = pivot_best_shifts
        self.max_corrs_ = max_corrs

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
                data_t[col] = data_t[col].shift(self.best_shifts_[col][self.target_col])

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
