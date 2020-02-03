import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from gtime.causality.base import CausalityMixin


class ShiftedPearsonCorrelation(BaseEstimator, TransformerMixin, CausalityMixin):
    """Class responsible for assessing the shifted Pearson correlations (PPMCC) between
    two or more series. For more info about the test, click
    `here <https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php>`_.

    Parameters
    ----------
    min_shift : int, optional, default: ``1``
        The minimum number of shifts to check for.

    max_shift : int, optional, default: ``10``
        The maximum number of shifts to check for.

    target_col : str, optional, default: ``'y'``
            The column to use as the a reference (i.e., the columns which is not
            shifted).

    dropna : bool, optional, default: ``False``
        Determines if the Nan values created by shifting are retained or dropped.

    bootstrap_iterations : int, optional, default: ``None``
        If not None, compute the p_values of the test, by performing bootstrap.

    Examples
    --------
    >>> from gtime.causality.pearson_correlation import ShiftedPearsonCorrelation
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
        min_shift: int = 1,
        max_shift: int = 10,
        target_col: str = "y",
        dropna: bool = False,
        bootstrap_iterations: int = None,
    ):
        super().__init__(bootstrap_iterations=bootstrap_iterations)
        self.min_shift = min_shift
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
        best_shifts = self._compute_best_shifts(data, self._get_max_corr_shift)

        pivot_tables = self._create_pivot_tables(best_shifts)

        self.best_shifts_ = pivot_tables["best_shifts"]
        self.max_corrs_ = pivot_tables["max_corrs"]

        if self.bootstrap_iterations:
            self.p_values_ = pivot_tables["p_values"]

        return self

    def _get_max_corr_shift(self, data: pd.DataFrame, x, y):
        shifts = pd.DataFrame()

        for shift in range(self.min_shift, self.max_shift):
            shifts[shift] = data[x].shift(shift)

        shifts = shifts.dropna()
        self.shifted_corrs = shifts.corrwith(data[y])

        q = self.shifted_corrs.max(), self.shifted_corrs.idxmax()
        return q
