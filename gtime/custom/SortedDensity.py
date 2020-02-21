import pandas as pd
from sklearn.utils.validation import check_is_fitted
from gtime.feature_extraction import MovingCustomFunction

class SortedDensity(MovingCustomFunction):
    """For each row in ``time_series``, compute the sorted density function of the
    previous ``window_size`` rows. If there are not enough rows, the value is ``Nan``.
    Sorted density measured is defined in (eq. 1) of: H. P. Tukuljac, V. Pulkki,
    H. Gamper, K. Godin, I. J. Tashev and N. Raghuvanshi, "A Sparsity Measure for Echo 
    Density Growth in General Environments," ICASSP 2019 - 2019 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United 
    Kingdom, 2019, pp. 1-5.
    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the sorted density.    
    is_causal : bool, optional, default: ``True``
        Whether the current sample is computed based only on the past or also on the future.
    Examples
    --------
    >>> import pandas as pd
    >>> from gtime.feature_extraction import SortedDensity
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> mv_avg = SortedDensity(window_size=2)
    >>> mv_avg.fit_transform(ts)
       0__SortedDensity
    0                      NaN
    1                 0.500000
    2                 0.666667
    3                 0.700000
    4                 0.714286
    5                 0.722222
    --------
    """    
    def __init__(self, window_size: int = 1, is_causal: bool = True):        
        super().__init__(self.sorted_density)
        self.window_size = window_size
        self.is_causal = is_causal
        
    def sorted_density(self, signal):
        import numpy as np
        t = (np.array(range(len(signal))) + 1)
        signal = signal[signal.argsort()[::-1]]
        t = np.reshape(t, signal.shape)
        SD = np.sum(np.multiply(t, signal))/np.sum(signal) # (eq. 2)
        SD = SD/(len(signal))
        return SD
    
    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """For every row of ``time_series``, compute the moving sorted density function of the
         previous ``window_size`` elements.
        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving custom function.
        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``time_series``, containing the rolling
            moving custom function for each element.
        """
        check_is_fitted(self)

        if(self.is_causal):
            time_series_mvg_sd = time_series.rolling(self.window_size).apply(
                self.sorted_density, raw=self.raw
            )
        else:
            time_series_mvg_sd = time_series.rolling(self.window_size, min_periods = int(self.window_size/2)).apply(
                self.sorted_density, raw=self.raw
            )
            time_series_mvg_sd = time_series_mvg_sd.dropna()
            
        time_series_t = time_series_mvg_sd
        return time_series_t