import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def loglikelihood(y_pred, y_true):
    """Helper function to calculate the loglikelihood for the log likelihood chi2 test

    """

    diff = y_true - y_pred
    std_predictions = np.std(diff)
    llh = -(len(y_true) / 2) * np.log(2.0 * np.pi * std_predictions * std_predictions) \
        - ((np.dot(diff.T, diff)) / (2.0 * std_predictions * std_predictions))
    return llh

def whiten(x):
    """Helper function to whiten data

    """

    x = np.append(x, np.ones((len(x), 1)), axis=1)
    weights = np.array([1.])
    if x.ndim == 1:
        return x * np.sqrt(weights)
    elif x.ndim == 2:
        return np.sqrt(weights)[:, None] * x
        
def pinv_extended(X, ratio=1e-15):
    """Calculate pseudoinverse. Code adapted from statstools and numpy

    """

    X = np.asarray(X)
    X = X.conjugate()
    u, s, vt = np.linalg.svd(X, 0)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = ratio * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i]>cutoff:
            s[i] = 1. / s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res

def _ssr_f(**kwargs):
    lr_single_residues, lr_joint_residues, max_shift, dof_joint = kwargs['lr_single_residues'], kwargs['lr_joint_residues'], \
                                                                  kwargs['max_shift'], kwargs['dof_joint']
    
    result_df = pd.DataFrame()
    f_stat = ((lr_single_residues - lr_joint_residues) / 
            lr_joint_residues / max_shift * dof_joint)                            
    
    result_df['ssr F-test'] = [f_stat, stats.f.sf(f_stat, max_shift, dof_joint), int(dof_joint), int(max_shift)]
    result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

def _ssr_chi2(**kwargs):
    data_single, lr_single_residues, lr_joint_residues, dof_joint, max_shift = kwargs['data_single'], kwargs['lr_single_residues'], \
                                                                               kwargs['lr_joint_residues'], kwargs['dof_joint'], \
                                                                               kwargs['max_shift']
    
    result_df = pd.DataFrame()
    fgc2 = len(data_single) * (lr_single_residues - lr_joint_residues) / lr_joint_residues
    result_df['ssr_chi2test'] = [fgc2, stats.chi2.sf(fgc2, max_shift), int(dof_joint), int(max_shift)]
    result_df.index = ['chi2', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

def _likelihood_chi2(**kwargs):
    y_pred_single, y_pred_joint, data = kwargs['y_pred_single'], kwargs['y_pred_joint'], kwargs['data']
    data_single, data_joint, dof_joint = kwargs['data_single'], kwargs['data_joint'], kwargs['dof_joint']
    max_shift, x_col = kwargs['max_shift'], kwargs['x_col']
    
    result_df = pd.DataFrame()
    lr_single_loglikelihood = loglikelihood(y_pred=y_pred_single, y_true=data[x_col].loc[data_single.index])
    lr_joint_loglikelihood = loglikelihood(y_pred=y_pred_joint, y_true=data[x_col].loc[data_joint.index])
    
    lr = -2 * (lr_single_loglikelihood - lr_joint_loglikelihood)

    result_df['likelihood ratio test'] = [lr, stats.chi2.sf(lr, max_shift), int(dof_joint), int(max_shift)]
    result_df.index = ['chi2', 'p-value', 'degrees of freedom','number of shifts']
    return result_df

def _zero_f(**kwargs):
    data_joint, lr_joint, data, y_pred_joint = kwargs['data_joint'], kwargs['lr_joint'], kwargs['data'], kwargs['y_pred_joint']
    lr_joint_residues, dof_joint, max_shift, x_col = kwargs['lr_joint_residues'], kwargs['dof_joint'], kwargs['max_shift'], kwargs['x_col']
    
    result_df = pd.DataFrame()
    r_matrix = np.column_stack((np.zeros((max_shift, max_shift)),
                                np.eye(max_shift, max_shift),
                                np.zeros((max_shift, 1))))
    
    y_true = data[x_col].loc[data_joint.index].values 
    q_matrix = np.zeros(len(r_matrix)) 
    
    params = list(lr_joint.coef_)
    params.append(lr_joint.intercept_)
    params = np.array(params) 
    cparams = np.dot(r_matrix, params[:, None])
    rbq = cparams - q_matrix
    
    scale = mean_squared_error(y_pred_joint, data[x_col].loc[data_joint.index])
    scale = lr_joint_residues/dof_joint
    params = params.reshape(1, -1)
    
    pinv_wexog = pinv_extended(whiten(data_joint.values.reshape(len(data_joint), -1)))
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog)) 
    
    cov_p = normalized_cov_params * scale 
    cov_p = np.dot(r_matrix, np.dot(cov_p, np.transpose(r_matrix)))
    invcov = np.linalg.pinv(cov_p)

    f = np.dot(np.dot(rbq.T, invcov), rbq) 
    f /= len(r_matrix)
    f = np.unique(f)[0]

    pvalue = stats.f.sf(f, len(r_matrix), dof_joint)
    pvalue = np.unique(pvalue)[0]
    
    result_df['F-test'] = [f, pvalue, int(dof_joint), int(max_shift)]
    result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
    
    return result_df

class GrangerCausality(BaseEstimator):
    """Class to check for Granger causality between two time series, i.e. 
    to check if a time series X causes Y: X->Y.

    Parameters
    ----------
    target_col : str, 
        The column to use as the reference, i.e. the time series Y
    x_col : str
        The column to test for Granger causality, i.e. the time 
        series X.
    max_shift : int
        The maximal number of shifts to check for Granger causality
    statistics : str, optional, default: 'ssr_f'
        The statistical test to perform for Granger causality. Either 'ssr_f'
        (sum squared residuals with F-test), 'ssr_chi2' (sum squared residuals 
        with chi square test), 'likelihood_chi2' (likelihood ratio test with 
        chi square distribution), 'zero_F' (F-test that all lag coefficients of 
        the time series X are zero) or 'all' (perform all the tests mentioned 
        above)

    Examples
    --------
    >>> from gtime.causality.granger_causality import GrangerCausality
    >>> import pandas.util.testing as testing
    >>> data = testing.makeTimeDataFrame(freq="s", nper=1000)
    >>> GrangerCausality(target_col='A', x_col='B', max_shift=10, statistics='ssr_f').fit(data)
                        ssr F-test
    F-value               0.372640
    p-value               0.958527
    degrees of freedom  969.000000
    number of shifts     10.000000

    """

    def __init__(self, 
                 target_col: str, 
                 x_col: str, 
                 max_shift: int, 
                 statistics='ssr_f'):
        self.target_col = target_col
        self.x_col = x_col
        self.max_shift = max_shift
        self.statistics = statistics
        self.results_ = []

    def fit(self, data: pd.DataFrame):
        """Create a dataframe with the results of the Granger causality test with the specified
        statistical test(s).

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The dataframe containing the time series.

        Returns
        -------
        result : pd.DataFrame (if only one test is selected) or tuple if statics=='all'
            A dataframe or a tuple of dataframes for each selected statistical test.

        """

        x = self.x_col
        y = self.target_col

        shifts = pd.DataFrame()
        
        for shift in range(0, self.max_shift+1):
            shifts[str(x)+'_'+str(shift)] = data[x].shift(shift)
            if shift > 0:
                shifts[str(y)+'_'+str(shift-1)] = data[y].shift(shift)
        shifts = shifts.dropna()
                
        y_columns = [c for c in shifts.columns if c.split('_')[0]==y]
        x_columns = [c for c in shifts.columns if c.split('_')[0]==x]
        
        data_single = shifts[x_columns].copy()
        data_joint = shifts[x_columns + y_columns].copy()
        data_single.drop([str(x)+'_'+str(0)], inplace=True, axis='columns')
        data_joint.drop([str(x)+'_'+str(0)], inplace=True, axis='columns')

        lr_single = LinearRegression()
        lr_joint = LinearRegression()
        lr_single.fit(data_single, data[x].loc[data_single.index])
        lr_joint.fit(data_joint, data[x].loc[data_joint.index])

        y_pred_single = lr_single.predict(data_single)
        y_pred_joint = lr_joint.predict(data_joint)
        
        try:
            dof_single = float(data_single.shape[0] - data_single.shape[1]) 
        except:
            dof_single = float(data_single.shape[0] - 1)
        dof_joint = float(data_joint.shape[0] - data_joint.shape[1]) -1  
                
        lr_single_residues = lr_single._residues
        lr_joint_residues = lr_joint._residues

        if self.statistics=='all':
            stat = ['ssr_f', 'ssr_chi2', 'likelihood_chi2', 'zero_f']
        else:
            stat = self.statistics

        stats = {'ssr_f': _ssr_f,
                 'ssr_chi2': _ssr_chi2,
                 'likelihood_chi2': _likelihood_chi2,
                 'zero_f': _zero_f}

        if isinstance(stat, list):
            for s in stat:
                self.results_.append(stats[s](lr_single_residues = lr_single_residues, lr_joint_residues = lr_joint_residues, 
                                              dof_joint = dof_joint, max_shift = self.max_shift, data_single = data_single, 
                                              y_pred_single = y_pred_single, y_pred_joint = y_pred_joint, data = data, x_col = self.x_col,
                                              data_joint=data_joint, lr_joint=lr_joint))
        elif isinstance(self.statistics, str):
            self.results_ = stats[self.statistics](lr_single_residues = lr_single_residues, lr_joint_residues = lr_joint_residues, 
                                                   dof_joint = dof_joint, max_shift = self.max_shift, data_single = data_single, 
                                                   y_pred_single = y_pred_single, y_pred_joint = y_pred_joint, data = data, x_col = self.x_col,
                                                   data_joint=data_joint, lr_joint=lr_joint)
        else:
            raise TypeError('Parameter statistics must be either one of the specified tests or a list thereof.')
        return self
