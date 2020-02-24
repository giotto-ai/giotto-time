import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def _loglikelihood(y_pred, y_true):
    """Helper function to calculate the loglikelihood for the log likelihood chi2 test

    """

    diff = y_true - y_pred
    std_predictions = np.std(diff)
    llh = -(len(y_true) / 2) * np.log(2.0 * np.pi * std_predictions * std_predictions) \
        - ((np.dot(diff.T, diff)) / (2.0 * std_predictions * std_predictions))
    return llh

def _whiten(x):
    """Helper function to whiten the data (i.e. )

    """

    x = np.append(x, np.ones((len(x), 1)), axis=1)
    weights = np.array([1.])
    if x.ndim == 1:
        return x * np.sqrt(weights)
    elif x.ndim == 2:
        return np.sqrt(weights)[:, None] * x
        
def pseudoinv_extended(X, ratio=1e-15):
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
    res = np.dot(np.transpose(vt), s[:, np.core.newaxis]*np.transpose(u))
    return res

def _ssr_f(**kwargs):
    linreg_single_residues, linreg_joint_residues, max_shift, dof_joint = kwargs['linreg_single_residues'], kwargs['linreg_joint_residues'], \
                                                                          kwargs['max_shift'], kwargs['dof_joint']
    
    f_stat = ((linreg_single_residues - linreg_joint_residues) / 
               linreg_joint_residues / max_shift * dof_joint)                            
    
    result_df = pd.DataFrame()
    result_df['ssr F-test'] = [f_stat, stats.f.sf(f_stat, max_shift, dof_joint), int(dof_joint), int(max_shift)]
    result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

def _ssr_chi2(**kwargs):
    data_single, linreg_single_residues, linreg_joint_residues, dof_joint, max_shift = kwargs['data_single'], kwargs['linreg_single_residues'], \
                                                                                       kwargs['linreg_joint_residues'], kwargs['dof_joint'], \
                                                                                       kwargs['max_shift']
    
    fgc2 = len(data_single) * (linreg_single_residues - linreg_joint_residues) / linreg_joint_residues
    
    result_df = pd.DataFrame()
    result_df['ssr_chi2test'] = [fgc2, stats.chi2.sf(fgc2, max_shift), int(dof_joint), int(max_shift)]
    result_df.index = ['chi2', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

def _likelihood_chi2(**kwargs):
    y_pred_single, y_pred_joint, data = kwargs['y_pred_single'], kwargs['y_pred_joint'], kwargs['data']
    data_single, data_joint, dof_joint = kwargs['data_single'], kwargs['data_joint'], kwargs['dof_joint']
    max_shift, x_col = kwargs['max_shift'], kwargs['x_col']
    
    linreg_single_loglikelihood = _loglikelihood(y_pred=y_pred_single, y_true=data[x_col].loc[data_single.index])
    linreg_joint_loglikelihood = _loglikelihood(y_pred=y_pred_joint, y_true=data[x_col].loc[data_joint.index])
    
    likelihood_ratio = -2 * (linreg_single_loglikelihood - linreg_joint_loglikelihood)

    result_df = pd.DataFrame()
    result_df['likelihood ratio test'] = [likelihood_ratio, stats.chi2.sf(likelihood_ratio, max_shift), int(dof_joint), int(max_shift)]
    result_df.index = ['chi2', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

def _zero_f(**kwargs):
    """
    Link: http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/4-2-Multiple-OLS.html (especially the part about hypothesis testing)

    """
    data_joint, linreg_joint, data, y_pred_joint = kwargs['data_joint'], kwargs['linreg_joint'], kwargs['data'], kwargs['y_pred_joint']
    linreg_joint_residues, dof_joint, max_shift, x_col = kwargs['linreg_joint_residues'], kwargs['dof_joint'], kwargs['max_shift'], kwargs['x_col']
    
    constraint_matrix = np.column_stack((np.zeros((max_shift, max_shift)),
                                         np.eye(max_shift, max_shift),
                                         np.zeros((max_shift, 1))))
    y_true = data[x_col].loc[data_joint.index].values 
    value_restriction = np.zeros(len(constraint_matrix)) 
    
    # Parameters of the fitted linear regression model
    linreg_params = list(linreg_joint.coef_)
    linreg_params.append(linreg_joint.intercept_)
    linreg_params = np.array([linreg_params])
    constraint_params = np.dot(constraint_matrix, linreg_params.T)
    params_diff = constraint_params - value_restriction
        
    pseudoinv_data = pseudoinv_extended(_whiten(data_joint.values))
    
    # Covariance matrix
    scale = linreg_joint_residues / dof_joint
    covar = np.dot(pseudoinv_data, np.transpose(pseudoinv_data)) * scale
    covar = np.dot(constraint_matrix, np.dot(covar, constraint_matrix.T))
    invcov = np.linalg.pinv(covar)

    f = (np.dot(np.dot(params_diff.T, invcov), params_diff) / len(constraint_matrix))[0, 0]
    pvalue = stats.f.sf(f, len(constraint_matrix), dof_joint)
    
    result_df = pd.DataFrame()
    result_df['F-test'] = [f, pvalue, int(dof_joint), int(max_shift)]
    result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
    return result_df

STAT_TESTS = {'ssr_f': _ssr_f,
              'ssr_chi2': _ssr_chi2,
              'likelihood_chi2': _likelihood_chi2,
              'zero_f': _zero_f}

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
    statistics : list, optional, default: ['ssr_f']
        The statistical test(s) to perform for Granger causality. A list with elements
        from the set: 'ssr_f' (sum squared residuals with F-test), 'ssr_chi2' (sum squared 
        residuals with chi square test), 'likelihood_chi2' (likelihood ratio test with 
        chi square distribution), 'zero_F' (F-test that all lag coefficients of 
        the time series X are zero).

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
                 statistics=['ssr_f']):
        self.target_col = target_col
        self.x_col = x_col
        self.max_shift = max_shift
        self.statistics = statistics
        
    def fit(self, data: pd.DataFrame):
        """Create a dataframe with the results of the Granger causality test with the specified
        statistical test(s).

        Parameters
        ----------
        data : pd.DataFrame, shape (n_samples, n_time_series), required
            The dataframe containing the time series.

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

        linreg_single = LinearRegression()
        linreg_joint = LinearRegression()
        linreg_single.fit(data_single, data[x].loc[data_single.index])
        linreg_joint.fit(data_joint, data[x].loc[data_joint.index])

        y_pred_single = linreg_single.predict(data_single)
        y_pred_joint = linreg_joint.predict(data_joint)
        
        dof_single = float(data_single.shape[0] - data_single.shape[1]) 
        dof_joint = float(data_joint.shape[0] - data_joint.shape[1]) -1  
                
        linreg_single_residues = linreg_single._residues
        linreg_joint_residues = linreg_joint._residues
        
        self.results_ = []

        for s in self.statistics:
            self.results_.append(STAT_TESTS[s](linreg_single_residues=linreg_single_residues, linreg_joint_residues=linreg_joint_residues, 
                                               dof_joint=dof_joint, max_shift=self.max_shift, data_single=data_single, 
                                               y_pred_single=y_pred_single, y_pred_joint=y_pred_joint, data = data, x_col=self.x_col,
                                               data_joint=data_joint, linreg_joint=linreg_joint))

        return self
