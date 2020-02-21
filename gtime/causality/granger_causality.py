import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
    statistics : str, optional, default: 'ssr_F'
        The statistical test to perform for Granger causality. Either 'ssr_F'
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
    >>> GrangerCausality(target_col='A', x_col='B', max_shift=10, statistics='ssr_F').fit(data)
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
                 statistics='ssr_F'):
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
            DOF_single = float(data_single.shape[0] - data_single.shape[1]) 
        except:
            DOF_single = float(data_single.shape[0] - 1)
        DOF_joint = float(data_joint.shape[0] - data_joint.shape[1]) -1  
                
        lr_single_residues = lr_single._residues
        lr_joint_residues = lr_joint._residues

        result = []
        if self.statistics=='all':
            stat = ['ssr_F', 'ssr_chi2', 'likelihood_chi2', 'zero_F']
        else:
            stat = [self.statistics]

        if 'ssr_F' in stat:
            result_df = pd.DataFrame()
            f_stat = ((lr_single_residues - lr_joint_residues) / 
                    lr_joint_residues / self.max_shift * DOF_joint)                            
            
            result_df['ssr F-test'] = [f_stat, stats.f.sf(f_stat, self.max_shift, DOF_joint), int(DOF_joint), int(self.max_shift)]
            result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
            result.append(result_df)

        if 'ssr_chi2' in stat:
            result_df = pd.DataFrame()
            fgc2 = len(data_single) * (lr_single_residues - lr_joint_residues) / lr_joint_residues
            result_df['ssr_chi2test'] = [fgc2, stats.chi2.sf(fgc2, self.max_shift), int(DOF_joint), int(self.max_shift)]
            result_df.index = ['chi2', 'p-value', 'degrees of freedom', 'number of shifts']
            result.append(result_df)
        
        if 'likelihood_chi2' in stat:
            result_df = pd.DataFrame()
            lr_single_loglikelihood = loglikelihood(y_pred=y_pred_single, y_true=data[x].loc[data_single.index])
            lr_joint_loglikelihood = loglikelihood(y_pred=y_pred_joint, y_true=data[x].loc[data_joint.index])
            
            lr = -2 * (lr_single_loglikelihood - lr_joint_loglikelihood)

            result_df['likelihood ratio test'] = [lr, stats.chi2.sf(lr, self.max_shift), int(DOF_joint), int(self.max_shift)]
            result_df.index = ['chi2', 'p-value', 'degrees of freedom','number of shifts']
            result.append(result_df)

        if 'zero_F' in stat:
            result_df = pd.DataFrame()
            r_matrix = np.column_stack((np.zeros((self.max_shift, self.max_shift)),
                                        np.eye(self.max_shift, self.max_shift),
                                        np.zeros((self.max_shift, 1))))
            
            y_true = data[x].loc[data_joint.index].values 
            q_matrix = np.zeros(len(r_matrix)) 
            
            params = list(lr_joint.coef_)
            params.append(lr_joint.intercept_)
            params = np.array(params) 
            cparams = np.dot(r_matrix, params[:, None])
            Rbq = cparams - q_matrix
            
            scale = mean_squared_error(y_pred_joint, data[x].loc[data_joint.index])
            scale = lr_joint_residues/DOF_joint
            params = params.reshape(1, -1)
            
            pinv_wexog = pinv_extended(whiten(data_joint.values.reshape(len(data_joint), -1)))
            normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog)) 
            
            cov_p = normalized_cov_params * scale 
            cov_p = np.dot(r_matrix, np.dot(cov_p, np.transpose(r_matrix)))
            invcov = np.linalg.pinv(cov_p)

            F = np.dot(np.dot(Rbq.T, invcov), Rbq) 
            F /= len(r_matrix)
            F = np.unique(F)[0]

            pvalue = stats.f.sf(F, len(r_matrix), DOF_joint)
            pvalue = np.unique(pvalue)[0]
            
            result_df['F-test'] = [F, pvalue, int(DOF_joint), int(self.max_shift)]
            result_df.index = ['F-value', 'p-value', 'degrees of freedom', 'number of shifts']
            result.append(result_df)

        if len(result)==1:
            return result[0]
        else:
            return tuple(result)

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