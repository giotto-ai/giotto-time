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
    loglikelihood = -(len(y_true) / 2) * np.log(
        2.0 * np.pi * std_predictions * std_predictions
    ) - ((np.dot(diff.T, diff)) / (2.0 * std_predictions * std_predictions))
    return loglikelihood


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
        if s[i] > cutoff:
            s[i] = 1.0 / s[i]
        else:
            s[i] = 0.0

    res = np.dot(np.transpose(vt), s[:, np.core.newaxis] * np.transpose(u))
    return res


def _ssr_f(params_dict):
    linreg_single_residues, linreg_joint_residues, max_shift, dof_joint = (
        params_dict["linreg_single_residues"],
        params_dict["linreg_joint_residues"],
        params_dict["max_shift"],
        params_dict["dof_joint"],
    )

    f_stat = (
        (linreg_single_residues - linreg_joint_residues)
        / linreg_joint_residues
        / max_shift
        * dof_joint
    )

    result_df = pd.DataFrame()
    result_df["ssr F-test"] = [
        f_stat,
        stats.f.sf(f_stat, max_shift, dof_joint),
        int(dof_joint),
        int(max_shift),
    ]
    result_df.index = ["F-value", "p-value", "degrees of freedom", "number of shifts"]
    return result_df


def _ssr_chi2(params_dict):
    data_single, linreg_single_residues, linreg_joint_residues, dof_joint, max_shift = (
        params_dict["data_single"],
        params_dict["linreg_single_residues"],
        params_dict["linreg_joint_residues"],
        params_dict["dof_joint"],
        params_dict["max_shift"],
    )

    chi2_stat = (
        len(data_single)
        * (linreg_single_residues - linreg_joint_residues)
        / linreg_joint_residues
    )

    result_df = pd.DataFrame()
    result_df["ssr_chi2test"] = [
        chi2_stat,
        stats.chi2.sf(chi2_stat, max_shift),
        int(dof_joint),
        int(max_shift),
    ]
    result_df.index = ["chi2", "p-value", "degrees of freedom", "number of shifts"]
    return result_df


def _likelihood_chi2(params_dict):
    y_pred_single, y_pred_joint, data = (
        params_dict["y_pred_single"],
        params_dict["y_pred_joint"],
        params_dict["data"],
    )
    data_single, data_joint, dof_joint = (
        params_dict["data_single"],
        params_dict["data_joint"],
        params_dict["dof_joint"],
    )
    max_shift, target_col = params_dict["max_shift"], params_dict["target_col"]

    linreg_single_loglikelihood = _loglikelihood(
        y_pred=y_pred_single, y_true=data[target_col].loc[data_single.index]
    )
    linreg_joint_loglikelihood = _loglikelihood(
        y_pred=y_pred_joint, y_true=data[target_col].loc[data_joint.index]
    )

    likelihood_ratio = -2 * (linreg_single_loglikelihood - linreg_joint_loglikelihood)

    result_df = pd.DataFrame()
    result_df["likelihood ratio test"] = [
        likelihood_ratio,
        stats.chi2.sf(likelihood_ratio, max_shift),
        int(dof_joint),
        int(max_shift),
    ]
    result_df.index = ["chi2", "p-value", "degrees of freedom", "number of shifts"]
    return result_df


def _zero_f(params_dict):
    data_joint, linreg_joint, data, y_pred_joint = (
        params_dict["data_joint"],
        params_dict["linreg_joint"],
        params_dict["data"],
        params_dict["y_pred_joint"],
    )
    linreg_joint_residues, dof_joint, max_shift, target_col = (
        params_dict["linreg_joint_residues"],
        params_dict["dof_joint"],
        params_dict["max_shift"],
        params_dict["target_col"],
    )

    constraint_matrix = np.column_stack(
        (
            np.zeros((max_shift, max_shift)),
            np.eye(max_shift, max_shift),
            np.zeros((max_shift, 1)),
        )
    )
    y_true = data[target_col].loc[data_joint.index].values
    value_restriction = np.zeros(len(constraint_matrix))

    # Parameters of the fitted linear regression model
    linreg_params = list(linreg_joint.coef_)
    linreg_params.append(linreg_joint.intercept_)
    linreg_params = np.array([linreg_params])
    constraint_params = np.dot(constraint_matrix, linreg_params.T)
    params_diff = constraint_params - value_restriction

    pseudoinv_data = pseudoinv_extended(
        np.append(data_joint.values, np.ones((len(data_joint.values), 1)), axis=1)
    )

    # Covariance matrix
    scale = linreg_joint_residues / dof_joint
    covar = np.dot(pseudoinv_data, np.transpose(pseudoinv_data)) * scale
    covar = np.dot(constraint_matrix, np.dot(covar, constraint_matrix.T))
    invcov = np.linalg.pinv(covar)

    f = (np.dot(np.dot(params_diff.T, invcov), params_diff) / len(constraint_matrix))[
        0, 0
    ]
    pvalue = stats.f.sf(f, len(constraint_matrix), dof_joint)

    result_df = pd.DataFrame()
    result_df["F-test"] = [f, pvalue, int(dof_joint), int(max_shift)]
    result_df.index = ["F-value", "p-value", "degrees of freedom", "number of shifts"]
    return result_df


STAT_TESTS = {
    "ssr_f": _ssr_f,
    "ssr_chi2": _ssr_chi2,
    "likelihood_chi2": _likelihood_chi2,
    "zero_f": _zero_f,
}


class GrangerCausality(BaseEstimator):
    """Class to check for Granger causality between two time series, i.e. 
    to check if a time series X causes Y: X->Y.

    Parameters
    ----------
    target_col : str
        The column to use as the reference, i.e. the time series Y

    x_col : str
        The column to test for Granger causality, i.e. the time 
        series X.

    max_shift : int, optional, default: 10
        The maximal number of shifts to check for Granger causality. 

    statistics : list, optional, default: ['ssr_f']
        The statistical test(s) to perform for Granger causality. A list with elements
        from the set: 'ssr_f' (sum squared residuals with F-test), 'ssr_chi2' (sum squared 
        residuals with chi square test), 'likelihood_chi2' (likelihood ratio test with 
        chi square distribution), 'zero_F' (F-test that all lag coefficients of 
        the time series X are zero).

    Attributes
    ----------
    results_ : list
        A list of pandas dataframes with the results for all the listed tests in 'statistics'.

    Examples
    --------
    >>> from gtime.causality.granger_causality import GrangerCausality
    >>> import pandas.util.testing as testing
    >>> data = testing.makeTimeDataFrame(freq="s", nper=1000)
    >>> gc = GrangerCausality(target_col='A', x_col='B', max_shift=10, statistics=['ssr_f']).fit(data)
    >>> gc.results_[0]
                        ssr F-test
    F-value               0.372640
    p-value               0.958527
    degrees of freedom  969.000000
    number of shifts     10.000000

    """

    def __init__(self, target_col: str, x_col: str, max_shift=10, statistics=["ssr_f"]):
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
        self : object
            Returns the instance itself.

        """

        shifts = data.copy()
        x_columns, y_columns = [], []
        for i in range(1, self.max_shift + 1):
            shifts[f"x_shift_{i}"] = data[self.target_col].shift(i)
            shifts[f"y_shift_{i-1}"] = data[self.x_col].shift(i)
            x_columns.append(f"x_shift_{i}")
            y_columns.append(f"y_shift_{i-1}")
        shifts.drop([self.target_col, self.x_col], axis="columns", inplace=True)
        shifts = shifts.dropna()

        data_single = shifts[x_columns].copy()
        data_joint = shifts[x_columns + y_columns].copy()

        linreg_single = LinearRegression()
        linreg_joint = LinearRegression()
        y_pred_single = linreg_single.fit(
            data_single, data[self.target_col].loc[data_single.index]
        ).predict(data_single)
        y_pred_joint = linreg_joint.fit(
            data_joint, data[self.target_col].loc[data_joint.index]
        ).predict(data_joint)

        dof_single = float(data_single.shape[0] - data_single.shape[1])
        dof_joint = float(data_joint.shape[0] - data_joint.shape[1]) - 1

        linreg_single_residues = np.sum(
            (y_pred_single - data[self.target_col].loc[data_single.index])**2)
        linreg_joint_residues = np.sum(
            (y_pred_joint - data[self.target_col].loc[data_joint.index])**2)

        self.results_ = []
        stat_test_input = {
            "linreg_single_residues": linreg_single_residues,
            "linreg_joint_residues": linreg_joint_residues,
            "dof_joint": dof_joint,
            "max_shift": self.max_shift,
            "data_single": data_single,
            "y_pred_single": y_pred_single,
            "y_pred_joint": y_pred_joint,
            "data": data,
            "target_col": self.target_col,
            "data_joint": data_joint,
            "linreg_joint": linreg_joint,
        }

        for s in self.statistics:
            self.results_.append(STAT_TESTS[s](stat_test_input))

        return self
