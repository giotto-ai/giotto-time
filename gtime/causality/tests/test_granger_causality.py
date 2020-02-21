import numpy as np
import pytest
import pandas.util.testing as testing
from gtime.causality import GrangerCausality


def test_granger_pvalues():
    # Set random seed, otherwise testing creates a new dataframe each time.
    np.random.seed(12)

    data = testing.makeTimeDataFrame(freq="s", nper=1000)
    result_ssr_F = GrangerCausality(target_col='B', x_col='A', max_shift=10, statistics='ssr_F').fit(data)
    result_ssr_chi2 = GrangerCausality(target_col='B', x_col='A', max_shift=10, statistics='ssr_chi2').fit(data)
    result_lh_chi2 = GrangerCausality(target_col='B', x_col='A', max_shift=10, statistics='likelihood_chi2').fit(data)
    result_zero_F = GrangerCausality(target_col='B', x_col='A', max_shift=10, statistics='zero_F').fit(data)

    p_ssr_F = result_ssr_F.loc['p-value'].values[0]
    p_ssr_chi2 = result_ssr_chi2.loc['p-value'].values[0]
    p_lh_chi2 = result_lh_chi2.loc['p-value'].values[0]
    p_zero_F = result_zero_F.loc['p-value'].values[0]

    test_list = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    # Expected values from comparison with stattools
    expected_p_values = [0.8420421667509344, 0.8327660223526767, 0.8341270186135072, 0.8420421667508992] 

    results_lst = list(np.array([[p_ssr_F, p_ssr_chi2, p_lh_chi2, p_zero_F], expected_p_values]).T)
    results_dict = dict(zip(test_list, results_lst))

    # Not exactly equal but up test to 7 digits
    np.testing.assert_almost_equal(results_dict['ssr_ftest'][0], results_dict['ssr_ftest'][1], decimal=7)
    np.testing.assert_almost_equal(results_dict['ssr_chi2test'][0], results_dict['ssr_chi2test'][1], decimal=7)
    np.testing.assert_almost_equal(results_dict['lrtest'][0], results_dict['lrtest'][1], decimal=7)
    np.testing.assert_almost_equal(results_dict['params_ftest'][0], results_dict['params_ftest'][1], decimal=7)

