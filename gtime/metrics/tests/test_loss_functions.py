import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from gtime.metrics import smape, max_error, mse, log_mse, r_square, mae, mape


class TestSmape:
    def _correct_smape(self, y_true, y_pred):
        non_normalized_smape = sum(
            np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
        )
        non_normalized_smape_filled = np.nan_to_num(non_normalized_smape, nan=0)
        error = (2 / len(y_pred)) * non_normalized_smape_filled
        return error

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            smape(y_true, y_pred)

    def test_nan_values(self):
        y_true = np.array([np.nan, 1, 2, 3])
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            smape(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = np.array([0, np.inf, 2, 3])

        with pytest.raises(ValueError):
            smape(y_true, y_pred)

    def test_correct_smape_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([3, 2, 4, 3, 5, 7])

        error = smape(y_true, y_pred)
        expected_error = (1 + 1 / 3 + 1 / 3 + 0 + 1 / 9 + 1 / 6) * (2 / 6)

        assert expected_error == error

    def test_correct_smape_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([3, 2, 4, 3, 5, 7])

        error = smape(y_true, y_pred)
        expected_error = (1 + 1 / 3 + 1 / 3 + 0 + 1 / 9 + 1 / 6) * (2 / 6)

        assert expected_error == error

    def test_correct_smape_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [3, 2, 4, 3, 5, 7]

        error = smape(y_true, y_pred)
        expected_error = (1 + 1 / 3 + 1 / 3 + 0 + 1 / 9 + 1 / 6) * (2 / 6)

        assert expected_error == error

    def test_smape_is_symmetric(self):
        y_true = np.random.random(10)
        y_pred = np.random.random(10)

        first_error = smape(y_true, y_pred)
        second_error = smape(y_pred, y_true)

        assert first_error == second_error

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_between_zero_and_one(self, y_true, y_pred):
        error = smape(y_true, y_pred)

        assert error >= 0
        assert error <= 2

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_smape_random_arrays(self, y_true, y_pred):
        error = smape(y_true, y_pred)
        expected_error = self._correct_smape(y_true, y_pred)

        assert expected_error == error

class TestMaxError:
    def _correct_max_error(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        error = np.amax(np.absolute((y_true - y_pred)))
        return error

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            max_error(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            max_error(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            max_error(y_true, y_pred)

    def test_max_error_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [-1, 4, 5, 10, 4, 1]

        error = max_error(y_true, y_pred)
        expected_max_error = 7

        assert expected_max_error == error

    def test_max_error_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([-1, 4, 5, 10, 4, 1])

        error = max_error(y_true, y_pred)
        expected_max_error = 7

        assert expected_max_error == error

    def test_max_error_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 4, 5, 10, 4, 1])

        error = max_error(y_true, y_pred)
        expected_max_error = 7

        assert expected_max_error == error

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_max_error_random_arrays_finite_values(self, y_true, y_pred):
        error = max_error(y_true, y_pred)
        expected_error = self._correct_max_error(y_true, y_pred)
        print(y_true)
        print(y_pred)
        assert expected_error == error


class TestMSE:
    def _correct_mse(self, y_true, y_pred, rmse=False):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sum_squared_error = sum(((y_true - y_pred)) ** 2)
        mse = sum_squared_error / len(y_true)
        return np.sqrt(mse) if rmse else mse

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mse(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mse(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            mse(y_true, y_pred)

    def test_mse_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [-1, 4, 5, 10, 4, 1]

        mse_value = np.round(mse(y_true, y_pred), decimals=2)
        expected_mse = 14
        rmse_value = np.round(mse(y_true, y_pred, rmse=True), decimals=2)
        expected_rmse = 3.74
        
        assert expected_mse == mse_value
        assert expected_rmse == rmse_value

    def test_mse_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([-1, 4, 5, 10, 4, 1])

        mse_value = np.round(mse(y_true, y_pred), decimals=2)
        expected_mse = 14
        rmse_value = np.round(mse(y_true, y_pred, rmse=True), decimals=2)
        expected_rmse = 3.74
        
        assert expected_mse == mse_value
        assert expected_rmse == rmse_value

    def test_mse_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 4, 5, 10, 4, 1])

        mse_value = np.round(mse(y_true, y_pred), decimals=2)
        expected_mse = 14
        rmse_value = np.round(mse(y_true, y_pred, rmse=True), decimals=2)
        expected_rmse = 3.74
        
        assert expected_mse == mse_value
        assert expected_rmse == rmse_value

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_mse_random_arrays_finite_values(self, y_true, y_pred):
        mse_value = mse(y_true, y_pred)
        expected_mse = self._correct_mse(y_true, y_pred)
        rmse_value = mse(y_true, y_pred, rmse=True)
        expected_rmse = self._correct_mse(y_true, y_pred, rmse=True)
        print(y_true)
        print(y_pred)

        assert expected_mse == mse_value
        assert expected_rmse == rmse_value


class TestLogMSE:
    def _correct_log_mse(self, y_true, y_pred, rmsle=False):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (np.any(y_true < 0)) or (np.any(y_pred < 0)):
            raise ValueError("MSLE can not be used when inputs contain Negative values") 
        log_y_true = np.log(y_true + 1)
        log_y_pred = np.log(y_pred + 1)
        log_mse = mse(log_y_true, log_y_pred)
        return np.sqrt(log_mse) if rmsle else log_mse

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            log_mse(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            log_mse(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            log_mse(y_true, y_pred)

    def test_log_mse_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [1, 4, 5, 10, 4, 1]

        log_mse_value = np.round(log_mse(y_true, y_pred), decimals=2)
        expected_log_mse = 0.67
        rmsle_value = np.round(log_mse(y_true, y_pred, rmsle=True), decimals=2)
        expected_rmsle = 0.82

        assert expected_log_mse == log_mse_value
        assert expected_rmsle == rmsle_value

    def test_log_mse_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([1, 4, 5, 10, 4, 1])

        log_mse_value = np.round(log_mse(y_true, y_pred), decimals=2)
        expected_log_mse = 0.67
        rmsle_value = np.round(log_mse(y_true, y_pred, rmsle=True), decimals=2)
        expected_rmsle = 0.82

        assert expected_log_mse == log_mse_value
        assert expected_rmsle == rmsle_value

    def test_log_mse_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([1, 4, 5, 10, 4, 1])

        log_mse_value = np.round(log_mse(y_true, y_pred), decimals=2)
        expected_log_mse = 0.67
        rmsle_value = np.round(log_mse(y_true, y_pred, rmsle=True), decimals=2)
        expected_rmsle = 0.82

        assert expected_log_mse == log_mse_value
        assert expected_rmsle == rmsle_value

    def test_log_mse_negative_values(self):
        y_true = np.array([0, 1, 2, 3, 4, -5])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        with pytest.raises(ValueError):
            log_mse(y_true, y_pred) 

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False, min_value=0)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False, min_value=0)),
    )
    def test_log_mse_random_arrays_finite_values(self, y_true, y_pred):
        log_mse_value = log_mse(y_true, y_pred)
        expected_log_mse = self._correct_log_mse(y_true, y_pred)
        rmsle_value = log_mse(y_true, y_pred, rmsle=True)
        expected_rmsle = self._correct_log_mse(y_true, y_pred, rmsle=True)
        print(y_true)
        print(y_pred)

        assert expected_log_mse == log_mse_value
        assert expected_rmsle == rmsle_value


class TestRSquare:
    def _correct_r_squared(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = sum(((y_true - y_pred)) ** 2)
        ss_tot = sum((y_true - np.mean(y_true)) ** 2)
        if not np.any(ss_tot):
            if not np.any(ss_res):
                return 1.0
            else:
                return 0.0
        if np.isnan((ss_res / ss_tot)):
            return np.NINF 
        r_square = 1 - (ss_res / ss_tot)
        return r_square

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            r_square(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            r_square(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            r_square(y_true, y_pred)

    def test_r_square_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [-1, 4, 5, 10, 4, 1]

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = -3.8

        assert expected_r_square == r_square_value

    def test_r_square_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([-1, 4, 5, 10, 4, 1])

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = -3.8

        assert expected_r_square == r_square_value

    def test_r_square_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 4, 5, 10, 4, 1])

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = -3.8

        assert expected_r_square == r_square_value

    def test_r_square_all_zero_values(self):
        # The test checks for all zero inputs covering the 'if' condition 
        # to check if both ss_res and ss_tot are all Zeros
        y_true = np.array([0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = 1.0

        assert expected_r_square == r_square_value

    def test_r_square_ss_tot_zero(self):
        # The test checks for 'if' condition 
        # to check if ss_res is certain value and ss_tot are all Zeros
        y_true = np.array([0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1])

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = 0.0

        assert expected_r_square == r_square_value

    def test_r_square_ss_res_ss_tot_infinity(self):
        # The test checks for 'if' condition 
        # to check if both ss_res and ss_tot are Infinite
        y_true = [0.00000000e+000, 0.00000000e+000, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200, 9.81351055e+200, 9.81351055e+200,
                  9.81351055e+200, 9.81351055e+200]
        y_pred = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

        r_square_value = np.round(r_square(y_true, y_pred), decimals=2)
        expected_r_square = np.NINF

        assert expected_r_square == r_square_value

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_r_square_random_arrays_finite_values(self, y_true, y_pred):
        r_square_value = r_square(y_true, y_pred)
        expected_r_square = self._correct_r_squared(y_true, y_pred)
        print(y_true)
        print(y_pred)
       
        assert expected_r_square == r_square_value


class TestMAE:
    def _correct_mae(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mae(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mae(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            mae(y_true, y_pred)

    def test_mae_list(self):
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [-1, 4, 5, 10, 4, 1]

        mae_value = np.round(mae(y_true, y_pred), decimals=2)
        expected_mae = 3
        
        assert expected_mae == mae_value

    def test_mae_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([-1, 4, 5, 10, 4, 1])

        mae_value = np.round(mae(y_true, y_pred), decimals=2)
        expected_mae = 3
        
        assert expected_mae == mae_value

    def test_mae_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 4, 5, 10, 4, 1])

        mae_value = np.round(mae(y_true, y_pred), decimals=2)
        expected_mae = 3
        
        assert expected_mae == mae_value

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_mae_random_arrays_finite_values(self, y_true, y_pred):
        mae_value = mae(y_true, y_pred)
        expected_mae = self._correct_mae(y_true, y_pred)
        print(y_true)
        print(y_pred)

        assert expected_mae == mae_value


class TestMAPE:
    def _correct_mape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ratio_list = np.abs((y_pred - y_true)/y_true)
        if (0 in y_true):
            if np.isnan(ratio_list).any():
                raise ValueError("MAPE can not be calculated due to Zero/Zero")
            else:
                return np.inf
        else:
            mape_value = np.mean(ratio_list) * 100
        return mape_value

    def test_wrong_vector_length(self):
        y_true = np.random.random(5)
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mape(y_true, y_pred)

    def test_nan_values(self):
        y_true = [np.nan, 1, 2, 3]
        y_pred = np.random.random(4)

        with pytest.raises(ValueError):
            mape(y_true, y_pred)

    def test_infinite_values(self):
        y_true = np.random.random(4)
        y_pred = [0, np.inf, 2, 3]

        with pytest.raises(ValueError):
            mape(y_true, y_pred)

    def test_mape_list(self):
        y_true = [1, 5, 7, 3, 4, 5]
        y_pred = [-1, 3, 5, 4, 4, 3]

        mape_value = np.round(mape(y_true, y_pred), decimals=2)
        expected_mape = 56.98
        
        assert expected_mape == mape_value

    def test_mape_array(self):
        y_true = np.array([1, 5, 7, 3, 4, 5])
        y_pred = np.array([-1, 3, 5, 4, 4, 3])

        mape_value = np.round(mape(y_true, y_pred), decimals=2)
        expected_mape = 56.98
        
        assert expected_mape == mape_value

    def test_mape_dataframe(self):
        y_true = pd.DataFrame([1, 5, 7, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 3, 5, 4, 4, 3])

        mape_value = np.round(mape(y_true, y_pred), decimals=2)
        expected_mape = 56.98
        
        assert expected_mape == mape_value

    def test_y_true_zero_and_ratio_zero(self):
        # If y_true = 0 and (y_pred - y_true) = 0
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 2.3, 0.4, 3.9, 3.1, 4.6]

        with pytest.raises(ValueError):
            mape(y_true, y_pred)

    def test_y_true_contains_zero(self):
        # If y_true = 0 
        y_true = [0, 2, 2, 3, 4, 5]
        y_pred = [2, 2.3, 0.4, 3.9, 3.1, 4.6]

        mape_value = np.round(mape(y_true, y_pred))
        expected_mape = np.inf
        
        assert expected_mape == mape_value

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False, min_value=1)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False, min_value=1)),
    )
    def test_mape_random_arrays_finite_values(self, y_true, y_pred):
        mape_value = mape(y_true, y_pred)
        expected_mape = self._correct_mape(y_true, y_pred)
        print(y_true)
        print(y_pred)

        assert expected_mape == mape_value
