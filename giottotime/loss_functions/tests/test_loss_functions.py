import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


from giottotime.loss_functions import smape, max_error


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
        error = np.amax(np.absolute(np.subtract(y_true, y_pred)))
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

        error = max_error(y_pred, y_true)
        expected_max_error = 7

        assert expected_max_error, error

    def test_max_error_array(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([-1, 4, 5, 10, 4, 1])

        error = max_error(y_pred, y_true)
        expected_max_error = 7

        assert expected_max_error, error

    def test_max_error_dataframe(self):
        y_true = pd.DataFrame([0, 1, 2, 3, 4, 5])
        y_pred = pd.DataFrame([-1, 4, 5, 10, 4, 1])

        error = max_error(y_pred, y_true)
        expected_max_error = 7

        assert expected_max_error, error

    @given(
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
        arrays(float, shape=30, elements=floats(allow_nan=False, allow_infinity=False)),
    )
    def test_smape_random_arrays_finite_values(self, y_true, y_pred):
        error = max_error(y_true, y_pred)
        expected_error = self._correct_max_error(y_true, y_pred)
        print(y_true)
        print(y_pred)
        assert expected_error == error
