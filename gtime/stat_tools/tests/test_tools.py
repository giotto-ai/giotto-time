import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis import given
from gtime.utils.hypothesis.time_indexes import giotto_time_series
from gtime.stat_tools.tools import (
    normalize,
    autocorrelation,
    solve_yw_equation,
    yule_walker,
    acf,
    pacf,
    arma_polynomial_roots,
    durbin_levinson_recursion,
)


class TestAcf:
    @given(x=st.lists(st.floats(allow_nan=False), min_size=1))
    def test_autocorrelation(self, x):
        autocorr = autocorrelation(np.array(x))
        expected = np.correlate(x, x, mode="full")[-len(x) :] / len(x)
        np.testing.assert_array_equal(autocorr, expected)

    @given(
        x=st.lists(
            st.floats(
                allow_nan=False, allow_infinity=False, max_value=1e20, min_value=1e20
            ),
            min_size=1,
        )
    )
    def test_scale(self, x):
        scaled_x = normalize(np.array(x))
        assert scaled_x.mean() == pytest.approx(0.0)
        assert scaled_x.std() == pytest.approx(1.0) or scaled_x.std() == pytest.approx(
            0.0
        )

    @given(x=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    def test_solve_yw(self, x):
        rho = solve_yw_equation(np.array(x))
        if not np.isnan(np.sum(rho)):
            assert len(rho) == len(x) - 1

    @given(
        x=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2),
        order=st.integers(min_value=1),
    )
    def test_yule_walker_abs(self, x, order):
        pacf = yule_walker(np.array(x), order)
        if not (np.isnan(np.sum(pacf)) or len(pacf) == 0):
            assert all(abs(pacf) <= 1)

    @given(
        df=giotto_time_series(min_length=1, allow_nan=False, allow_infinity=False),
        max_lag=st.one_of(st.integers(min_value=1, max_value=100), st.none()),
    )
    def test_acf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = acf(df_array, max_lag)
        if max_lag is None:
            max_lag = len(df)
        assert len(res) == min(max_lag, len(df))

    @given(
        df=giotto_time_series(
            min_length=1, allow_nan=False, allow_infinity=False, max_length=50
        ),
        max_lag=st.one_of(st.integers(min_value=1, max_value=100), st.none()),
    )
    def test_pacf_len(self, df, max_lag):
        df_array = np.ravel(df.values)
        res = pacf(df_array, max_lag)
        if max_lag is None:
            max_lag = len(df)
        assert len(res) == min(max_lag, len(df))


@st.composite
def arma_params(draw, max_dim):
    p = draw(st.integers(min_value=0, max_value=max_dim))
    q = draw(st.integers(min_value=0, max_value=max_dim))
    params = draw(
        arrays(
            np.float,
            shape=(p + q + 2),
            elements=st.floats(
                min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return params, p


class TestMLETools:
    @given(data=arma_params(max_dim=10))
    def test_arma_polynomial_roots_abs(self, data):
        params, len_p = data
        res = arma_polynomial_roots(params, len_p)
        assert all(res >= 0)

    @given(
        x=arrays(
            dtype=float,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1,
                max_value=1,
                exclude_min=True,
                exclude_max=True,
            ),
        )
    )
    def test_durbin_levinson_recursion(self, x):
        transformed_x = durbin_levinson_recursion(x)
        y = transformed_x.copy()
        y2 = transformed_x.copy()
        for j in range(len(y) - 1, 0, -1):
            b = y[j]
            for kiter in range(j):
                y2[kiter] = (y[kiter] - b * y[j - kiter - 1]) / (1 - b ** 2)
            y[:j] = y2[:j]
        np.testing.assert_array_almost_equal(x, y, decimal=4)
