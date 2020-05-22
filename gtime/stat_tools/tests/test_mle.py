import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis import given, settings
from gtime.stat_tools import ARMAMLEModel


class TestMLEModel:
    def test_zeros(self):
        x = np.zeros(100)
        model = ARMAMLEModel(order=(0, 0), method="css")
        model.fit(x)
        assert model.mu == pytest.approx(0.0)

    def test_ar(self):
        x = np.array([(-1) ** i for i in range(100)])
        model = ARMAMLEModel(order=(1, 0), method="css")
        model.fit(x)
        assert model.phi == pytest.approx(-1.0)

    @given(
        x=arrays(
            dtype=float,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(
                max_value=1e15, min_value=-1e15, allow_infinity=False, allow_nan=False
            ),
            unique=True,
        ),
        order=st.tuples(
            st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)
        ),
    )
    @settings(deadline=None)
    def test_random_likelihood(self, x, order):
        model = ARMAMLEModel(order=order, method="css")
        model.fit(x)
        assert not np.isnan(model.ml)

    @given(
        x=arrays(
            dtype=float,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(
                max_value=1e15, min_value=-1e15, allow_infinity=False, allow_nan=False
            ),
            unique=True,
        ),
        order=st.tuples(
            st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)
        ),
    )
    @settings(deadline=None)
    def test_random_errors_len(self, x, order):
        model = ARMAMLEModel(order=order, method="css")
        model.fit(x)
        errors = model.get_errors(x)
        assert len(errors) == len(x) - order[0]
