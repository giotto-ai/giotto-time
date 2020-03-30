import numpy as np
import pandas as pd
import pytest
import sklearn
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from pytest import fixture

from gtime.hierarchical import HierarchicalNaive, HierarchicalBase
from gtime.utils.fixtures import (
    time_series_forecasting_model1_no_cache,
    features1,
    model1,
)
from gtime.utils.hypothesis.time_indexes import giotto_time_series, period_indexes


@fixture(scope="function")
def hierarchical_naive_model(time_series_forecasting_model1_no_cache):
    return HierarchicalNaive(time_series_forecasting_model1_no_cache)


@st.composite
def n_time_series_with_same_index(
    draw, min_length: int = 5, min_n: int = 1, max_n: int = 5,
):
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    index = draw(period_indexes(min_length=min_length))
    dictionary = {}
    for _ in range(n):
        key = draw(st.text(min_size=4, max_size=12))
        df_values = draw(
            arrays(
                dtype=np.float64,
                shape=index.shape[0],
                elements=st.floats(allow_nan=False, allow_infinity=False, width=32),
            )
        )
        value = pd.DataFrame(index=index, data=df_values)
        dictionary[key] = value
    return dictionary


class TestHierarchicalBase:
    def test_class_abstract(self, model1):
        HierarchicalBase(model1, {})


class TestHierarchicalNaive:
    def test_constructor(self, time_series_forecasting_model1_no_cache):
        HierarchicalNaive(model=time_series_forecasting_model1_no_cache)

    def test_constructor_no_hierarchy_tree(
        self, time_series_forecasting_model1_no_cache
    ):
        hierarchy_tree = {}
        with pytest.raises(TypeError):
            HierarchicalNaive(
                model=time_series_forecasting_model1_no_cache,
                hierarchy_tree=hierarchy_tree,
            )

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_dataframe(self, time_series, hierarchical_naive_model):
        with pytest.raises(ValueError):
            hierarchical_naive_model.fit(time_series)

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_key_not_string(self, time_series, hierarchical_naive_model):
        with pytest.raises(ValueError):
            hierarchical_naive_model.fit({1: time_series})

    def test_error_fit_value_not_dataframe(self, hierarchical_naive_model):
        with pytest.raises(ValueError):
            hierarchical_naive_model.fit({"wrong_field": 12})

    @given(dataframes=n_time_series_with_same_index())
    def test_fit_n_dataframes(self, dataframes, hierarchical_naive_model):
        hierarchical_naive_model.fit(dataframes)

    @given(dataframes=n_time_series_with_same_index())
    def test_fit_predict_n_dataframes_on_different_data(
        self, dataframes, hierarchical_naive_model
    ):
        hierarchical_naive_model.fit(dataframes).predict(dataframes)

    @given(dataframes=n_time_series_with_same_index())
    def test_fit_predict_n_dataframes(self, dataframes, hierarchical_naive_model):
        hierarchical_naive_model.fit(dataframes).predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_fit_predict_on_subset_of_time_series(
        self, dataframes, hierarchical_naive_model
    ):
        key = np.random.choice(list(dataframes.keys()), 1)[0]
        hierarchical_naive_model.fit(dataframes)
        hierarchical_naive_model.predict({key: dataframes[key]})

    def test_error_predict_not_fitted(self, hierarchical_naive_model):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            hierarchical_naive_model.predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_error_with_bad_predict_key(self, dataframes, hierarchical_naive_model):
        correct_key = np.random.choice(list(dataframes.keys()), 1)[0]
        bad_key = "".join(dataframes.keys()) + "bad_key"
        hierarchical_naive_model.fit(dataframes)
        with pytest.raises(KeyError):
            hierarchical_naive_model.predict({bad_key: dataframes[correct_key]})
