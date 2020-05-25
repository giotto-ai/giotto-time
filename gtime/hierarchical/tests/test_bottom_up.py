import numpy as np
import pandas as pd
import pytest
import sklearn
from hypothesis import given
import networkx as nx
import random
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays
from pytest import fixture

from gtime.hierarchical import HierarchicalBottomUp
from gtime.utils.fixtures import (
    time_series_forecasting_model1_no_cache,
    features1,
    model1,
)
from gtime.utils.hypothesis.time_indexes import giotto_time_series, period_indexes

@st.composite
def n_time_series_with_same_index(
    draw, min_length: int = 5, min_n: int = 1, max_n: int = 5,
):
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    index = draw(period_indexes(min_length=min_length))
    dictionary = {}
    for i in range(n):
        key = str(i)
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


@st.composite
def tree_construction(
        draw, dictionary
):
    tree_nodes = list(dictionary.keys())
    tree = nx.DiGraph()
    n = len(tree_nodes)
    for i in range(n):
        selected_key = random.choice(tree_nodes)
        if len(tree) > 0:
            selected_node = random.choice(list(tree.nodes))
            tree.add_edge(selected_node, selected_key)
        tree.add_node(selected_key)
        tree_nodes.remove(selected_key)
    return tree


@st.composite
def hierarchical_bottom_up_model(draw, time_series_forecasting_model1_no_cache):
    dataframes = draw(n_time_series_with_same_index(min_n=5))
    tree = draw(tree_construction(dataframes))
    return HierarchicalBottomUp(time_series_forecasting_model1_no_cache, tree)

@fixture(scope="function")
def hierarchical_basic_bottom_up_model(time_series_forecasting_model1_no_cache):
    return HierarchicalBottomUp(time_series_forecasting_model1_no_cache, 'infer')


class TestHierarchicalBottomUp:
    def test_basic_constructor(self, time_series_forecasting_model1_no_cache):
        HierarchicalBottomUp(model=time_series_forecasting_model1_no_cache, hierarchy_tree='infer')

    @given(dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_basic_bottom_up_on_different_data(self, dataframes, hierarchical_basic_bottom_up_model):
        hierarchical_basic_bottom_up_model.fit(dataframes).predict(dataframes)

    @given(dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_basic_bottom_up(self, dataframes, hierarchical_basic_bottom_up_model):
        hierarchical_basic_bottom_up_model.fit(dataframes).predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_constructor(self, time_series_forecasting_model1_no_cache, dataframes):
        tree = tree_construction(dataframes)
        HierarchicalBottomUp(time_series_forecasting_model1_no_cache, tree)

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_bottom_up(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_bottom_up_model(time_series_forecasting_model1_no_cache))
        prediction = model.fit(dataframes).predict()
        for key in dataframes.keys():
            if key not in prediction.keys():
                raise ValueError

    @given(dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_on_subset_of_time_series(
            self, dataframes, hierarchical_basic_bottom_up_model
    ):
        key = np.random.choice(list(dataframes.keys()), 1)[0]
        hierarchical_basic_bottom_up_model.fit(dataframes)
        hierarchical_basic_bottom_up_model.predict({key: dataframes[key]})

    def test_error_predict_not_fitted(self, hierarchical_basic_bottom_up_model):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            hierarchical_basic_bottom_up_model.predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_error_with_bad_predict_key(self, dataframes, hierarchical_basic_bottom_up_model):
        correct_key = np.random.choice(list(dataframes.keys()), 1)[0]
        bad_key = "".join(dataframes.keys()) + "bad_key"
        hierarchical_basic_bottom_up_model.fit(dataframes)
        with pytest.raises(KeyError):
            hierarchical_basic_bottom_up_model.predict({bad_key: dataframes[correct_key]})

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_dataframe(self, time_series, hierarchical_basic_bottom_up_model):
        with pytest.raises(ValueError):
            hierarchical_basic_bottom_up_model.fit(time_series)

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_key_not_string(self, time_series, hierarchical_basic_bottom_up_model):
        with pytest.raises(ValueError):
            hierarchical_basic_bottom_up_model.fit({1: time_series})

    def test_error_fit_value_not_dataframe(self, hierarchical_basic_bottom_up_model):
        with pytest.raises(ValueError):
            hierarchical_basic_bottom_up_model.fit({"wrong_field": 12})