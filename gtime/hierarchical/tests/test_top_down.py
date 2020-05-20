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

from gtime.hierarchical import HierarchicalTopDown
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

@fixture(scope="function")
def hierarchical_basic_top_down_model(time_series_forecasting_model1_no_cache):
    return HierarchicalTopDown(time_series_forecasting_model1_no_cache, 'infer')

@st.composite
def hierarchical_top_down_model_sga(draw, time_series_forecasting_model1_no_cache):
    dataframes = draw(n_time_series_with_same_index(min_n=5))
    tree = draw(tree_construction(dataframes))
    method = 'tdsga'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, hierarchy_tree=tree, method=method)


@st.composite
def hierarchical_top_down_model_sgf(draw, time_series_forecasting_model1_no_cache):
    dataframes = draw(n_time_series_with_same_index(min_n=5))
    tree = draw(tree_construction(dataframes))
    method = 'tdsgf'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, hierarchy_tree=tree, method=method)


@st.composite
def hierarchical_top_down_model_fp(draw, time_series_forecasting_model1_no_cache):
    dataframes = draw(n_time_series_with_same_index(min_n=5))
    tree = draw(tree_construction(dataframes))
    method = 'tdfp'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, hierarchy_tree=tree, method=method)


@st.composite
def hierarchical_top_down_model_sga_tree_by_hand(draw, time_series_forecasting_model1_no_cache):
    tree_adjacency_list = {'0': {}, '1': {}, '2': {'1': {}, '4': {}}, '3': {'0': {}, '2': {}}, '4': {}}
    tree = nx.DiGraph(tree_adjacency_list)
    root = '3'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, hierarchy_tree=tree, root=root)


@st.composite
def hierarchical_top_down_model_sgf_tree_by_hand(draw, time_series_forecasting_model1_no_cache):
    tree_adjacency_list = {'0': {}, '1': {}, '2': {'1': {}, '3': {}}, '4': {'0': {}, '2': {}}, '3': {}}
    tree = nx.DiGraph(tree_adjacency_list)
    root = '4'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, method='tdsgf', hierarchy_tree=tree, root=root)

@st.composite
def hierarchical_top_down_model_fp_tree_by_hand(draw, time_series_forecasting_model1_no_cache):
    tree_adjacency_list = {'0': {}, '1': {}, '2': {'1': {}, '3': {}}, '4': {'0': {}, '2': {}}, '3': {}}
    tree = nx.DiGraph(tree_adjacency_list)
    root = '4'
    return HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, method='tdfp', hierarchy_tree=tree, root=root)

@fixture(scope="function")
def hierarchical_basic_top_down_model(time_series_forecasting_model1_no_cache):
    return HierarchicalTopDown(time_series_forecasting_model1_no_cache, 'infer')


class TestHierarchicalTopDown:
    def test_basic_constructor(self, time_series_forecasting_model1_no_cache):
        HierarchicalTopDown(model=time_series_forecasting_model1_no_cache, hierarchy_tree='infer')

    @given(dataframes=n_time_series_with_same_index())
    def test_fit_predict_basic_top_down_on_different_data(self, dataframes, hierarchical_basic_top_down_model):
        hierarchical_basic_top_down_model.fit(dataframes).predict(dataframes)

    @given(dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_basic_top_down(self, dataframes, hierarchical_basic_top_down_model):
        hierarchical_basic_top_down_model.fit(dataframes).predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_constructor(self, time_series_forecasting_model1_no_cache, dataframes):
        tree = tree_construction(dataframes)
        HierarchicalTopDown(time_series_forecasting_model1_no_cache, tree)

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_top_down_sga(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_top_down_model_sga(time_series_forecasting_model1_no_cache))
        model.fit(dataframes).predict()

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_top_down_sgf(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_top_down_model_sgf(time_series_forecasting_model1_no_cache))
        model.fit(dataframes).predict()

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_top_down_sga_tree_by_hands(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_top_down_model_sga_tree_by_hand(time_series_forecasting_model1_no_cache))
        model.fit(dataframes).predict()

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_top_down_sgf_tree_by_hands(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_top_down_model_sgf_tree_by_hand(time_series_forecasting_model1_no_cache))
        model.fit(dataframes).predict()

    @given(data=st.data(), dataframes=n_time_series_with_same_index(min_n=5))
    def test_fit_predict_top_down_fp_tree_by_hands(self, data, dataframes, time_series_forecasting_model1_no_cache):
        model = data.draw(hierarchical_top_down_model_fp_tree_by_hand(time_series_forecasting_model1_no_cache))
        model.fit(dataframes).predict()

    def test_error_predict_not_fitted(self, hierarchical_basic_top_down_model):
        with pytest.raises(sklearn.exceptions.NotFittedError):
            hierarchical_basic_top_down_model.predict()

    @given(dataframes=n_time_series_with_same_index())
    def test_error_with_bad_predict_key(self, dataframes, hierarchical_basic_top_down_model):
        correct_key = np.random.choice(list(dataframes.keys()), 1)[0]
        bad_key = "".join(dataframes.keys()) + "bad_key"
        hierarchical_basic_top_down_model.fit(dataframes)
        with pytest.raises(KeyError):
            hierarchical_basic_top_down_model.predict({bad_key: dataframes[correct_key]})

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_dataframe(self, time_series, hierarchical_basic_top_down_model):
        with pytest.raises(ValueError):
            hierarchical_basic_top_down_model.fit(time_series)

    @given(time_series=giotto_time_series(min_length=5))
    def test_error_fit_key_not_string(self, time_series, hierarchical_basic_top_down_model):
        with pytest.raises(ValueError):
            hierarchical_basic_top_down_model.fit({1: time_series})

    def test_error_fit_value_not_dataframe(self, hierarchical_basic_top_down_model):
        with pytest.raises(ValueError):
            hierarchical_basic_top_down_model.fit({"wrong_field": 12})