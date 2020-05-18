from gtime.hierarchical import HierarchicalNaive
from typing import Union, Dict
import networkx as nx
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy


class HierarchicalBottomUp(HierarchicalNaive):
    """ Hierarchical model with prediction following the Bottom-Up procedure.
    It computes the forecast of the leaves in the hierarchical tree structure
    and it performs the sums of these forecasts to recover the prediction on the
    higher level.

    Parameters
    ----------
    model: BaseEstimator, required
        time series forecasting model that is applied to each of the time series. A cross validation model
        can also be passed.
    hierarchy_tree: String or Dict[str, List], optional, default: 'Infer'
        DiGraph to import the structure of the hierarchical tree. Keys must be the same as in the
        dictionary X containing the dataframes. The items are lists containing the name of the children
        of each key. In case there are no children (the key is a leaf) the item must be an empty list.
    Examples
    --------
    >>> import pandas._testing as testing
    >>> from gtime.time_series_models import AR
    >>> from gtime.hierarchical import HierarchicalBottomUp
    >>>
    >>> testing.N, testing.K = 20, 1
    >>>
    >>> data1 = testing.makeTimeDataFrame(freq="s")
    >>> data2 = testing.makeTimeDataFrame(freq="s")
    >>> data3 = testing.makeTimeDataFrame(freq="s")
    >>> data = {'data1': data1, 'data2': data2, 'data3' : data3}
    >>>
    >>> time_series_model = AR(p=2, horizon=3)
    >>> tree_adj = {'data1': {'data2': {}, 'data3': {}}, 'data2': {}, 'data3': {}}
    >>> tree = nx.DiGraph(tree_adj)
    >>> hierarchical_model = HierarchicalBottomUp(model=time_series_model, hierarchy_tree=tree)
    >>> hierarchical_model.fit(data)
    >>> hierarchical_model.predict()
    {'data2':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.809728 -0.211003 -0.079735
    2000-01-01 00:00:18  0.938604  0.000904 -0.536674
    2000-01-01 00:00:19  0.263110 -0.190320 -0.377745, 'data3':                           y_1       y_2       y_3
    2000-01-01 00:00:17 -0.415473  0.494154  0.058705
    2000-01-01 00:00:18 -0.018414  0.451427 -0.430473
    2000-01-01 00:00:19  0.759329  0.139795  0.257968, 'data1':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.394255  0.283151 -0.021031
    2000-01-01 00:00:18  0.920190  0.452331 -0.967147
    2000-01-01 00:00:19  1.022438 -0.050525 -0.119777}

    """
    def __init__(self, model: BaseEstimator,
                 hierarchy_tree: Union[str, nx.DiGraph] = "infer"
                 ):
        super().__init__(model=model)
        self.hierarchy_tree = hierarchy_tree


    def fit(self, X: Dict[str, pd.DataFrame], y: pd.DataFrame = None):
        """ Fit method

        Parameters
        ----------
        X : Dict[str, pd.DataFrame], required
            A dictionary of time series. Each is fitted independently
        y : pd.DataFrame, optional, default = ``None``
            only for compatibility

        Returns
        -------
        self
        """
        self._check_is_dict_of_dataframes_with_str_key(X)
        if self.hierarchy_tree == 'infer':
            self._infer_hierarchy_tree(X)
        super()._initialize_models(X)
        for key, time_series in X.items():
            self.models_[key].fit(time_series)
        return self

    def predict(self, X: Dict[str, pd.DataFrame] = None):
        """ Predict method

        Parameters
        ----------
        X : Dict[str, pd.DataFrame], optional, default = ``None``
            time series to predict. If ``None`` all the fitted time series are predicted.
            The keys in ``X`` have to match the ones used to fit.

        Returns
        -------
        predictions : Dict[str, pd.DataFrame]
        """
        check_is_fitted(self)
        if X is None:
            return self._predict_fitted_time_series()
        else:
            return super()._predict_new_time_series(X)

    def _infer_hierarchy_tree(self, X: Dict[str, pd.DataFrame]):
        self.hierarchy_tree = nx.DiGraph()
        for key in list(X.keys()):
            self.hierarchy_tree.add_node(key)
        for key in list(X.keys())[1:]:
            self.hierarchy_tree.add_edge(list(X.keys())[0], key)
        # First key of X is the root, all other keys are leaves of the tree

    def _predict_fitted_time_series(self) -> Dict[str, pd.DataFrame]:
        bottom_up_dictionary = {}
        for key, model in self.models_.items():
            self._bottom_up_addiction_to_dictionary(key, model, bottom_up_dictionary)
        return bottom_up_dictionary

    def _bottom_up_addiction_to_dictionary(self, key, model, dict_bottom_up):
        if self._is_a_leaf(key):
            dict_bottom_up[key] = model.predict()
        else:
            self._check_predict_children_computed(dict_bottom_up, key)
            self._sum_children_prediction(key, dict_bottom_up)

    def _is_a_leaf(self, key) -> bool:
        if self.hierarchy_tree.out_degree[key] == 0:
            return True
        else:
            return False

    def _sum_children_prediction(self, parent_key, dict_bottom_up):
        temp = deepcopy(dict_bottom_up[list(self.hierarchy_tree[parent_key])[0]])
        for item in list(self.hierarchy_tree[parent_key])[1:]:
            temp = temp + dict_bottom_up[item]
        dict_bottom_up[parent_key] = temp
        return

    def _check_predict_children_computed(self, dict_bottom_up, tested_key):
        for child in list(self.hierarchy_tree[tested_key]):
            if child not in dict_bottom_up:  # If child model not computed yet
                if self._is_a_leaf(child):
                    dict_bottom_up[child] = self.models_[child].predict()
                else:
                    self._check_predict_children_computed(dict_bottom_up, child)
                    self._sum_children_prediction(child, dict_bottom_up)
        return
