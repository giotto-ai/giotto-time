from typing import Union, Dict
import networkx as nx
import pandas as pd
from numpy import mean
from sklearn.base import BaseEstimator
from gtime.hierarchical import HierarchicalBottomUp
from sklearn.utils.validation import check_is_fitted



class HierarchicalTopDown(HierarchicalBottomUp):
    """ Hierarchical model with prediction following the Top-Down procedure.
       It computes the forecast of the root and finds, with different methods the proportion between
       the root model and the children ones. Then it proceed to compute the forecast of these using the
       same proportion

       Parameters
       ----------
       model: BaseEstimator, required
           time series forecasting model that is applied to each of the time series. A cross validation model
           can also be passed.
       hierarchy_tree: String or nx.DiGraph, optional, default: 'Infer'
           Networkx Digraph containing the structure of the hierarchy tree.
           If 'Infer' the selected root will have as many children as the remaining keys.
        root: String, optional, default: first key of the dictionary of data
        method: String, optional, default='tdsga'
            Different top-down approaches are possible. Possible choices are:
            -'tdsga': the proportion between the parent time series and the child is computed as the mean
                      of every proportion at each timestep in the data provided.
            -'tdsgf': the proportion between the parent time series and the child is computed as proportion
                      of the means in the data provided.
            -'tdfp':  Iterative method computing the proportions using the forecast of each
                      time series as if it is indipendent.
       Examples
       --------
       >>> import pandas._testing as testing
       >>> from gtime.time_series_models import AR
       >>> from gtime.hierarchical import HierarchicalTopDown
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
       >>> hierarchical_model = HierarchicalTopDown(model=time_series_model, hierarchy_tree=tree, root='data1', method='tdsgf')
       >>> hierarchical_model.fit(data)
       >>> hierarchical_model.predict()
       """
    def __init__(self, model: BaseEstimator,
                 hierarchy_tree: Union[str, nx.DiGraph] = "infer",
                 root: str = None,
                 method: str = 'tdsga'):
        super().__init__(model=model, hierarchy_tree=hierarchy_tree)
        self.root = root
        self.method = method
        self.proportions = {}

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
        if self.root is None:
            self.root = list(X.keys())[0]
        if self.hierarchy_tree == 'infer':
            self._infer_hierarchy_tree(X)
        self._initialize_models(X)
        for key, time_series in X.items():
            self.models_[key].fit(time_series)
        if self.method == 'tdsga':
            self._proportions_sga_method(X)
        elif self.method == 'tdsgf':
            self._proportions_sgf_method(X)
        return self

    def _proportions_sga_method(self, X):
        self.proportions[self.root] = 1.0
        self._compute_children_proportions_sga_method(self.root, X)
        return

    def _proportions_sgf_method(self, X):
        self.proportions[self.root] = 1.0
        self._compute_children_proportions_sgf_method(self.root, X)

    def _compute_children_proportions_sga_method(self, parent, X):
        for child in self.hierarchy_tree[parent]:
            try:
                self.proportions[child] = mean(X[child].values / X[parent].values)
            except ZeroDivisionError:
                self.proportions[child] = 0
            if not self._is_a_leaf(child):
                self._compute_children_proportions_sga_method(child, X)
        return

    def _compute_children_proportions_sgf_method(self, parent, X):
        for child in self.hierarchy_tree[parent]:
            try:
                self.proportions[child] = mean(X[child].values)/mean(X[parent].values)
            except ZeroDivisionError:
                self.proportions[child] = 0
            if not self._is_a_leaf(child):
                self._compute_children_proportions_sgf_method(child, X)
        return

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
            if self.method == 'tdfp':
                return self._predict_fitted_time_series_fp()
            else:
                return self._predict_fitted_time_series()
        else:
            return self._predict_new_time_series(X)

    def _predict_fitted_time_series(self) -> Dict[str, pd.DataFrame]:
        top_down_sga_dictionary = {}
        top_down_sga_dictionary.update({self.root: self.models_[self.root].predict()})
        self._top_down_children_predict_computation(top_down_sga_dictionary, self.root)
        return top_down_sga_dictionary

    def _top_down_children_predict_computation(self, top_down_dictionary, parent_key):
        for child in list(self.hierarchy_tree[parent_key]):
            top_down_dictionary[child] = top_down_dictionary[parent_key] * self.proportions[child]
            if not self._is_a_leaf(child):
                self._top_down_children_predict_computation(top_down_dictionary, child)
        return

    def _predict_fitted_time_series_fp(self):
        basic_predictions = {key: model.predict() for key, model in self.models_.items()}
        self.proportions[self.root] = pd.Series([1.0 for i in range(self.model.horizon)], index=basic_predictions[self.root].index)
        self._compute_children_proportions_fp_method(basic_predictions, self.root)
        return self._top_down_fp_children_predict_computation(basic_predictions)

    def _compute_children_proportions_fp_method(self, prediction, parent_key):
        for child in self.hierarchy_tree[parent_key]:
            self.proportions[child] = self.proportions[parent_key] * self._compute_parent_child_proportion_fp_method(parent_key, child, prediction)
            if not self._is_a_leaf(child):
                self._compute_children_proportions_fp_method(prediction, child)
        return

    def _compute_parent_child_proportion_fp_method(self, parent_key, child_key, base_predictions):
        try:
            return mean(base_predictions[child_key], axis=1)/mean(base_predictions[parent_key], axis=1)
        except ZeroDivisionError:
            return 0

    def _top_down_fp_children_predict_computation(self, prediction) -> Dict[str, pd.DataFrame]:
        _top_down_fp_dictionary = {}
        for key in prediction.keys():
            _top_down_fp_dictionary[key] = prediction[self.root] * self.proportions[key].values.T
        return _top_down_fp_dictionary
