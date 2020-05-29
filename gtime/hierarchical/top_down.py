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
       hierarchy_tree: String, nx.DiGraph, Dict[str, dict] or Dict[str, list], optional, default: 'Infer'
            Networkx Digraph containing the structure of the hierarchy tree.
            If 'Infer' the selected root will have as many children as the remaining keys.
       root: String, optional: if hierarchy_tree is 'infer' it is the first key of the dictionary of data
              otherwise it will look for the root of the tree (this procedure may slow down the code.)
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

       'data1':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.005146 -0.297901 -0.440032
    2000-01-01 00:00:18  0.011849 -0.332548 -0.203615
    2000-01-01 00:00:19 -0.013333 -0.042139 -0.123846, 'data2':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.009618 -0.556743 -0.822369
    2000-01-01 00:00:18  0.022144 -0.621494 -0.380533
    2000-01-01 00:00:19 -0.024918 -0.078752 -0.231454, 'data3':                           y_1       y_2       y_3
    2000-01-01 00:00:17  0.001977 -0.114453 -0.169059
    2000-01-01 00:00:18  0.004552 -0.127764 -0.078228
    2000-01-01 00:00:19 -0.005122 -0.016190 -0.047581
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
            if self.hierarchy_tree == 'infer':
                self.root = list(X.keys())[0]
            else:
                self.root = self._extract_root_from_hierarchy_tree()
        if self.hierarchy_tree == 'infer':
            self._infer_hierarchy_tree(X)
        self._initialize_models(X)
        for key, time_series in X.items():
            self.models_[key].fit(time_series)
        if self.method == 'tdsga':
            self._compute_proportions_sga_method(X)
        elif self.method == 'tdsgf':
            self._compute_proportions_sgf_method(X)
        return self

    def _extract_root_from_hierarchy_tree(self):
        for node, degree in self.hierarchy_tree.in_degree:
            if degree == 0:
                return node
        raise ValueError

    def _compute_proportions_sga_method(self, X):
        self.proportions[self.root] = 1.0
        self._compute_children_proportions_sga_method(self.root, X)

    def _compute_proportions_sgf_method(self, X):
        self.proportions[self.root] = 1.0
        self._compute_children_proportions_sgf_method(self.root, X)

    def _compute_children_proportions_sga_method(self, parent, X):
        for child in self.hierarchy_tree[parent]:
            try:
                self.proportions[child] = mean(X[child] / X[parent]).values[0]
            except ZeroDivisionError:
                self.proportions[child] = 0
            if not self._is_a_leaf(child):
                self._compute_children_proportions_sga_method(child, X)

    def _compute_children_proportions_sgf_method(self, parent, X):
        for child in self.hierarchy_tree[parent]:
            try:
                self.proportions[child] = (mean(X[child])/mean(X[parent])).values[0]
            except ZeroDivisionError:
                self.proportions[child] = 0
            if not self._is_a_leaf(child):
                self._compute_children_proportions_sgf_method(child, X)

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
            return self._predict_new_time_series(X)

    def _predict_fitted_time_series(self) -> Dict[str, pd.DataFrame]:
        top_down_dictionary = {}
        if self.method == 'tdfp':
            self._predict_fitted_time_series_fp(top_down_dictionary)
        else:
            top_down_dictionary.update({self.root: self.models_[self.root].predict()})
            self._top_down_children_predict_computation(top_down_dictionary, self.root)
        return top_down_dictionary

    def _top_down_children_predict_computation(self, top_down_dictionary, parent_key):
        for child in list(self.hierarchy_tree[parent_key]):
            top_down_dictionary[child] = top_down_dictionary[parent_key] * self.proportions[child]
            if not self._is_a_leaf(child):
                self._top_down_children_predict_computation(top_down_dictionary, child)

    def _predict_fitted_time_series_fp(self, top_down_dictionary) -> Dict[str, pd.DataFrame]:
        basic_predictions = {key: model.predict() for key, model in self.models_.items()}
        self.proportions[self.root] = pd.Series([1.0 for i in range(self.model.horizon)], index=basic_predictions[self.root].columns)
        self._compute_children_proportions_fp_method(basic_predictions, self.root)
        self._top_down_fp_children_predict_computation(top_down_dictionary, basic_predictions)
        return top_down_dictionary

    def _compute_children_proportions_fp_method(self, prediction, parent_key):
        for child in self.hierarchy_tree[parent_key]:
            self.proportions[child] = self.proportions[parent_key] * self._compute_parent_child_proportion_fp_method(parent_key, child, prediction)
            if not self._is_a_leaf(child):
                self._compute_children_proportions_fp_method(prediction, child)

    def _compute_parent_child_proportion_fp_method(self, parent_key, child_key, base_predictions):
        try:
            return (mean(base_predictions[child_key])/mean(base_predictions[parent_key])).values
        except ZeroDivisionError:
            return 0

    def _top_down_fp_children_predict_computation(self, top_down_dictionary, prediction):
        for key in prediction.keys():
            top_down_dictionary[key] = prediction[self.root] * self.proportions[key].values

    def _predict_new_time_series(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        new_time_series_dictionary = {}
        if self.method == 'tdfp':
            self._predict_new_time_series_fp(new_time_series_dictionary, X)
        else:
            new_time_series_dictionary.update({self.root: self.models_[self.root].predict(X[self.root])})
            self._top_down_children_predict_computation(new_time_series_dictionary, self.root)
        return new_time_series_dictionary
    
    def _predict_new_time_series_fp(self, top_down_dictionary, X) -> Dict[str, pd.DataFrame]:
        basic_predictions = {key: self.models_[key].predict(timeseries) for key, timeseries in X.items()}
        proportion_length = self.model.horizon
        self.proportions[self.root] = pd.Series([1.0 for i in range(proportion_length)], index=basic_predictions[self.root].columns)
        self._compute_children_proportions_fp_method(basic_predictions, self.root)
        self._top_down_fp_children_predict_computation(top_down_dictionary, basic_predictions)
        return top_down_dictionary


