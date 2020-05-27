from typing import Union, Dict
import networkx as nx
import pandas as pd
from numpy import mean
from sklearn.base import BaseEstimator
from gtime.hierarchical import HierarchicalTopDown
from sklearn.utils.validation import check_is_fitted



class HierarchicalMiddleOut(HierarchicalTopDown):
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
           level: Int, optional default = 0
    Examples
       --------
       >>> import pandas._testing as testing
       >>> from gtime.time_series_models import AR
       >>> from gtime.hierarchical import HierarchicalMiddleOut
       >>>
       >>> testing.N, testing.K = 20, 1
       >>> data1 = testing.makeTimeDataFrame(freq="s")
       >>> data2 = testing.makeTimeDataFrame(freq="s")
       >>> data3 = testing.makeTimeDataFrame(freq="s")
       >>> data4 = testing.makeTimeDataFrame(freq="s")
       >>> data5 = testing.makeTimeDataFrame(freq="s")
       >>> data6 = testing.makeTimeDataFrame(freq="s")
       >>> data = {'data1': data1, 'data2': data2, 'data3' : data3, 'data4' : data4, 'data5' : data5, 'data6' : data6}
       >>> time_series_model = AR(p=2, horizon=3)
       >>> tree_adj = {'data1': {'data2': {}, 'data3': {}}, 'data2': {'data4': {}, 'data5': {}}, 'data3': {'data6': {}}, 'data4': {}, 'data5': {}, 'data6': {}}
       >>> tree = nx.DiGraph(tree_adj)
       >>> hierarchical_model = HierarchicalMiddleOut(model=time_series_model, hierarchy_tree=tree, root='data1', method='tdsgf', level=1)
       >>> hierarchical_model.fit(data)
       >>> hierarchical_model.predict()
    'data2':                           y_1       y_2       y_3
 2000-01-01 00:00:17 -0.113171  0.162990 -0.036698
 2000-01-01 00:00:18  0.218730 -0.056611 -0.032642
 2000-01-01 00:00:19 -0.008568 -0.038176 -0.018069,
 'data4':                           y_1       y_2       y_3
 2000-01-01 00:00:17 -0.042989  0.061913 -0.013940
 2000-01-01 00:00:18  0.083086 -0.021504 -0.012399
 2000-01-01 00:00:19 -0.003255 -0.014501 -0.006864,
 'data5':                           y_1       y_2       y_3
 2000-01-01 00:00:17 -0.070182  0.101077 -0.022758
 2000-01-01 00:00:18  0.135644 -0.035107 -0.020243
 2000-01-01 00:00:19 -0.005313 -0.023675 -0.011205,
 'data3':                           y_1       y_2       y_3
 2000-01-01 00:00:17 -0.038946 -0.108634  0.014530
 2000-01-01 00:00:18 -0.117246 -0.029228  0.007229
 2000-01-01 00:00:19 -0.076983 -0.004052 -0.007851,
 'data6':                           y_1       y_2       y_3
 2000-01-01 00:00:17 -0.038946 -0.108634  0.014530
 2000-01-01 00:00:18 -0.117246 -0.029228  0.007229
 2000-01-01 00:00:19 -0.076983 -0.004052 -0.007851,
 'data1':                           y_1       y_2       y_3
 2000-01-01 00:00:17  0.762066 -0.227920  0.103735
 2000-01-01 00:00:18 -0.456526  0.432652  0.122335
 2000-01-01 00:00:19  0.448537  0.209102  0.130292
    """
    def __init__(self, model: BaseEstimator,
                 hierarchy_tree: Union[str, nx.DiGraph] = "infer",
                 root: str = None,
                 method: str = 'tdsga',
                 level: int = 0):
        super().__init__(model=model, hierarchy_tree=hierarchy_tree, root=root, method=method)
        self.level = level
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

        super().fit(X, y)
        self._find_levels()
        self._sub_trees_dictionary_construction()
        return self

    def _find_levels(self):
        self.level_nodes = {0: self.root}
        self._add_nodes_to_level_nodes_dictionary(self.root, 0)
        return

    def _add_nodes_to_level_nodes_dictionary(self, parent_key, parent_level):
        for child in self.hierarchy_tree[parent_key]:
            if parent_level+1 in self.level_nodes.keys():
                self.level_nodes[parent_level+1].append(child)
            else:
                self.level_nodes[parent_level+1] = [child]
            if not self._is_a_leaf(child):
                self._add_nodes_to_level_nodes_dictionary(child, parent_level+1)
        return

    def _sub_trees_dictionary_construction(self):
        self.sub_trees_dictionary = {}
        for node in self.level_nodes[self.level]:
            self.sub_trees_dictionary[node] = self._sub_tree_construction(node)
        return

    def _sub_tree_construction(self, new_root) -> nx.DiGraph():
        sub_tree = nx.DiGraph()
        sub_tree.add_node(new_root)
        self._add_edge_sub_tree(sub_tree, new_root)
        return sub_tree

    def _add_edge_sub_tree(self, tree, parent):
        if not self._is_a_leaf(parent):
            for child in self.hierarchy_tree[parent]:
                tree.add_node(child)
                tree.add_edge(parent, child)
                self._add_edge_sub_tree(tree, child)
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
            return self._predict_fitted_time_series()
        else:
            return self._predict_new_time_series(X)

    def _predict_fitted_time_series(self) -> Dict[str, pd.DataFrame]:
        prediction_dictionary = {}
        self._predict_fitted_time_series_top_down(prediction_dictionary)
        self._predict_fitted_time_series_bottom_up(prediction_dictionary)
        return prediction_dictionary

    def _predict_fitted_time_series_top_down(self, dictionary):
        for key in self.level_nodes[self.level]:
            submodel = HierarchicalTopDown(model=self.model, hierarchy_tree=self.sub_trees_dictionary[key],
                                           root=key, method=self.method)
            submodel.models_ = self._extract_fitted_models(key)
            submodel.proportions = self._extract_proportions(key)
            submodel_prediction = submodel.predict()
            for submodel_key, prediction in submodel_prediction.items():
                dictionary.update({submodel_key: prediction})
        return

    def _extract_fitted_models(self, sub_tree_root):
        submodels_dictionary = {}
        for node in self.sub_trees_dictionary[sub_tree_root].nodes:
            submodels_dictionary[node] = self.models_[node]
        return submodels_dictionary

    def _extract_proportions(self, sub_tree_root):
        submodel_proportions = {}
        if self.proportions:
            for node in self.sub_trees_dictionary[sub_tree_root].nodes:
                submodel_proportions[node] = self.proportions[node]
        return submodel_proportions

    def _predict_fitted_time_series_bottom_up(self, dictionary):
        for key, model in self.models_.items():
            self._bottom_up_addiction_to_dictionary(key, model, dictionary)
        return
