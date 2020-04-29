from typing import List

import pytest
from hypothesis import given
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
import numpy as np
import pandas as pd

from gtime.explainability import _LimeExplainer, _ShapExplainer
from gtime.forecasting.tests.test_gar import df_transformer
from gtime.model_selection import FeatureSplitter
from gtime.regressors import ExplainableRegressor
from gtime.utils.hypothesis.feature_matrices import (
    numpy_X_matrices,
    numpy_X_y_matrices,
    X_y_matrices,
)
from gtime.utils.hypothesis.general_strategies import regressors
from gtime.utils.hypothesis.time_indexes import samples_from


def bad_regressors():
    return samples_from([DBSCAN(), SpectralClustering(), PCA(),])


@given(bad_regressors())
def test_bad_regressors(bad_regressor):
    assert hasattr(bad_regressor, "fit")
    assert not hasattr(bad_regressor, "predict")


class TestExplainableRegressor:
    @pytest.mark.parametrize("explainer_type", ["lime", "shap"])
    @given(estimator=regressors())
    def test_constructor(self, estimator, explainer_type):
        regressor = ExplainableRegressor(estimator, explainer_type)
        if explainer_type == "lime":
            assert isinstance(regressor.explainer, _LimeExplainer)
        elif explainer_type == "shap":
            assert isinstance(regressor.explainer, _ShapExplainer)

    @given(estimator=regressors())
    def test_constructor_bad_explainer(self, estimator):
        with pytest.raises(ValueError):
            ExplainableRegressor(estimator, "bad")

    @pytest.mark.parametrize("explainer_type", ["lime", "shap"])
    @given(bad_estimator=bad_regressors())
    def test_constructor_bad_regressor(self, bad_estimator, explainer_type):
        with pytest.raises(TypeError):
            ExplainableRegressor(bad_estimator, explainer_type)

    @pytest.mark.parametrize("explainer_type", ["lime", "shap"])
    @given(estimator=regressors(), X=numpy_X_matrices())
    def test_error_predict_not_fitted(self, estimator, explainer_type, X):
        regressor = ExplainableRegressor(estimator, explainer_type)
        with pytest.raises(NotFittedError):
            regressor.predict(X)

    def _get_fit_attributes(self, estimator: BaseEstimator) -> List[str]:
        return [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    @pytest.mark.parametrize("explainer_type", ["lime", "shap"])
    @given(
        estimator=regressors(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100)
    )
    def test_fit_values(self, estimator, explainer_type, X_y):
        X, y = X_y
        regressor = ExplainableRegressor(estimator, explainer_type)
        regressor.fit(X, y)

        cloned_estimator = clone(estimator)
        cloned_estimator.fit(X, y)

        estimator_fit_attributes = self._get_fit_attributes(regressor.estimator)
        cloned_estimator_fit_attributes = self._get_fit_attributes(cloned_estimator)

        np.testing.assert_array_equal(
            estimator_fit_attributes, cloned_estimator_fit_attributes
        )

    @pytest.mark.parametrize("explainer_type", ["lime", "shap"])
    @given(
        estimator=regressors(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100)
    )
    def test_predict_values(self, estimator, explainer_type, X_y):
        X, y = X_y
        X_test = X[:1, :]
        regressor = ExplainableRegressor(estimator, explainer_type)
        regressor_predictions = regressor.fit(X, y).predict(X_test)

        cloned_estimator = clone(estimator)
        estimator_predictions = cloned_estimator.fit(X, y).predict(X_test)

        assert regressor_predictions.shape == estimator_predictions.shape
        assert regressor_predictions.shape[0] == len(regressor.explanations_)
