import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, sampled_from, data, lists, text
from lime.explanation import Explanation
from shap.explainers._explainer import Explainer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    BayesianRidge,
)
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils.validation import check_is_fitted

from gtime.explainability import _LimeExplainer, _ShapExplainer
from gtime.utils.fixtures import lazy_fixtures
from gtime.utils.hypothesis.feature_matrices import numpy_X_matrices, numpy_X_y_matrices


@pytest.fixture(scope="function")
def lime_explainer():
    return _LimeExplainer()


@pytest.fixture(scope="function")
def shap_explainer():
    return _ShapExplainer()


@pytest.fixture(scope="function")
def unrecognized_regressor():
    return AdaBoostRegressor()


@st.composite
def models(draw):
    regressors = [
        LinearRegression(),
        Ridge(alpha=draw(floats(0.00001, 2))),
        BayesianRidge(),
        # AdaBoostRegressor(), not supported
        # BaggingRegressor(), not supported
        ExtraTreeRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
    ]
    return draw(sampled_from(regressors))


@given(models())
def test_models(regressor):
    assert hasattr(regressor, "fit")
    assert hasattr(regressor, "predict")


class TestAllExplainers:
    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    def test_constructor(self, explainer):
        pass

    def _check_all_parameters_fitted(self, explainer):
        assert hasattr(explainer, "model_")
        assert hasattr(explainer, "explainer_")
        assert hasattr(explainer, "feature_names_")

    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    @given(regressor=models(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
    def test_fit_no_feature_names(self, explainer, regressor, X_y):
        X, y = X_y
        regressor.fit(X, y)
        explainer.fit(regressor, X)
        check_is_fitted(explainer)
        self._check_all_parameters_fitted(explainer)
        np.testing.assert_array_equal(
            explainer.feature_names_, [f"{i}" for i in range(X.shape[1])]
        )

    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    @given(
        data=data(),
        regressor=models(),
        X_y=numpy_X_y_matrices(min_value=-100, max_value=100),
    )
    def test_fit_feature_names(self, data, explainer, regressor, X_y):
        X, y = X_y
        feature_names = data.draw(
            lists(elements=text(), min_size=X.shape[1], max_size=X.shape[1])
        )
        regressor.fit(X, y)
        explainer.fit(regressor, X, feature_names)
        check_is_fitted(explainer)
        self._check_all_parameters_fitted(explainer)

    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    @given(regressor=models(), X=numpy_X_matrices())
    def test_error_fit_regressor_not_fitted(self, explainer, regressor, X):
        with pytest.raises(NotFittedError):
            explainer.fit(regressor, X)

    def _check_predict_output(
        self, explainer: Explainer, predictions: np.ndarray, test_matrix: np.ndarray
    ):
        assert predictions.shape[0] == test_matrix.shape[0]
        assert isinstance(explainer.explanations_, list)
        assert len(explainer.explanations_) == predictions.shape[0]
        assert all(
            isinstance(key, str)
            for explanation in explainer.explanations_
            for key in explanation.keys()
        )
        assert all(
            [
                len(explanation) == test_matrix.shape[1]
                for explanation in explainer.explanations_
            ]
        )

    @settings(deadline=pd.Timedelta(milliseconds=5000), max_examples=7)
    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    @given(regressor=models(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
    def test_predict(self, explainer, regressor, X_y):
        X, y = X_y
        regressor.fit(X, y)
        explainer.fit(regressor, X)

        test_matrix = X[:2, :]
        predictions = explainer.predict(test_matrix)
        self._check_predict_output(explainer, predictions, test_matrix)

    @pytest.mark.parametrize(
        "explainer", lazy_fixtures([lime_explainer, shap_explainer])
    )
    @given(X=numpy_X_matrices(min_value=-100, max_value=100))
    def test_error_predict_not_fit(self, explainer, X):
        with pytest.raises(NotFittedError):
            explainer.predict(X[:2, :])


class TestLime:
    @settings(deadline=pd.Timedelta(milliseconds=10000), max_examples=7)
    @given(regressor=models(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
    def test_predict(self, lime_explainer, regressor, X_y):
        X, y = X_y
        regressor.fit(X, y)
        lime_explainer.fit(regressor, X)

        test_matrix = X[:2, :]
        lime_explainer.predict(test_matrix)
        self._check_explanations(lime_explainer)

    def _check_explanations(self, lime_explainer: _LimeExplainer):
        assert isinstance(lime_explainer._explanations_, list)
        assert all(
            isinstance(explanation, Explanation)
            for explanation in lime_explainer._explanations_
        )


class TestShap:
    @given(regressor=models(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
    def test_predict(self, shap_explainer, regressor, X_y):
        X, y = X_y
        regressor.fit(X, y)
        shap_explainer.fit(regressor, X)

        test_matrix = X[:2, :]
        shap_explainer.predict(test_matrix)
        self._check_shap_values(shap_explainer, test_matrix)

    def _check_shap_values(self, shap_explainer: Explainer, test_matrix: np.ndarray):
        assert isinstance(shap_explainer.shap_values_, np.ndarray)
        assert shap_explainer.shap_values_.shape == test_matrix.shape

    @given(X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
    def test_fit_no_feature_names(self, shap_explainer, unrecognized_regressor, X_y):
        X, y = X_y
        unrecognized_regressor.fit(X, y)
        with pytest.raises(ValueError):
            shap_explainer.fit(unrecognized_regressor, X)
