import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, sampled_from, data, lists, text
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    BayesianRidge,
)
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils.validation import check_is_fitted

from gtime.explainability import LimeExplainer, ShapExplainer
from gtime.utils.hypothesis.feature_matrices import numpy_X_matrices, numpy_X_y_matrices


@pytest.fixture(scope="function")
def lime_explainer():
    return LimeExplainer()


@pytest.fixture(scope="function")
def shap_explainer():
    return ShapExplainer()


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
        "explainer",
        [pytest.lazy_fixture("lime_explainer"), pytest.lazy_fixture("shap_explainer"),],
    )
    def test_constructor(self, explainer):
        pass

    def _check_all_parameters_fitted(self, explainer):
        assert hasattr(explainer, "model_")
        assert hasattr(explainer, "explainer_")
        assert hasattr(explainer, "feature_names_")

    @pytest.mark.parametrize(
        "explainer",
        [pytest.lazy_fixture("lime_explainer"), pytest.lazy_fixture("shap_explainer"),],
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
        "explainer",
        [pytest.lazy_fixture("lime_explainer"), pytest.lazy_fixture("shap_explainer"),],
    )
    @given(data=data(), regressor=models(), X_y=numpy_X_y_matrices(min_value=-100, max_value=100))
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
        "explainer",
        [pytest.lazy_fixture("lime_explainer"), pytest.lazy_fixture("shap_explainer"),],
    )
    @given(regressor=models(), X=numpy_X_matrices())
    def test_error_fit_regressor_not_fitted(self, explainer, regressor, X):
        with pytest.raises(NotFittedError):
            explainer.fit(regressor, X)


class TestShap:
    def test_constructor(self, shap_explainer):
        assert len(shap_explainer.allowed_explainer) == 2
