import pytest
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.strategies import floats, sampled_from
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    LogisticRegression,
    BayesianRidge,
)
from sklearn.tree import ExtraTreeRegressor

from gtime.explainability import LimeExplainer, ShapExplainer
from gtime.utils.hypothesis.feature_matrices import numpy_X_matrices


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
        AdaBoostRegressor(),
        BaggingRegressor(),
        ExtraTreeRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
    ]
    return sampled_from(regressors)


@given(models())
def test_models(regressor):
    assert hasattr(regressor, 'fit')
    assert hasattr(regressor, 'predict')


class TestAllExplainers:
    @pytest.mark.parametrize("explainer", [lime_explainer, shap_explainer])
    def test_constructor(self, explainer):
        pass


    @pytest.mark.parametrize("explainer", [lime_explainer, shap_explainer])
    @given(regressor=models(), X=numpy_X_matrices())
    def test_fit(self):
        pass

class TestShap:
    def test_constructor(self, shap_explainer):
        assert len(shap_explainer.allowed_explainer) == 2
