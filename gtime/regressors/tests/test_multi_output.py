import numpy as np
import pytest
from hypothesis import strategies as st, given, assume
from hypothesis.strategies import integers, data
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputRegressor

from gtime.regressors.multi_output import MultiFeatureMultiOutputRegressor
from gtime.utils.hypothesis.feature_matrices import numpy_X_y_matrices, numpy_X_matrices
from gtime.utils.fixtures import estimator
from gtime.utils.hypothesis.general_strategies import shape_X_y_matrices


@st.composite
def numeric_target_to_features_dicts(
    draw,
    n_targets: int,
    n_features: int,
    min_features_per_target: int = 1,
    max_features_per_target: int = None,
):
    if max_features_per_target is None:
        max_features_per_target = n_features
    target_to_features_dict = {}
    for i in range(n_targets):
        n_target_features = draw(
            integers(min_features_per_target, max_features_per_target)
        )
        target_features = sorted(
            np.random.choice(n_features, n_target_features, replace=False)
        )
        target_to_features_dict[i] = target_features
    return target_to_features_dict


@given(
    data=data(),
    n_targets=integers(1, 10),
    n_features=integers(1, 10),
    min_features_per_target=integers(1, 4),
    max_features_per_target=integers(5, 10),
)
def test_numeric_target_to_features_dicts(
    data, n_targets, n_features, min_features_per_target, max_features_per_target
):
    assume(min_features_per_target <= n_features)
    assume(max_features_per_target <= n_features)
    target_to_features_dict = data.draw(
        numeric_target_to_features_dicts(
            n_targets, n_features, min_features_per_target, max_features_per_target
        )
    )
    assert len(target_to_features_dict) == n_targets
    for target, features in target_to_features_dict.items():
        assert max(features) < n_features
        assert min(features) >= 0
        assert len(set(features)) == len(features)
        assert min_features_per_target <= len(features) <= max_features_per_target


class TestMultiFeatureMultiOutputRegressor:
    def test_constructor(self, estimator):
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator
        )
        assert multi_feature_multi_output_regressor.n_jobs == 1

    @given(
        data=data(),
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        ),
    )
    def test_fit_bad_y(self, data, estimator, X_y):
        X, y = X_y
        y = y[:, 0].flatten()
        target_to_features_dict = data.draw(
            numeric_target_to_features_dicts(n_targets=1, n_features=X.shape[1])
        )
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator, target_to_features_dict=target_to_features_dict
        )
        with pytest.raises(ValueError):
            multi_feature_multi_output_regressor.fit(
                X, y
            )

    @given(
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        )
    )
    def test_fit_as_multi_output_regressor_if_target_to_feature_none(
        self, estimator, X_y
    ):
        X, y = X_y
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator
        )
        multi_feature_multi_output_regressor.fit(X, y)

        multi_output_regressor = MultiOutputRegressor(estimator)
        multi_output_regressor.fit(X, y)

        assert_almost_equal(
            multi_feature_multi_output_regressor.predict(X),
            multi_output_regressor.predict(X),
        )

    @given(X=numpy_X_matrices(min_value=-10000, max_value=10000))
    def test_error_predict_with_no_fit(self, estimator, X):
        regressor = MultiFeatureMultiOutputRegressor(estimator)
        with pytest.raises(NotFittedError):
            regressor.predict(X)

    @given(
        data=data(),
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        ),
    )
    def test_fit_target_to_features_dict_working(self, data, X_y, estimator):
        X, y = X_y
        target_to_features_dict = data.draw(
            numeric_target_to_features_dicts(n_targets=y.shape[1], n_features=X.shape[1])
        )
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator
        )
        multi_feature_multi_output_regressor.target_to_features_dict = target_to_features_dict
        multi_feature_multi_output_regressor.fit(
            X, y
        )

    @given(
        data=data(),
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        ),
    )
    def test_fit_target_to_features_dict_consistent(self, data, X_y, estimator):
        X, y = X_y
        target_to_features_dict = data.draw(
            numeric_target_to_features_dicts(n_targets=y.shape[1], n_features=X.shape[1])
        )
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator, target_to_features_dict=target_to_features_dict
        )
        multi_feature_multi_output_regressor.fit(
            X, y
        )
        for i, estimator_ in enumerate(
            multi_feature_multi_output_regressor.estimators_
        ):
            expected_n_features = len(target_to_features_dict[i])
            assert len(estimator_.coef_) == expected_n_features

    @given(
        data=data(),
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        ),
    )
    def test_predict_target_to_features_dict(self, data, X_y, estimator):
        X, y = X_y
        target_to_features_dict = data.draw(
            numeric_target_to_features_dicts(n_targets=y.shape[1], n_features=X.shape[1])
        )
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator
        )
        multi_feature_multi_output_regressor.target_to_features_dict = target_to_features_dict
        multi_feature_multi_output_regressor.fit(
            X, y
        )
        X_predict = data.draw(numpy_X_matrices([100, X.shape[1]]))
        multi_feature_multi_output_regressor.predict(X_predict)

    @given(
        data=data(),
        X_y=numpy_X_y_matrices(
            X_y_shapes=shape_X_y_matrices(y_as_vector=False),
            min_value=-10000,
            max_value=10000,
        ),
    )
    def test_error_predict_target_to_features_dict_wrong_X_shape(
        self, data, X_y, estimator
    ):
        X, y = X_y
        target_to_features_dict = data.draw(
            numeric_target_to_features_dicts(n_targets=y.shape[1], n_features=X.shape[1])
        )
        multi_feature_multi_output_regressor = MultiFeatureMultiOutputRegressor(
            estimator, target_to_features_dict=target_to_features_dict
        )
        multi_feature_multi_output_regressor.fit(
            X, y
        )
        X_predict = data.draw(numpy_X_matrices([100, 30]))
        with pytest.raises(ValueError):
            multi_feature_multi_output_regressor.predict(X_predict)
