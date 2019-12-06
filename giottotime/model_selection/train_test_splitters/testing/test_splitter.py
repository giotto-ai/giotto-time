import datetime
from typing import Tuple

from hypothesis import given, settings
import pytest
import pandas as pd
from hypothesis._strategies import integers, datetimes, floats
from pandas.testing import assert_frame_equal

from giottotime.feature_creation import ShiftFeature, MovingAverageFeature
from ..base import Splitter
from ..splitters import DatetimeSplitter, PercentageSplitter, TrainSizeSplitter
from giottotime.core.hypothesis.feature_matrices import X_y_matrices

features_to_consider = [
    ShiftFeature(1, "shift_feature_1"),
    ShiftFeature(2, "shift_feature_2"),
    MovingAverageFeature(4, "moving_average_4"),
]


def test_splitter_is_abstract_class():
    with pytest.raises(TypeError):
        Splitter()


# noinspection PyPep8Naming
class TestDatetimeSplitter:
    def test_datetime_splitter_initialization(self):
        DatetimeSplitter()

    @settings(max_examples=10)
    @given(X_y_matrices(horizon=3, time_series_features=features_to_consider))
    def test_transform_with_split_at_time_none(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame]
    ):
        X, y = matrices

        datetime_splitter = DatetimeSplitter()
        X_train, y_train, X_test, y_test = datetime_splitter.transform(
            X, y, split_at_time=None
        )

        assert_frame_equal(X_train, X)
        assert_frame_equal(y_train, y)
        assert X_test.shape[0] == 0
        assert y_test.shape[0] == 0

    @settings(max_examples=10)
    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider),
        integers(0, 200),
    )
    def test_transform_at_datetime_in_X_index(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], split_at_element: int
    ):
        X, y = matrices
        try:
            datetime_to_split_at = X.index[split_at_element]
        except IndexError:
            split_at_element = X.shape[0] - 1
            datetime_to_split_at = X.index[-1]

        datetime_splitter = DatetimeSplitter()
        X_train, y_train, X_test, y_test = datetime_splitter.transform(
            X, y, split_at_time=datetime_to_split_at
        )

        try:
            assert X_train.shape[0] == split_at_element + 1
            assert X_test.shape[0] == X.shape[0] - (split_at_element + 1)
            assert X_train.shape[0] == y_train.shape[0]
            assert X_test.shape[0] == y_test.shape[0]
        except Exception as e:
            print(datetime_to_split_at, split_at_element)
            print(e)
            raise e

    @settings(max_examples=10)
    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider), datetimes()
    )
    def test_transform_at_random_datetime(
        self,
        matrices: Tuple[pd.DataFrame, pd.DataFrame],
        split_at_datetime: datetime.datetime,
    ):
        X, y = matrices
        expected_X_train_shape = sum(X.index <= split_at_datetime)

        datetime_splitter = DatetimeSplitter()
        X_train, y_train, X_test, y_test = datetime_splitter.transform(
            X, y, split_at_time=split_at_datetime
        )

        assert X_train.shape[0] == expected_X_train_shape
        assert X_test.shape[0] == X.shape[0] - expected_X_train_shape
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


class TestPercentageSplitter:
    def test_percentage_splitter_initialization(self):
        PercentageSplitter()

    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider),
        floats(-1000001, -0.001),
    )
    def test_error_with_negative_percentage(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], percentage: float
    ):
        X, y = matrices

        percentage_splitter = PercentageSplitter()
        with pytest.raises(ValueError):
            X_train, y_train, X_test, y_test = percentage_splitter.transform(
                X, y, split_at_percentage=percentage
            )

    @settings(max_examples=10)
    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider),
        floats(1.0001, 100000),
    )
    def test_error_with_percentage_greater_than_1(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], percentage: float
    ):
        X, y = matrices

        percentage_splitter = PercentageSplitter()
        with pytest.raises(ValueError):
            X_train, y_train, X_test, y_test = percentage_splitter.transform(
                X, y, split_at_percentage=percentage
            )

    @settings(max_examples=10)
    @given(X_y_matrices(horizon=3, time_series_features=features_to_consider))
    def test_transform_with_default_percentage(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame]
    ):
        X, y = matrices

        percentage_splitter = PercentageSplitter()
        X_train, y_train, X_test, y_test = percentage_splitter.transform(X, y)

        assert_frame_equal(X_train, X)
        assert_frame_equal(y_train, y)
        assert X_test.shape[0] == 0
        assert y_test.shape[0] == 0

    @settings(max_examples=10)
    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider), floats(0, 1)
    )
    def test_transform_with_random_percentage(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], percentage: float
    ):
        X, y = matrices

        percentage_splitter = PercentageSplitter()
        X_train, y_train, X_test, y_test = percentage_splitter.transform(
            X, y, split_at_percentage=percentage
        )
        expected_train_shape = min(int(X.shape[0] * percentage) + 1, X.shape[0])

        assert X_train.shape[0] == expected_train_shape
        assert y_train.shape[0] == expected_train_shape
        assert X_test.shape[0] == X.shape[0] - expected_train_shape
        assert y_test.shape[0] == X.shape[0] - expected_train_shape


class TestTrainSizeSplitter:
    def test_train_size_splitter_initialization(self):
        TrainSizeSplitter()

    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider),
        integers(-1000, -1),
    )
    def test_error_with_negative_train_elements(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], train_elements: int
    ):
        X, y = matrices

        train_size_splitter = TrainSizeSplitter()
        with pytest.raises(ValueError):
            X_train, y_train, X_test, y_test = train_size_splitter.transform(
                X, y, train_elements=train_elements
            )

    @settings(max_examples=10)
    @given(X_y_matrices(horizon=3, time_series_features=features_to_consider))
    def test_transform_with_default_number_of_elements(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame]
    ):
        X, y = matrices

        train_size_splitter_splitter = TrainSizeSplitter()
        X_train, y_train, X_test, y_test = train_size_splitter_splitter.transform(X, y)

        assert_frame_equal(X_train, X)
        assert_frame_equal(y_train, y)
        assert X_test.shape[0] == 0
        assert y_test.shape[0] == 0

    @settings(max_examples=10)
    @given(
        X_y_matrices(horizon=3, time_series_features=features_to_consider),
        integers(0, 100000),
    )
    def test_transform_with_random_number_of_elements(
        self, matrices: Tuple[pd.DataFrame, pd.DataFrame], train_elements: int
    ):
        X, y = matrices

        train_size_splitter_splitter = TrainSizeSplitter()
        X_train, y_train, X_test, y_test = train_size_splitter_splitter.transform(
            X, y, train_elements=train_elements
        )

        expected_train_elements = min(train_elements, X.shape[0])

        assert X_train.shape[0] == expected_train_elements
        assert y_train.shape[0] == expected_train_elements
        assert X_test.shape[0] == X.shape[0] - expected_train_elements
        assert y_test.shape[0] == X.shape[0] - expected_train_elements
