from copy import deepcopy
from typing import Union, Dict, Optional

import pandas as pd
import pandas.util.testing as testing
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import \
    MovingAverageFeature, ExogenousFeature, ShiftFeature
from giottotime.models.utils import check_is_fitted


def check_input(X, y=None):
    # TODO: Call Stefano's function to check whether X is a valid input or not
    pass


class GAR:
    """
    This class implements the Generalized Auto Regression model.

    Parameters
    ----------
    base_model: object
        The model used to make the predictions step by step. This class must
        have a ``fit``and ``predict`` method.
    feed_forward: bool
        If true, feed-forward the predictions of the models at training and
        prediction time

    """

    def __init__(self,
                 base_model: object,
                 feed_forward: bool = False):

        if not hasattr(base_model, 'fit') or \
                not hasattr(base_model, 'predict'):
            raise TypeError(f"{base_model} must implement both 'fit' "
                            f"and 'predict' methods")

        self._base_model = base_model
        self._feed_forward = feed_forward

    # TODO: change from time-series to features as input
    def fit(self, X: Union[pd.DataFrame, pd.Series],
            y: Union[pd.DataFrame, pd.Series, str],
            **kwargs: Dict[str, object]) -> object:
        """
        Fit the GAR model according to the training data.

        Parameters
        ----------
        X: Union[pd.DataFrame, pd.Series]
            Features used to fit the model
        y: Union[pd.DataFrame, pd.Series, str]
            If a DataFrame or a Series, target values to fit on. If a string,
            the y represent the name of the target column contained in the X
        kwargs: Dict[str, object]
            Optional parameters to be passed to the base model during the
            fit procedure

        Returns
        -------
        self: object

        """

        check_input(X, y)

        if isinstance(y, str):
            y = X[y]

        features = deepcopy(X)
        self.models_per_predstep_ = [deepcopy(self._base_model)
                                     for _ in range(y.shape[1])]

        for pred_step in range(len(self.models_per_predstep_)):
            model_for_pred_step = self.models_per_predstep_[pred_step]
            target_y = y[f"y_{pred_step}"]
            model_for_pred_step.fit(features, target_y, **kwargs)

            if self._feed_forward:
                predictions = model_for_pred_step.predict(features)
                features[f"preds_{pred_step}"] = predictions

        self.train_features_ = X

        return self

    # TODO: change from time-series to features as input
    def predict(self, X: Union[pd.DataFrame, pd.Series],
                start_date: Optional[Union[pd.Timestamp, str]] = None)\
            -> pd.DataFrame:
        """
        Make predictions for each sample and for each prediction step

        Parameters
        ----------
        X: Union[pd.DataFrame, pd.Series]
            Features used to predict
        start_date: Union[pd.Timestamp, str], optional
            If provided, start predicting from this date.

        Returns
        -------
        predictions: pd.DataFrame
            The predictions of the model.

        Raises
        ------
        NotFittedError
            Thrown if the model has not been previously fitted

        Notes
        -----
        If start_date has been provided, the predictions are going to be
        starting the specified date and is going to contain as many predictions
        as number of samples with a greater date. Otherwise, the DataFrame of
        the predictions has shape (n_samples, horizon), where n_samples is the
        length ``ts``.

        """

        check_is_fitted(self)

        test_features = deepcopy(X)
        # TODO: check this if is correct
        if start_date is not None:
            test_features = X[X.index >= start_date]

        predictions = pd.DataFrame(index=test_features.index)

        for pred_step in range(len(self.models_per_predstep_)):
            model_for_pred_step = self.models_per_predstep_[pred_step]
            model_predictions = model_for_pred_step.predict(test_features)
            predictions[f"y_{pred_step}"] = model_predictions

            if self._feed_forward:
                test_features.loc[:, f"preds_{pred_step}"] = model_predictions

        return predictions


if __name__ == "__main__":
    base_m = LinearRegression()
    testing.N, testing.K = 200, 1

    t = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_1 = testing.makeTimeDataFrame(freq='MS')
    x_exogenous_2 = testing.makeTimeDataFrame(freq='MS')

    time_series_features = [MovingAverageFeature(2),
                            MovingAverageFeature(4),
                            ShiftFeature(-1),
                            ExogenousFeature(x_exogenous_1, "test_ex1"),
                            ExogenousFeature(x_exogenous_2, "test_ex2")]

    h = 4
    feature_creation = FeaturesCreation(h, time_series_features)

    x_train, y_train, x_test = feature_creation.fit_transform(t)

    ar = GAR(base_m, feed_forward=False)
    ar.fit(x_train, y_train)

    preds = ar.predict(x_test)
    print(preds)
