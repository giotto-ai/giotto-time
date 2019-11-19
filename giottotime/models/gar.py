from copy import deepcopy
from typing import Union, Dict, Optional

import pandas as pd
import pandas.util.testing as testing
from sklearn.linear_model import LinearRegression

from giottotime.feature_creation.feature_creation import FeaturesCreation
from giottotime.feature_creation.time_series_features import \
    MovingAverageFeature, ExogenousFeature, ShiftFeature
from giottotime.models.utils import check_is_fitted


def check_x(X):
    # TODO: Call Stefano's function to check whether X is a valid input or not
    pass


class GAR:
    """
    This class implements the Generalized Auto Regression model.

    Parameters
    ----------
    feature_creator: FeaturesCreation
        This object is responsible for the creation of the features
    base_model: object
        The model used to make the predictions step by step. This class must
        have a ``fit``and ``predict`` method.
    horizon: int
        The horizon represents how many steps into the future to predict
    feed_forward: bool
        If true, feed-forward the predictions of the models at training and
        prediction time

    """

    def __init__(self, feature_creator: FeaturesCreation, base_model: object,
                 horizon: int, feed_forward: bool = False):

        if not hasattr(base_model, 'fit') or \
                not hasattr(base_model, 'predict'):
            raise TypeError(f"{base_model} must implement both 'fit' "
                            f"and 'predict' methods")

        if horizon <= 0:
            raise ValueError("The horizon should be greater than 0, but "
                             f"has value {horizon}.")

        self._feature_creator = feature_creator
        self._base_model = base_model
        self._models_per_predstep = [deepcopy(base_model)
                                     for _ in range(horizon)]
        self._feed_forward = feed_forward

    def fit(self, ts: Union[pd.DataFrame, pd.Series],
            **kwargs: Dict[str, object]) -> object:
        """
        Fit the GAR model according to the training data.

        Parameters
        ----------
        ts: Union[pd.DataFrame, pd.Series]
            Time series of shape (n_samples, 1) on which to fit the model
        kwargs: Dict[str, object]
            Optional parameters to be passed to the base model during the
            fit procedure

        Returns
        -------
        self: object

        """

        check_x(ts)
        features, y = self._feature_creator.fit_transform(ts)

        for pred_step in range(len(self._models_per_predstep)):
            model_for_pred_step = self._models_per_predstep[pred_step]
            model_for_pred_step.fit(features, y, **kwargs)

            if self._feed_forward:
                predictions = model_for_pred_step.predict(features)
                features = pd.concat([features, predictions], axis=1)

        self.x_features_ = features

        return self

    def predict(self, ts: Optional[Union[pd.DataFrame, pd.Series]] = None,
                start_date: Optional[Union[pd.Timestamp, str]] = None)\
            -> pd.DataFrame:
        """
        Make predictions for each sample and for each prediction step

        Parameters
        ----------
        ts: Union[pd.DataFrame, pd.Series], optional
            Time series of shape (n_samples, 1) from which to start to predict
        start_date: Union[pd.Timestamp, str], optional
            If provided, start predicting from this date.

        Returns
        -------
        predictions: pd.DataFrame
            The predictions of the model.

        Notes
        -----
        If start_date has been provided, the predictions are going to be
        starting the specified date and is going to contain as many predictions
        as number of samples with a greater date. Otherwise, the DataFrame of
        the predictions has shape (n_samples, horizon), where n_samples is the
        length ``ts``.

        """

        check_x(ts)
        check_is_fitted(self)

        features, _ = self._feature_creator.fit_transform(ts)

        # TODO: check this if is correct
        if start_date is not None:
            features = features[features.index >= start_date]

        predictions = pd.DataFrame(index=features.index)

        for pred_step in range(len(self._models_per_predstep)):
            model_for_pred_step = self._models_per_predstep[pred_step]
            model_predictions = model_for_pred_step.predict(features)
            predictions[f"y_{pred_step}"] = model_predictions

            if self._feed_forward:
                features = pd.concat([features, model_predictions], axis=1)

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

    ar = GAR(feature_creation, base_m, h)
    ar.fit(t[: 100])

    preds = ar.predict(t[100:])
    print(preds)