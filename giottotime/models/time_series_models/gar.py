from copy import deepcopy

import pandas as pd
from sklearn.utils.validation import check_is_fitted


class GAR:
    """This class implements the Generalized Auto Regression model.

    Parameters
    ----------
    base_model: object, required
        The model used to make the predictions step by step. This class must have a
        ``fit``and ``predict`` method.

    feed_forward: bool, optional, default: ``False``
        If true, feed-forward the predictions of the time_series_models at training and
        prediction time.

    """

    def __init__(self, base_model: object, feed_forward: bool = False):
        if not hasattr(base_model, "fit") or not hasattr(base_model, "predict"):
            raise TypeError(
                f"{base_model} must implement both 'fit' " f"and 'predict' methods"
            )

        self._base_model = base_model
        self._feed_forward = feed_forward

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs: object) -> "GAR":
        """Fit the GAR model according to the training data.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, n_features), required
            Features used to fit the model.

        y: pd.DataFrame, shape (n_samples, horizon), required
            Target values to fit on.

        kwargs: dict, optional
            Optional parameters to be passed to the base model during the fit procedure.

        Returns
        -------
        self: GAR
            The fitted GAR object.

        """
        features = X.copy()
        models_per_predstep = [deepcopy(self._base_model) for _ in range(y.shape[1])]

        for pred_step, model_for_pred_step in enumerate(models_per_predstep):
            target_y = y[f"y_{pred_step}"]
            model_for_pred_step.fit(features, target_y, **kwargs)

            if self._feed_forward:
                predictions = model_for_pred_step.predict(features)
                features[f"preds_{pred_step}"] = predictions

        self.models_per_predstep_ = models_per_predstep
        self.train_features_ = X

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for each sample and for each prediction step.

        Parameters
        ----------
        X: pd.DataFrame, shape (n_samples, n_features), required
            Features used to predict.

        Returns
        -------
        predictions: pd.DataFrame, shape (n_samples, 1)
            The predictions of the model.

        Raises
        ------
        NotFittedError
            Thrown if the model has not been previously fitted.

        """
        check_is_fitted(self, ["models_per_predstep_", "train_features_"])

        test_features = X.copy()

        predictions = pd.DataFrame(index=test_features.index)

        for pred_step, model_for_pred_step in enumerate(self.models_per_predstep_):
            model_predictions = model_for_pred_step.predict(test_features)
            predictions[f"y_{pred_step}"] = model_predictions

            if self._feed_forward:
                test_features[f"preds_{pred_step}"] = model_predictions

        return predictions
