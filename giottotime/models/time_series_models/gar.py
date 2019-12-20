from copy import deepcopy

import pandas as pd
from sklearn.utils.validation import check_is_fitted


class GAR:
    """Implementation of the Generalized Auto Regression model.

    Parameters
    ----------
    base_model: object, required
        The model used to make the predictions step by step. This object must have a
        ``fit``and ``predict`` method.

    feed_forward: bool, optional, default: ``False``
        If true, feed-forward the predictions of the time_series_models at training and
        prediction time.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from giottotime.models import GAR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> time_index = pd.date_range("2020-01-01", "2020-01-30")
    >>> X = pd.DataFrame(np.random.random((30, 5)), index=time_index)
    >>> y_columns = ["y_1", "y_2", "y_3"]
    >>> y = pd.DataFrame(np.random.random((30, 3)), index=time_index, columns=y_columns)
    >>> X_train, y_train = X[:20], y[:20]
    >>> X_test, y_test = X[20:], y[20:]
    >>> random_forest = RandomForestRegressor()
    >>> gar = GAR(base_model=random_forest)
    >>> gar.fit(X_train, y_train)
    >>> predictions = gar.predict(X_test)
    >>> predictions.shape
    (10, 3)

    """

    def __init__(self, base_model: object, feed_forward: bool = False):
        if not hasattr(base_model, "fit") or not hasattr(base_model, "predict"):
            raise TypeError(
                f"{base_model} must implement both 'fit' " f"and 'predict' methods"
            )

        self.base_model = base_model
        self.feed_forward = feed_forward

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
        models_per_predstep = [deepcopy(self.base_model) for _ in range(y.shape[1])]

        for pred_step, model_for_pred_step in enumerate(models_per_predstep, 1):
            target_y = y[f"y_{pred_step}"]
            model_for_pred_step.fit(features, target_y, **kwargs)

            if self.feed_forward:
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
        check_is_fitted(self)

        test_features = X.copy()

        predictions = pd.DataFrame(index=test_features.index)

        for pred_step, model_for_pred_step in enumerate(self.models_per_predstep_, 1):
            model_predictions = model_for_pred_step.predict(test_features)
            predictions[f"y_{pred_step}"] = model_predictions

            if self.feed_forward:
                test_features[f"preds_{pred_step}"] = model_predictions

        return predictions
