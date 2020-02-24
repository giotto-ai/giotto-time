import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y, check_random_state


def l1(a, b):
    return np.abs(np.subtract(a, b))


class HedgeForecaster(BaseEstimator):
    """Regressor model using Hedge algorithm.

    This algorithm is based on a multiplicative weight update method to create a dynamic combination of regressive
    models. In theory, there is no common training phase on data, only the loss is necessary to update the model.

    Parameters
    ----------

    learning_rate : float, (default=0.001)
        The factor to use for the weight update.

    loss : callable, optional (default=`gtime.forecasting.online.l1`)
        Loss function use to compute loss matrix.

    random_state : int, RandomState instance or None, optional (default=None)
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        # TODO: write glossary
        See :term:`Glossary <random_state>` for details.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gtime.forecasting.online import HedgeForecaster
    >>> time_index = pd.date_range("2020-01-01", "2020-01-20")
    >>> X = pd.DataFrame(np.random.randint(4, size=(20, 3)), index=time_index)
    >>> y = pd.DataFrame(np.random.randint(4, size=(20, 1)), index=time_index, columns=["y_1"])
    >>> hr = HedgeForecaster(random_state=42)
    >>> hr.fit_predict(X, y).head()
                0
    2020-01-01  2
    2020-01-02  0
    2020-01-03  3
    2020-01-04  3
    2020-01-05  2
    >>> print(f"Estimator weights: {hr.weights_}")
    Estimator weights: [0.97713925 0.97723619 0.97980439]
    >>> print(f"Decisions: {hr.decisions_}")
    Decisions: [1 2 2 1 0 0 0 2 1 2 0 2 2 0 0 0 0 1 1 0]
    >>> print(f"Total loss: {hr.total_loss_}")
    Total loss: 30

    """

    def __init__(self, learning_rate: float = 0.001, loss: callable = l1, random_state=None):
        self.eps = learning_rate
        self.loss = loss
        self.random_state = random_state
        pass

    def hedge(self, timestamps, n_experts, loss, eps, random_state):
        weights = np.ones(n_experts)
        self.decisions_ = np.zeros(timestamps, dtype=int)

        total_loss = 0
        for t in range(timestamps):
            self.decisions_[t] = random_state.choice(n_experts, p=weights / np.sum(weights))
            total_loss += loss[t][np.int(self.decisions_[t])]
            weights *= np.exp(-eps * loss[t])
        return total_loss, weights

    def fit(self, X, y):
        """ Fit the model to data, compute weights and decisions iteratively.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        self : object
        """

        random_state = check_random_state(self.random_state)

        #  FIXME: multi_output is not currently supported but mono-column dataframe is 2D (n, 1) so multi_output=True
        #  makes it easier to handle
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.loss_matrix_ = self.loss(X, y)

        timestamps = len(X)
        n_experts = X.shape[1]

        self.total_loss_, self.weights_ = self.hedge(timestamps=timestamps, n_experts=n_experts,
                                                     loss=self.loss_matrix_, eps=self.eps, random_state=random_state)

        return self

    def fit_predict(self, X, y):
        """Fit and predict variable using Hedge algorithm.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Predictions.

        Returns
        -------

        """
        self.fit(X, y)

        predictions = pd.DataFrame(np.take_along_axis(check_array(X), self.decisions_.reshape(-1, 1), axis=1),
                                   index=X.index)

        return predictions
