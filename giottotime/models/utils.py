from sklearn.exceptions import NotFittedError


def check_is_fitted(estimator: object) -> None:
    """Checks if the ``estimator`` is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a ``NotFittedError`` with the given message.

    Parameters
    ----------
    estimator: ``object``, required.
        An instance of an estimator for which the check is performed

    Returns
    -------
    None

    Raises
    ------
    ``NotFittedError``
        If the attributes are not found.

    """
    if not hasattr(estimator, "fit"):
        raise TypeError(
            f"{estimator} must implement both 'fit' " f"and 'predict' methods"
        )

    attrs = [v for v in vars(estimator) if v.endswith("_")]

    if not attrs:
        raise NotFittedError("Not fitted")
