from neural_networks_from_scratch.gradient_variable import (
    GradientTensor,
    GradientVariable,
)


def _clean_y(y: list[GradientVariable | float]) -> list[GradientVariable | float]:
    return [y_.to_gvariable() if isinstance(y_, GradientTensor) else y_ for y_ in y]


def mean_squared_error(
    y_true: GradientTensor | list[GradientVariable | float],
    y_pred: GradientTensor | list[GradientVariable | float],
) -> GradientVariable:
    """
    Calculate the mean squared error between the true and predicted values.

    The outer dimensions of the true and predicted values must match, each
    corresponding to a data sample.

    Parameters
    ----------
    y_true : GradientTensor | list[GradientVariable | float]
        The true values.
    y_pred : GradientTensor | list[GradientVariable | float]
        The predicted values.

    Returns
    -------
    GradientVariable
        The mean squared error.
    """
    return sum(
        [
            (y_p - y_t) ** 2
            for y_p, y_t in zip(_clean_y(y_pred), _clean_y(y_true), strict=True)
        ]
    ) / len(y_true)
