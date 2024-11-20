from neural_networks_from_scratch.gradient_variable import GradientVariable
from neural_networks_from_scratch.tensor import GradientTensor


def _clean_y(y: list[GradientVariable | float]) -> list[GradientVariable | float]:
    return [y_.to_gvariable() if isinstance(y_, GradientTensor) else y_ for y_ in y]


def mean_squared_error(
    y_true: list[GradientVariable | float],
    y_pred: GradientTensor | list[GradientVariable | float],
) -> GradientVariable:
    return sum(
        [
            (y_p - y_t) ** 2
            for y_p, y_t in zip(_clean_y(y_pred), _clean_y(y_true), strict=True)
        ]
    ) / len(y_true)
