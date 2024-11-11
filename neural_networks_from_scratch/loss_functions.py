from gradient_tracking.gradient_float import GFloat
from tensor import GTensor


def mean_squared_error(
    y_true: list[GTensor | GFloat | float], y_pred: list[GTensor | GFloat | float]
) -> GFloat:
    y_true = [y_.to_gfloat() if isinstance(y_, GTensor) else y_ for y_ in y_true]
    y_pred = [y_.to_gfloat() if isinstance(y_, GTensor) else y_ for y_ in y_pred]

    return sum(
        [(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_true, strict=True)]
    ) / len(y_true)
