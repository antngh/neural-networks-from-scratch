from math import log

from .config import NumericDifferentiationConfig

GRAD_FUNCS = {
    "add__left": lambda self_var, downstream_var, other_var: 1.0,
    "add__right": lambda self_var, downstream_var, other_var: 1.0,
    "mul__left": lambda self_var, downstream_var, other_var: other_var,
    "mul__right": lambda self_var, downstream_var, other_var: other_var,
    "sub__left": lambda self_var, downstream_var, other_var: 1.0,
    "sub__right": lambda self_var, downstream_var, other_var: -1.0,
    "div__left": lambda self_var, downstream_var, other_var: 1.0 / other_var,
    "div__right": lambda self_var, downstream_var, other_var: -other_var / self_var**2,
    "pow__left": lambda self_var, downstream_var, other_var: (
        (other_var * self_var ** (other_var - 1))
        if other_var.is_integer() or self_var >= 0
        else calculate_grad_numerically(
            x=self_var, y=downstream_var, func=lambda x: x**other_var
        )
    ),
    "pow__right": lambda self_var, downstream_var, other_var: (
        (other_var**self_var * log(other_var))
        if other_var > 0 and self_var > 0
        else calculate_grad_numerically(
            x=self_var, y=downstream_var, func=lambda x: other_var**x
        )
    ),
}


def calculate_grad_numerically(
    x,
    y,
    func,
    epsilon=NumericDifferentiationConfig.numeric_diff_epsilon,
    accurate=NumericDifferentiationConfig.numeric_diff_accuracte,
) -> float:
    res_plus = func(x + epsilon)
    res_minus = func(x - epsilon) if accurate else y
    diff_x = 2 * epsilon if accurate else epsilon
    return (res_plus - res_minus) / diff_x
