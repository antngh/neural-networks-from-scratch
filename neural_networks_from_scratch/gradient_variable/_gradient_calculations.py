from math import log
from typing import Callable

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
    x: float,
    y: float,
    func: Callable[[float], float],
    epsilon: float = NumericDifferentiationConfig.numeric_diff_epsilon,
    accurate: float = NumericDifferentiationConfig.numeric_diff_accurate,
) -> float:
    """
    Calculate the gradient of an arbitrary function of one variable.

    Parameters
    ----------
    x : float
        The value at which to calculate the gradient.
    y : float
        The value of the function at x.
        Provided for efficiency, as it may have already been calculated.
        Ignored if accurate is True
    func : Callable[[float], float]
        The function of which to calculate the gradient.
    epsilon : float, optional
        The step size for the numerical differentiation.
    accurate : bool, optional
        Whether to use a more accurate method but it requires another function call.
        Calculate the gradient symmetrically around x if True.
        Will ignore the y parameter.

    Returns
    -------
    float
        The gradient of the function at x.
    """
    res_plus = func(x + epsilon)
    res_minus = func(x - epsilon) if accurate else y
    diff_x = 2 * epsilon if accurate else epsilon
    return (res_plus - res_minus) / diff_x
