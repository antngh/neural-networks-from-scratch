from math import log
from typing import Callable

from .config import NumericDifferentiationConfig


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


def _grad_one(self_var: float, downstream_var: float, other_var: float) -> float:
    """
    The gradient is one regardless of the input values
    """
    return 1.0


def _grad_other_var(self_var: float, downstream_var: float, other_var: float) -> float:
    """
    The gradient is the "other" variable
    """
    return other_var


# known gradient calculations for common operations, quicker and more robust than
# numerical differentiation
GRAD_FUNCS = {
    "add__left": _grad_one,  # d(self + other)/d(self)
    "add__right": _grad_one,  # d(other + self)/d(self)
    "mul__left": _grad_other_var,  # d(self * other)/d(self)
    "mul__right": _grad_other_var,  # d(other * self)/d(self)
    "sub__left": _grad_one,  # d(self - other)/d(self)
    "sub__right": lambda self_var, downstream_var, other_var: -1.0,  # d(other - self)/d(self)
    "div__left": lambda self_var, downstream_var, other_var: 1.0
    / other_var,  # d(self/other)/d(self)
    "div__right": lambda self_var, downstream_var, other_var: -other_var
    / self_var**2,  # d(other/self)/d(self)
    "pow__left": lambda self_var, downstream_var, other_var: (  # d(self^other)/d(self)
        (other_var * self_var ** (other_var - 1))
        if other_var.is_integer() or self_var >= 0
        else calculate_grad_numerically(
            x=self_var, y=downstream_var, func=lambda x: x**other_var
        )
    ),
    "pow__right": lambda self_var, downstream_var, other_var: (  # d(other^self)/d(self)
        (other_var**self_var * log(other_var))
        if other_var > 0 and self_var > 0
        else calculate_grad_numerically(
            x=self_var, y=downstream_var, func=lambda x: other_var**x
        )
    ),
}
