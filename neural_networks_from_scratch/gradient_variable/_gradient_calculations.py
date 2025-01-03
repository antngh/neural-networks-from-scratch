from math import e, log
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


def get_numerical_gradient_func(
    func: Callable[[float], float] | Callable[[float, float], float],
    other_var: float | None = None,
    left: bool = True,
):
    """
    Get a function to calculate the gradient of a function numerically.

    Parameters
    ----------
    func : Callable[[float], float] | Callable[[float, float], float]
        The function of which to calculate the gradient.
        Either a unary function or a binary function.
    other_var : float | None, optional
        The other variable in the binary function.
        If provided assumes that func is binary, otherwise assumes it is unary.
    left : bool, optional
        Whether self is the first or second variable in the binary function.
        Is ignored if func is a unary function.

    Returns
    -------
    Callable[[float, float, float], float]
        The function to calculate the gradient of the function numerically.
        Takes the self variable, the downstream variable, and the other variable,
        and returns the gradient.
    """
    func = (
        func
        if not other_var
        else lambda x: (func(x, other_var) if left else lambda x: func(other_var, x))
    )
    return lambda self_var, downstream_var, other_var: calculate_grad_numerically(
        x=self_var, y=downstream_var, func=func
    )


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


def _pow_left(
    self_var,
    downstream_var,
    other_var,
    epsilon=NumericDifferentiationConfig.numeric_diff_epsilon,
):
    if other_var.is_integer() or self_var >= 0:
        return other_var * self_var ** (other_var - 1)

    return calculate_grad_numerically(
        x=self_var, y=downstream_var, func=lambda x: x**other_var, epsilon=epsilon
    )


def _pow_right(
    self_var,
    downstream_var,
    other_var,
    epsilon=NumericDifferentiationConfig.numeric_diff_epsilon,
):
    if abs(other_var - e) < epsilon:
        return downstream_var

    if other_var > 0 and self_var > 0:
        return downstream_var * log(other_var)

    return calculate_grad_numerically(
        x=self_var, y=downstream_var, func=lambda x: other_var**x, epsilon=epsilon
    )


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
    "pow__left": _pow_left,  # d(self^other)/d(self)
    "pow__right": _pow_right,  # d(other^self)/d(self)
}
