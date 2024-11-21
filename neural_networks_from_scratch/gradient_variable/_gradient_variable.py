from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from ._gradient_calculations import GRAD_FUNCS, calculate_grad_numerically


class GradientVariable:
    """
    A class to hold a float and track it's gradient.

    Attributes
    ----------
    val : float
        The value of the variable.
    is_updatable : bool
        Whether the variable can be updated, i.e is it a variable or a constant
    name : str
        The name of the variable, used for debugging mainly
    track_gradients : bool
        Whether to track gradients. By default you want this to be true, but should be
        false at inference time (as it gets very inneficient otherwise)
    grad_funcs : dict[str, Callable]
        A dictionary of functions used to calculate the gradient of the variable
    """

    def __init__(
        self, val: float, is_updatable: bool = True, name: str | None = None
    ) -> None:
        """
        Initialize the GradientVariable.

        Parameters
        ----------
        val : float
            The (initial) value of the variable.
        is_updatable : bool, optional
            Whether the variable can be updated, i.e is it a variable or a constant
        name : str, optional
            The name of the variable, used for debugging mainly
        """
        if val != val:
            raise ValueError("NaNs aren't valid")

        if not isinstance(val, (float, int)):
            raise ValueError(f"Expected float or int, got {type(val)}")
        self.val = float(val)
        self.is_updatable = is_updatable
        self.name = name
        self.track_gradients = True

        self.grad_funcs = GRAD_FUNCS.copy()

        self._grad = None
        self._is_output = False

        self._downstream_data: list[tuple[GradientVariable, str, Any]] = []
        self._upstream_vars: list[GradientVariable] = []

    @property
    def downstream_data(self) -> list[tuple[GradientVariable, str, Any]]:
        """
        Get the downstream data (if tracking gradients) of the variable, corresponding
        to all directly downstream variables.

        Returns
        -------
        list[tuple[GradientVariable, str, Any]]
            The downstream data, a list of tuples of the form:
                - downstream_var: the result of the calculation of self and the other
                  variable
                - mathematical relation: The function name
                - other variable: The other variable in the calculation.
            If the function is a unary function, the other variable will be unary
            function itself.
        """
        return self._downstream_data if self.track_gradients else []

    @downstream_data.setter
    def downstream_data(self, value: list[tuple[GradientVariable, str, Any]]) -> None:
        """
        Set the downstream data.

        This is mainly for internal use, the downstream data should be set by the
        mathematical operations or the "apply func" methods
        """
        self._downstream_data = value

    @property
    def downstream_vars(self) -> list[GradientVariable]:
        """
        Get the directly downstream variables if tracking gradients.

        Returns
        -------
        list[GradientVariable]
            The directly downstream variables.
        """
        return [gvar for gvar, *_ in self.downstream_data]

    @property
    def upstream_vars(self) -> list[GradientVariable]:
        """
        A list of the GradientVariable that are directly upstream of this one, if
        tracking gradients.

        Returns
        -------
        list[GradientVariable]
            The directly upstream variables.
            Will be empty if not tracking gradients, or if there are no upstream
            variables.
            Will have one entry if this variable is the result of a unary function on a
            GradientVariable or a binary function on a GradientVariable and a float
            Will have two entries if this variable is the result of a binary function
            on two GradientVariables.
            Will never have more than two parents.
        """
        return self._upstream_vars if self.track_gradients else []

    @upstream_vars.setter
    def upstream_vars(self, value: list[GradientVariable]) -> None:
        """
        Set the upstream variables.

        Parameters
        ----------
        value : list[GradientVariable]
            The upstream variables.
        """
        self._upstream_vars = value

    def _get_all_upstream_or_downstream_vars(
        self, upstream: bool, with_counts: bool, _var_counts=None
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        """
        Get all the upstream or downstream variables of this variable across the entire
        network.

        Parameters
        ----------
        upstream : bool
            Whether to get the upstream or downstream variables
        with_counts : bool
            Whether to return the counts of each variable
        _var_counts : dict[GradientVariable, int], optional
            The dictionary to store the counts of each variable, defaults to None
        """
        vars = self.upstream_vars if upstream else self.downstream_vars
        _var_counts = _var_counts or defaultdict(int)
        for child_gvar in vars:
            _var_counts[child_gvar] += 1
            child_gvar._get_all_upstream_or_downstream_vars(
                with_counts=True, upstream=upstream, _var_counts=_var_counts
            )
        return _var_counts if with_counts else set(_var_counts.keys())

    def get_all_upstream_vars(
        self, with_counts=False
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        """
        Get all the upstream variables of this variable across the entire network.

        Parameters
        ----------
        with_counts : bool, optional
            Whether to return the counts of each variable, defaults to False

        Returns
        -------
        set[GradientVariable] | dict[GradientVariable, int]
            The upstream variables, either as a set or as a dictionary with counts,
            where the count is the number of paths from the upstream variable to this
            one.
        """
        return self._get_all_upstream_or_downstream_vars(
            upstream=True, with_counts=with_counts
        )

    def get_all_downstream_vars(
        self, with_counts=False
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        """
        Get all the downstream variables of this variable across the entire network.

        Parameters
        ----------
        with_counts : bool, optional
            Whether to return the counts of each variable, defaults to False

        Returns
        -------
        set[GradientVariable] | dict[GradientVariable, int]
            The downstream variables, either as a set or as a dictionary with counts,
            where the count is the number of paths from this variable to the downstream
            one.
        """
        return self._get_all_upstream_or_downstream_vars(
            upstream=False, with_counts=with_counts
        )

    def _calculate_grad_from_downstream_var(
        self, downstream_var: GradientVariable, relation: str, other: Any
    ) -> float:
        """
        Calculate the gradient of the downstream variable with respect to self

        Parameters
        ----------
        downstream_var : GradientVariable
            The downstream variable
        relation : str
            The relation between the variables (i.e. the function name)
        other : Any
            The other variable in the calculation, can be a float or a function if
            downstream is a unary function of self.
        """
        if downstream_var not in self.downstream_vars:
            raise RuntimeError("Variable not downstream")

        # The relation should have been set in the call to applyfunc or
        # applyfunc_two_inputs (or the math methods that call this)
        return self.grad_funcs[relation](
            float(self),
            float(downstream_var),
            (
                float(other)
                if isinstance(other, (GradientVariable, float, int))
                else other
            ),
        )

    def calculate_grad(self) -> float:
        """
        Calculate the gradient of the output of the network with respect to this
        variable.

        Uses the chain rule to calculate the gradient of the downstream var with
        respect to self, and multiplying this by the gradient of the network output with
        to the downstream var, and summing over all downstream vars.

        Returns
        -------
        float
            The gradient of the output of the network with respect to this variable.
        """
        if not self.downstream_data:
            raise RuntimeError("No downstream data")

        grad = 0.0
        for downstream_var, relation, other_var in self.downstream_data:
            grad += downstream_var.grad * self._calculate_grad_from_downstream_var(
                downstream_var, relation, other_var
            )
        return grad

    @property
    def grad(self) -> float:
        """
        Calculates the gradient of the output of the network with respect to this one if
        it hasn't been calculated yet.

        Returns
        -------
        float
            The gradient of the output of the network with respect to this variable.
        """
        if self._grad is None:
            self.grad = self.calculate_grad()
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        """
        Set the gradient of the output of the network with respect to this variable.

        Parameters
        ----------
        value : float
            The gradient
        """
        if self._grad is not None:
            raise RuntimeError(f"Gradient already set. {self=}")
        self._grad = value

    def set_is_network_output(self) -> None:
        """
        Set this variable as the output of the network.

        You probably want to do this for the purposes of training a network, in which
        case this variable should be the loss.
        """
        self.grad = 1.0  # Gradient of the output with respect to itself is 1
        self._is_output = True

        if any(gf._is_output for gf in self.get_all_upstream_vars()):
            raise RuntimeWarning(
                "Multiple outputs detected, only the final output (i.e. the loss) "
                "should be set as output"
            )

    def clear_grad(self) -> None:
        """
        Clear the gradient of the variable (i.e. after a training step)
        """
        self._grad = None

    def clear_downstream_data(self) -> None:
        """
        Clear the downstream data (i.e. after a training step)
        """
        self.downstream_data = []

    def update(self, lr: float, clear_grad: bool, clear_downstream_data: bool) -> None:
        """
        Update the variable in the direction of the negative gradient.

        Parameters
        ----------
        lr : float
            The learning rate (step size)
        clear_grad : bool, optional
            Whether to clear the gradient after updating, defaults to True
        clear_downstream_data : bool, optional
            Whether to clear the downstream data after updating, defaults to True
        """
        if self.is_updatable:
            self.val = self.val - lr * self.grad

        if clear_grad:
            self.clear_grad()

        if clear_downstream_data:
            self.clear_downstream_data()

    def get_all_vars_in_network(self) -> list[GradientVariable]:
        """
        Get all the variables in the network, upstream and downstream of this
        one, including this one.

        Returns
        -------
        list[GradientVariable]
            All the variables in the network
        """
        return self.get_network_output().get_all_upstream_vars()

    def update_full_network(
        self, lr: float, clear_grad=True, clear_downstream_data=True
    ) -> None:
        """
        Update the all vars in the network, upstream and downstream of this one.

        Parameters
        ----------
        lr : float
            The learning rate (step size)
        clear_grad : bool, optional
            Whether to clear the gradient after updating, defaults to True
            You probably want to do this after each training step
        clear_downstream_data : bool, optional
            Whether to clear the downstream data after updating, defaults to True
            You probably want to do this after each training step
        """
        all_vars_in_network = self.get_all_vars_in_network()
        for gvar in all_vars_in_network:
            gvar.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad_full_network(all_vars_in_network=all_vars_in_network)

        if clear_downstream_data:
            self.clear_downstream_data_full_network(
                all_vars_in_network=all_vars_in_network
            )

    def clear_grad_full_network(
        self, all_vars_in_network: list[GradientVariable] | None = None
    ) -> None:
        """
        Clear the gradient of every variable in the network.

        Parameters
        ----------
        all_vars_in_network : list[GradientVariable] | None, optional
            All the variables in the network. Use this if you have already fetched them.
        """
        for gvar in all_vars_in_network or self.get_all_vars_in_network():
            gvar.clear_grad()

    def clear_downstream_data_full_network(
        self, all_vars_in_network: list[GradientVariable] = None
    ) -> None:
        """
        Clear the downstream data of every variable in the network.

        Parameters
        ----------
        all_vars_in_network : list[GradientVariable] | None, optional
            All the variables in the network. Use this if you have already fetched them.
        """
        for gvar in all_vars_in_network or self.get_all_vars_in_network():
            gvar.clear_downstream_data()

    def get_network_output(self) -> GradientVariable:
        """
        Get the output variable of the full network (probably the loss).

        Returns
        -------
        GradientVariable
            The output variable
        """
        if self._is_output:
            return self

        for gvar in self.get_all_downstream_vars(with_counts=False):
            if gvar._is_output:
                return gvar

        raise RuntimeError("No network output found")

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        """
        Set whether or not to track gradients for all variables in the network.

        Parameters
        ----------
        track_gradients : bool
            Whether to track gradients
        """
        for gvar in self.get_all_vars_in_network():
            gvar.track_gradients = track_gradients

    def __float__(self) -> float:
        """
        Get the value of the variable as a float.

        Returns
        -------
        float
            The value of the variable
        """
        return float(self.val)

    def __pos__(self) -> GradientVariable:
        """
        What to return on `+self`

        Returns
        -------
        GradientVariable
            The variable itself
        """
        return self

    def __neg__(self) -> GradientVariable:
        """
        What to return on `-self`

        Returns
        -------
        GradientVariable
            A new variable with this same value but negative
        """
        return self * -1

    def __add__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `self + other`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to add to this one

        Returns
        -------
        GradientVariable
            A new variable with the result of the addition
        """
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x + y,
            func_name="add",
        )

    def __radd__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `other + self`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to add to this one

        Returns
        -------
        GradientVariable
            A new variable with the result of the addition
        """
        return self + other

    def __sub__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `self - other`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to subtract from this one

        Returns
        -------
        GradientVariable
            A new variable with the result of the subtraction
        """
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x - y,
            func_name="sub",
        )

    def __rsub__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `other - self`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to subtract this one from

        Returns
        -------
        GradientVariable
            A new variable with the result of the subtraction
        """
        return -self + other

    def __mul__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `self * other`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to multiply this one by

        Returns
        -------
        GradientVariable
            A new variable with the result of the multiplication
        """
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x * y,
            func_name="mul",
        )

    def __rmul__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `other * self`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to multiply this one by

        Returns
        -------
        GradientVariable
            A new variable with the result of the multiplication
        """
        return self * other

    def __truediv__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `self / other`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to divide this one by

        Returns
        -------
        GradientVariable
            A new variable with the result of the division
        """
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x / y,
            func_name="div",
        )

    def __rtruediv__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `other / self`

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to divide by this one
        """
        return self ** (-1) * other

    def __pow__(self, other: int | float) -> GradientVariable:
        """
        What to run on `self ** other`

        Parameters
        ----------
        other : int | float
            The exponent

        Returns
        -------
        GradientVariable
            A new variable with the result of the exponentiation
        """
        if self.val <= 0 and (float(other) <= 0 or not float(other).is_integer()):
            raise ValueError(
                "If GVar GradientVariable not positive, the exponent must be a positive integer"
            )

        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x**y,
            func_name="pow",
        )

    def __rpow__(self, other: GradientVariable | float) -> GradientVariable:
        """
        What to run on `other ** self`

        Parameters
        ----------
        other : GradientVariable | float
            The base

        Returns
        -------
        GradientVariable
            A new variable with the result of the exponentiation
        """
        if not isinstance(other, GradientVariable):
            other = GradientVariable(other, is_updatable=False, name="other_rpow_base")

        # Just use the __pow__ method from other
        return other**self

    def __repr__(self) -> str:
        """
        Get a string representation of the variable.

        Returns
        -------
        str
            The string representation
        """
        # Truncate the name if it's too long
        name = self.name if len(self.name) < 100 else f"{self.name[:50]}..."
        return f"GVar({self.val}, is_updatable={self.is_updatable}, name={name})"

    @staticmethod
    def _validate_func_name(func: Callable, func_name: str | None) -> None:
        """
        Validate the function naming.

        Parameters
        ----------
        func : Callable
            The function to validate
        func_name : str, optional
            The name of the function
        """
        if func_name is None and func.__name__ == "<lambda>":
            raise ValueError(
                "If func is a lambda function, you must provide a func_name"
            )

    def applyfunc(
        self,
        func: Callable[[float], float],
        func_name: str | None = None,
        grad_func: Callable | None = None,
    ) -> GradientVariable:
        """
        Apply a unary function to the variable.

        Parameters
        ----------
        func : Callable[[float], float]
            The function to apply. Must take a float and return a float.
        func_name : str, optional
            The name of the function that is applied, if not provided gets it is gotten
            automatically. This will cause errors if the different functions of the
            same name are used.
        grad_func : Callable | None, optional
            The function to calculate the gradient of the result with respect to self.
            If not provided, it will be calculated numerically.

        Returns
        -------
        GradientVariable
            A new variable with the result of the function applied to the original
        """
        self._validate_func_name(func, func_name)
        func_name = func_name or func.__name__
        result = GradientVariable(
            val=func(self.val),
            is_updatable=False,
            name=f"{self.name}_applyfunc_{func_name}",
        )
        self.downstream_data.append((result, func_name, func))
        result.upstream_vars.append(self)

        if func_name in self.grad_funcs:
            return result

        if grad_func is None:
            grad_func = (
                lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                    x=self_var, y=downstream_var, func=func
                )
            )
        self.grad_funcs[func_name] = grad_func
        return result

    def applyfunc_two_inputs(
        self,
        other: GradientVariable | float,
        func: Callable[[float, float], float],
        func_name: str | None = None,
        grad_func_left: Callable | None = None,
        grad_func_right: Callable | None = None,
    ) -> GradientVariable:
        """
        Apply a binary function to the variable and another variable.

        Parameters
        ----------
        other : GradientVariable | float
            The other variable to apply the function to
        func : Callable[[float, float], float]
            The function to apply. Must take two floats and return a float.
        func_name : str | None, optional
            The name of the function that is applied, if not provided gets it is gotten
            automatically. This will cause errors if the different functions of the
            same name are used.
        grad_func_left : Callable | None, optional
            The function to calculate the gradient of the result with respect to self.
            If not provided, it will be calculated numerically.
        grad_func_right : Callable | None, optional
            The function to calculate the gradient of the result with respect to the other
            variable. If not provided, it will be calculated numerically.

        Returns
        -------
        GradientVariable
            A new variable with the result of the function applied to the original
        """
        self._validate_func_name(func, func_name)
        func_name = func_name or func.__name__
        func_name_left = f"{func_name}__left"
        func_name_right = f"{func_name}__right"
        result = GradientVariable(
            val=func(float(self), float(other)),
            is_updatable=False,
            name=f"{self.name}_applyfunc_two_inputs_{func_name}",
        )
        self.downstream_data.append((result, func_name_left, other))
        result.upstream_vars.append(self)

        if isinstance(other, GradientVariable):
            other.downstream_data.append((result, func_name_right, self))
            result.upstream_vars.append(other)
            result.name = result.name + f"_{other.name}"
            if func_name_right not in self.grad_funcs:
                if grad_func_right is None:
                    lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                        x=self_var, y=downstream_var, func=lambda x: func(other_var, x)
                    )
                other.grad_funcs[func_name_right] = grad_func_right
        else:
            result.name = result.name + f"_{other}"

        if func_name_left in self.grad_funcs:
            return result

        if grad_func_left is None:
            grad_func_left = (
                lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                    x=self_var, y=downstream_var, func=lambda x: func(x, other_var)
                )
            )
        self.grad_funcs[func_name_left] = grad_func_left
        return result
