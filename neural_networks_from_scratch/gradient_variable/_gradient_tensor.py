from __future__ import annotations

import pprint
from typing import Any, Callable

from ._gradient_variable import GradientVariable

NUMERIC_TYPES = (float, int, GradientVariable)
NUMERIC_TYPES_TYPING = float | int | GradientVariable


def _is_scalar(value: Any) -> bool:
    """
    Check if a value is a scalar

    Parameters
    ----------
    value : Any
        The value to check

    Returns
    -------
    bool
        Whether the value is a scalar
    """
    return isinstance(value, NUMERIC_TYPES)


def _get_dims_from_nested_list(values: list) -> tuple[int, ...]:
    """
    Get the dimensions of a tensor represented as a nested list

    Note: This function does some checks but doesn't check all entries
    i.e. it assumes that the tensor is well formed

    Parameters
    ----------
    values : list
        A nested list representing a tensor

    Returns
    -------
    tuple[int, ...]
        The dimensions of the tensor
    """
    if not values:
        raise NotImplementedError("Zero length dimensions not handled")

    types = {type(entry) for entry in values}
    for type_ in types:
        if type_ not in (list,) + NUMERIC_TYPES:
            raise TypeError("Invalid entry type, only numeric or list allowed")

    if list not in types:
        return (len(values),)

    if len(types) > 1:
        raise TypeError("Cannot mix lists and numbers")

    lens = [len(entry) for entry in values]
    if len(set(lens)) > 1:
        raise ValueError("All rows must have the same length")

    # We only take the first row, meaning we don't check all rows
    return (len(values),) + _get_dims_from_nested_list(values[0])


def _create_tensor_from_dims(
    dims: tuple[int, ...], value: float | Callable[[], float] = 0.0
) -> list:
    """
    Initialise a tensor with the given dimensions

    Parameters
    ----------
    dims : tuple[int, ...]
        The dimensions of the tensor
    value : float | Callable, optional
        The initial value of the tensor, by default 0.0
        Either a float or a callable of no args that returns a float

    Returns
    -------
    list
        The initialised tensor
    """
    if len(dims) == 0:
        raise NotImplementedError("Zero length dimensions not handled")
    if len(dims) == 1:
        return [value() if callable(value) else value for _ in range(dims[0])]
    return [_create_tensor_from_dims(dims[1:], value=value) for _ in range(dims[0])]


def _get_all_elements(values: list) -> set[NUMERIC_TYPES_TYPING]:
    """
    Unpack the tensor into a set of all elements

    Parameters
    ----------
    values : list
        The tensor to unpack

    Returns
    -------
    set[NUMERIC_TYPES_TYPING]
        A set of all elements in the tensor
    """
    if _is_scalar(values):
        return values

    chained_vals = set()
    for list_ in values:
        res = _get_all_elements(list_)
        if isinstance(res, set):
            chained_vals |= res
        else:
            chained_vals.add(res)

    return chained_vals


def _is_tensor_of_gvars(values, check_all=True):
    """
    Check if all/any elements of the tensor are GradientVariables

    Parameters
    ----------
    values : list
        The tensor to check
    check_all : bool, optional
        Whether to check all elements or any
    """
    check = all if check_all else any
    return check(isinstance(el, GradientVariable) for el in _get_all_elements(values))


def _numeric_tensor_to_tensor_of_gvars(
    values: list | float | int,
    name: str | None = None,
    level="i",
    **kwargs,
) -> GradientVariable:
    """
    Convert a tensor of numbers to a tensor of GradientVariables

    Parameters
    ----------
    values : list | float | int
        The tensor to convert
    name : str | None
        An optional label for the tensor
    level : str
        The outer dimension (i.e. i,j,k ...), used for naming
    kwargs : dict
        kwargs to pass to GradientVariable
    """
    name = name or ""
    if isinstance(values, (float, int)):
        return GradientVariable(
            float(values),
            name=name,
            **kwargs,
        )

    return [
        _numeric_tensor_to_tensor_of_gvars(
            list_,
            name=name + f"{level}{index_}",
            level=chr(ord(level) + 1),
            **kwargs,
        )
        for index_, list_ in enumerate(values)
    ]


def _check_dimensions_align(values1: list, values2: list) -> bool:
    """
    Check if two tensors have the same shape

    Parameters
    ----------
    values1 : list
        The first tensor
    values2 : list
        The second tensor

    Returns
    -------
    bool
        Whether the tensors have the same shape
    """
    v1_is_num = _is_scalar(values1)
    v2_is_num = _is_scalar(values2)
    if v1_is_num != v2_is_num:
        return False

    if v1_is_num:  # both are scalars
        return True

    if len(values1) != len(values2):
        return False

    # if both are lists and align
    return all(_check_dimensions_align(v1, v2) for v1, v2 in zip(values1, values2))


def _elementwise_function_tensor(
    values: list | NUMERIC_TYPES_TYPING,
    func: Callable[[float], float],
    **applyfunc_kwargs,
) -> list | NUMERIC_TYPES_TYPING:
    """
    Apply a unary method to all elements in a tensor

    Parameters
    ----------
    values : list | NUMERIC_TYPES_TYPING
        The tensor to apply the method to
    func : Callable[[float], float]
        The function to apply
    applyfunc_kwargs : dict
        kwargs to pass to GradientVariable.applyfunc

    Returns
    -------
    list | NUMERIC_TYPES_TYPING
        The tensor with each element being the result of the function applied to the
        same element in the input tensor
    """
    if isinstance(values, GradientVariable):
        return values.applyfunc(func, **applyfunc_kwargs)
    elif isinstance(values, (float, int)):
        return func(values)

    return [_elementwise_function_tensor(v, func, **applyfunc_kwargs) for v in values]


def _elementwise_method_two_tensors(
    values1: list | NUMERIC_TYPES_TYPING,
    values2: list | NUMERIC_TYPES_TYPING,
    func: Callable[[float, float], float],
):
    """
    Apply a function of two arguments element-wise across two tensors

    Parameters
    ----------
    values1 : list | NUMERIC_TYPES_TYPING
        The first tensor
    values2 : list | NUMERIC_TYPES_TYPING
        The second tensor
    func : Callable[[float, float], float]
        The function to apply

    Returns
    -------
    list | NUMERIC_TYPES_TYPING
        The tensor with each element being the result of the function applied to the
        same elements in the input tensors
    """
    assert _check_dimensions_align(
        values1, values2
    ), f"Dimensions must match {values1=} {values2=}"

    if _is_scalar(values1) and _is_scalar(values2):
        return func(values1, values2)

    return [
        _elementwise_method_two_tensors(v1, v2, func)
        for v1, v2 in zip(values1, values2)
    ]


class GradientTensor:
    """
    A tensor of GradientVariables

    Attributes
    ----------
    values : list
        The tensor (defined as nested lists) of GradientVariables
    dims : tuple[int, ...]
        The dimensions of the tensor
    is_updatable : bool
        Whether the variables in the tensor can be updated, i.e are they variable
        of constant
    name : str
        The name of the tensor
    """

    def __init__(
        self,
        values: list[float, GradientVariable | list] | None = None,
        dims: tuple[int, ...] | None = None,
        initial_value: float | Callable[[], None] | None = None,
        is_updatable: bool | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialise the values of the tensor

        Parameters
        ----------
        values : list[float, GradientVariable | list] | None, optional
            The values of the elements tensor.
            Provide either this or dims.
        dims : tuple[int, ...] | None, optional
            The dimensions of the tensor.
            Provide either this or values.
        initial_value : float | Callable[[], None] | None, optional
            The initial value (or function to generate initial value) of the elements.
            Only provide this is initialising with dims.
        is_updatable : bool | None, optional
            Whether the variables in the tensor can be updated, i.e are they variable
            of constant
        name : str | None, optional
            The name of the tensor
        """
        self.name = name

        if not values and not dims:
            raise TypeError("Must specify either dims or values")

        if dims:
            if values:
                raise TypeError("Cannot specify both dims and values")

            self.dims = dims
            values = _create_tensor_from_dims(
                dims, value=0.0 if initial_value is None else initial_value
            )
            self.is_updatable = True if is_updatable is None else is_updatable
            self.values = _numeric_tensor_to_tensor_of_gvars(
                values,
                is_updatable=self.is_updatable,
                name=name,
            )
            return

        if initial_value:
            raise TypeError("Only specify initial_value if dims are provided")

        self.dims = _get_dims_from_nested_list(values)

        all_are_gvars = _is_tensor_of_gvars(values, check_all=True)
        if not all_are_gvars and _is_tensor_of_gvars(values, check_all=False):
            raise ValueError(
                "Cannot have a mix of GradientVariables and floats in the tensor"
            )
        if all_are_gvars:
            if is_updatable is not None:
                raise ValueError(
                    "if values are GradientVariables, is_updatable must not be provided"
                )
            self.values = values
            return

        self.is_updatable = True if is_updatable is None else is_updatable
        self.values = _numeric_tensor_to_tensor_of_gvars(
            values,
            is_updatable=self.is_updatable,
            name=name,
        )

    def transpose(self) -> GradientTensor:
        """
        Get a new GradientTensor that is the transpose of the current tensor

        Returns
        -------
        GradientTensor
            The transpose of the current
        """
        if len(self.dims) != 2:
            raise ValueError("Transpose only supported for 2D tensors")

        new_values = []
        for j in range(self.dims[1]):
            row = []
            for i in range(self.dims[0]):
                row.append(self.values[i][j])
            new_values.append(row)

        return GradientTensor(
            values=new_values,
            name=f"{self.name}_transpose",
        )

    def _clean_other_tensor(
        self, other: list | GradientTensor | NUMERIC_TYPES_TYPING, method_name: str
    ) -> GradientTensor:
        """
        Clean the other tensor to ensure it is a GradientTensor of the right format

        Parameters
        ----------
        other : list | NUMERIC_TYPES_TYPING
            The other tensor to clean
        method_name : str
            The name of the method calling this function

        Returns
        -------
        GradientTensor
            The cleaned other tensor
        """
        return (
            GradientTensor(
                values=other,
                is_updatable=False,
                name=f"{self.name}_{method_name}_other",
            )
            if isinstance(other, list)
            else (
                GradientVariable(
                    val=other,
                    is_updatable=False,
                    name=f"{self.name}_{method_name}_other",
                )
                if isinstance(other, (float, int))
                else other  # GradientTensor
            )
        )

    def applyfunc(
        self,
        func: Callable[[float], float],
        func_name: str | None = None,
        **gvar_applyfunc_kwargs,
    ) -> GradientTensor:
        """
        Apply a function element-wise to the tensor

        Parameters
        ----------
        func : Callable[[float], float]
            The function to apply
        func_name : str | None, optional
            The name of the function
        gvar_applyfunc_kwargs : dict
            kwargs to pass to GradientVariable.applyfunc

        Returns
        -------
        GradientTensor
            Each element is the output of the function applied to the corresponding
            element in self.
        """
        func_name = func_name or func.__name__
        return GradientTensor(
            values=_elementwise_function_tensor(
                self.values, func, func_name=func_name, **gvar_applyfunc_kwargs
            ),
            name=f"{self.name}_applyfunc_{func_name}",
        )

    def applyfunc_two_inputs(
        self,
        other: GradientTensor | float,
        func: Callable[[float, float], float],
        method_name: str | None = None,
    ) -> GradientTensor:
        """
        Apply a two-argument function element-wise self and another tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor, elements will be the second arg of func
            If a float then broadcast to all elements
        func : Callable[[float, float], float]
            The function to apply
        method_name : str | None, optional
            The name of the method

        Returns
        -------
        GradientTensor
            Each element is the output of the function applied to the corresponding
            elements in self and other.
        """
        other = self._clean_other_tensor(other, method_name)

        other_dims = other.dims if isinstance(other, GradientTensor) else ()
        other_values = other.values if isinstance(other, GradientTensor) else other

        if self.dims != other_dims and other_dims != ():
            raise ValueError("Dimensions must match")

        return GradientTensor(
            values=_elementwise_method_two_tensors(
                self.values, other_values, lambda x, y: func(x, y)
            ),
            name=f"{self.name}_{method_name}_{other.name}",
        )

    def __pos__(self) -> GradientTensor:
        """
        What to return on `+self`

        Returns
        -------
        GradientTensor
            The tensor itself
        """
        return self

    def __neg__(self) -> GradientTensor:
        """
        What to return on `-self`

        Returns
        -------
        GradientTensor
            A new tensor with all elements with this same value but negative
        """
        return self * -1

    def __add__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Add each element of the tensor to the corresponding element of another tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The sum of the two tensors element-wise
        """
        return self.applyfunc_two_inputs(other, lambda x, y: x + y, method_name="add")

    def __radd__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Add each element of the tensor to the corresponding element of another tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The sum of the two tensors element-wise
        """
        return self + other

    def __sub__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Subtract each element of the other tensor from the corresponding element of this tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The difference of the two tensors element-wise
        """
        return self.applyfunc_two_inputs(other, lambda x, y: x - y, method_name="sub")

    def __rsub__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Subtract each element of the this tensor from the corresponding element of the other tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The difference of the two tensors element-wise
        """
        return -self + other

    def __mul__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Multiply each element of the tensor by the corresponding element of another tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The product of the two tensors element-wise
        """
        return self.applyfunc_two_inputs(other, lambda x, y: x * y, method_name="elmul")

    def __rmul__(self, other: GradientTensor | float) -> GradientTensor:
        """
        Multiply each element of the tensor by the corresponding element of another tensor

        Parameters
        ----------
        other : GradientTensor | float
            The other tensor.
            If a float then broadcast to all elements

        Returns
        -------
        GradientTensor
            The product of the two tensors element-wise
        """
        return self * other

    def matmul(self, other: GradientTensor) -> GradientTensor:
        """
        Matrix multiplication of this tensor with another tensor

        Both this tensor and the other must be 2D.

        Parameters
        ----------
        other : GradientTensor
            The other tensor to multiply with

        Returns
        -------
        GradientTensor
            The result of the matrix multiplication
        """
        other = self._clean_other_tensor(other, method_name="matmul")

        if not isinstance(other, GradientTensor):
            raise ValueError("matmul only supported for tensors")

        if len(self.dims) != 2 or len(other.dims) != 2:
            raise ValueError("matmul only supported for 2D tensors")

        if self.dims[1] != other.dims[0] or self.dims[0] != other.dims[1]:
            raise ValueError("Dimensions must match for matmul")

        new_values = []
        for i in range(self.dims[0]):
            row = []
            for j in range(other.dims[1]):
                row.append(
                    sum(
                        self.values[i][k] * other.values[k][j]
                        for k in range(self.dims[1])
                    )
                )
            new_values.append(row)

        return GradientTensor(
            values=new_values,
            name=f"{self.name}_matmul_{other.name}",
        )

    def vecmul(self, other: GradientTensor | list) -> GradientTensor:
        """
        Apply this tensor to a vector (1D tensor).

        The tensor must be 2D.

        Parameters
        ----------
        other : GradientTensor | list
            The other tensor to multiply with, must be 1d (a vector)

        Returns
        -------
        GradientTensor
            The result of the vector multiplication, a 1D tensor (vector)
        """
        other = self._clean_other_tensor(other, method_name="vecmul")

        if not isinstance(other, GradientTensor):
            raise ValueError("matmul only supported for tensors")

        if len(self.dims) != 2 or len(other.dims) != 1:
            raise ValueError("vecmul only supported for 2D on to 1D tensors")

        if self.dims[1] != other.dims[0]:
            raise ValueError("Dimensions must match for vecmul")

        return GradientTensor(
            values=[
                sum(
                    self_val * other_val
                    for self_val, other_val in zip(self_row, other.values)
                )
                for self_row in self.values
            ],
            name=f"{self.name}_vecmul_{other.name}",
        )

    def __matmul__(self, other: GradientTensor) -> GradientTensor:
        """
        Matrix multiply. Allows us to use self @ other.

        See matmul or vecmul for implementation details

        self must be 2D, other must be 1D or 2D

        Parameters
        ----------
        other : GradientTensor
            The other tensor to multiply with.
            Must be either 1D or 2D

        Returns
        -------
        GradientTensor
            The result of the matrix or vector multiplication
        """
        other = self._clean_other_tensor(other, method_name="__matmul__")
        err_msg = (
            "Only matmul with two 2d tensors or vecmul with one 2d and one "
            "1d tensor is supported"
        )
        if not len(self.dims) == 2:
            raise NotImplementedError(err_msg)

        if len(other.dims) == 1:
            return self.vecmul(other)
        elif len(other.dims) == 2:
            return self.matmul(other)

        raise NotImplementedError(err_msg)

    def __rmatmul__(self, other: GradientTensor) -> GradientTensor:
        """
        Matrix multiply. Allows us to use other @ self.

        See matmul or vecmul for implementation details

        other must be 2D, self must be 1D or 2D

        Parameters
        ----------
        other : GradientTensor
            The other tensor to multiply with.
            Must be 2D

        Returns
        -------
        GradientTensor
            The result of the matrix or vector multiplication
        """
        return self._clean_other_tensor(other, method_name="__rmatmul__") @ self

    def __repr__(self) -> str:
        """
        Get a string representation of the tensor

        Returns
        -------
        str
            The string representation of the tensor
        """
        vals_str = pprint.pformat(self.values)
        return f"GTensor(name={self.name}, dims={self.dims},\nvalues=\n{vals_str}\n)"

    def __len__(self) -> int:
        """
        Get the length of outer dimension of the tensor

        Returns
        -------
        int
            The length of the tensor
        """
        return len(self.values)

    def update(
        self, lr: float, clear_grad: bool = True, clear_downstream_data: bool = True
    ) -> None:
        """
        Update all the variables in the tensor using the given learning rate and their
        gradients

        Parameters
        ----------
        lr : float
            The learning rate (step size)
        clear_grad : bool, optional
            Whether to clear the gradients after updating
        clear_downstream_data : bool, optional
            Whether to clear the downstream data after updating
        """
        if not self.is_updatable:
            return

        for el in _get_all_elements(self.values):
            el.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad()

        if clear_downstream_data:
            self.clear_downstream_data()

    def clear_grad(self) -> None:
        """
        Clear the gradients of all the variables in the tensor
        """
        for el in _get_all_elements(self.values):
            el.clear_grad()

    def clear_downstream_data(self) -> None:
        """
        Clear the downstream data of all the variables in the tensor
        """
        for el in _get_all_elements(self.values):
            el.clear_downstream_data()

    def to_gvariable(self) -> GradientVariable:
        """
        Convert a 1D tensor with 1 entry to a GradientVariable

        Returns
        -------
        GradientVariable
            The value of the single entry in the tensor
        """
        if self.dims != (1,):
            raise ValueError("Only scalar tensors can be converted to GVar")

        return self.values[0]

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        """
        Set whether to track gradients for the full network, defined by all
        GradientVariables connected together, regardless of whether they are held in
        this tensor or not

        Parameters
        ----------
        track_gradients : bool
            Whether to track gradients for the full network
        """
        _get_all_elements(self.values).pop().set_track_gradients_full_network(
            track_gradients
        )
