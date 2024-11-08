from __future__ import annotations

import pprint
from typing import Callable

from gradient_float import GFloat


def _get_dims_from_nested_list(values: list) -> tuple[int, ...]:
    is_numeric = None
    list_lengths = set()
    for entry in values:
        if isinstance(entry, (float, int, GFloat)):
            if is_numeric is None:
                is_numeric = True
            elif not is_numeric:
                raise TypeError("Cannot mix numbers and lists")
        elif isinstance(entry, list):
            if is_numeric is None:
                is_numeric = False
            elif is_numeric:
                raise TypeError("Cannot mix numbers and lists")
            list_lengths.add(len(entry))
        else:
            raise TypeError("Invalid entry type, only numeric or list allowed")

    if is_numeric:
        return (len(values),)
    if list_lengths:
        if len(list_lengths) != 1:
            raise ValueError("All rows must have the same length")
        return (len(values),) + _get_dims_from_nested_list(values[0])


def _create_tensor_from_dims(
    dims: tuple[int, ...], value: float | Callable = 0.0
) -> list:
    if len(dims) == 0:
        return []
    if len(dims) == 1:
        return [value() if callable(value) else value for _ in range(dims[0])]
    return [_create_tensor_from_dims(dims[1:], value=value) for _ in range(dims[0])]


def _get_all_elements(values):
    if isinstance(values, (float, int, GFloat)):
        return values

    chained_vals = set()
    for list_ in values:
        res = _get_all_elements(list_)
        if isinstance(res, set):
            chained_vals |= res
        else:
            chained_vals.add(res)

    return chained_vals


def _is_tensor_of_gfloats(values):
    return all(isinstance(el, GFloat) for el in _get_all_elements(values))


def _float_tensor_to_tensor_of_gfloats(
    values: list | float | int,
    name=None,
    level="i",
    # index=0,
    **kwargs,
) -> GFloat:
    name = name or ""
    if isinstance(values, (float, int)):
        return GFloat(
            float(values),
            name=name,
            **kwargs,
        )

    return [
        _float_tensor_to_tensor_of_gfloats(
            list_,
            name=name + f"{level}{index_}",
            level=chr(ord(level) + 1),
            # index=index_,
            **kwargs,
        )
        for index_, list_ in enumerate(values)
    ]


def _is_scalar(values):
    if isinstance(values, (float, int, GFloat)):
        return True
    return False


def _check_dimensions_align(values1, values2):
    v1_is_num = _is_scalar(values1)
    v2_is_num = _is_scalar(values2)
    if v1_is_num != v2_is_num:
        return False

    if v1_is_num and v2_is_num:
        return True

    if not isinstance(values1, list):
        return False

    if len(values1) != len(values2):
        return False

    return all(_check_dimensions_align(v1, v2) for v1, v2 in zip(values1, values2))


def _elementwise_method_tensors(values, func, **kwargs):
    if isinstance(values, (float, int)):
        return func(values, **kwargs)
    elif isinstance(values, GFloat):
        return values.applyfunc(func, **kwargs)

    return [_elementwise_method_tensors(v, func, **kwargs) for v in values]


def _elementwise_method_two_tensors(values1, values2, func):
    assert _check_dimensions_align(
        values1, values2
    ), f"Dimensions must match {values1=} {values2=}"

    if isinstance(values1, (float, int, GFloat)) and isinstance(
        values2, (float, int, GFloat)
    ):
        return func(values1, values2)

    return [
        _elementwise_method_two_tensors(v1, v2, func)
        for v1, v2 in zip(values1, values2)
    ]


class GTensor:

    def __init__(
        self,
        values: list[float, GFloat | list] | None = None,
        dims: tuple[int, ...] | None = None,
        initial_value: float | Callable | None = None,
        is_updateable: bool | None = None,
        name: str | None = None,
    ) -> None:
        self.name = name
        if dims and values:
            raise TypeError("Cannot specify both dims and values")
        if not dims and not values:
            raise TypeError("Must specify either dims or values")

        if values:
            if dims:
                raise TypeError("Cannot specify both dims and values")

            if initial_value:
                raise TypeError("Cannot specify both initial_value and values")

            self.dims = _get_dims_from_nested_list(values)

        if dims:
            self.dims = dims
            values = _create_tensor_from_dims(
                dims, value=0.0 if initial_value is None else initial_value
            )

        if _is_tensor_of_gfloats(values):
            if is_updateable:
                raise ValueError(
                    "if values are GFloats, is_updateable must not be provided"
                )
            self.values = values
            return

        self.is_updateable = True if is_updateable is None else is_updateable
        self.values = _float_tensor_to_tensor_of_gfloats(
            values,
            is_updateable=self.is_updateable,
            name=name,
        )

    def transpose(self) -> GTensor:
        if len(self.dims) != 2:
            raise ValueError("Transpose only supported for 2D tensors")

        new_values = []
        for j in range(self.dims[1]):
            row = []
            for i in range(self.dims[0]):
                row.append(self.values[i][j])
            new_values.append(row)

        return GTensor(
            values=new_values,
            name=f"{self.name}_transpose",
        )

    def _clean_other_tensor(self, other: GTensor | float, method_name) -> GTensor:
        return (
            GTensor(
                values=other,
                is_updateable=False,
                name=f"{self.name}_{method_name}_other",
            )
            if isinstance(other, list)
            else (
                GFloat(
                    val=other,
                    is_updateable=False,
                    name=f"{self.name}_{method_name}_other",
                )
                if isinstance(other, float)
                else other
            )
        )

    def applyfunc(
        self, func: Callable, func_name: str | None = None, **kwargs
    ) -> GTensor:
        func_name = func_name or func.__name__
        return GTensor(
            values=_elementwise_method_tensors(
                self.values, func, func_name=func_name, **kwargs
            ),
            name=f"{self.name}_applyfunc_{func_name}",
        )

    def _elementwise_method(
        self, other: GTensor | float, func, method_name=None
    ) -> GTensor:
        other = self._clean_other_tensor(other, method_name)

        other_dims = other.dims if isinstance(other, GTensor) else ()
        other_values = other.values if isinstance(other, GTensor) else other

        if self.dims != other_dims and other_dims != ():
            raise ValueError("Dimensions must match")

        return GTensor(
            values=_elementwise_method_two_tensors(
                self.values, other_values, lambda x, y: func(x, y)
            ),
            name=f"{self.name}_{method_name}_{other.name}",
        )

    def __add__(self, other: GTensor | float) -> GTensor:
        return self._elementwise_method(other, lambda x, y: x + y, method_name="add")

    def __radd__(self, other: GTensor | float) -> GTensor:
        return self._elementwise_method(other, lambda x, y: x + y, method_name="radd")

    def __sub__(self, other: GTensor | float) -> GTensor:
        return self._elementwise_method(other, lambda x, y: x - y, method_name="sub")

    def __rsub__(self, other: GTensor | float) -> GTensor:
        return self._elementwise_method(other, lambda x, y: y - x, method_name="rsub")

    def __mul__(self, other: GTensor | float) -> GTensor:
        if not isinstance(other, float):
            raise ValueError("* only used for elementwise multiplication")

        return self._elementwise_method(other, lambda x, y: x * y, method_name="elmul")

    def __rmul__(self, other: GTensor | float) -> GTensor:
        if not isinstance(other, float):
            raise ValueError("* only used for elementwise multiplication")

        return self._elementwise_method(other, lambda x, y: x * y, method_name="relmul")

    def matmul(self, other: GTensor) -> GTensor:
        other = self._clean_other_tensor(other, method_name="matmul")

        if not isinstance(other, GTensor):
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

        return GTensor(
            values=new_values,
            name=f"{self.name}_matmul_{other.name}",
        )

    def vecmul(self, other: GTensor | list) -> GTensor:
        other = self._clean_other_tensor(other, method_name="vecmul")

        if not isinstance(other, GTensor):
            raise ValueError("matmul only supported for tensors")

        if len(self.dims) != 2 or len(other.dims) != 1:
            raise ValueError("vecmul only supported for 2D on to 1D tensors")

        if self.dims[1] != other.dims[0]:
            raise ValueError("Dimensions must match for vecmul")

        return GTensor(
            values=[
                sum(
                    self_val * other_val
                    for self_val, other_val in zip(self_row, other.values)
                )
                for self_row in self.values
            ],
            name=f"{self.name}_vecmul_{other.name}",
        )

    def __repr__(self) -> str:
        vals_str = pprint.pformat(self.values)
        return f"GTensor(name={self.name}, dims={self.dims},\nvalues=\n{vals_str}\n)"

    def update(self, lr: float, clear_grad=True, clear_downstream_data=True) -> None:
        if not self.is_updateable:
            return

        for el in _get_all_elements(self.values):
            el.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad()

        if clear_downstream_data:
            self.clear_downstream_data()

    def clear_grad(self) -> None:
        for el in _get_all_elements(self.values):
            el.clear_grad()

    def clear_downstream_data(self) -> None:
        for el in _get_all_elements(self.values):
            el.clear_downstream_data()

    def to_gfloat(self) -> GTensor:
        if self.dims != (1,):
            raise ValueError("Only scalar tensors can be converted to GFloat")

        return self.values[0]
