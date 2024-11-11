from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from .gradient_calculations import GRAD_FUNCS, calculate_grad_numerically


class GFloat:

    def __init__(
        self, val: float, is_updatable: bool = True, name: str | None = None
    ) -> None:
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

        self._downstream_data = []
        self._upstream_floats = []

    @property
    def downstream_data(self) -> list[tuple[GFloat, str, Any]]:
        return self._downstream_data if self.track_gradients else []

    @downstream_data.setter
    def downstream_data(self, value: list[tuple[GFloat, str, Any]]) -> None:
        self._downstream_data = value

    @property
    def upstream_floats(self) -> list[GFloat]:
        return self._upstream_floats if self.track_gradients else []

    @upstream_floats.setter
    def upstream_floats(self, value: list[GFloat]) -> None:
        self._upstream_floats = value

    @property
    def downstream_floats(self) -> list[GFloat]:
        return [gfloat for gfloat, *_ in self.downstream_data]

    @property
    def downstream_relations(self) -> list[tuple[GFloat, str]]:
        return [relation for _, relation, *_ in self.downstream_data]

    def _get_all_upstream_or_downstream_floats(
        self, upstream: bool, with_counts: bool, floats_counts=None
    ) -> set[GFloat] | dict[GFloat, int]:
        from collections import defaultdict

        floats = self.upstream_floats if upstream else self.downstream_floats
        floats_counts = floats_counts or defaultdict(int)
        for child_gfloat in floats:
            floats_counts[child_gfloat] += 1
            child_gfloat._get_all_upstream_or_downstream_floats(
                with_counts=True, upstream=upstream, floats_counts=floats_counts
            )

        return floats_counts if with_counts else set(floats_counts.keys())

    def get_all_upstream_floats(
        self, with_counts=False
    ) -> set[GFloat] | dict[GFloat, int]:
        return self._get_all_upstream_or_downstream_floats(
            upstream=True, with_counts=with_counts
        )

    def get_all_downstream_floats(
        self, with_counts=False
    ) -> set[GFloat] | dict[GFloat, int]:
        return self._get_all_upstream_or_downstream_floats(
            upstream=False, with_counts=with_counts
        )

    @property
    def grad(self) -> float | None:
        if self._grad is None:
            self.grad = self.calculate_grad()
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        if self._grad is not None:
            raise RuntimeError(f"Gradient already set. {self=}")
        self._grad = value

    def set_is_graph_output(self) -> None:
        self.grad = 1.0
        self._is_output = True

        if any(gf._is_output for gf in self.get_all_upstream_floats()):
            raise RuntimeWarning(
                "Multiple outputs detected, only the final output (i.e. the loss) "
                "should be set as output"
            )

    def clear_grad(self) -> None:
        self._grad = None

    def clear_downstream_data(self) -> None:
        self.downstream_data = []

    def _calculate_grad_from_downstream_var(
        self, downstream_var: GFloat, relation: str, other: Any
    ) -> float:
        if downstream_var not in self.downstream_floats:
            raise RuntimeError("Variable not downstream")

        return self.grad_funcs[relation](
            float(self),
            float(downstream_var),
            float(other) if isinstance(other, (GFloat, float, int)) else other,
        )

    def calculate_grad(self) -> float:
        if not self.downstream_data:
            raise RuntimeError("No downstream data")

        grad = 0.0
        for downstream_var, relation, other_var in self.downstream_data:
            grad += downstream_var.grad * self._calculate_grad_from_downstream_var(
                downstream_var, relation, other_var
            )
        return grad

    def update(self, lr: float, clear_grad=True, clear_downstream_data=True) -> None:
        if self.is_updatable:
            self.val = self.val - lr * self.grad

        if clear_grad:
            self.clear_grad()

        if clear_downstream_data:
            self.clear_downstream_data()

    def update_full_network(
        self, lr: float, clear_grad=True, clear_downstream_data=True
    ) -> None:
        if not self._is_output:
            raise RuntimeError("Only the output can be used to update the network")

        for gfloat in self.get_all_upstream_floats():
            gfloat.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad_full_network()

        if clear_downstream_data:
            self.clear_downstream_data_full_network()

    def clear_grad_full_network(self) -> None:
        for gfloat in self.get_all_upstream_floats():
            gfloat.clear_grad()

    def clear_downstream_data_full_network(self) -> None:
        for gfloat in self.get_all_upstream_floats():
            gfloat.clear_downstream_data()

    def get_graph_output(self) -> GFloat:
        if self._is_output:
            return self
        for gfloat in self.get_all_downstream_floats(with_counts=False):
            if gfloat._is_output:
                return gfloat
        raise RuntimeError("No graph output found")

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        for gfloat in self.get_graph_output().get_all_upstream_floats():
            gfloat.track_gradients = track_gradients

    def __float__(self) -> float:
        return float(self.val)

    def __pos__(self) -> GFloat:
        return self

    def __neg__(self) -> GFloat:
        return self * -1

    def __add__(self, other: GFloat | float) -> GFloat:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x + y,
            func_name="add",
        )

    def __radd__(self, other: GFloat | float) -> GFloat:
        return self + other

    def __sub__(self, other: GFloat | float) -> GFloat:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x - y,
            func_name="sub",
        )

    def __rsub__(self, other: GFloat | float) -> GFloat:
        return -self + other

    def __mul__(self, other: GFloat | float) -> GFloat:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x * y,
            func_name="mul",
        )

    def __rmul__(self, other: GFloat | float) -> GFloat:
        return self * other

    def __truediv__(self, other: GFloat | float) -> GFloat:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x / y,
            func_name="div",
        )

    def __rtruediv__(self, other: GFloat | float) -> GFloat:
        return self ** (-1) * other

    def __pow__(self, other: int | float) -> GFloat:
        if self.val <= 0 and (float(other) <= 0 or not float(other).is_integer()):
            raise ValueError(
                "If GFloat is not positive, the exponent must be a positive integer"
            )

        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x**y,
            func_name="pow",
        )

    def __rpow__(self, other: GFloat | float) -> GFloat:
        if not isinstance(other, GFloat):
            other = GFloat(other, is_updatable=False, name="other_rpow_base")
        return other**self

    def __repr__(self) -> str:
        return f"GFloat({self.val}, is_updatable={self.is_updatable}, name={self.name})"

    def applyfunc(
        self, func, func_name: str | None = None, grad_func: Callable | None = None
    ) -> GFloat:
        # note that func must be a function that takes a single float and returns a float,
        # it should not include any other variables
        func_name = func_name or func.__name__
        result = func(self.val)
        result = GFloat(
            val=result,
            is_updatable=False,
            name=f"{self.name}_applyfunc_{func_name}",
        )
        self.downstream_data.append((result, func_name, func))
        result.upstream_floats.append(self)

        if func_name not in self.grad_funcs:
            self.grad_funcs[func_name] = (
                grad_func
                if grad_func is not None
                else (
                    lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                        x=self_var, y=downstream_var, func=func
                    )
                )
            )
        return result

    def applyfunc_two_inputs(
        self,
        other,
        func,
        func_name: str | None = None,
        grad_func_left: Callable | None = None,
        grad_func_right: Callable | None = None,
    ) -> GFloat:
        func_name = func_name or func.__name__
        func_name_left = f"{func_name}__left"
        func_name_right = f"{func_name}__right"
        result = GFloat(
            val=func(float(self), float(other)),
            is_updatable=False,
            name=f"{self.name}_applyfunc_two_inputs_{func_name}",
        )

        if isinstance(other, GFloat):
            other.downstream_data.append((result, func_name_right, self))
            result.upstream_floats.append(other)
            result.name = result.name + f"_{other.name}"
            if func_name_right not in self.grad_funcs:
                grad_func_right = (
                    grad_func_right
                    if grad_func_right is not None
                    else lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                        x=self_var, y=downstream_var, func=lambda x: func(other_var, x)
                    )
                )
                other.grad_funcs[func_name_right] = grad_func_right
        else:
            result.name = result.name + f"_{other}"

        self.downstream_data.append((result, func_name_left, other))
        result.upstream_floats.append(self)

        if func_name_left not in self.grad_funcs:
            grad_func_left = (
                grad_func_left
                if grad_func_left is not None
                else lambda self_var, downstream_var, other_var: calculate_grad_numerically(
                    x=self_var, y=downstream_var, func=lambda x: func(x, other_var)
                )
            )
            self.grad_funcs[func_name_left] = grad_func_left

        return result
