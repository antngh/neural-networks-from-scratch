from __future__ import annotations

from typing import Any, Callable

from ._gradient_calculations import GRAD_FUNCS, calculate_grad_numerically


class GradientVariable:

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
    def downstream_data(self) -> list[tuple[GradientVariable, str, Any]]:
        return self._downstream_data if self.track_gradients else []

    @downstream_data.setter
    def downstream_data(self, value: list[tuple[GradientVariable, str, Any]]) -> None:
        self._downstream_data = value

    @property
    def upstream_vars(self) -> list[GradientVariable]:
        return self._upstream_floats if self.track_gradients else []

    @upstream_vars.setter
    def upstream_vars(self, value: list[GradientVariable]) -> None:
        self._upstream_floats = value

    @property
    def downstream_vars(self) -> list[GradientVariable]:
        return [gvar for gvar, *_ in self.downstream_data]

    @property
    def downstream_relations(self) -> list[tuple[GradientVariable, str]]:
        return [relation for _, relation, *_ in self.downstream_data]

    def _get_all_upstream_or_downstream_vars(
        self, upstream: bool, with_counts: bool, var_counts=None
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        from collections import defaultdict

        vars = self.upstream_vars if upstream else self.downstream_vars
        var_counts = var_counts or defaultdict(int)
        for child_gvar in vars:
            var_counts[child_gvar] += 1
            child_gvar._get_all_upstream_or_downstream_vars(
                with_counts=True, upstream=upstream, var_counts=var_counts
            )

        return var_counts if with_counts else set(var_counts.keys())

    def get_all_upstream_vars(
        self, with_counts=False
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        return self._get_all_upstream_or_downstream_vars(
            upstream=True, with_counts=with_counts
        )

    def get_all_downstream_vars(
        self, with_counts=False
    ) -> set[GradientVariable] | dict[GradientVariable, int]:
        return self._get_all_upstream_or_downstream_vars(
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

        if any(gf._is_output for gf in self.get_all_upstream_vars()):
            raise RuntimeWarning(
                "Multiple outputs detected, only the final output (i.e. the loss) "
                "should be set as output"
            )

    def clear_grad(self) -> None:
        self._grad = None

    def clear_downstream_data(self) -> None:
        self.downstream_data = []

    def _calculate_grad_from_downstream_var(
        self, downstream_var: GradientVariable, relation: str, other: Any
    ) -> float:
        if downstream_var not in self.downstream_vars:
            raise RuntimeError("Variable not downstream")

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

        for gvar in self.get_all_upstream_vars():
            gvar.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad_full_network()

        if clear_downstream_data:
            self.clear_downstream_data_full_network()

    def clear_grad_full_network(self) -> None:
        for gvar in self.get_all_upstream_vars():
            gvar.clear_grad()

    def clear_downstream_data_full_network(self) -> None:
        for gvar in self.get_all_upstream_vars():
            gvar.clear_downstream_data()

    def get_graph_output(self) -> GradientVariable:
        if self._is_output:
            return self

        for gvar in self.get_all_downstream_vars(with_counts=False):
            if gvar._is_output:
                return gvar

        raise RuntimeError("No graph output found")

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        for gvar in self.get_graph_output().get_all_upstream_vars():
            gvar.track_gradients = track_gradients

    def __float__(self) -> float:
        return float(self.val)

    def __pos__(self) -> GradientVariable:
        return self

    def __neg__(self) -> GradientVariable:
        return self * -1

    def __add__(self, other: GradientVariable | float) -> GradientVariable:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x + y,
            func_name="add",
        )

    def __radd__(self, other: GradientVariable | float) -> GradientVariable:
        return self + other

    def __sub__(self, other: GradientVariable | float) -> GradientVariable:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x - y,
            func_name="sub",
        )

    def __rsub__(self, other: GradientVariable | float) -> GradientVariable:
        return -self + other

    def __mul__(self, other: GradientVariable | float) -> GradientVariable:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x * y,
            func_name="mul",
        )

    def __rmul__(self, other: GradientVariable | float) -> GradientVariable:
        return self * other

    def __truediv__(self, other: GradientVariable | float) -> GradientVariable:
        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x / y,
            func_name="div",
        )

    def __rtruediv__(self, other: GradientVariable | float) -> GradientVariable:
        return self ** (-1) * other

    def __pow__(self, other: int | float) -> GradientVariable:
        if self.val <= 0 and (float(other) <= 0 or not float(other).is_integer()):
            raise ValueError(
                "If GVar is not positive, the exponent must be a positive integer"
            )

        return self.applyfunc_two_inputs(
            other=other,
            func=lambda x, y: x**y,
            func_name="pow",
        )

    def __rpow__(self, other: GradientVariable | float) -> GradientVariable:
        if not isinstance(other, GradientVariable):
            other = GradientVariable(other, is_updatable=False, name="other_rpow_base")
        return other**self

    def __repr__(self) -> str:
        name = self.name if len(self.name) < 100 else f"{self.name[:50]}..."
        return f"GVar({self.val}, is_updatable={self.is_updatable}, name={name})"

    def applyfunc(
        self, func, func_name: str | None = None, grad_func: Callable | None = None
    ) -> GradientVariable:
        # note that func must be a function that takes a single float and returns a float,
        # it should not include any other variables
        func_name = func_name or func.__name__
        result = func(self.val)
        result = GradientVariable(
            val=result,
            is_updatable=False,
            name=f"{self.name}_applyfunc_{func_name}",
        )
        self.downstream_data.append((result, func_name, func))
        result.upstream_vars.append(self)

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
    ) -> GradientVariable:
        func_name = func_name or func.__name__
        func_name_left = f"{func_name}__left"
        func_name_right = f"{func_name}__right"
        result = GradientVariable(
            val=func(float(self), float(other)),
            is_updatable=False,
            name=f"{self.name}_applyfunc_two_inputs_{func_name}",
        )

        if isinstance(other, GradientVariable):
            other.downstream_data.append((result, func_name_right, self))
            result.upstream_vars.append(other)
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
        result.upstream_vars.append(self)

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
