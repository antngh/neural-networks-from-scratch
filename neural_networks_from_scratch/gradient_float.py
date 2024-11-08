from __future__ import annotations

from typing import Any, Callable


class GFloat:
    grad_funcs = {
        "add": lambda self_var, downstream_var, other_var: 1.0,
        "radd": lambda self_var, downstream_var, other_var: 1.0,
        "mul": lambda self_var, downstream_var, other_var: other_var,
        "rmul": lambda self_var, downstream_var, other_var: other_var,
        "sub": lambda self_var, downstream_var, other_var: 1.0,
        "rsub": lambda self_var, downstream_var, other_var: 1.0,
        "div": lambda self_var, downstream_var, other_var: 1.0 / other_var,
        "rdiv": lambda self_var, downstream_var, other_var: -other_var / self_var**2,
        "pow": lambda self_var, downstream_var, other_var: other_var
        * self_var ** (other_var - 1),
    }

    def __init__(
        self, val: float, is_updateable: bool = True, name: str | None = None
    ) -> None:
        if val != val:
            raise ValueError("NaNs aren't valid")

        if not isinstance(val, (float, int)):
            raise ValueError(f"Expected float or int, got {type(val)}")
        self.val = float(val)
        self.is_updateable = is_updateable
        self.name = name
        self.track_gradients = True

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
    def downstream_floats(self) -> list[tuple[GFloat, str]]:
        return [gfloat for gfloat, *_ in self.downstream_data]

    @property
    def downstream_relations(self) -> list[tuple[GFloat, str]]:
        return [relation for _, relation, *_ in self.downstream_data]

    @property
    def all_upstream_floats(self) -> set[GFloat]:
        upstream_floats = set(self.upstream_floats)
        upstream_floats_copy = upstream_floats.copy()
        for gfloat in upstream_floats:
            upstream_floats_copy |= gfloat.all_upstream_floats
        return upstream_floats_copy

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

    def set_is_output(self) -> None:
        self.grad = 1.0
        self._is_output = True

        if any(gf._is_output for gf in self.all_upstream_floats):
            raise RuntimeWarning(
                "Multiple outputs detected, only the final output (i.e. the loss) "
                "should be set as output"
            )

    def clear_grad(self) -> None:
        self._grad = None

    def clear_downstream_data(self) -> None:
        self.downstream_data = []

    def _grad_fn(self, downstream_var: GFloat, relation: str, other: Any) -> float:
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
            grad += downstream_var.grad * self._grad_fn(
                downstream_var, relation, other_var
            )
        return grad

    def update(self, lr: float, clear_grad=True, clear_downstream_data=True) -> None:
        if self.is_updateable:
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

        for gfloat in self.all_upstream_floats:
            gfloat.update(lr=lr, clear_grad=False, clear_downstream_data=False)

        if clear_grad:
            self.clear_grad_full_network()

        if clear_downstream_data:
            self.clear_downstream_data_full_network()

    def clear_grad_full_network(self) -> None:
        for gfloat in self.all_upstream_floats:
            gfloat.clear_grad()

    def clear_downstream_data_full_network(self) -> None:
        for gfloat in self.all_upstream_floats:
            gfloat.clear_downstream_data()

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        if not self._is_output:
            raise RuntimeError("Only the output can be used to update the network")

        for gfloat in self.all_upstream_floats:
            gfloat.track_gradients = track_gradients

    def _math_method(self, other: GFloat | float, func, method_name) -> GFloat:
        if self is other:
            raise NotImplementedError

        result = GFloat(
            val=func(self.val, float(other)),
            is_updateable=False,
            name=f"{self.name}_{method_name}",
        )

        if isinstance(other, GFloat):
            name_ = (
                method_name[1:] if method_name.startswith("r") else f"r{method_name}"
            )
            other.downstream_data.append((result, name_, self))
            result.upstream_floats.append(other)
            result.name = result.name + f"_{other.name}"
        else:
            result.name = result.name + f"_{other}"

        self.downstream_data.append((result, method_name, other))
        result.upstream_floats.append(self)
        return result

    def __float__(self) -> float:
        return float(self.val)

    def __add__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x + y, "add")

    def __radd__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x + y, "radd")

    def __sub__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x - y, "sub")

    def __rsub__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: y - x, "rsub")

    def __mul__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x * y, "mul")

    def __rmul__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x * y, "rmul")

    def __truediv__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x / y, "div")

    def __rtruediv__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: y / x, "rdiv")

    def __pow__(self, other: int | float) -> GFloat:
        if not isinstance(other, int) and not other.is_integer():
            raise NotImplementedError(f"{self=}, {other=}")
        return self._math_method(other, lambda x, y: x**y, "pow")

    def __rpow__(self, other: GFloat | float) -> GFloat:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"GFloat({self.val}, is_updateable={self.is_updateable}, name={self.name})"
        )

    def applyfunc(
        self, func, func_name: str | None = None, grad_func: Callable | None = None
    ) -> GFloat:
        func_name = func_name or func.__name__
        result = GFloat(
            val=func(self.val),
            is_updateable=False,
            name=f"{self.name}_applyfunc_{func_name}",
        )
        self.downstream_data.append((result, func_name, func))
        result.upstream_floats.append(self)

        if grad_func is not None:
            self.grad_funcs[func_name] = grad_func
        return result
        return result
