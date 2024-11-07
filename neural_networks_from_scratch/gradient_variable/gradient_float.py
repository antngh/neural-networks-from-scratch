from __future__ import annotations


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
        self.val = val
        self.is_updateable = is_updateable
        self.name = name

        self._grad = None

        self.downstream_data = []
        self.upstream_floats = []

    @property
    def downstream_floats(self) -> list[tuple[GFloat, str]]:
        return [gfloat for gfloat, *_ in self.downstream_data]

    @property
    def downstream_relations(self) -> list[tuple[GFloat, str]]:
        return [relation for _, relation, *_ in self.downstream_data]

    @property
    def all_upstream_floats(self) -> list[GFloat]:
        upstream_floats = self.upstream_floats.copy()
        for gfloat in self.upstream_floats:
            upstream_floats += gfloat.all_upstream_floats
        return list(upstream_floats)

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

    def clear_grad(self) -> None:
        self._grad = None
        self.downstream_data = []

    def _grad_fn(
        self, downstream_var: GFloat, relation: str, other: GFloat | float | int
    ) -> float:
        if downstream_var not in self.downstream_floats:
            raise RuntimeError("Variable not downstream")

        return self.grad_funcs[relation](
            float(self), float(downstream_var), float(other)
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

    def update(self, lr: float, clear_grad=True) -> None:
        if not self.is_updateable:
            return

        self.val = self.val - lr * self.grad

        if clear_grad:
            self.clear_grad()

    def _math_method(self, other: GFloat | float, func, name) -> GFloat:
        if self is other:
            raise NotImplementedError

        result = GFloat(func(self.val, float(other)), is_updateable=False)

        if isinstance(other, GFloat):
            name_ = name[1:] if name.startswith("r") else f"r{name}"
            other.downstream_data.append((result, name_, self))
            result.upstream_floats.append(other)

        self.downstream_data.append((result, name, other))
        result.upstream_floats.append(self)
        return result

    def __float__(self) -> float:
        return float(self.val)

    def __add__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x + y, "add")

    def __radd__(self, other: GFloat | float) -> GFloat:
        return self.__add__(other)

    def __sub__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x - y, "sub")

    def __rsub__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: y - x, "rsub")

    def __mul__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x * y, "mul")

    def __rmul__(self, other: GFloat | float) -> GFloat:
        return self.__mul__(other)

    def __truediv__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: x / y, "div")

    def __rtruediv__(self, other: GFloat | float) -> GFloat:
        return self._math_method(other, lambda x, y: y / x, "rdiv")

    def __pow__(self, other: GFloat | float) -> GFloat:
        if isinstance(other, GFloat) or not isinstance(other, (int)):
            raise NotImplementedError
        return self._math_method(other, lambda x, y: x**y, "pow")

    def __rpow__(self, other: GFloat | float) -> GFloat:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"GFloat({self.val}, is_updateable={self.is_updateable}, name={self.name})"
        )
