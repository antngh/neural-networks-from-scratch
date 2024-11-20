import random
from abc import ABC, abstractmethod
from typing import Callable

from neural_networks_from_scratch.gradient_variable import GradientVariable
from neural_networks_from_scratch.gradient_variable._gradient_tensor import (
    GradientTensor,
)


class GModelBase(ABC):
    def __init__(
        self,
        is_updatable: bool = True,
        name: str | None = None,
        weight_initialiser: Callable | None = None,
    ):
        self.name = name
        self.is_updatable = is_updatable

        self.weight_value_initialiser = (
            weight_initialiser
            if weight_initialiser is not None
            else lambda: random.normalvariate(0, 1)
        )

        self._tracking_gradients = None

        self._create_variables()

    @abstractmethod
    def _create_variables(self):
        pass

    @abstractmethod
    def forward(self, input_data: list | GradientTensor | GradientVariable) -> list:
        return self._forward_calculation(input_data)
