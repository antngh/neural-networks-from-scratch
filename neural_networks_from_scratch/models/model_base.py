import random
from abc import ABC, abstractmethod
from typing import Callable

from neural_networks_from_scratch.gradient_variable._gradient_tensor import (
    GradientTensor,
)


class GradientModelBase(ABC):
    """
    A base class for all models that can be trained using gradient descent.

    Attributes
    ----------
    name : str
        The name of the model.
    is_updatable : bool
        Whether the model is updatable or not.
    weight_value_initialiser :float | Callable[[], float]
        The initialiser for the weights of the model.
        Either a float or a function that returns a float.
        By default it will be initialised with a normal distribution).
    """

    def __init__(
        self,
        is_updatable: bool = True,
        name: str | None = None,
        weight_initialiser: float | Callable[[], float] | None = None,
    ):
        """
        Initialises the model.

        Parameters
        ----------
        is_updatable : bool, optional
            Whether the model is updatable or not,
        name : str | None, optional
            The name of the model.
        weight_initialiser : float | Callable[[], float] | None, optional
            The initialiser for the weights of the model.
            Either a float or a function that returns a float.
            By default it will be initialised with a normal distribution).
        """
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
        """
        Create all the variables of the model.
        Initialise all GradientVariables here.
        """

    @abstractmethod
    def forward(self, input_data: list | GradientTensor) -> list:
        """
        Forward pass through the model.

        Use the variables created earlier (do not create them here) to calculate the
        output.

        Parameters
        ----------
        input_data : list | GradientTensor
            The input data to the model.

        Returns
        -------
        list
            The output of the model.
        """
