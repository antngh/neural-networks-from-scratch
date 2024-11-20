from typing import Callable

from neural_networks_from_scratch.gradient_variable import GradientTensor
from neural_networks_from_scratch.models.model_base import GModelBase

from .activation_functions import ActivationBase, Relu


class MLP(GModelBase):
    def __init__(
        self,
        n_inputs: int,
        layers: tuple[int, ...],
        is_updatable: bool = True,
        name: str | None = None,
        weight_initialiser: Callable | None = None,
        internal_activation_function: ActivationBase | None = Relu(),
        final_activation_function: ActivationBase | None = None,
    ):
        self.n_inputs = n_inputs
        self.layers = layers

        self.activation_function = internal_activation_function
        self.final_activation_function = final_activation_function

        super().__init__(
            is_updatable=is_updatable, name=name, weight_initialiser=weight_initialiser
        )

    def _create_variables(self):
        self.weights = []
        self.biases = []
        n_nodes_previous_layer = self.n_inputs
        for i, n_nodes_layer in enumerate(self.layers):
            self.weights.append(
                GradientTensor(
                    dims=(n_nodes_layer, n_nodes_previous_layer),
                    initial_value=self.weight_value_initialiser,
                    name=f"{self.name}_W{i}",
                    is_updatable=self.is_updatable,
                )
            )
            self.biases.append(
                GradientTensor(
                    dims=(n_nodes_layer,),
                    initial_value=self.weight_value_initialiser,
                    name=f"{self.name}_B{i}",
                    is_updatable=self.is_updatable,
                )
            )
            n_nodes_previous_layer = n_nodes_layer

    def _activation(self, x: GradientTensor, final_layer=False) -> GradientTensor:
        activation_function_obj = (
            self.final_activation_function if final_layer else self.activation_function
        )
        if activation_function_obj is None:
            return x

        return x.applyfunc(
            activation_function_obj.forward,
            func_name=activation_function_obj.__class__.__name__,
            grad_func=activation_function_obj.grad,
        )

    def forward(self, input_data: list | GradientTensor) -> list:
        output_data = []
        input_data = input_data if isinstance(input_data, list) else input_data.values
        for row in input_data:
            output = row
            for layer_index, (weights, bias) in enumerate(
                zip(self.weights, self.biases)
            ):
                output = self._activation(
                    weights.vecmul(output) + bias,
                    final_layer=layer_index + 1 == len(self.layers),
                )
            output_data.append(output)
        return output_data
