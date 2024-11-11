import random
from typing import Callable

from neural_networks_from_scratch.activation_functions import ActivationBase, Relu
from neural_networks_from_scratch.tensor import GTensor


class MLP:
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
        self.name = name
        self.is_updatable = is_updatable

        self.n_inputs = n_inputs
        self.layers = layers

        self.activation_function = internal_activation_function
        self.final_activation_function = final_activation_function

        self.weight_value_initialiser = (
            weight_initialiser
            if weight_initialiser is not None
            else lambda: random.normalvariate(0, 1)
        )

        self._create_weights_and_biases()

    def _create_weights_and_biases(self):
        self.weights = []
        self.biases = []
        n_nodes_previous_layer = self.n_inputs
        for i, n_nodes_layer in enumerate(self.layers):
            self.weights.append(
                GTensor(
                    dims=(n_nodes_layer, n_nodes_previous_layer),
                    initial_value=self.weight_value_initialiser,
                    name=f"{self.name}_W{i}",
                    is_updatable=self.is_updatable,
                )
            )
            self.biases.append(
                GTensor(
                    dims=(n_nodes_layer,),
                    initial_value=self.weight_value_initialiser,
                    name=f"{self.name}_B{i}",
                    is_updatable=self.is_updatable,
                )
            )
            n_nodes_previous_layer = n_nodes_layer

    def _activation(self, x: GTensor, final_layer=False) -> GTensor:
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

    def forward(self, input_data: list | GTensor) -> list[GTensor]:
        output_data = []
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

    def set_track_gradients_full_network(self, track_gradients: bool) -> None:
        self.weights[-1].set_track_gradients_full_network(track_gradients)
        self.weights[-1].set_track_gradients_full_network(track_gradients)
