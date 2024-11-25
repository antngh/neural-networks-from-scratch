from typing import Callable

from neural_networks_from_scratch.gradient_variable import GradientTensor
from neural_networks_from_scratch.models.model_base import GradientModelBase

from .activation_functions import ActivationBase, Relu


class MLP(GradientModelBase):
    """
    A multi layer perceptron model (feed forward neural network).

    Attributes
    ----------
    n_inputs : int
        The number of inputs to the model.
    layers : tuple[int, ...]
        The number of nodes in each layer.
        i.e. a 3 layer model with 2 nodes in the first hidden layer, 3 in the second
        and 1 in the output would be (2, 3, 1).
    name : str
        The name of the model.
    is_updatable : bool
        Whether the model is updatable or not.
    weight_value_initialiser :float | Callable[[], float]
        The initialiser for the weights of the model.
        Either a float or a function that returns a float.
        By default it will be initialised with a normal distribution).
    internal_activation_function : ActivationBase | None
        The activation function to use for the internal connections.
    final_activation_function : ActivationBase | None
        The activation function to use for the final output.
    """

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
        """
        Initialise the MLP

        Parameters
        ----------
        n_inputs : int
            The number of inputs to the model.
        layers : tuple[int, ...]
            The number of nodes in each layer.
        is_updatable : bool | None, optional
            Whether the model is updatable or not
        name : str | None, optional
            The name of the model.
        weight_initialiser : Callable | None, optional
            The initialiser for the weights of the model.
        internal_activation_function : ActivationBase | None
            The activation function to use for the internal connections.
        final_activation_function : ActivationBase | None
            The activation function to use for the final output.
        """
        self.n_inputs = n_inputs
        self.layers = layers

        self.activation_function = internal_activation_function
        self.final_activation_function = final_activation_function

        super().__init__(
            is_updatable=is_updatable, name=name, weight_initialiser=weight_initialiser
        )

    def _create_variables(self):
        """
        Initialise all weights and biases of the model..
        """
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
        """
        Apply the activation function to all elements of x.

        Parameters
        ----------
        x : GradientTensor
            The tensor to apply the activation function to
        final_layer : bool, optional
            Whether this is the final layer of the model.

        Returns
        -------
        GradientTensor
            The tensor with the activation function applied element-wise
        """
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
        """
        Run the data through the model.

        Defines how the network is wired up.

        Parameters
        ----------
        input_data : list | GradientTensor
            The input data to the model.

        Returns
        -------
        list
            The output of the model.
        """
        output_data = []
        input_data = input_data if isinstance(input_data, list) else input_data.values
        for row in input_data:
            output = row
            for layer_index, (weights, bias) in enumerate(
                zip(self.weights, self.biases)
            ):
                output = self._activation(
                    (weights @ output) + bias,
                    final_layer=layer_index + 1 == len(self.layers),
                )
            output_data.append(output)
        return output_data
