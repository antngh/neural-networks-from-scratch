from abc import ABC, abstractmethod


class ActivationBase(ABC):
    """
    Utility class for activation functions
    """

    @staticmethod
    @abstractmethod
    def forward(self_var: float) -> float:
        """
        Implements the forward pass of the activation function

        Parameters
        ----------
        self_var : float
            The input value to the activation function

        Returns
        -------
        float
            The output value of the activation function
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def grad(self_var: float, downstream_var: float, other_var: float):
        """
        The gradient of the output of the activation function with respect to the input

        Parameters
        ----------
        self_var : float
            The input value to the activation function
        downstream_var : float
            The output of the activation function
        other_var : float
            The other variable in a calculation.
            Not used, is here to match the signature grad functions.
        """
        raise NotImplementedError


class Linear(ActivationBase):
    """
    Linear, i.e. do nothing.
    """

    @staticmethod
    def forward(self_var):
        return self_var

    @staticmethod
    def grad(self_var, downstream_var, other_var):
        return 1.0


class Relu(ActivationBase):
    """
    Rectified Linear Unit.

    f(x) = x if x > 0 else 0

    The gradient is 1 if x > 0, else 0.
    """

    @staticmethod
    def forward(self_var):
        return self_var if self_var > 0 else 0

    @staticmethod
    def grad(self_var, downstream_var, other_var):
        return 1.0 if self_var > 0 else 0
