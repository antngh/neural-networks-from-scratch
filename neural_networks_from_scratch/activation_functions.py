from abc import ABC, abstractmethod


class ActivationBase(ABC):
    @staticmethod
    @abstractmethod
    def forward(self_var):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def grad(self_var, downstream_var, other_var):
        raise NotImplementedError


class Linear(ActivationBase):
    @staticmethod
    def forward(self_var):
        return self_var

    @staticmethod
    def grad(self_var, downstream_var, other_var):
        return 1.0


class Relu(ActivationBase):
    @staticmethod
    def forward(self_var):
        return self_var if self_var > 0 else 0

    @staticmethod
    def grad(self_var, downstream_var, other_var):
        return 1.0 if self_var > 0 else 0
