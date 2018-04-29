from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
from torch.nn import functional

from typing import Generator, Iterable, List

from cse547.tensor import truncated_normal

class Model(ABC):
    """
    It's not permitted to use torch.nn.Module, so I implement a more
    basic version myself.
    """
    @abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Variable]:
        pass

    def __call__(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)    


class LinearClassifier(Model):
    def __init__(self, n_features: int, n_classes: int) -> None:
        shape = (n_features) if n_classes == 1 else (n_features, n_classes)
        self._weights: Variable = Variable(
            truncated_normal(shape)/256,
            requires_grad = True)
        self._parameters: List[Variable] = [self._weights]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.matmul(x, self._weights)

    def parameters(self) -> Generator[Variable, None, None]:
        for param in self._parameters:
            yield param

class MultiLayerPerceptron(Model):
    def __init__(self, n_features: int, n_classes: int, hidden_units: Iterable[int]) -> None:
        self._parameters: List[Variable] = []

        previous_units = n_features
        for units in hidden_units:
            shape = (previous_units, units)
            self._parameters.append(
                Variable(truncated_normal(shape)/256, requires_grad=True))
            previous_units = units

        shape = (previous_units) if n_classes == 1 else (previous_units, n_classes)
        self._parameters.append(
            Variable(truncated_normal(shape)/256, requires_grad=True))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        output = torch.matmul(x, self._parameters[0])
        for weights in self._parameters[1:]:
            output = torch.matmul(functional.relu(output), weights)
        return output

    def parameters(self) -> Generator[Variable, None, None]:
        for param in self._parameters:
            yield param
