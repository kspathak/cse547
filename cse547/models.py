from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
from torch.nn import functional

from typing import Generator, Iterable, List

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
    def __init__(self, n_features: int) -> None:
        self._weights: Variable = Variable(
            torch.normal(means=torch.zeros(n_features),
                         std=torch.ones(n_features)/256),
            requires_grad = True)

        self._parameters: List[Variable] = [self._weights]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.matmul(x, self._weights)

    def parameters(self) -> Generator[Variable, None, None]:
        for param in self._parameters:
            yield param

class MultiLayerPerceptron(Model):
    def __init__(self, n_features: int, hidden_units: int) -> None:
        self._weights1: Variable = Variable(
            torch.normal(means=torch.zeros(n_features, hidden_units),
                         std=torch.ones(n_features, hidden_units)/256),
            requires_grad=True)

        self._weights2: Variable = Variable(
            torch.normal(means=torch.zeros(hidden_units),
                         std=torch.ones(hidden_units)/256),
            requires_grad=True)

        self._parameters: List[Variable] = [self._weights1, self._weights2]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.matmul(functional.relu(torch.matmul(x, self._weights1)), self._weights2)

    def parameters(self) -> Generator[Variable, None, None]:
        for param in self._parameters:
            yield param
