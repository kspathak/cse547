from abc import ABC, abstractmethod
import collections
from typing import Generator, Iterable, List, Dict

import torch
from torch.autograd import Variable
from torch.nn import functional

from cse547.tensor import truncated_normal

class Model(ABC):
    _parameters: List[Variable]
    _state_dict: Dict[str, torch.FloatTensor] 

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

    def state_dict(self) -> Dict[str, torch.FloatTensor]:
        return collections.OrderedDict([
            (key, value.clone()) for key, value in self._state_dict.items()
        ])

    def load_state_dict(self, state_dict: Dict[str, torch.FloatTensor]) -> None:
        assert len(state_dict) == len(self._state_dict)
        for key, value in state_dict.items():
            assert key in self._state_dict
            self._state_dict[key].copy_(value)
    
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                pp.grad.detach_()
                p.grad.zero_()

    def __call__(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)    


class LinearClassifier(Model):
    def __init__(self, n_features: int, n_classes: int) -> None:
        shape = (n_features) if n_classes == 1 else (n_features, n_classes)
        self._weights: Variable = Variable(
            truncated_normal(shape)/512,
            requires_grad = True)
        self._parameters: List[Variable] = [self._weights]
        self._state_dict = collections.OrderedDict([
            ('fc{0}.weights'.format(index + 1), param.data)
            for index, param in enumerate(self._parameters)
        ])

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
