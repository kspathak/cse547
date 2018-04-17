from typing import Callable

import torch
from torch.nn import functional

class BinaryCrossEntropy(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, output: torch.FloatTensor, labels: torch.FloatTensor):
        return functional.binary_cross_entropy(functional.sigmoid(output), labels)

class MeanSquaredError(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, predictions: torch.FloatTensor, labels: torch.FloatTensor):
        return functional.mse_loss(predictions, labels)
