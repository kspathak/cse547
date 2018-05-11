from typing import Callable

import torch
from torch.nn import functional

class BinaryCrossEntropy(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    """Assumes labels are +1 or -1.
    """
    def __call__(self, output: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        return functional.soft_margin_loss(output, labels)

class MeanSquaredError(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, predictions: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        return functional.mse_loss(predictions, labels)

class MultiLabelCrossEntropy(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    """Assumes the label for each observation is a vector of 0s and 1s.
    """
    def __call__(self, output: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        return functional.multilabel_soft_margin_loss(output, labels)
