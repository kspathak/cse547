import torch
from typing import List

def truncated_normal(shape: List[int], level=2) -> torch.Tensor:
    return torch.clamp(
        torch.normal(torch.ones(shape)),
        min=-level, max=level)
