from typing import Callable, Tuple

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from cse547.data import OneShotDataLoader
from cse547.models import Model

def evaluate_binary_classifier(model: Model, loss_fn: Callable, dataset: Dataset) -> Tuple[float, float]:
    samples = iter(OneShotDataLoader(dataset)).next()
    features = Variable(samples['features'], volatile=True)
    labels = Variable(samples['label'].float(), volatile=True)
    model_output = model(features)
    loss = loss_fn(model_output, labels).data[0]
    
    predictions = torch.sign(model_output)
    correct_predictions = torch.sum(
        (predictions == labels).float()).data[0]

    return loss, correct_predictions/len(dataset)
