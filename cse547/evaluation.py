from typing import Callable, Dict

from sklearn.metrics import average_precision_score

import torch
from torch.autograd import Variable
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader

from cse547.data import OneShotDataLoader
from cse547.models import Model

def evaluate_binary_classifier(model: Model, loss_fn: Callable, dataset: Dataset) -> Dict[str, float]:
    samples = iter(OneShotDataLoader(dataset)).next()
    features = Variable(samples['features'], volatile=True)
    labels = Variable(samples['label'].float(), volatile=True)
    model_output = model(features)
    loss = loss_fn(model_output, labels).data[0]
    
    predictions = torch.sign(model_output)
    correct_predictions = torch.sum(
        (predictions == labels).float()).data[0]
    return {'loss': loss, 'accuracy': correct_predictions/len(dataset)}

def evaluate_multilabel_classifier(model: Model, loss_fn: Callable, dataset: Dataset) -> Dict[str, float]:
    """Computes loss and the average precision score.
    """
    samples = iter(OneShotDataLoader(dataset)).next()
    features = Variable(samples['features'], volatile=True)
    labels = Variable(samples['label'], volatile=True)
    model_output = model(features)
    loss = loss_fn(model_output, labels).data[0]

    predictions = functional.sigmoid(model_output)
    precision = average_precision_score(labels.data.numpy(), predictions.data.numpy())
    return {'loss': loss, 'average_precision_score': precision}
