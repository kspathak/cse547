from abc import ABC, abstractmethod
import collections
import logging
import math
from typing import Callable, Dict, Iterable

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from cse547.models import Model

_logger = logging.getLogger(__name__)

class TrainingContext:
    step = 0
    previous_batch_loss = 0

    def __init__(self, model: Model) -> None:
        self.model = model

class TrainingHook(ABC, Callable[[TrainingContext], None]):
    @abstractmethod
    def __call__(self, context: TrainingContext) -> None:
        pass

    @abstractmethod
    def should_run(self, context: TrainingContext) -> bool:
        pass

class TrainingEvaluator(TrainingHook):
    _log = []

    def __init__(self, frequency_steps, model, loss_fn, evaluation_fn,
                 datasets: Dict[str, Dataset]):
        self._frequency_steps = frequency_steps

        self._model = model
        self._loss_fn = loss_fn
        self._evaluation_fn = evaluation_fn
        self._datasets = datasets

    @property
    def log(self):
        return self._log

    def __call__(self, context: TrainingContext) -> None:
        self._model.eval()

        evaluation_results = {'step': context.step}
        for dataset_key, dataset in self._datasets.items():
            results = self._evaluation_fn(self._model, self._loss_fn, dataset)
            for result_key, value in results.items():
                key = '_'.join([dataset_key, result_key])
                evaluation_results[key] = value

        self._model.train()
        _logger.info(str(evaluation_results))
        self._log.append(evaluation_results)

    def should_run(self, context: TrainingContext):
        return context.step % self._frequency_steps == 0

class TrainingSummarizer(TrainingHook):
    _running_loss = 0

    def __init__(self, frequency_steps):
        self._frequency_steps = frequency_steps

    def __call__(self, context: TrainingContext) -> None:
        self._running_loss += context.previous_batch_loss
        if context.step % self._frequency_steps == 0:
            self._running_loss /= self._frequency_steps
            _logger.info("{'step': %d, 'running_loss': %f}", context.step, self._running_loss)
            self._running_loss = 0

    def should_run(self, _: TrainingContext) -> bool:
        return True

class ModelSaver(TrainingHook):
    def __init__(self, evaluator: TrainingEvaluator, key: str) -> None:
        self._evaluator = evaluator
        self._key = key

        self.state_dict = collections.OrderedDict()
        self._loss = math.inf

    def __call__(self, context: TrainingContext):
        new_loss = self._evaluator.log[-1]['{0}_loss'.format(self._key)]
        if new_loss < self._loss:
            self._loss = new_loss
            self.state_dict = context.model.state_dict()
            _logger.info('Best model state updated.')

    def should_run(self, context: TrainingContext):
        return self._evaluator.should_run(context)
        
        
def train(model: Model,
          data_loader: DataLoader,
          optimizer: optim.Optimizer,
          loss_fn: Callable,
          epochs: int,
          hooks: Iterable[TrainingHook] = []):
    context = TrainingContext(model)

    for _ in range(epochs):
        for batch in data_loader:
            # Pytorch framework expects variables.
            inputs = Variable(batch['features'], requires_grad=False)
            labels = Variable(batch['label'].float(), requires_grad=False)

            # Zeros out the gradients.
            optimizer.zero_grad()

            # Do a training step.
            model_output = model(inputs)
            batch_loss = loss_fn(model_output, labels)
            batch_loss.backward()
            optimizer.step()

            # Update context.
            context.step += 1
            context.previous_batch_loss = batch_loss

            # Run hooks if they should be run.
            for hook in hooks:            
                if hook.should_run(context):
                    hook(context)
