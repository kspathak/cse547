#!/usr/bin/env python

import logging
import time

from absl import app
from absl import flags

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cse547.data import CocoFeaturesDataset, TensorTransform
from cse547.loss import BinaryCrossEntropy
from cse547.models import LinearClassifier, MultiLayerPerceptron
from cse547.train import train, TrainingEvaluator, TrainingSummarizer

# Mode
flags.DEFINE_enum('mode', 'train', ['train', 'evaluation', 'predict'],
                  'Specifies the task.')

# Data flags
flags.DEFINE_string('data_dir', 'data', "Data directory.")
flags.DEFINE_enum('dataset', 'train',
                  ['train', 'test', 'validation'],
                  'Specifies the dataset.')
flags.DEFINE_enum('size', 'tiny',
                  ['tiny', 'small'],
                  'Specifies the size of the dataset to use.')

# Model flags
flags.DEFINE_enum('model', 'linear', ['linear', 'multilayer_perceptron'],
                  'The model type to use.')
flags.DEFINE_integer('model_multilayer_perceptron_hidden_units', 200,
                     'The number of hidden units for the multi-layer perceptron.')

# Training flags, ignored by evaluation jobs
flags.DEFINE_integer('train_batch_size', 8, 'Batch sizes during training.')
flags.DEFINE_float('train_l2_regularization', 4e-4,
                   'L2 regularization in the loss function.')
flags.DEFINE_integer('train_epochs', 32,
                     'The number of times to iterate over the data in training.')
flags.DEFINE_float('train_optimizer_learning_rate', 1e-5,
                   'The learning rate for stochastic gradient descent.')
flags.DEFINE_float('train_optimizer_momentum', 0.7,
                   'Nesterov\'s momentum for acceleration.')
flags.DEFINE_integer('train_summary_steps', 250,
                     'How often to summarize the model.')
flags.DEFINE_integer('train_evaluation_steps', 1000,
                     'How often to evaluate the model.')

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


def main(argv):
    dataset = CocoFeaturesDataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.size,
                                  transform=TensorTransform())

    # TODO(phillypham): Move this to a different job, so it doesn't block training.
    test_dataset = CocoFeaturesDataset(FLAGS.data_dir, 'test', FLAGS.size,
                                       transform=TensorTransform())

    data_loader = DataLoader(dataset, batch_size=FLAGS.train_batch_size,
                             shuffle=True, num_workers=4)

    n_features = dataset[0]['features'].size()[0]
    hidden_units = FLAGS.model_multilayer_perceptron_hidden_units
    model = (LinearClassifier(n_features)
             if FLAGS.model == 'linear' else
             MultiLayerPerceptron(n_features, hidden_units))

    loss_fn = BinaryCrossEntropy()
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.train_optimizer_learning_rate,
        momentum=FLAGS.train_optimizer_momentum,
        weight_decay=FLAGS.train_l2_regularization)

    # Define training hooks
    training_evaluator = TrainingEvaluator(
        FLAGS.train_evaluation_steps,
        model=model,
        loss_fn=loss_fn,
        training_dataset=dataset,
        test_dataset=test_dataset)
    hooks = [
        TrainingSummarizer(FLAGS.train_summary_steps),
        training_evaluator,
    ]

    training_start_time = time.time()
    logger.info("Training is starting.")
    train(
        model=model, 
        data_loader=data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=FLAGS.train_epochs,
        hooks = hooks)
    logger.info("Training completed after %d seconds.", time.time() - training_start_time)
    logger.info("Model hyperparameters: {'l2_penalty': %f}", FLAGS.train_l2_regularization)
    logger.info("Training evaluation log: %s", training_evaluator.log)

if __name__ == '__main__':
    app.run(main)
    
