#!/usr/bin/env python

import logging
import time

from absl import app
from absl import flags

from torch import optim
from torch.utils.data import DataLoader

from cse547.evaluation import evaluate_multilabel_classifier
from cse547.data import CocoMultiLabelFeaturesDataset, FlattenTensorTransform
from cse547.loss import MeanSquaredError, MultiLabelCrossEntropy
from cse547.models import LinearClassifier, MultiLayerPerceptron
from cse547.s3 import deserialize_object, serialize_object
from cse547.train import train, TrainingEvaluator, TrainingSummarizer

# Data flags
flags.DEFINE_string('data_dir', 'data', "Data directory.")
flags.DEFINE_enum('dataset', 'train',
                  ['train', 'test', 'validation'],
                  'Specifies the dataset.')
flags.DEFINE_enum('size', 'tiny',
                  ['tiny', 'small'],
                  'Specifies the size of the dataset to use.')
flags.DEFINE_string('data_output_s3_bucket', 'cse-547',
                    'Where to put the results of training.')
flags.DEFINE_string('data_output_s3_key', 'hw2/train',
                    'Key prefix in S3. It should not end with a trailing slash.')

# Model flags
flags.DEFINE_enum('model', 'linear', ['linear', 'multilayer_perceptron'],
                  'The model type to use.')
flags.DEFINE_multi_integer('model_multilayer_perceptron_hidden_units', [256],
                           'The number of hidden units for the multi-layer perceptron.')

# Training flags, ignored by evaluation jobs
flags.DEFINE_integer('train_batch_size', 8, 'Batch sizes during training.')
flags.DEFINE_integer('train_epochs', 32,
                     'The number of times to iterate over the data in training.')
flags.DEFINE_enum('train_loss_function', 'cross_entropy', ['cross_entropy', 'mean_squared_error'],
                  'Which loss function to use when training.')
flags.DEFINE_float('train_l2_regularization', 4e-4,
                   'L2 regularization in the loss function.')
flags.DEFINE_integer('train_summary_steps', 250,
                     'How often to summarize the model.')
flags.DEFINE_integer('train_evaluation_steps', 1000,
                     'How often to evaluate the model.')

# Training optimizer flags.
flags.DEFINE_float('train_optimizer_learning_rate', 1e-4,
                   'The learning rate for stochastic gradient descent.')

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


def main(argv):
    dataset = CocoMultiLabelFeaturesDataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.size,
                                            transform=FlattenTensorTransform())

    data_loader = DataLoader(dataset, batch_size=FLAGS.train_batch_size,
                             shuffle=True, num_workers=2)

    # Specify the model from the data.
    n_features = dataset[0]['features'].size()[0]
    n_classes = dataset[0]['label'].size()[0]
    hidden_units = FLAGS.model_multilayer_perceptron_hidden_units
    model = (LinearClassifier(n_features, n_classes)
             if FLAGS.model == 'linear' else
             MultiLayerPerceptron(n_features, n_classes, hidden_units))

    loss_fn = (MultiLabelCrossEntropy() if FLAGS.train_loss_function == 'cross_entropy'
               else MeanSquaredError())

    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.train_optimizer_learning_rate,
        weight_decay=FLAGS.train_l2_regularization)

    # Define training hooks
    training_evaluator = TrainingEvaluator(
        FLAGS.train_evaluation_steps,
        model=model,
        loss_fn=loss_fn,
        evaluation_fn=evaluate_multilabel_classifier,
        datasets = {
            'training': dataset,
        })
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
        hooks=hooks)
    logger.info("Training completed after %d seconds.", time.time() - training_start_time)

if __name__ == '__main__':
    app.run(main)
