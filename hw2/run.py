#!/usr/bin/env python

import logging
import os
import time

from absl import app
from absl import flags

from torch import optim
from torch.utils.data import DataLoader

from cse547.evaluation import evaluate_multilabel_classifier
from cse547.data import CocoMultiLabelFeaturesDataset, FlattenTensorTransform
from cse547.loss import MeanSquaredError, MultiLabelCrossEntropy
from cse547.models import LinearClassifier, MultiLayerPerceptron
from cse547.s3 import serialize_object
from cse547.train import train, TrainingEvaluator, TrainingSummarizer

# Data flags
flags.DEFINE_string('data_dir', 'data', "Data directory.")
flags.DEFINE_enum('dataset', 'train', ['train', 'test', 'validation'],
                  'Specifies the dataset.')
flags.DEFINE_enum('size', 'tiny', ['tiny', 'small'],
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
flags.DEFINE_integer('train_epochs', 16,
                     'The number of times to iterate over the data in training.')
flags.DEFINE_enum('train_loss_function', 'cross_entropy', ['cross_entropy', 'mean_squared_error'],
                  'Which loss function to use when training.')
flags.DEFINE_float('train_l2_regularization', 0.5,
                   'L2 regularization in the loss function.')
flags.DEFINE_integer('train_summary_steps', 500,
                     'How often to summarize the model.')
flags.DEFINE_integer('train_evaluation_steps', 1000,
                     'How often to evaluate the model.')

# Training optimizer flags.
flags.DEFINE_enum('train_optimizer', 'sgd', ['sgd', 'adagrad', 'adam'],
                  'The gradient descent algorithm to use.')
flags.DEFINE_float('train_optimizer_learning_rate', 1e-4,
                   'The learning rate for stochastic gradient descent.')
flags.DEFINE_float('train_optimizer_momentum', 0,
                   'If non-zero, use momentum. Only applicable with SGD.')
flags.DEFINE_boolean('train_optimizer_nesterov', False,
                     'If true, uses Nesterov\'s momentum. Only applicable with SGD.')

# Training output
flags.DEFINE_boolean('train_s3_output', False,
                     'Whether to output training results to AWS S3.')
flags.DEFINE_string('train_s3_bucket', 'cse-547',
                    'AWS S3 bucket to dump output to')
flags.DEFINE_string('train_s3_key',
                    'hw2/train/{0}.pkl'.format(os.getenv('AWS_BATCH_JOB_ID', time.strftime('%s'))),
                    'Key in AWS S3 bucket to dump ouput to.')

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


def main(argv):
    dataset = CocoMultiLabelFeaturesDataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.size,
                                            transform=FlattenTensorTransform())
    test_dataset = CocoMultiLabelFeaturesDataset(FLAGS.data_dir, 'test', FLAGS.size,
                                                 transform=FlattenTensorTransform())
    # validation_dataset = CocoMultiLabelFeaturesDataset(FLAGS.data_dir, 'validation', FLAGS.size,
    #                                                    transform=FlattenTensorTransform())

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

    if FLAGS.train_optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=FLAGS.train_optimizer_learning_rate,
            momentum=FLAGS.train_optimizer_momentum,
            nesterov=FLAGS.train_optimizer_nesterov,
            weight_decay=FLAGS.train_l2_regularization)
    elif FLAGS.train_optimizer == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=FLAGS.train_optimizer_learning_rate,
            weight_decay=FLAGS.train_l2_regularization)
    elif FLAGS.train_optimizer == 'adam':
        optimizer = optim.Adam(
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
            'test': test_dataset,
            # 'validation': validation_dataset,
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

    training_run_output = {
        'evaluations': training_evaluator.log,
        'loss_function': {
            'type': FLAGS.train_loss_function,
            'l2_regularization': FLAGS.train_l2_regularization,
        },
        'model': {'type': FLAGS.model},
        'optimizer': {
            'type': FLAGS.train_optimizer,
            'learning_rate': FLAGS.train_optimizer_learning_rate,
            'batch_size': FLAGS.train_batch_size,
        },
    }
    if FLAGS.model == 'multilayer_perceptron':
        training_run_output['model']['hidden_units'] = FLAGS.model_multilayer_perceptron_hidden_units

    logger.info(training_run_output)
    if FLAGS.train_s3_output:
        serialize_object(training_run_output, FLAGS.train_s3_bucket, FLAGS.train_s3_key)

if __name__ == '__main__':
    app.run(main)
