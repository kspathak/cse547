#!/usr/bin/env python

import logging
import time

from absl import app
from absl import flags

flags.DEFINE_multi_integer('model_multilayer_perceptron_hidden_units', [256],
                           'The number of hidden units for the multi-layer perceptron.')

FLAGS = flags.FLAGS


def main(argv):
    print(FLAGS.model_multilayer_perceptron_hidden_units)

if __name__ == '__main__':
    app.run(main)
