#!/usr/bin/env python

import gc
import pickle

from absl import app
from absl import flags

import numpy as np

import torch
from torch.autograd import Variable

from cse547.data import CocoPatchesDataset, OneShotDataLoader
from cse547.models import MultiLayerPerceptron
from cse547.s3 import deserialize_object


flags.DEFINE_string('s3_bucket', 'cse-547', 'Where to retrieve model from.')
flags.DEFINE_string('s3_training_log_key', '', 'Model specification.')
flags.DEFINE_string('s3_model_key', '', 'Model parameters to use for embedding.')

flags.DEFINE_list('data', ['data/patch_features_tiny/train2014.p'],
                  'Features to embed.')
flags.DEFINE_string('output', '', 'Where to dump the pickled embeddings.')

FLAGS = flags.FLAGS


def main(argv):
    assert FLAGS.output, 'Output must be specified.'
    assert FLAGS.s3_training_log_key, 'Training log must be provided.'
    assert FLAGS.s3_model_key, 'Model key must be provided.'
    training_log = deserialize_object(FLAGS.s3_bucket, FLAGS.s3_training_log_key)
    state_dict = deserialize_object(FLAGS.s3_bucket, FLAGS.s3_model_key)

    embeddings = []
    labels = []

    for data_filename in FLAGS.data:
        samples = iter(OneShotDataLoader(
            CocoPatchesDataset.from_state_dict_files([data_filename]))).next()
        gc.collect()            # Make sure to free previous data subset

        labels.append(samples['label'].data.numpy())
        model = MultiLayerPerceptron(
            samples['features'].size()[1],
            samples['label'].size()[1],
            training_log['model']['hidden_units'],
            training=False,
            dropout=training_log['model']['dropout'])
        model.load_state_dict(state_dict)
        with torch.no_grad():
            embeddings.append(model.embed(samples['features']).data.numpy())

    with open(FLAGS.output, 'wb') as f:
        pickle.dump({
            'embeddings': np.vstack(embeddings),
            'labels': np.vstack(labels),
        }, f)

if __name__ == '__main__':
    app.run(main)
