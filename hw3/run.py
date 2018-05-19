#!/usr/bin/env python

import logging
import pickle
import time

from absl import app
from absl import flags

from torch.utils.data import DataLoader

from cse547.data import CocoPatchesDataset

# Data flags
flags.DEFINE_string('training_data', 'data/patch_features_tiny/train2014.p',
                    'Data to train model.')
flags.DEFINE_string('validation_data', 'data/patch_features_tiny/val2014.p',
                    'Data to validate model.')

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


def main(argv):
    with open(FLAGS.training_data, 'rb') as f:
        dataset = CocoPatchesDataset.from_state_dict(pickle.load(f))

    data_loader = DataLoader(dataset, batch_size=8,
                             shuffle=True, num_workers=2)
    for batch in data_loader:
        print(batch)

if __name__ == '__main__':
    app.run(main)
