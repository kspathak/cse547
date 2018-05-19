#!/usr/bin/env python

import logging
import pickle
import time

from absl import app, flags

from cse547.data import CocoPatchesDataset

# Data flags
flags.DEFINE_string('data_dir', 'data', "Data directory.")
flags.DEFINE_enum('dataset', 'train', ['train', 'test', 'validation'],
                  'Specifies the dataset.')
flags.DEFINE_enum('size', 'tiny', ['tiny', 'small'],
                  'Specifies the size of the dataset to use.')
flags.DEFINE_list('supercategories', ['vehicle', 'animal'],
                  'Filters out which categories to consider.')
flags.DEFINE_float('iou_threshold', 0.5,
                   'Determines the intersection over union (IoU) threshold for positive examples.')
flags.DEFINE_string('examples_output', '',
                    'Where to output examples.')

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


def main(argv):
    assert len(FLAGS.examples_output) > 0, 'An output path must be specified.'

    start_time = time.time()
    dataset = CocoPatchesDataset.from_images(
        FLAGS.data_dir, FLAGS.dataset, FLAGS.size,
        supercategories = frozenset(FLAGS.supercategories) if len(FLAGS.supercategories) > 0 else None,
        iou_threshold=FLAGS.iou_threshold)

    with open(FLAGS.examples_output, 'wb') as f:
        pickle.dump(dataset.state_dict(), f)

    logger.info('%d examples generated in %d seconds.', len(dataset), time.time() - start_time)

if __name__ == '__main__':
    app.run(main)
