from collections import defaultdict
import os
import pickle
import logging
from typing import Callable, Optional

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO

_ANNOTATIONS_FILE_PATH = {
    'train': os.path.join('annotations', 'instances_train2014.json'),
    'test': os.path.join('annotations', 'instances_test2014.json'),
    'validation': os.path.join('annotations', 'instances_val2014.json'),
}

_FEATURES_FILE_PATH = {
    'small': {
        'train': os.path.join('features_small', 'train2014.p'),
        'test': os.path.join('features_small', 'test2014.p'),
        'validation': os.path.join('features_small', 'val2014.p'),
    },
    'tiny': {
        'train': os.path.join('features_tiny', 'train2014.p'),
        'test': os.path.join('features_tiny', 'test2014.p'),
        'validation': os.path.join('features_tiny', 'val2014.p'),
    },
}

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class CocoFeaturesDataset(Dataset):
    def __init__(self, data_dir: str, mode: str, size: str,
                 transform: Optional[Callable] = None) -> None:
        self._transform = transform

        annotations_file_path: str = os.path.join(
            data_dir, _ANNOTATIONS_FILE_PATH[mode])
        self._coco = COCO(annotations_file_path)

        features_file_path = os.path.join(data_dir, _FEATURES_FILE_PATH[size][mode])
        with open(features_file_path, 'rb') as f:
            img_ids, features = pickle.load(f, encoding="bytes")

        self._img_ids = img_ids
        self._features = features

        imgs = self._coco.loadImgs(self._img_ids)

        categories = self._coco.loadCats(self._coco.getCatIds())        
        category_id_to_supercategory = {c['id']: c['supercategory'] for c in categories}
        category_tree = defaultdict(list)
        for c in categories:
            category_tree[c['supercategory']].append(c['name'])

        self._labels = []
        for img_id in img_ids:
            annotation_ids = self._coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            annotations = self._coco.loadAnns(annotation_ids)
            img_category_ids = frozenset([a['category_id'] for a in annotations])
            img_supercategories = set([category_id_to_supercategory[c]
                                       for c in img_category_ids])
            assert len(img_supercategories) == 1, 'Each image should only have one label.'
            self._labels.append(img_supercategories.pop())

        _logger.info(
            'Loaded the features and labels for %d images.', self.__len__())

        _logger.debug('Manually inspect a random sample of images as a sanity check.')
        for i in np.random.choice(np.arange(len(self._labels)), 10):
            _logger.debug("{'index': %d, 'id': %d, 'url': '%s', 'label': '%s'}",
                          i, imgs[i]['id'], imgs[i]['coco_url'], self._labels[i])


    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        sample = {'features': self._features[i], 'label': self._labels[i]}
        if self._transform:
            return self._transform(sample)
        return sample

class TensorTransform(Callable):
    def __call__(self, sample):
        return {
            'features': torch.from_numpy(sample['features'].reshape(-1)),
            'label': (1.0 if sample['label'] == 'vehicle' else -1.0),
        }

class OneShotDataLoader(DataLoader):
    """Used to evaluate small datasets.
    """
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset, batch_size=len(dataset), shuffle=False)
