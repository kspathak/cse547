from collections import OrderedDict, defaultdict
import cv2
from math import ceil, floor
import os
import pickle
import logging
from typing import Callable, Optional, Set

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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

_FEATURES2_FILE_PATH = {
    'small': {
        'train': os.path.join('features2_small', 'train2014.p'),
        'test': os.path.join('features2_small', 'test2014.p'),
        'validation': os.path.join('features2_small', 'val2014.p'),
    },
    'tiny': {
        'train': os.path.join('features2_tiny', 'train2014.p'),
        'test': os.path.join('features2_tiny', 'test2014.p'),
        'validation': os.path.join('features2_tiny', 'val2014.p'),
    },
}

_IMG_DIR_PATH = {
    'train': os.path.join('train2014_2'),
    'test': os.path.join('test2014_2'),
    'validation': os.path.join('val2014_2'),
}

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class CocoSingleLabelFeaturesDataset(Dataset):
    def __init__(self, data_dir: str, mode: str, size: str,
                 transform: Optional[Callable] = None) -> None:
        self._transform = transform

        annotations_file_path: str = os.path.join(
            data_dir, _ANNOTATIONS_FILE_PATH[mode])
        coco = COCO(annotations_file_path)

        features_file_path = os.path.join(data_dir, _FEATURES_FILE_PATH[size][mode])
        with open(features_file_path, 'rb') as f:
            img_ids, features = pickle.load(f, encoding='bytes')

        self._features = features
        imgs = coco.loadImgs(img_ids)

        categories = coco.loadCats(coco.getCatIds())
        category_id_to_supercategory = {c['id']: c['supercategory'] for c in categories}
        category_tree = defaultdict(list)
        for c in categories:
            category_tree[c['supercategory']].append(c['name'])

        self._labels = []
        for img_id in img_ids:
            annotation_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            annotations = coco.loadAnns(annotation_ids)
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

class CocoMultiLabelFeaturesDataset(Dataset):
    def __init__(self, data_dir: str, mode: str, size: str,
                 transform: Optional[Callable] = None) -> None:
        self._transform = transform

        annotations_file_path: str = os.path.join(
            data_dir, _ANNOTATIONS_FILE_PATH[mode])
        coco = COCO(annotations_file_path)

        features_file_path = os.path.join(data_dir, _FEATURES2_FILE_PATH[size][mode])
        with open(features_file_path, 'rb') as f:
            img_ids, features = pickle.load(f, encoding="bytes")

        self._features = features

        self._label_names = []
        categories = {}
        category_ids = []
        for category in filter(
                lambda category: category['supercategory'] == 'animal' or category['supercategory'] == 'vehicle',
                coco.loadCats(coco.getCatIds())):
            categories[category['id']] = {'name': category['name'], 'index': len(self._label_names)}
            category_ids.append(category['id'])
            self._label_names.append(category['name'])

        self._labels = []
        for img_id in img_ids:
            annotation_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            annotations = coco.loadAnns(annotation_ids)
            img_category_ids = frozenset([a['category_id'] for a in annotations])
            indices = [categories[i]['index'] for i in img_category_ids if i in categories]
            self._labels.append(torch.sparse.FloatTensor(
                torch.LongTensor([indices]),
                torch.ones(len(indices)),
                torch.Size([len(categories)])).to_dense())

        _logger.info(
            'Loaded the features and labels for %d images.', self.__len__())

        imgs = coco.loadImgs(img_ids)
        _logger.debug('Manually inspect a random sample of images as a sanity check.')
        for i in np.random.choice(np.arange(len(self._labels)), 10):
            labels = [categories[category_ids[j[0]]]['name']
                      for j in np.argwhere(self._labels[i].numpy())]
            _logger.debug("{'index': %d, 'id': %d, 'url': '%s', 'label': '%s'}",
                          i, imgs[i]['id'], imgs[i]['coco_url'], labels)

    @property
    def label_names(self):
        return self._label_names

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        sample = {'features': self._features[i], 'label': self._labels[i]}
        if self._transform:
            return self._transform(sample)
        return sample

class FlattenTensorTransform(Callable):
    def __call__(self, sample):
        return {
            'features': torch.from_numpy(sample['features'].reshape(-1)),
            'label': sample['label'],
        }

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

class CocoPatchesDataset(Dataset):
    def __init__(self, categories, features, labels) -> None:
        self._categories = categories
        self._features = features
        self._labels = labels
        
    @staticmethod
    def from_images(data_dir: str, mode: str, size: str,
                    iou_threshold = 0.5,
                    supercategories: Optional[Set[str]] = None) -> 'CocoPatchesDataset':
        img_dir_path = os.path.join(data_dir, _IMG_DIR_PATH[mode])
        annotations_file_path: str = os.path.join(
            data_dir, _ANNOTATIONS_FILE_PATH[mode])
        coco = COCO(annotations_file_path)

        features_file_path = os.path.join(data_dir, _FEATURES2_FILE_PATH[size][mode])
        with open(features_file_path, 'rb') as f:
            img_ids, img_features = pickle.load(f, encoding='bytes')

        # We'll encode each category as a one-hot vector.
        categories = [
            category for category in coco.loadCats(coco.getCatIds())
            if supercategories is None or category['supercategory'] in supercategories
        ]
        category_index = {
            category['id']: i
            for i, category in enumerate(categories)
        }

        features = []
        labels = []

        featurizer = _Featurizer()
        imgs = coco.loadImgs(img_ids)
        for img_index, img in enumerate(imgs):
            if img_index % 10 == 0 and img_index > 0:
                _logger.info('Patch features extracted for %d images.', img_index)

            try:
                img_array = cv2.imread(os.path.join(img_dir_path, img['file_name']))
                bboxes = _get_bboxes(img_array, num_rects=512)
            except Exception as e:
                print(e)
                _logger.warn('%s raised when getting bounding boxes for image: %s',
                             e, img)
                continue                
            ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            annotations = [
                annotation for annotation in coco.loadAnns(ann_ids)
                if annotation['category_id'] in category_index
            ]
                
            img_pil = Image.open(os.path.join(img_dir_path, img['file_name']))
            for bbox in bboxes:
                bbox_category_indices: Set[int] = set()
                for annotation in annotations:
                    if _iou(bbox, annotation['bbox']) > iou_threshold:
                        bbox_category_indices.add(category_index[annotation['category_id']])
                        projected_bbox = _project_onto_feature_space(bbox, img_pil.size)
                        bbox_features = featurizer(projected_bbox, img_features[img_index])
                if len(bbox_category_indices) == 0:
                    # Negative
                    continue
                label = torch.sparse.FloatTensor(
                    torch.LongTensor([list(bbox_category_indices)]),
                    torch.ones(len(bbox_category_indices)),
                    torch.Size([len(categories)])).to_dense()
                features.append(bbox_features)
                labels.append(label)
        return CocoPatchesDataset(categories, features, labels)

    @staticmethod
    def from_state_dict(state_dict) -> 'CocoPatchesDataset':
        return CocoPatchesDataset(
            state_dict['categories'], state_dict['features'], state_dict['labels'])

    def state_dict(self):
        return OrderedDict([
            ('categories', self._categories),
            ('features', self._features),
            ('labels', self._labels),
        ])

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        return {'features': self._features[i], 'label': self._labels[i]}

    @property
    def categories(self):
        return self._categories

cv2.setNumThreads(4)
_SELECTIVE_SEARCHER = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
def _get_bboxes(img: np.array, num_rects: Optional[int] = None) -> np.array:
    _SELECTIVE_SEARCHER.setBaseImage(img)
    # _SELECTIVE_SEARCHER.switchToSelectiveSearchQuality()
    _SELECTIVE_SEARCHER.switchToSelectiveSearchFast()
    bboxes = _SELECTIVE_SEARCHER.process()
    return bboxes if num_rects is None else bboxes[:num_rects]

def _iou(rect1, rect2) -> float: # rect = [x, y, w, h]
    """Computes intersection over union.
    """
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1+w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2+w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2) 
    X_int = min(X1, X2) 
    y_int = max(y1, y2) 
    Y_int = min(Y1, Y2) 
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0 
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0 
    return a_int / (a1 + a2 - a_int)  

# nearest neighbor in 1-based indexing
def _nnb_1(x):                                                                                                                               
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def _project_onto_feature_space(rect, image_dims):
    """Projects bounding box onto convolutional network.
    Args:
      rect: Bounding box (x, y, w, h).
      image_dims: Image size, (imgx, imgy).
    
    Returns:
      Projected coordinates, (x, y, x'+1, y'+1) where the box is x:x', y:y'.
    """
    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1 # inclusive
    py = _nnb_1(y + 1) - 1 # inclusive
    px1 = _nnb_1(x1 + 1) # exclusive
    py1 = _nnb_1(y1 + 1) # exclusive
    return [px, py, px1, py1]

class _Featurizer(Callable):
    dim = 11776 # for small features
    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)                                                                                                 
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]
        
    def __call__(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1] 
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),  
                          self.pool6(crop).view(-1)], dim=0).data.numpy() # returns numpy array
