
"""Provides data for the OCR handwritting dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import pascalvoc_common

import os
from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'ocr_%s_*.tfrecord'

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

#Dataset account: [1872:  1685/187]
#Phase2_1:{'train': 1479, 'test': }
#Phase2_2: {'train': 392, 'test': 50}
_SPLITS_TO_SIZES = {'train': 392, 'test': 50}

NUM_CLASSES = 11


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading PASCAL VOC.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      _SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
    
    