# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the MNIST dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'captcha_%s.tfrecord' #train_8_11_num.tfrecord

_SPLITS_TO_SIZES = {'train': 64*2000, 'test': 500}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 256 x 3] grayscale image.',
    'label': 'A single integer between 0 and 9',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.

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
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  #filename_queue = tf.train.string_input_producer(['./data/cifar10_train.tfrecord'])


  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  """
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
    'image/height': tf.FixedLenFeature((), tf.int64),
    'image/width': tf.FixedLenFeature((), tf.int64),
    'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  })
  """
    
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      #'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      #'image/height': tf.FixedLenFeature((), tf.int64),
      #'image/width': tf.FixedLenFeature((), tf.int64),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }

  items_to_handlers = {
      #'image': slim.tfexample_decoder.Image(shape=[32, 128, 3], channels=3),
      #'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      #'image': tf.image.decode_png(features['image/encoded'], channels=3)
      'image': slim.tfexample_decoder.Image(image_key = 'image/encoded', format_key = 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)
