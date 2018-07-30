
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import crnn_preprocessing
#from preprocessing import attention_preprocessing

from preprocessing import inception_preprocessing
#from preprocessing import vgg_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):
    """
    Args:
      name: The name of the preprocessing function.
      is_training: `True`for training, 'False' for others.
    Returns:
      preprocessing_fn: A function that preprocessing image.
        e.g:
          image = preprocessing_fn(image, output_height, output_width, ...).
    """
    preprocessing_fn_map = {
      'crnn': crnn_preprocessing,
#      'attention': attention_preprocessing,      
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'pnasnet_large': inception_preprocessing,
 #     'resnet_v2_101': vgg_preprocessing,
 #     'resnet_v2_152': vgg_preprocessing,
 #     'resnet_v2_200': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(
                image, output_height, output_width, is_training=is_training, **kwargs)
    
    return preprocessing_fn
