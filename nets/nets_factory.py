
"""Contains a factory for building various models.
"""

import functools
import tensorflow as tf

from nets.crnn import crnn
#from nets.attention import attention
from nets import inception

from nets import resnet_v2
#from nets.mobilenet import mobilenet_v2
from nets.nasnet import nasnet
from nets.nasnet import pnasnet

slim = tf.contrib.slim

networks_map = {'crnn': crnn.CRNNnet,
               # 'attention': attention.Attention,
                'inception_v3': inception.inception_v3,
               # 'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'nasnet_large': nasnet.build_nasnet_large,
                'pnasnet_large': pnasnet.build_pnasnet_large,
#                'mobilenet_v2': mobilenet_v2.mobilenet,
                }

arg_scopes_map = {'crnn': crnn.crnn_arg_scope,
                #  'attention': attention.attention_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                #  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'nasnet_large': nasnet.nasnet_large_arg_scope,
                  'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
#                  'mobilenet_v2': mobilenet_v2.training_scope,
                  }

networks_obj = {'crnn': crnn.CRNNnet,
               # 'attention': attention.Attention,
                }

def get_network(name):
    """Get a network object from a name.
    """
    # params = networks_obj[name].default_params if params is None else params
    return networks_obj[name]


def get_network_fn(name, num_classes, is_training=False, **kwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      is_training: `True` for training and `False` for otherwise.
    Returns:
      network_fn: A function that applies the model to images. 
                  e.g. logits, end_points = network_fn(images)
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](**kwargs)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
