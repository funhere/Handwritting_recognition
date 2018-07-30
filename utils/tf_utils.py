
"""Diverse TensorFlow utils, for training, evaluation and so on!
"""
import os
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

slim = tf.contrib.slim

import os.path as ops

from tensorflow.python.ops import image_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from utils import establish_char_dict

image_shape = (32, 256, 3)
image_encoding = 'jpeg' # 'png'
DIGITS='0123456789'

# =========================================================================== #
# General tools.
# =========================================================================== #
def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


# =========================================================================== #
# Training utils.
# =========================================================================== #
def print_configuration(flags, data_sources, save_dir=None):
    """Print the training configuration.
    """
    def print_config(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(flags, stream=stream)
        """
        print('\n# =========================================================================== #', file=stream)
        print('# model parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(dict(net_params._asdict()), stream=stream)
        """
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    print_config(None)
    # Save to a text file as well.
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, 'training_config.txt')
        with open(path, "w") as out:
            print_config(out)


def configure_learning_rate(flags, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch / flags.batch_size *
                      flags.num_epochs_per_decay)

    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          flags.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         flags.learning_rate_decay_type)


def configure_optimizer(flags, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=flags.adam_beta1,
            beta2=flags.adam_beta2,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer


def add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def update_model_scope(var, ckpt_scope, new_scope):
    return var.op.name.replace(new_scope,'vgg_16')


def get_init_fn(flags):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if flags.checkpoint_path is None:
        return None
    
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None

    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(flags.model_name,
                                 flags.checkpoint_model_scope): var
             for var in variables_to_restore}


    if tf.gfile.IsDirectory(flags.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
    else:
        checkpoint_path = flags.checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def int_to_char(number):
    """
    :param number:
    :return:
    """
    char_dict_path=ops.join(os.getcwd(), 'data/char_dict/char_dict.json')
    __char_list = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        
    if number == '1':
        return '*'
    if number == 1:
        return '*'
    else:
        return __char_list[str(number)]

def sparse_tensor_to_str(spares_tensor: tf.SparseTensor):
    """
    :param spares_tensor:
    :return: a str
    """
    ord_map_dict_path=ops.join(os.getcwd(), 'data/char_dict/ord_map.json')
    __ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
    
    
    indices = spares_tensor.indices
    values = spares_tensor.values
    values = np.array([__ord_map[str(tmp)] for tmp in values])
    dense_shape = spares_tensor.dense_shape
    
    number_lists = np.ones(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val) for val in number_list])
    for str_list in str_lists:
        res.append(''.join(c for c in str_list if c != '*'))
    return res


def decode_sparse_tensor(sparse_tensor):
    #print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        #print(result)
    return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    #str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    #str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded



# convert sequence list to sparse tuple.   
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
    
    return indices, values, shape

def int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_features(tfrecords_path, num_epochs):
    """
    :param tfrecords_path:
    :param num_epochs:
    :return:
     """
    assert ops.exists(tfrecords_path)

    filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
   
    
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        #'image': tf.FixedLenFeature((), tf.string),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        #'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/label': tf.VarLenFeature(tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        #'image': slim.tfexample_decoder.Image('image'),
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    [images] = decoder.decode(serialized_example, ['image'])

    features = tf.parse_single_example(serialized_example,
                                           features={
                                               #'imagenames': tf.FixedLenFeature([1], tf.string),
                                               'image/object/bbox/label': tf.VarLenFeature(tf.int64),
                                               #'image': tf.FixedLenFeature((), tf.string),
                                           })
    labels = features['image/object/bbox/label']
    labels = tf.cast(labels, tf.int32)
    #labels = tf.cast(labels, tf.int32)
    #labels = labels.eval()
    
    return images, labels
               
# =========================================================================== #
# Evaluation utils.
# =========================================================================== #
