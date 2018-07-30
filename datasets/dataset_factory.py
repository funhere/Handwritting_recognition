
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from datasets import cifar10
#from datasets import imagenet

from datasets import pascalvoc_2007
from datasets import pascalvoc_2012

from datasets import mnist
from datasets import captcha
from datasets import ocr

datasets_map = {
#    'imagenet': imagenet,
    'pascalvoc_2007': pascalvoc_2007,
    'pascalvoc_2012': pascalvoc_2012,
    'mnist': mnist,
    'captcha':captcha,
    'ocr':ocr,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.
    Args:
        name: String, the name of the dataset.
        split_name: A train/test split name.
        dataset_dir: The directory where the dataset files are stored.
        file_pattern: The file pattern to use for matching the dataset source files.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        reader)
