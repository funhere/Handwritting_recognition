#!/usr/bin/env python
"""
# run python setup.py install to install the package and its dependencies
"""
__author__ = "xxx"
__license__ = "GPL"

from setuptools import setup, find_packages

setup(name='handwritting OCR',
      version='0.5.0',
      license='GPL',
      author='Simon',
      description='TensorFlow Convolutional Recurrent Neural Network (CRNN) for OCR',
      install_requires=[
            'tensorflow-gpu',
            'numpy',
            'imageio',
            'matplotlib',
            'easydict',
            'tqdm',
            'sacred',
            'tensorflow-tensorboard',
            'tensorflow',
            'better_exceptions',
            'opencv-python'
      ],
      packages=find_packages(where='.'),
      zip_safe=False)