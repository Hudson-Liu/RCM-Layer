#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainer.py: Trains a model using the MNIST dataset"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple

tf.random.set_seed(952)

train_data, val_data, test_data = tfds.load("mnist", split=['train[10000:]', 'train[0:10000]', 'test'], batch_size=64, as_supervised=True)

def preprocess(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simple preprocessing func for MNIST"""
    x = tf.reshape(x, shape=[-1, 784])
    print(x)
    x /= 255
    return x, y

train_data = train_data.map(preprocess)
val_data = val_data.map(preprocess)

