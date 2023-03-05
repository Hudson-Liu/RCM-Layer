#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainer.py: Trains a model using the MNIST dataset"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import numpy as np
from tensorflow import keras
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout
)

from jimmy_mk_iv import JimmyMarkIV

# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert to categorical data
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Design model
inputs = keras.Input(shape=input_shape)
conv = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
conv = Conv2D(64, kernel_size=(3, 3), activation="relu")(pool)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
flatten = Flatten()(pool)
dropout = Dropout(0.5)(flatten)
outputs = JimmyMarkIV(hidden_units=50, output_units=num_classes, num_propagations=10)(dropout)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name="Jimmy_MK_IV_Test")
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)