#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""basic_mnist.py: Trains a model using the MNIST dataset"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import numpy as np
from tensorflow import keras
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense
)
import datetime
import tensorflow as tf

from rcm_layer import RCM

# TODO Clean this program up

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
rcm = RCM(hidden_units=5, output_units=10, num_propagations=5, activation="relu")(dropout)
outputs = Dense(10, activation="softmax")(rcm)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name="Jimmy_MK_IV_Test")
keras.utils.plot_model(model, to_file="jimmy.png", show_shapes=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

batch_size = 128
epochs = 15

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback])
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Add TensorBoard