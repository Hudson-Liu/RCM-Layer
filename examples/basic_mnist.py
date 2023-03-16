#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
basic_mnist.py: Compares the RCM layer accuracy to the Dense layer accuracy on the MNIST dataset. 
The code is heavily based off the example code provided by Keras.
"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import numpy as np
import os
import shutil
from keras.layers import (
    Flatten,
    Dropout,
    Dense
)
import tensorflow as tf

# NOTE: This script must be executed from the "examples" module-level, as __init__.py must be executed prior
from rcm_layer import RCM, ActivationTable

# Defines Model Hyperparameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
ACTIVATION_TABLE: ActivationTable = { # NOTE: The "ActivationTable" type annotation is not required, but recommended
    "inputs": "relu", 
    "hidden": "relu", 
    "outputs": "softmax"
}
BATCH_SIZE = 128
EPOCHS = 15
HIDDEN_UNITS = 10
NUM_PROPAGATIONS = 2

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

def equivalent_dense_nodes(input_nodes: int, hidden_units: int, output_nodes: int) -> int:
    """Calculates the necessary number of hidden Dense nodes needed to equal the same number of parameters as the RCM layer"""
    return (input_nodes + hidden_units + output_nodes) ** 2 // (input_nodes + output_nodes)

def create_model(model_type: str) -> tf.keras.Model:
    """Creates a model with either an RCM or Dense layer"""
    # Flattens input image
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    flatten = Flatten()(inputs)
    dropout = Dropout(0.7)(flatten)

    # Creates either a Dense layer or an RCM layer
    if model_type == "rcm":
        outputs = RCM(
            hidden_units=HIDDEN_UNITS, 
            output_units=NUM_CLASSES,
            num_propagations=NUM_PROPAGATIONS, 
            activations=ACTIVATION_TABLE,
        )(dropout)
    elif model_type == "dense":
        nodes = equivalent_dense_nodes(
            dropout.get_shape()[-1], 
            HIDDEN_UNITS, 
            NUM_CLASSES
        )
        hidden_layer = Dense(nodes, activation="relu")(dropout)
        outputs = Dense(NUM_CLASSES, activation="softmax")(hidden_layer)
    else:
        raise ValueError(f"'model_type' must either be 'rcm' or 'dense', got {model_type}.")
    
    # Returns final model
    return tf.keras.Model(inputs, outputs)

# Runs both a Dense layer model and an RCM layer model to compare accuracies
MODEL_TYPES = ["dense", "rcm"]
for model_type in MODEL_TYPES:
    # Creates a model of the specified model type
    model = create_model(model_type)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    # Creates visualizations and logs
    tf.keras.utils.plot_model(model, to_file=f"models/MNIST_{model_type}.png", show_shapes=True)
    log_dir = f"models/logs/fit/MNIST_{EPOCHS}_{HIDDEN_UNITS}_{model_type}"
    if model_type == "rcm":
        log_dir += f"_{NUM_PROPAGATIONS}"
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Trains model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, callbacks=[tensorboard_callback])
    
    # Evaluates model on test set
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model_type} test loss:", score[0])
    print(f"{model_type} test accuracy:", score[1])
