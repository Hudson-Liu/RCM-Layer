#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""basic_visualizations.py: Creates some basic visualizations of the model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
from rcm_layer import plot_rcm_layer, plot_and_save_logo, plot_from_weights

# Imports CIFAR model
model = tf.keras.models.load_model('models/CIFAR_600_100_rcm_adam_DenseNorm')
rcm_layer = model.layers[-3]
weights = rcm_layer.get_weights()[0]
print(weights[:2000])

# Get model configuration
input_units = rcm_layer.input_shape[-1]
hidden_units = rcm_layer.get_config()["hidden_units"]
output_units = rcm_layer.output_shape[-1]

# Create visualization
plot_from_weights(weights, input_units, hidden_units)

# Plots an RCM Layer of 12 nodes
plot_rcm_layer(12, colors=["limegreen", "darkgreen"], circle_color="green")

# Saves a version of the logo
PATH = "."
plot_and_save_logo(PATH)
