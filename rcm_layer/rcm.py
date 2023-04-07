#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rcm.py: Contains a Keras layer implementation of the Recurrent Complete Multidigraph Layer"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
from overrides import overrides
from typing import TypedDict
import math


class ActivationTable(TypedDict):
    """Defines the Activation Table dictionary used by the RCM layer"""
    inputs: str
    hidden: str
    outputs: str


class RCM(tf.keras.layers.Layer):
    """A Keras implementation of the Recurrent Complete Multidigraph Layer"""
    
    @overrides
    def __init__(self, 
            hidden_units: int, 
            output_units: int, 
            num_propagations: int,
            activations: ActivationTable | str | None = "relu",
            weight_initializer: str = "glorot_uniform",
            normalization_position: str = None,
            dropout_rate: float = None,
            name: str = "Jimmy",
            **kwargs
        ):
        """Initializes params necessary for Jimmy MK IV"""
        super().__init__(name=name, **kwargs)
        
        # Initialize parameters
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_propagations = num_propagations
        self.weight_initializer = weight_initializer #TODO try a new weight initiation scheme

        # Initialize activation
        if isinstance(activations, str):
            self.activation_mode = "uniform"
            self.activations = tf.keras.activations.get(activations)
        elif isinstance(activations, dict):
            self.activation_mode = "table"
            self.activations = {key: tf.keras.activations.get(value) for key, value in activations.items()}
        else:
            self.activation_mode = "none"
        
        # Initialize normalization position and verify validity
        if (normalization_position != "preactivation") and (normalization_position != "postactivation") and (normalization_position != None):
            raise ValueError(
                "The normalization_position value you entered was invalid." + 
                f"The value must either be 'preactivation' or 'postactivation', received {normalization_position}"
            )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.normalization_position = normalization_position

        # Initialize dropout rate and verify validity
        if dropout_rate != None:
            if dropout_rate > 1.0 and not math.isclose(dropout_rate, 1.0):
                raise ValueError(f"The dropout rate cannot be greater than 1, received {dropout_rate}")
            elif dropout_rate < 0.0 and not math.isclose(dropout_rate, 0.0):
                raise ValueError(f"The dropout rate cannot be less than 0, received {dropout_rate}")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    @overrides
    def build(self, input_shape):
        """Creates weights, biases, and untracked node states"""
        # Generates input shape and total units and saves offset for usage in call
        self.input_nodes = input_shape[-1] #make this private
        total_units = self.hidden_units + self.output_units + self.input_nodes
        self.offset = self.hidden_units + self.output_units
        
        # Weights and biases
        self.w = self.add_weight(
            shape=(total_units, total_units),
            initializer=self.weight_initializer,
            trainable=True,
            name="weights"
        )
        self.b = self.add_weight(
            shape=(total_units,),
            initializer=self.weight_initializer,
            trainable=True,
            name="biases"
        )

    @overrides
    @tf.function
    def call(self, inputs, *args, **kwargs):
        """
        Performs a self-recurrent forward propagation.
        Using the @tf.function decorator allows it to be
        ran as a graph, greatly accelerating training speed.
        """
        # Set initial states
        states = tf.pad(inputs, paddings=[[0, 0], [0, self.offset]], mode="CONSTANT", constant_values=0.)
        
        # Iteratively propagate data
        for _ in range(self.num_propagations):
            # Apply dropout
            if self.dropout_rate != None:
                states = self.dropout(states)

            # Calculate new states
            states = tf.nn.bias_add(tf.matmul(states, self.w), self.b)

            # Run preactivation normalization
            if self.normalization_position == "preactivation":
                states = self.batch_norm(states)
            
            # Apply activation functions
            if self.activation_mode == "uniform":
                states = self.activations(states)
            elif self.activation_mode == "table":
                states = tf.concat(
                    [
                        self.activations["inputs"](states[:, :self.input_nodes]),
                        self.activations["hidden"](states[:, self.input_nodes:-self.output_units]),
                        self.activations["outputs"](states[:, -self.output_units:])
                    ],
                    axis=1
                )
            
            # Run postactivation normalization
            if self.normalization_position == "postactivation":
                states = self.batch_norm(states)

        # Returns output
        return states[:, -self.output_units:]

    @overrides
    def get_config(self):
        """Allows for serialization"""
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "output_units": self.output_units,
                "num_propagations": self.num_propagations,
                "activations": self.activations,
                "weight_initializer": self.weight_initializer,
                "normalization_position": self.normalization_position,
                "dropout_rate": self.dropout_rate,
                "name": self.name
            }
        )
        return config
