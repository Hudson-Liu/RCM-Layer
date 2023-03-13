#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rcm.py: Contains a Keras layer implementation of the Recurrent Complete Multidigraph Layer"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
from overrides import overrides


class RCM(tf.keras.layers.Layer):
    """A Keras implementation of the Recurrent Complete Multidigraph Layer"""
    
    @overrides
    def __init__(self, 
            hidden_units: int, 
            output_units: int, 
            num_propagations: int,
            activation: str = None,
            weight_initializer: str = "glorot_uniform",
            name: str = "Jimmy",
            **kwargs
        ):
        """Initializes params necessary for Jimmy MK IV"""
        super().__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_propagations = num_propagations
        self.weight_initializer = weight_initializer #TODO try a new weight initiation scheme
        self.activation = tf.keras.activations.get(activation)

    @overrides
    def build(self, input_shape):
        """Creates weights, biases, and untracked node states"""
        # Generates input shape and total units and saves offset for usage in call
        input_nodes = input_shape[-1] #make this private
        total_units = self.hidden_units + self.output_units + input_nodes
        self.offset = total_units - input_nodes
        
        # Weights and biases
        self.w = self.add_weight(
            shape=(total_units, total_units),
            initializer=self.weight_initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(total_units,),
            initializer=self.weight_initializer,
            trainable=True
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
        states = tf.pad(
            inputs, 
            paddings=[[0, 0], [0, self.offset]], 
            mode="CONSTANT",
            constant_values=0.
        )
        
        # Iteratively propagate data
        for _ in range(self.num_propagations):
            #tf.print(states) # TODO remove this
            # Calculate new states
            states = tf.nn.bias_add(tf.matmul(states, self.w), self.b)
            # Modify propagation based on configuration
            if self.activation != None:
                states = self.activation(states) #TODO varying activations per iteration
               
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
                "weight_initializer": self.weight_initializer
            }
        )
        return config
