#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""jimmy_mk_iv.py: Contains the tools for creating an Auto-Recurrent Network"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
from overrides import overrides

class Jimmy_Mark_IV(tf.keras.layers.Layer): # Make this into a keras.Layer
    """The Jimmy Auto-Recurrent Network Layer."""
    
    RANDOM_SEED = 952
    
    @overrides
    def __init__(self, 
            hidden_units: int, 
            output_units: int, 
            num_propagations: int,
            weight_initializer: str = "glorot_normal",
            name: str = "Jimmy",
            **kwargs
        ):
        """Initializes params necessary for Jimmy MK IV"""
        super().__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_propagations = num_propagations
        self.weight_initializer = weight_initializer

    @overrides
    def build(self, input_shape):
        """Allows lazy creation of weights and biases"""
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.weight_initializer,
            trainable=True
        )
        self.b = self.add_weights(
            shape=(self.units,),
            initializer=self.weight_initializer,
            trainable=True
        )

    @overrides
    def call(self, x):
        """
        Performs an auto-recursive forward propagation.
        Using the @tf.function decorator allows it to be
        ran as a graph, greatly accelerating training speed.
        """
        #for passthrough in range(self.num_propagations):

        self.states = [] 

    @overrides
    def get_config(self):
        """Allows for serializing this layer"""
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