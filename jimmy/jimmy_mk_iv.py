#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""jimmy_mk_iv.py: Contains the tools for creating an Auto-Recurrent Network"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
import numpy as np
from overrides import overrides


class JimmyMarkIV(tf.keras.layers.Layer):
    """The Jimmy Auto-Recurrent Network Layer"""
    
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
        self.weight_initializer = weight_initializer
        self.activation = tf.keras.activations.get(activation)

    @overrides
    def build(self, input_shape):
        """Creates weights, biases, and untracked node states"""
        # Saves input shape and total units for call usage
        self.input_shape = input_shape[-1]
        self.total_units = self.hidden_units + self.output_units + self.input_shape
        
        # Weights and biases
        self.w = self.add_weight(
            shape=(input_shape[-1], self.total_units),
            initializer=self.weight_initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.total_units,),
            initializer=self.weight_initializer,
            trainable=True
        )
        
        # Node states
        self.states = tf.Variable(
            initial_value=[0.] * self.total_units,
            trainable=False
        )
        self.new_states = tf.Variable(
            initial_value=[0.] * self.total_units,
            trainable=False
        )

        # Create Node-to-Weight-Index table
        weight_table = self.generate_node_to_weight_table(self.total_units)
        self.weight_table = tf.convert_to_tensor(weight_table)

    def generate_node_to_weight_table(self, total_units):
        """Behold, Hell on Earth: Generating the Node-to-Weight-Index Table"""
        # Generate Sequence 1
        seq_one = []
        cur_val = 0
        for unit in range(total_units): # Each unit is a row
            inverse = abs(unit - total_units) # Inverse is the number of columns (with values/not np.NAN) for this given row
            row = [col for col in range(cur_val, cur_val + inverse - 1)] # Iterates through each column
            cur_val = row[-1] + 1 if len(row) > 0 else 0 # Updates starting point for next list
            row.extend([np.NAN for _ in range(unit)]) # Adds inverse np.NAN variables for all missing values in this list
            seq_one.append(row)
        seq_one = np.array(seq_one)

        # Generate Sequence 2
        flipped = np.fliplr(seq_one)
        seq_two = []
        for col in range(total_units):
            seq_two.append(flipped.diagonal(col))
        seq_two = seq_two[::-1]

        # Remove NaN from array
        seq_one_reshaped = []
        for row in seq_one:
            seq_one_reshaped.append(row[~np.isnan(row)])
        seq_one = seq_one_reshaped

        # Merge row-by-row ragged lists
        assert len(seq_one) == len(seq_two)
        merged = []
        for ind in range(len(seq_one)):
            merge_row = np.concatenate((seq_one[ind], seq_two[ind]))
            merged.append(merge_row)

        # Return final weight table
        return np.array(merged)

    @overrides
    @tf.function
    def call(self, inputs, *args, **kwargs):
        """
        Performs an auto-recursive forward propagation.
        Using the @tf.function decorator allows it to be
        ran as a graph, greatly accelerating training speed.
        """
        # Update inputs
        self.states[:self.input_shape].assign(inputs)
        
        # Begin propagation
        for _ in range(self.num_propagations):
            # Update each node's state
            for node in range(self.total_units):
                # Calculate the corresponding weight indexes using the weight table
                weight_inds = self.weight_table[node]
                
                # Get the neighboring nodes by getting all nodes minus the current one
                neighbor_nodes = tf.concat(self.states[:node], self.states[(node + 1):])

                # Calculate new state of node
                new_state = tf.tensordot(self.w[weight_inds], neighbor_nodes) + self.b[node] # TODO Accelerate this, also make sure that all of the neighboring nodes are in the same pattern as the weights are in
                if self.activation != None:
                    new_state = self.activation(new_state)
                self.new_states[node]
            
            # Update all states
            self.states.assign(self.new_states)

        # Returns output 
        return self.states[-self.output_units:]

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
