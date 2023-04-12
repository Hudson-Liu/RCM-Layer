#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
layerless_supremacy.py: Proves the superiority of layerless networks. 
This file generates diagrams used throughout the documentation
"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import math
from typing import NamedTuple, Callable
import numpy as np
from overrides import overrides
import tensorflow as tf
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
import random


class Equation(NamedTuple):
    """Represents an equation with a set interval and sample frequency"""
    f: Callable # Function
    interval: tuple[float, float] # Interval
    samp_freq: float # Sample Frequency


class Layerless(tf.keras.layers.Layer):
    """A Keras implementation of a Layerless Layer"""
    
    @overrides
    def __init__(self, 
            hidden_units: int, 
            output_units: int, 
            num_propagations: int,
            num_connections: int,
            activation: str = "relu",
            weight_initializer: str = "glorot_uniform",
            **kwargs
        ):
        """Initializes params necessary for Jimmy MK IV"""
        super().__init__(**kwargs)
        
        # Initialize parameters
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_propagations = num_propagations
        self.num_connections = num_connections
        self.weight_initializer = weight_initializer
        self.activation = activation

    @overrides
    def build(self, input_shape):
        """Creates weights, biases, and untracked node states"""
        # Generates input shape and total units and saves offset for usage in call
        self.input_nodes = input_shape[-1]
        total_units = self.hidden_units + self.output_units + self.input_nodes
        self.offset = self.hidden_units + self.output_units
        
        # Generates connection table
        self.generate_connection_table(total_units)

        # Weights and biases
        self.w = self.add_weight(
            shape=(self.num_connections,),
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
    
    def generate_connection_table(self, num_nodes: int):
        """Generates a random assortment of connections"""
        # Randomly generates unique pairs of nodes
        connections = []
        for _ in range(self.num_connections):
            random_pair = (random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1))
            if not random_pair in connections:
                connections.append(random_pair)
        return connections

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
            # MODIFY THIS
            # Calculate new states
            states = tf.nn.bias_add(tf.matmul(states, self.w), self.b)
            
            # Apply activation function
            states = self.activations(states)

        # Returns output
        return states[:, -self.output_units:]

def make_dense(node_list: list[int]):
    """Procedurally generate Dense layer"""
    inputs = Input(shape=(1,))
    dense = Dense(node_list[1], activation="relu")(inputs)
    for num_nodes in node_list[2:]:
        dense = Dense(num_nodes, activation="relu")(dense)
    model = tf.keras.Model(inputs=inputs, outputs=dense, name="Dense_Model")
    return model

def make_layerless(num_connections: int, propagations: int):
    """Generate layerless model"""
    inputs = Input(shape=(1,))
    outputs = Layerless(NUM_NODES - 2, 10, num_propagations=propagations, num_connections=num_connections)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Layerless_Model")
    return model
    
def train_model(model: tf.keras.Model, x: np.ndarray, y: np.ndarray):
    """Train model (Dense or RCM)"""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    history = model.fit(x, y, epochs=100, verbose=0)
    return history


# Each equation is represented by an anonymous function
FUNCS: list[Equation] = [
    # Transcendentals
    Equation(lambda x: math.sin(x), (-10, 10), 0.2),
    Equation(lambda x: math.log(x), (0.5, 4), 0.0035),
    # Polynomials
    Equation(lambda x: x ** 2, (-2, 2), 0.04),
    Equation(lambda x: x ** 3, (-2, 2), 0.04),
    # Composites/Piecewise
    Equation(lambda x: -1 if x < 0 else 1, (-1, 1), 0.02),
    Equation(lambda x: 3 * (x ** 3) + 2 * (x ** 2) - 6 * x + 2 if x < 0 else x ** 2, (-2, 2), 0.04)
]

# Define hyperparameters
NUM_NODES = 9
MAX_PROPAGATIONS = 10

# Generate all possible hidden Layer configurations
layer_permutations = []
int_set = [i for i in range(1, NUM_NODES - 1)]
for i in tqdm(range(1, NUM_NODES - 1), desc="Permutation Generation"):
    for s in tqdm(itertools.product(int_set, repeat=i), total=len(int_set) ** i, desc="Permutations", leave=False): 
        if sum(s) == NUM_NODES - 2: 
            layer_permutations.append(s)

# Add input and ouput to every element
for ind, val in enumerate(layer_permutations):
    layer_permutations[ind] = (1,) + val + (1,)

# Iterates through equations and runs tests
for func in FUNCS:
    # Generate dataset
    data = np.array([[x, func.f(x)] for x in np.arange(func.interval[0], func.interval[1], func.samp_freq)])
    np.random.shuffle(data)
    x, y = data[:, 0], data[:, 1]

    # Iterate through each possible Dense configuration
    all_params, mae = [], []
    for node_list in layer_permutations:
        # Create dense layer
        model = make_dense(node_list)
        history = train_model(model, x, y)
        
        # Calculate number of weight parameters
        num_params = 0
        prev_val = 1
        for val in node_list[1:]:
            num_params += val * prev_val
            prev_val = val

        # Print corresponding info
        print(f"Model Configuration: {node_list} ({num_params} params) || Mean Absolute Error: {history.history['mae'][-1]}")

        # Save info for creating graphs
        all_params.append(num_params)
        mae.append(history.history['mae'][-1])

    # Calculate best graph approximation of line by retraining
    best_node_list = layer_permutations[np.argmin(mae)]
    model = make_dense(best_node_list)
    history = train_model(model, x, y)
    y_pred = []
    x_pred = np.arange(func.interval[0], func.interval[1], func.samp_freq / 10)
    for val in tqdm(x_pred, desc="Calculating Predictions"):
        y_pred.append(model.predict([val], verbose=0))
    plt.scatter(x_pred, y_pred)
    plt.scatter(x, y)
    plt.title(f"Best Line Approximation {best_node_list}")
    plt.show()
            
    # Plot training progression of highest accuracy model
    plt.plot(history.history['mae'])
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Best Model's Training History")
    plt.show()

    # Plot of number of parameters vs final mae
    plt.scatter(all_params, mae)
    plt.xlabel("Number of Parameters")
    plt.ylabel("Final MAE")
    plt.title("Num Parameters VS Final MAE")
    plt.show()

    # Create layerless networks based off of best Dense structure
    best_params = all_params[np.argmin(mae)]
    for prop in range(1, MAX_PROPAGATIONS + 1):
        model = make_layerless(best_params, prop)
        history = train_model(model, x, y)
        print(f"Model Configuration: ({prop} propagations) || Mean Absolute Error: {history.history['mae'][-1]}")