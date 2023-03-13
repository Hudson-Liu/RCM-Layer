#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualizations.py: Contains methods for creating a visualization"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import matplotlib.pyplot as plt
from math import sin, cos, pi

def plot_rcm_layer(num_nodes: int, colors: list[str] = ["red", "blue"], circle_color = "red"):
    """Creates plot of Recurrent Complete Multidigraph model"""
    _base_plot(num_nodes, colors, circle_color)
    plt.show()

def plot_and_save_logo(save_path: str):
    """Generates the RCM logo"""
    _base_plot(8, ["maroon", "tomato"], circle_color="red")
    plt.savefig(f'{save_path}/logo.svg', transparent=True, format="svg", bbox_inches="tight")

def _base_plot(num_nodes: int, colors: list[str], circle_color: str):
    """Creates plot of Recurrent Complete Multidigraph model"""
    # Initialize constants
    RADIUS = 0.1
    NUM_DOUBLE_LINES = 2

    # Assert that there's the correct amount of colors
    if len(colors) != NUM_DOUBLE_LINES:
        raise ValueError(f"There can only be {NUM_DOUBLE_LINES} colors, you entered {len(colors)}.")

    # Create list of angles for all nodes
    angles = [(2 * pi * x) / num_nodes for x in range(num_nodes)]

    # Generate points and create lines
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            color = colors[0]
            for line_num in range(NUM_DOUBLE_LINES):
                # Augment the angles by a certain offset denoted in radians
                offset = ((RADIUS * line_num) / (NUM_DOUBLE_LINES - 1)) - (RADIUS / 2)
                theta_1 = angles[i] + offset
                theta_2 = angles[j] - offset
                
                # Get the two points's x and y coords
                line_x = [cos(theta_1), cos(theta_2)]
                line_y = [sin(theta_1), sin(theta_2)]
                
                # Plot the line
                plt.plot(line_x, line_y, color=color, linewidth=3)
                
                # Swap the colors
                color = colors[1] if color == colors[0] else colors[0]
    
    # Add circles for nodes
    for i in range(num_nodes):
        circle = plt.Circle((cos(angles[i]), sin(angles[i])), RADIUS, facecolor="white", edgecolor=circle_color, zorder=10, linewidth=3)
        plt.gca().add_artist(circle)

    # Plot model
    plt.gca().set_aspect(1)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis('off')

# TODO add "plot_rcm_3d"

# TODO add "plot_from_model"; plots from keras model trained instance (figure out a way to detect from this layer specifically, and color weights accordingly)
