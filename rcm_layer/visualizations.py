#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualizations.py: Contains methods for creating a visualization"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import matplotlib.pyplot as plt
from math import sin, cos, pi

def plot_rcm_layer(num_nodes: int):
    """Creates plot of Recurrent Complete Multidigraph model"""
    # Initialize constants
    RADIUS = 0.1
    NUM_DOUBLE_LINES = 2
    COLORS = ["red", "blue"]

    # Create list of angles for all nodes
    angles = [(2 * pi * x) / num_nodes for x in range(num_nodes)]

    # Generate points and create lines
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            color = COLORS[0]
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
                color = COLORS[1] if color == COLORS[0] else COLORS[0]
    
    # Add circles for nodes
    for i in range(num_nodes):
        circle = plt.Circle((cos(angles[i]), sin(angles[i])), RADIUS, facecolor="white", edgecolor="red", zorder=10, linewidth=3)
        plt.gca().add_artist(circle)

    # Plot model
    plt.gca().set_aspect(1)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis('off')
    # plt.savefig('logo.svg', transparent=True, format="svg", bbox_inches="tight") # Uncomment this to generate logo
    plt.show()

# TODO add "plot_rcm_3d"
# TODO remove this
plot_rcm_layer(8)