#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualizer.py: Contains methods for creating a visualization"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import matplotlib.pyplot as plt
from math import sin, cos, pi

def plot_model(num_nodes):
    """Creates plot of the model"""
    # Create list of angles for all nodes
    angles = [(2 * pi * x) / num_nodes for x in range(num_nodes)]

    # Generate points and create lines
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            line_x = [cos(angles[i]), cos(angles[j])]
            line_y = [sin(angles[i]), sin(angles[j])]
            plt.plot(line_x, line_y, color="red", linewidth=1)
    
    # Add circles for nodes
    RADIUS = 0.075
    for i in range(num_nodes):
        circle = plt.Circle((cos(angles[i]), sin(angles[i])), RADIUS, facecolor="white", edgecolor="red", zorder=10)
        plt.gca().add_artist(circle)

    # Plot model
    plt.gca().set_aspect(1)
    plt.title("Jimmy MK IV")
    plt.show()


if __name__ == "__main__":
    plot_model(30)
