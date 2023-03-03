#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""visualizer.py: Contains methods for creating a visualization"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import matplotlib.pyplot as plt
from math import sin, cos, pi

def plot_circular_model(num_nodes):
    """Creates plot of circular  model"""
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

def plot_linear(nodes):
    """Part of the Node-To-Weight reference table creation"""
    b = []
    for x in range(nodes):
        a = []
        prev_adder = 0
        a.append(x)
        for y in range(1, nodes):#important part, replace a with node_inds
            if x >= y:
                adder = prev_adder + abs(y - nodes) - 1
                a.append(x + adder) #end of important part
                prev_adder = adder
        b.append(a) #change this to extend
    for e in b:
        plt.plot(e, color="red")
    plt.show()
#def plot_grid2d_model(input_nodes, output_nodes, side_length):
 #   pass


if __name__ == "__main__":
    plot_circular_model(20)
    plot_linear(20)
