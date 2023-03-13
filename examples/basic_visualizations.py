#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""basic_visualizations.py: Creates some basic visualizations of the model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from rcm_layer import plot_rcm_layer, plot_and_save_logo

# Plots an RCM Layer of 12 nodes
plot_rcm_layer(12, colors=["limegreen", "darkgreen"], circle_color="green")

# Saves a version of the logo
PATH = "."
plot_and_save_logo(PATH)
