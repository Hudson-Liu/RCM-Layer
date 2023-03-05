#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""enum_types.py: Contains useful enums for model configuration"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from enum import Enum, auto


class ModelShapes(Enum):
    """Represents different possible model shapes"""
    CIRCULAR = auto() # Each node is directly connected to all other nodes; similar to Boltzmann machine
    GRID_2D = auto() # Similar to SQUARE_DENSE, but each node is only connected to it's neighbors (with diagonals)
    GRID_3D = auto() # Nodes arranged in a 3D grid shape (with diagonals)
    SQUARE_DENSE = auto() # A normal dense network with a square-like structure
