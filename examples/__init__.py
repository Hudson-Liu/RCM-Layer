#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__.py: Allows the usage of the RCM class in examples w/o being placed in the same directory or having RCM installed to path"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
