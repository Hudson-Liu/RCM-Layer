#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_layers.py: Runs basic unittests on the layers module"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import unittest
from rcm_layer import RCM
from keras import Input


class ShapeTest(unittest.TestCase):
    """Tests if TensorShapes of RCM layer are correct"""

    def test_output_shape(self):
        """Checks if outputs are the expected shape"""
        INPUTS, HIDDEN, OUTPUTS = 9, 5, 2
        rcm = self.__get_rcm(HIDDEN, OUTPUTS)(Input(shape=(INPUTS,)))
        correct = [None, OUTPUTS]
        actual = rcm.shape.as_list()
        self.assertEqual(correct, actual)

    def test_internal_weight_shape(self):
        """Checks if internal weight tensors are the expected shape"""
        INPUTS, HIDDEN, OUTPUTS = 9, 5, 2
        rcm = self.__get_rcm(HIDDEN, OUTPUTS)
        rcm.build(input_shape=(INPUTS,))
        correct = [INPUTS + HIDDEN + OUTPUTS, INPUTS + HIDDEN + OUTPUTS]
        actual = rcm.w.shape.as_list()
        self.assertEqual(correct, actual)
    
    def test_internal_bias_shape(self):
        """Checks if internal bias tensors are the expected shape"""
        INPUTS, HIDDEN, OUTPUTS = 9, 5, 2
        rcm = self.__get_rcm(HIDDEN, OUTPUTS)
        rcm.build(input_shape=(INPUTS,))
        correct = [INPUTS + HIDDEN + OUTPUTS]
        actual = rcm.b.shape.as_list()
        self.assertEqual(correct, actual)
    
    def __get_rcm(self, hidden, outputs):
        """Creates an RCM instance for testing"""
        return RCM(hidden, outputs, 952, "relu")


class ConstructorTest(unittest.TestCase):
    """Tests if constructor of RCM layer is correct"""

    def test_invalid_input_shape_handling(self):
        """Checks if __init__() can handle an invalid input shape pass"""
        RCM(5, 2, 952)

    def test_invalid_activation_handling(self):
        """Checks if __init__() can handle an invalid activation input"""

    def test_invalid_dropout_handling(self):
        """Checks if __init__() can handle an invalid dropout input"""

    def test_invalid_output_shape_handling(self):
        """Checks if __init__() can handle an invalid output shape"""


class ConfigTest(unittest.TestCase):
    """Tests if get_config of RCM layer is correct"""

if __name__ == '__main__':
    """Runs unittests"""
    unittest.main()