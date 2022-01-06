#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    res = x
    for i in range(len(layer_sizes)):
        res = create_layer(res, layer_sizes[i], activations[i])
    return res
