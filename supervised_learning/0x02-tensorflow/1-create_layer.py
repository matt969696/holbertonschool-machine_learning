#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """returns two placeholders, x and y, for the neural network"""
    winit = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=winit, name='layer')
    return layer(prev)
