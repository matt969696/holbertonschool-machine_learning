#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    ret = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon=epsilon)
    return ret.minimize(loss)
