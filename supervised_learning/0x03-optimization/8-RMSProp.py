#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm"""
    ret = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return ret.minimize(loss)
