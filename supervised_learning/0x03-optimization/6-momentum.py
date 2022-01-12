#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation for a neural network
    in tensorflow using the gradient descent
    with momentum optimization algorithm"""
    ret = tf.train.MomentumOptimizer(alpha, beta1)
    return ret.minimize(loss)
