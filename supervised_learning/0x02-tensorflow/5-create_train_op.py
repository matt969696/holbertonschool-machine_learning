#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    res = tf.train.GradientDescentOptimizer(alpha,
                                      name='GradientDescent')
    return res.minimize(loss)
