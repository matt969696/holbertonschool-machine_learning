#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    res = tf.losses.softmax_cross_entropy(y, y_pred)
    return res
