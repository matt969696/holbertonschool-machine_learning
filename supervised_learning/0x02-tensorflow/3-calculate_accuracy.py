#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    prediction = tf.math.argmax(y_pred, axis=1)
    equality = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32), name='Mean')
    return accuracy
