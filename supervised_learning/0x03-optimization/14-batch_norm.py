#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network in tensorflow"""
    winit = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    baselayer = tf.keras.layers.Dense(n, kernel_initializer=winit)
    mu, sig = tf.nn.moments(baselayer(prev), axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8
    layer = tf.nn.batch_normalization(baselayer(prev), mu,
                                      sig, beta, gamma, epsilon)
    return activation(layer)
