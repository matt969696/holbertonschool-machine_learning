#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
    using batch normalization"""
    m = Z.shape[0]
    mu = np.mean(Z, axis=0)
    sig = np.std(Z, axis=0)
    Znorm = (Z - mu) / (sig ** 2 + epsilon) ** 0.5
    res = gamma * Znorm + beta
    return res
