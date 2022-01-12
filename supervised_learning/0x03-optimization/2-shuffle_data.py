#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    np.random.seed(0)
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]
