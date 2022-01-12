#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    p = np.random.permutation(len(X))
    return X[p], Y[p]
