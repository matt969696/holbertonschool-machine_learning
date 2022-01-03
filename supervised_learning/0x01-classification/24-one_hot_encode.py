#!/usr/bin/env python3
"""
Encode Module
"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    if len(Y.shape) != 1:
        return None
    try:
        hot = np.eye(classes)[Y]
        return hot.T
    except Exception:
        return None
