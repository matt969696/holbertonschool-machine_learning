#!/usr/bin/env python3
"""
Decode Module
"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    if one_hot.shape[0] == 0 or one_hot.shape[1] == 0:
        return None
    ret = np.argmax(one_hot, axis=0)
    return ret
