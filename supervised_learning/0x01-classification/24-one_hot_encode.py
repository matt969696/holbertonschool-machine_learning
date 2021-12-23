#!/usr/bin/env python3
"""
Encode Module
"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if type(classes) is not int:
        return None
    if classes <= 0:
        return None
    if type(Y) is not np.ndarray:
        return None
    ret = np.eye(classes)[Y]
    return ret.T
