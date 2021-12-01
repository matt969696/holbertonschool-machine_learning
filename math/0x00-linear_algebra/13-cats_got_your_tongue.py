#!/usr/bin/python3
"""
This module contains a simple function
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    cat = np.concatenate((mat1, mat2), axis=axis)
    return cat
