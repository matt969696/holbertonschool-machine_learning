#!/usr/bin/python3
"""
This module contains a simple function
"""


def np_slice(matrix, axes={}):
    """slices a matrix along specific axes"""
    temp = matrix
    slc = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        slc[key] = slice(*value)
    return temp[tuple(slc)]
