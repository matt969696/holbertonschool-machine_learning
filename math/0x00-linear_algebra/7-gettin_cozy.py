#!/usr/bin/python3
"""
This module contains a simple function
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        cat_mat = mat1 + mat2
        return cat_mat

    if len(mat1) != len(mat2):
        return None
    cat_mat = []
    for i in range(len(mat1)):
        cat_mat.append(mat1[i] + mat2[i])
    return cat_mat
