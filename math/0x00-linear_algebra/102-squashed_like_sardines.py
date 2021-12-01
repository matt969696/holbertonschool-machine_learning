#!/usr/bin/env python3
"""
This module contains a simple function
"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    shape = []
    if matrix == []:
        return shape
    temp = matrix
    while type(temp) != int and type(temp) != float:
        shape.append(len(temp))
        temp = temp[0]
    return shape


def cat_rec(mat1, mat2, axis):
    """does the efective concatenation"""
    if axis == 0:
        return(mat1 + mat2)
    cat = []
    for i in range(len(mat1)):
        cat.append(cat_rec(mat1[i], mat2[i], axis - 1))
    return cat


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    if axis >= len(shape1):
        return None
    for i in range(len(shape1)):
        if shape1[i] != shape2[i] and i != axis:
            return None
    return cat_rec(mat1, mat2, axis)
