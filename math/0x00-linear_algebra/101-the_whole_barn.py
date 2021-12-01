#!/usr/bin/env python3
"""
This module contains 2 simple functions
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


def add_matrices(mat1, mat2):
    """adds two matrices"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    if len(shape1) == 1:
        add = []
        for i in range(shape1[0]):
            add.append(mat1[i] + mat2[i])
        return add
    add = []
    for i in range(shape1[0]):
        add.append(add_matrices(mat1[i], mat2[i]))
    return add
