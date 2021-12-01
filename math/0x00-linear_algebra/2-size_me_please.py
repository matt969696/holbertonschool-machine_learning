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
