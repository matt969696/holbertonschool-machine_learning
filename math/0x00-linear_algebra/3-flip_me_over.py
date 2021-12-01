#!/usr/bin/env python3
"""
This module contains a simple function
"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    transp = []
    w = len(matrix)
    h = len(matrix[0])
    for j in range(h):
        vect = []
        for i in range(w):
            vect.append(matrix[i][j])
        transp.append(vect)
    return transp
