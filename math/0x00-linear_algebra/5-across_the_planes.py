#!/usr/bin/env python3
"""
This module contains a simple function
"""


def add_matrices2D(mat1, mat2):
    """adds two matrices element-wise"""
    w1 = len(mat1)
    h1 = len(mat1[0])
    w2 = len(mat2)
    h2 = len(mat2[0])

    if w1 != w2 or h1 != h2:
        return None
    add = []
    for i in range(w1):
        vect = []
        for j in range(h1):
            vect.append(mat1[i][j] + mat2[i][j])
        add.append(vect)
    return(add)
