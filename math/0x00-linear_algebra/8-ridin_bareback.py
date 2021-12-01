#!/usr/bin/python3
"""
This module contains a simple function
"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    w1 = len(mat1)
    h1 = len(mat1[0])
    w2 = len(mat2)
    h2 = len(mat2[0])

    if h1 != w2:
        return None
    mult = []
    for i in range(w1):
        vect = []
        for j in range(h2):
            elem = 0
            for k in range(h1):
                elem += mat1[i][k] * mat2[k][j]
            vect.append(elem)
        mult.append(vect)
    return(mult)
