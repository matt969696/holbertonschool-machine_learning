#!/usr/bin/env python3
"""
This module contains a simple function
"""


def np_elementwise(mat1, mat2):
    """ performs element-wise addition, subtraction, mult and division"""
    madd = mat1 + mat2
    msub = mat1 - mat2
    mmul = mat1 * mat2
    mdiv = mat1 / mat2
    return (madd, msub, mmul, mdiv)
