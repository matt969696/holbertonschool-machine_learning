#!/usr/bin/env python3
"""
This module contains a simple function
"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(e, (int, float)) for e in poly):
        return None
    if not isinstance(C, int):
        return None
    integ = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
    while integ[-1] == 0 and len(integ) > 1:
        integ.pop()
    return integ
