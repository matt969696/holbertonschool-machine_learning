#!/usr/bin/env python3
"""
This module contains a simple function
"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    if not all(isinstance(e, (int, float)) for e in poly):
        return None
    deriv = [poly[i] * i for i in range(1, len(poly))]
    return deriv
