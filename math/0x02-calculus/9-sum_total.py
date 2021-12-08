#!/usr/bin/env python3
"""
This module contains a simple function
"""


def summation_i_squared(n):
    """calculates the sum of i^2"""
    if not isinstance(n, int):
        return None
    if n < 1:
        return None
    sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sum)
