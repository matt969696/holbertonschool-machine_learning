#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    res = (X - m) / s
    return res
