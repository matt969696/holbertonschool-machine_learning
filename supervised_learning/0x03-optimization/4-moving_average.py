#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    ewa = 0
    res = []
    for i in range(len(data)):
        ewa = beta * ewa + (1 - beta) * data[i]
        res.append(ewa)
    res2 = [res[i] / (1 - beta ** (i + 1)) for i in range(len(data))]
    return res2
