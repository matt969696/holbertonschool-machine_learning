#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the
    RMSProp optimization algorithm"""
    sd = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (sd ** 0.5 + epsilon)
    return var, sd
