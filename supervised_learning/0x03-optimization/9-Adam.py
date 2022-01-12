#!/usr/bin/env python3
"""
This module contains a simple function
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable using the
    Adam optimization algorithm"""
    vd = beta1 * v + (1 - beta1) * grad
    sd = beta2 * s + (1 - beta2) * grad ** 2
    vdc = vd / (1 - beta1 ** t)
    sdc = sd / (1 - beta2 ** t)
    var = var - alpha * vdc / (sdc ** 0.5 + epsilon)
    return var, vd, sd
