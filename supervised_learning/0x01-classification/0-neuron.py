#!/usr/bin/env python3
"""
Neuron Module
Contains Neuron class :  single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Neuron Class implementation
    """
    def __init__(self, nx):
        """
        Neuron Init : initialize a Neuron object
        Attributes:
        nx (int) : number of input features to the neuron
        W : weights vector for the neuron
        b : bias for the neuron
        A : activated output of the neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
