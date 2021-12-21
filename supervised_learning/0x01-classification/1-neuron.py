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
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W attribute"""
        return self.__W

    @property
    def b(self):
        """Getter for b attribute"""
        return self.__b

    @property
    def A(self):
        """Getter for A attribute"""
        return self.__A
