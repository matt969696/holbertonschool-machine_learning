#!/usr/bin/env python3
"""
Neuron Module
Contains Neuron class :  single neuron performing binary classification
"""
import numpy as np


def sig(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A = sig(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        res = -1 / Y.shape[1] * (np.matmul(Y, np.log(A).T) +
                                 np.matmul((1 - Y), np.log(1.0000001 - A).T))
        return res[0, 0]

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        ret1 = self.forward_prop(X)
        ret2 = self.cost(Y, ret1)
        ret3 = np.rint(np.nextafter(ret1, ret1+1)).astype(int)
        return ret3, ret2
