#!/usr/bin/env python3
"""
NeuralNetwork Module
Contains NN class :  multiple neurons performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    NeuronNetwork Class implementation
    """
    def __init__(self, nx, nodes):
        """
        NN Init : initialize a NN object
        Attributes:
        nx (int) : number of input features
        nodes (int) : number of nodes found in the hidden layer
        W1 : weights vector for the hidden layer
        b1 : bias for the hidden layer
        A1 : activated output for the hidden layer
        W2 : weights vector for the output neuron
        b2 : bias for the output neuron
        A2 : activated output for the output neuron (prediction)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1 attribute"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1 attribute"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1 attribute"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2 attribute"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2 attribute"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2 attribute"""
        return self.__A2
