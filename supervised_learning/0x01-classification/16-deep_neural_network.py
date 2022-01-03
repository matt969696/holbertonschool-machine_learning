#!/usr/bin/env python3
"""
Deep NeuralNetwork Module
Contains Deep class :  deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork Class implementation
    """
    def __init__(self, nx, layers):
        """
        DNN Init : initialize a DNN object
        Attributes:
        nx (int) : number of input features
        layers (list of int) : number of nodes in each layer of the network
        L : number of layers in the neural network
        cache : dictionary to hold all intermediary values of the network
        weights :dictionary to hold all weights and biased of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int), layers))):
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if i == 0:
                tmp = nx
            else:
                tmp = layers[i - 1]
            self.weights["W" + str(i + 1)] = np.random.randn(layers[i], tmp) *\
                np.sqrt(2/tmp)
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
