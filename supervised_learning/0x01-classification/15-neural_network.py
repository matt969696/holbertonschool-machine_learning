#!/usr/bin/env python3
"""
NeuralNetwork Module
Contains NN class :  multiple neurons performing binary classification
"""
import numpy as np


def sig(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the NN"""
        self.__A1 = sig(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = sig(np.matmul(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        res = -1 / Y.shape[1] * (np.matmul(Y, np.log(A).T) +
                                 np.matmul((1 - Y), np.log(1.0000001 - A).T))
        return res[0, 0]

    def evaluate(self, X, Y):
        """Evaluates the NN’s predictions"""
        _, ret1 = self.forward_prop(X)
        ret2 = self.cost(Y, ret1)
        ret3 = np.rint(np.nextafter(ret1, ret1+1)).astype(int)
        return ret3, ret2

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the NN"""
        m = X.shape[1]
        dz2 = np.subtract(A2, Y)
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * np.multiply(A1, (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = np.subtract(self.__W2, np.multiply(alpha, dw2))
        self.__b2 = np.subtract(self.__b2, np.multiply(alpha, db2))
        self.__W1 = np.subtract(self.__W1, np.multiply(alpha, dw1))
        self.__b1 = np.subtract(self.__b1, np.multiply(alpha, db1))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the NN"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        A1, A2 = self.forward_prop(X)
        c = self.cost(Y, A2)
        costs = [c]
        steps = [0]
        if verbose:
            print("Cost after {} iterations: {}".format(0, c))
        for i in range(1, iterations + 1):
            self.gradient_descent(X, Y, A1, A2, alpha)
            A1, A2 = self.forward_prop(X)
            if i % step == 0:
                c = self.cost(Y, A2)
                costs.append(c)
                steps.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
        if iterations % step != 0:
            c = self.cost(Y, A2)
            costs.append(c)
            steps.append(iterations)
            if verbose:
                print("Cost after {} iterations: {}".format(iterations, c))

        if graph:
            plt.plot(steps, costs)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()

        return self.evaluate(X, Y)
