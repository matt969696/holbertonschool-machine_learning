#!/usr/bin/env python3
"""
Deep NeuralNetwork Module
Contains Deep class :  deep neural network performing binary classification
"""
import numpy as np
import pickle as pick


def sig(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax function"""
    return np.exp(x)


def tanha(x):
    """Tanh function"""
    return np.tanh(x)


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork Class implementation
    """
    def __init__(self, nx, layers, activation='sig'):
        """
        DNN Init : initialize a DNN object
        Attributes:
        nx (int) : number of input features
        layers (list of int) : number of nodes in each layer of the network
        activation : type of activation function used in the hidden layers
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
        if not all(map(lambda x: isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                tmp = nx
            else:
                tmp = layers[i - 1]
            self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], tmp)\
                * np.sqrt(2/tmp)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def activation(self):
        """Getter for activation attribute"""
        return self.__activation

    @property
    def L(self):
        """Getter for L attribute"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights attribute"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the DNN"""
        self.__cache["A0"] = X
        if self.activation == "sig":
            for i in range(self.L - 1):
                tmp = sig(np.matmul(self.weights["W" + str(i + 1)],
                                    self.cache["A" + str(i)]) +
                          self.weights["b" + str(i + 1)])
                self.__cache["A" + str(i + 1)] = tmp
            tmp = softmax(np.matmul(self.weights["W" + str(self.L)],
                                    self.cache["A" + str(self.L - 1)]) +
                          self.weights["b" + str(self.L)])
            sum = np.sum(tmp, axis=0)
            tmp = tmp / sum
            self.__cache["A" + str(self.L)] = tmp
        else:
            for i in range(self.L - 1):
                tmp = tanha(np.matmul(self.weights["W" + str(i + 1)],
                                      self.cache["A" + str(i)]) +
                            self.weights["b" + str(i + 1)])
                self.__cache["A" + str(i + 1)] = tmp
            tmp = softmax(np.matmul(self.weights["W" + str(self.L)],
                                    self.cache["A" + str(self.L - 1)]) +
                          self.weights["b" + str(self.L)])
            sum = np.sum(tmp, axis=0)
            tmp = tmp / sum
            self.__cache["A" + str(self.L)] = tmp
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        res = -1 / Y.shape[1] * np.sum(Y * np.log(A))
        return res

    def evaluate(self, X, Y):
        """Evaluates the NN’s predictions"""
        ret1, _ = self.forward_prop(X)
        ret2 = self.cost(Y, ret1)
        ret3 = np.rint(np.nextafter(ret1, ret1+1)).astype(int)
        return ret3, ret2

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the NN"""
        nbL = self.L
        m = cache["A0"].shape[1]
        for i in range(nbL, 0, -1):
            if i == nbL:
                dz = np.subtract(cache["A" + str(i)], Y)
            else:
                if self.activation == "sig":
                    dz = np.matmul(Wold.T, dzold) *\
                        np.multiply(cache["A" + str(i)],
                                    (1 - cache["A" + str(i)]))
                else:
                    dz = np.matmul(Wold.T, dzold) *\
                        (1 - np.multiply(cache["A" + str(i)],
                                         cache["A" + str(i)]))
            dzold = dz
            Wold = self.weights["W" + str(i)]
            dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights["W" + str(i)] = np.subtract(self.__weights["W" +
                                                                      str(i)],
                                                       np.multiply(alpha, dw))
            self.__weights["b" + str(i)] = np.subtract(self.__weights["b" +
                                                                      str(i)],
                                                       np.multiply(alpha, db))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the DNN"""
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

        A, cache = self.forward_prop(X)
        c = self.cost(Y, A)
        costs = [c]
        steps = [0]
        if verbose:
            print("Cost after {} iterations: {}".format(0, c))
        for i in range(1, iterations + 1):
            self.gradient_descent(Y, cache, alpha)
            A, cache = self.forward_prop(X)
            if i % step == 0:
                c = self.cost(Y, A)
                costs.append(c)
                steps.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
        if iterations % step != 0:
            c = self.cost(Y, A)
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

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as outp:
            pick.dump(self, outp)
            outp.close()

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        with open(filename, 'rb') as inp:
            ret = pick.load(inp)
            inp.close()
        return ret
