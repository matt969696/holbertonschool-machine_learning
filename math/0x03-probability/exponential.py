#!/usr/bin/env python3
"""
Exponential Module
Contains simple class that represents an exponential distribution
"""


def fact(n):
    """Simple factorial function"""
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


class Exponential:
    """
    Exponential Class : simple implementation
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Exponential Init : initialize an exponential distribution
        Attributes:
        data (list):  list of the data to be used to estimate the distribution
        lambtha (float): expected number of occurences in a given time frame
        """
        self.data = data
        if data is None:
            self.lambtha = float(lambtha)
        else:
            self.lambtha = float(len(self.data) / sum(self.data))

    @property
    def data(self):
        """Getter for data attribute"""
        return self.__data

    @data.setter
    def data(self, value):
        """Setter for data attribute of Exponential object"""
        if value is None:
            self.__data = None
        else:
            if not isinstance(value, list):
                raise TypeError("data must be a list")
            if len(value) < 2:
                raise ValueError("data must contain multiple values")
        self.__data = value

    @property
    def lambtha(self):
        """Getter for lambda attribute"""
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        """Setter for lambtha attribute of Poisson object"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("lambtha must be a positive value")
        else:
            self.__lambtha = float(value)

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        e = 2.7182818285
        if not isinstance(x, (int, float)):
            return 0
        if x < 0:
            return 0
        res = self.lambtha * e ** (-self.lambtha * x)
        return res

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        e = 2.7182818285
        if not isinstance(x, (int, float)):
            return 0
        if x < 0:
            return 0
        res = 1 - e ** (-self.lambtha * x)
        return res
