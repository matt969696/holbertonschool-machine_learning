#!/usr/bin/env python3
"""
Poisson Module - Contains simple class dthat represents a poisson distribution
"""


def fact(n):
    """Simple factorial function"""
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


class Poisson:
    """
    Poisson Class : simple implementation
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Poisson Init : initialize a poisson distribution
        Attributes:
        data (list):  list of the data to be used to estimate the distribution
        lambtha (float): expected number of occurences in a given time frame
        """
        self.data = data
        if data is None:
            self.lambtha = float(lambtha)
        else:
            self.lambtha = float(sum(self.data) / len(self.data))

    @property
    def data(self):
        """Getter for data attribute"""
        return self.__data

    @data.setter
    def data(self, value):
        """Setter for data attribute of Poisson object"""
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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        e = 2.7182818285
        if not isinstance(k, (int, float)):
            return 0
        if k < 0:
            return 0
        n = int(k)
        res = self.lambtha ** n * e ** (-self.lambtha) / fact(n)
        return res

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if not isinstance(k, (int, float)):
            return 0
        if k < 0:
            return 0
        n = int(k)
        res = 0
        for i in range(n + 1):
            res += self.pmf(i)
        return res
