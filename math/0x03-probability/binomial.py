#!/usr/bin/env python3
"""
Binomial Module
Contains simple class that represents an binomial distribution
"""


def fact(n):
    """Simple factorial function"""
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


class Binomial:
    """
    Binomial Class : simple implementation
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Binomial Init : initialize an binomial distribution
        Attributes:
        data (list):  list of the data to be used to estimate the distribution
        n (int): number of Bernouilli trials
        p (float): probability of a success
        """
        self.data = data
        if data is None:
            self.n = round(n)
            self.p = float(p)
        else:
            mean = sum(self.data) / len(self.data)
            res = 0
            for i in range(len(self.data)):
                res += (self.data[i] - mean) ** 2
            var = res / len(self.data)
            p1 = 1 - var / mean
            self.n = round(mean / p1)
            self.p = float(mean / self.n)

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
    def n(self):
        """Getter for n attribute"""
        return self.__n

    @n.setter
    def n(self, value):
        """Setter for n attribute of Binomial object"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("n must be a positive value")
        else:
            self.__n = int(round(value))

    @property
    def p(self):
        """Getter for p attribute"""
        return self.__p

    @p.setter
    def p(self, value):
        """Setter for stddev attribute of Normal object"""
        if not isinstance(value, (int, float)) or value <= 0 or value >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        else:
            self.__p = float(value)

    def pmf(self, k):
        """Calculates the value of the PDF for a given x-value"""
        if not isinstance(k, (int, float)) or k < 0:
            return 0
        k = int(k)
        res = fact(self.n) / (fact(k) * fact(self.n-k)) *\
            self.p ** k * (1 - self.p) ** (self.n - k)
        return res

    def cdf(self, k):
        """Calculates the value of the CDF for a given x-value"""
        if not isinstance(k, (int, float)) or k < 0:
            return 0
        k = int(k)
        res = 0
        for i in range(k + 1):
            res += self.pmf(i)
        return res
