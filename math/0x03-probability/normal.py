#!/usr/bin/env python3
"""
Normal Module
Contains simple class that represents an exponential distribution
"""


def fact(n):
    """Simple factorial function"""
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


def erf(x):
    """Simple error function approximation"""
    pi = 3.1415926536
    res = 2 * (x - x**3 / 3 + x**5 / 10 - x**7 / 42 + x**9 / 216) / pi**0.5
    return res


class Normal:
    """
    Normal Class : simple implementation
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Exponential Init : initialize an exponential distribution
        Attributes:
        data (list):  list of the data to be used to estimate the distribution
        mean (float): mean of the distribution
        stddev (float): standard deviation of the distribution
        """
        self.data = data
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            self.mean = float(sum(self.data) / len(self.data))
            res = 0
            for i in range(len(self.data)):
                res += (self.data[i] - self.mean) ** 2
            self.stddev = float((res / len(self.data)) ** 0.5)

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
    def mean(self):
        """Getter for mean attribute"""
        return self.__mean

    @mean.setter
    def mean(self, value):
        """Setter for mean attribute of Normal object"""
        if not isinstance(value, (int, float)):
            raise ValueError("mean must be a float or int")
        else:
            self.__mean = float(value)

    @property
    def stddev(self):
        """Getter for stddev attribute"""
        return self.__stddev

    @stddev.setter
    def stddev(self, value):
        """Setter for stddev attribute of Normal object"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("stddev must be a positive value")
        else:
            self.__stddev = float(value)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        res = (x - self.mean) / self.stddev
        return res

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        res = self.mean + z * self.stddev
        return res

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        e = 2.7182818285
        pi = 3.1415926536
        if not isinstance(x, (int, float)):
            return 0
        res = e ** (-0.5 * self.z_score(x) ** 2) /\
            (self.stddev * (2 * pi) ** 0.5)
        return res

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        if not isinstance(x, (int, float)):
            return 0
        res = 0.5 * (1 + erf((x - self.mean) / (self.stddev * 2 ** 0.5)))
        return res
