#!/usr/bin/python3
"""
This module contains a simple function
"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    add = []
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])
    return(add)
